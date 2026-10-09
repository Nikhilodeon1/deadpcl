"""XGBoost baseline on aggregate features (fixed hyperparameters, no tuning), CPU.

Same features as src/baselines.py (_aggregate_features: mean/min/max/observation
rate per variable) and the same hyperparameters; the features are computed here
because that function imports a module that does not exist in this repository.
Trains on the same 80% Site A split as the neural arms for the same seed, scores
the source validation split and the target sites, and saves per-stay predictions
(hashed id, label, score) under RB_OUT/predictions/xgb_s<seed>/.

    python rebuttal/scripts/run_xgb.py --seeds 42 43 44 45 46 47 48 49 50 51
"""
import argparse
import json
import os

import numpy as np

import rb_common as rb


def features(samples):
    rows, ys, ids = [], [], []
    for s in samples:
        x = np.asarray(s["values"], dtype=np.float32)     # normalised values, zeros where unobserved
        m = np.asarray(s["mask"], dtype=bool)
        row = []
        for v in range(x.shape[1]):
            obs = x[m[:, v], v]
            row.extend([0.0] * 4 if len(obs) == 0 else [obs.mean(), obs.min(), obs.max(), m[:, v].mean()])
        rows.append(row)
        ys.append(float(np.sum(s["label"]) > 0))
    return np.array(rows, dtype=np.float32), np.array(ys, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--sites", nargs="+", default=["site_b", "mimic", "eicu"])
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from src.data.dataset import ICUDataset, make_patient_split_loaders

    names = {"site_b": "physionet_b", "mimic": "mimic", "eicu": "eicu"}
    src = rb.load_site_cached("physionet_a", 1.0, 0)
    ds = ICUDataset(src)
    tgt = {k: rb.load_site_cached(names[k], 1.0, 0) for k in args.sites}
    tgt_feat = {k: features(v) for k, v in tgt.items()}
    tgt_ids = {k: [rb.hash_id(k, s.get("patient_id", ""), s.get("stay_id", 0)) for s in v] for k, v in tgt.items()}
    X_all, y_all = features(src)

    for seed in args.seeds:
        tag = f"xgb_s{seed}"
        out = os.path.join(rb.RB_OUT, "runs", f"{tag}.json")
        if os.path.exists(out):
            continue
        # same split call as the neural arms (patient-level, stratified, seeded)
        tr, va = make_patient_split_loaders(ds, train_frac=0.8, batch_size=64, seed=seed, stratify_key=rb.TASK)
        tr_idx, va_idx = list(tr.dataset.indices), list(va.dataset.indices)
        sc = StandardScaler().fit(X_all[tr_idx])
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, eval_metric="logloss", random_state=seed, verbosity=0)
        clf.fit(sc.transform(X_all[tr_idx]), y_all[tr_idx])
        pdir = os.path.join(rb.RB_OUT, "predictions", tag)
        os.makedirs(pdir, exist_ok=True)
        metrics = {}
        sets = {"site_a_val": (X_all[va_idx], y_all[va_idx],
                               [rb.hash_id("site_a", src[i].get("patient_id", ""), src[i].get("stay_id", 0)) for i in va_idx])}
        for k, (X, y) in tgt_feat.items():
            sets[k] = (X, y, tgt_ids[k])
        for k, (X, y, ids) in sets.items():
            p = clf.predict_proba(sc.transform(X))[:, 1]
            np.savez(os.path.join(pdir, f"{k}.npz"), hashed_id=np.array(ids), label=y, score=p, site=k)
            metrics[k] = {"n": int(len(y)), "n_pos": int(y.sum()), "auroc": float(roc_auc_score(y, p)),
                          "auprc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
                          "ece10": rb.ece(y, p)}
        rb.write_json(out, {"tag": tag, "seed": seed, "model": "xgboost", "metrics": metrics})
        rb.logging.info(f"{tag}: " + ", ".join(f"{k} {v['auroc']:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
