"""Fixed-hyperparameter XGBoost on per-variable aggregates (mean, min, max, observation rate),
the same recipe as src/baselines.py::run_classical_baselines (200 trees, depth 4, lr 0.05,
subsample 0.8, colsample 0.8, no tuning, no class weighting, standardized features), trained on
the same 80% training split of Site A as the neural arms and scored on Site B, MIMIC-IV and eICU.
CPU only; runs where the cached data live. Saves per-stay predictions (hashed id, label, score).

    python rebuttal/scripts/xgb_local.py --seeds 42 43 44 45 46 47 48 49 50 51
"""
import argparse
import os

import numpy as np

import rb_common as rb


def features(samples, site=""):
    X = np.empty((len(samples), 0), dtype=np.float32)
    rows = []
    for s in samples:
        x = np.asarray(s["values"], dtype=np.float32)
        m = np.asarray(s["mask"], dtype=bool)
        row = []
        for v in range(x.shape[1]):
            obs = x[m[:, v], v]
            if len(obs) == 0:
                row.extend([0.0, 0.0, 0.0, 0.0])
            else:
                row.extend([obs.mean(), obs.min(), obs.max(), m[:, v].mean()])
        rows.append(row)
    y = np.array([float(np.sum(s["label"]) > 0) for s in samples], dtype=np.float32)
    ids = [rb.hash_id(site, str(s.get("patient_id", "")), s.get("stay_id", 0)) for s in samples]
    return np.array(rows, dtype=np.float32), y, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--sites", nargs="+", default=["site_b", "mimic", "eicu"])
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from config import BATCH_SIZE
    from src.data.dataset import ICUDataset, make_patient_split_loaders
    from sklearn.metrics import roc_auc_score

    names = {"site_b": "physionet_b", "mimic": "mimic", "eicu": "eicu"}
    src = rb.load_site_cached("physionet_a", 1.0, 0)
    Xa, ya, _ = features(src)
    feats = {}
    for site in args.sites:
        feats[site] = features(rb.load_site_cached(names[site], 1.0, 0), site)
    ds = ICUDataset(src)

    for seed in args.seeds:
        tag = f"xgb_s{seed}"
        out = os.path.join(rb.RB_OUT, "runs", f"{tag}.json")
        if os.path.exists(out):
            continue
        tr, _ = make_patient_split_loaders(ds, train_frac=0.8, batch_size=BATCH_SIZE, seed=seed,
                                           stratify_key=rb.TASK)
        idx = np.array(tr.dataset.indices)
        sc = StandardScaler().fit(Xa[idx])
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, eval_metric="logloss", random_state=seed, verbosity=0)
        clf.fit(sc.transform(Xa[idx]), ya[idx])
        metrics = {}
        for site, (X, y, ids) in feats.items():
            p = clf.predict_proba(sc.transform(X))[:, 1]
            d = os.path.join(rb.RB_OUT, "predictions", tag)
            os.makedirs(d, exist_ok=True)
            np.savez(os.path.join(d, f"{site}.npz"), hashed_id=np.array(ids), label=y, score=p, site=site)
            metrics[site] = float(roc_auc_score(y, p))
        rb.write_json(out, {"tag": tag, "seed": seed, "metrics_auroc": metrics})
        rb.logging.info(f"{tag}: {metrics}")


if __name__ == "__main__":
    main()
