"""Aggregate per-stay predictions into the tables the response needs.

Reads RB_OUT/predictions/<arm>_s<seed>/<site>.npz (hashed id, label, score) and writes
  - per-arm, per-site mean and sd over seeds of AUROC, AUPRC, Brier, ECE (10 bins)
  - paired differences between arms, with the interval fixed in PREREG:
      two-level percentile bootstrap (resample seeds with replacement, then stays with
      replacement, the same stays for both arms), 1000 resamples, 95%;
      plus a t-interval over seeds as a robustness view.
  - a verdict per comparison: "worse" only if the whole interval is below 0, "better" only
    if the whole interval is above 0, otherwise "no evidence of a difference".
All seeds present for an arm are used; none are dropped. n seeds is printed in every row.

    python rebuttal/scripts/aggregate.py --arms ERM=lam0 PCL0.5=lam0.5 XGB=xgb \
        --pairs PCL0.5:ERM XGB:ERM --out rebuttal/pack/tables/main
Arm spec NAME=PREFIX means the folder <PREFIX>_s<seed>.
"""
import argparse
import csv
import json
import os

import numpy as np

import rb_common as rb

SITES = ["site_a_val", "site_b", "mimic", "eicu"]


def load(prefix, seed, site):
    p = os.path.join(rb.RB_OUT, "predictions", f"{prefix}_s{seed}", f"{site}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    return z["hashed_id"], z["label"].astype(float), z["score"].astype(float)


class Ranked:
    """Scores sorted once, so a bootstrap AUROC is a weighted rank sum (tie-aware)."""
    def __init__(self, label, score):
        order = np.argsort(score, kind="mergesort")
        self.order = order
        s = score[order]
        self.label = label[order]
        new = np.r_[True, s[1:] != s[:-1]]
        self.group = np.cumsum(new) - 1
        self.ng = int(self.group[-1]) + 1

    def auroc(self, counts):
        c = counts[self.order]
        wp = np.bincount(self.group, weights=c * self.label, minlength=self.ng)
        wn = np.bincount(self.group, weights=c * (1 - self.label), minlength=self.ng)
        Wp, Wn = wp.sum(), wn.sum()
        if Wp == 0 or Wn == 0:
            return np.nan
        below = np.cumsum(wn) - wn
        return float((wp * (below + 0.5 * wn)).sum() / (Wp * Wn))


def descriptive(label, score):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    return {"auroc": roc_auc_score(label, score), "auprc": average_precision_score(label, score),
            "brier": brier_score_loss(label, score), "ece10": rb.ece(label, score)}


def paired(a_prefix, b_prefix, seeds, site, nboot, rng):
    """Mean over seeds of AUROC(a) - AUROC(b), with two-level bootstrap and t intervals."""
    A, B, ids = {}, {}, None
    for sd in seeds:
        a, b = load(a_prefix, sd, site), load(b_prefix, sd, site)
        if a is None or b is None:
            continue
        ia = a[0]
        if not np.array_equal(ia, b[0]):                 # align on hashed id if order differs
            pos = {h: i for i, h in enumerate(b[0])}
            idx = np.array([pos[h] for h in ia])
            b = (b[0][idx], b[1][idx], b[2][idx])
        assert np.array_equal(a[1], b[1])
        A[sd], B[sd] = Ranked(a[1], a[2]), Ranked(b[1], b[2])
        ids = ia if ids is None else ids
    used = sorted(A)
    if len(used) < 2:
        return {"n_seeds": len(used), "note": "need at least 2 seeds"}
    n = len(next(iter(A.values())).label)
    same_stays = site != "site_a_val"       # source validation splits differ by seed -> seed level only
    ones = np.ones(n)
    per_seed = np.array([A[s].auroc(ones) - B[s].auroc(ones) for s in used])
    est = float(per_seed.mean())
    boots = np.empty(nboot)
    for i in range(nboot):
        chosen = rng.integers(0, len(used), len(used))
        counts = np.bincount(rng.integers(0, n, n), minlength=n).astype(float) if same_stays else ones
        boots[i] = np.nanmean([A[used[j]].auroc(counts) - B[used[j]].auroc(counts) for j in chosen])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    from scipy import stats
    se = per_seed.std(ddof=1) / np.sqrt(len(used))
    tcrit = stats.t.ppf(0.975, len(used) - 1)
    verdict = "worse" if hi < 0 else "better" if lo > 0 else "no evidence of a difference"
    return {"n_seeds": len(used), "seeds": used, "diff_auroc_mean": est,
            "boot95": [float(lo), float(hi)], "t95_over_seeds": [float(est - tcrit * se), float(est + tcrit * se)],
            "per_seed_diff": per_seed.tolist(), "verdict": verdict,
            "stay_level_resampling": same_stays}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="NAME=PREFIX")
    ap.add_argument("--pairs", nargs="*", default=[], help="NAME_A:NAME_B  (reports A minus B)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--sites", nargs="+", default=SITES)
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--out", required=True, help="output path stem")
    args = ap.parse_args()
    arms = dict(a.split("=") for a in args.arms)
    rng = np.random.default_rng(20260109)

    rows, result = [], {"descriptive": {}, "paired": {}}
    for name, prefix in arms.items():
        for site in args.sites:
            vals, used = [], []
            for sd in args.seeds:
                z = load(prefix, sd, site)
                if z is None:
                    continue
                vals.append(descriptive(z[1], z[2]))
                used.append(sd)
            if not vals:
                continue
            row = {"arm": name, "site": site, "n_seeds": len(used), "seeds": used}
            for m in ("auroc", "auprc", "brier", "ece10"):
                v = np.array([x[m] for x in vals])
                row[m + "_mean"], row[m + "_sd"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else None
            rows.append(row)
            result["descriptive"][f"{name}|{site}"] = row
    for pr in args.pairs:
        a, b = pr.split(":")
        for site in args.sites:
            r = paired(arms[a], arms[b], args.seeds, site, args.nboot, rng)
            result["paired"][f"{a}-{b}|{site}"] = r
            if "diff_auroc_mean" in r:
                print(f"{a} - {b} @ {site}: {r['diff_auroc_mean']*100:+.2f} pts, boot95 "
                      f"[{r['boot95'][0]*100:+.2f}, {r['boot95'][1]*100:+.2f}], n={r['n_seeds']} seeds -> {r['verdict']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rb.write_json(args.out + ".json", result)
    if rows:
        with open(args.out + "_descriptive.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "seeds"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
