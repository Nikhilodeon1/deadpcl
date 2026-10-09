"""Aggregate per-stay predictions into the pre-registered tables (PREREG A2 / R1-R5).

Reads <pred-root>/<tag>/<site>.npz (hashed_id, label, score) where tag = lam<L>_s<seed>.
Arms:
  ERM            lambda = 0
  PCL_0.5        lambda = 0.5
  PCL_srcsel     per seed, the lambda with the best source-validation AUROC over the grid
  PCL_oracle     per site, the lambda with the best mean target AUROC (target-selected, NOT deployable)
Paired contrasts use two-level percentile bootstrap (resample seeds, then stays; stays are
resampled once per draw and shared by both arms) with a t-interval over seeds as a robustness view.
A contrast is called "worse" / "better" only if the 95% interval excludes zero, and a "worse" claim
is only eligible with 10 seeds (PREREG R-rules).

    python rebuttal/scripts/analyze.py --pred-root <dir> --out rebuttal/pack/tables --n-boot 1000
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss

SITES = ["site_b", "mimic", "eicu"]
GRID = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]


def auroc_fast(y, s):
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ece10(y, p, n_bins=10):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


class Store:
    def __init__(self, root):
        self.root = root
        self.cache = {}
        self.tags = {}   # (lam, seed) -> tag
        for d in glob.glob(os.path.join(root, "lam*_s*")):
            m = re.match(r"lam([0-9.]+)_s(\d+)$", os.path.basename(d))
            if m:
                self.tags[(float(m.group(1)), int(m.group(2)))] = os.path.basename(d)
        for d in glob.glob(os.path.join(root, "xgb_s*")):
            m = re.match(r"xgb_s(\d+)$", os.path.basename(d))
            if m:
                self.tags[(-1.0, int(m.group(1)))] = os.path.basename(d)   # XGBoost arm

    def get(self, lam, seed, site):
        key = (lam, seed, site)
        if key not in self.cache:
            p = os.path.join(self.root, self.tags[(lam, seed)], f"{site}.npz")
            if not os.path.exists(p):
                self.cache[key] = None
            else:
                z = np.load(p, allow_pickle=False)
                self.cache[key] = (z["hashed_id"], z["label"].astype(int), z["score"].astype(float))
        return self.cache[key]

    def has(self, lam, seed, site):
        return (lam, seed) in self.tags and self.get(lam, seed, site) is not None


def metrics(store, lam, seed, site):
    _, y, s = store.get(lam, seed, site)
    return {"auroc": float(auroc_fast(y, s)), "auprc": float(average_precision_score(y, s)),
            "brier": float(brier_score_loss(y, s)), "ece10": ece10(y, s)}


def make_arms(store, seeds_all):
    """Each arm: dict site -> dict seed -> lambda used for that seed."""
    arms = {}
    arms["ERM"] = {site: {sd: 0.0 for sd in seeds_all if store.has(0.0, sd, site)} for site in SITES}
    arms["PCL_0.5"] = {site: {sd: 0.5 for sd in seeds_all if store.has(0.5, sd, site)} for site in SITES}
    # source-validation selection: needs every grid lambda for the seed
    srcsel, oracle = {}, {}
    full_seeds = [sd for sd in seeds_all if all(store.has(l, sd, "site_a_val") for l in GRID)]
    for site in SITES:
        srcsel[site] = {}
        for sd in full_seeds:
            if all(store.has(l, sd, site) for l in GRID):
                best = max(GRID, key=lambda l: (metrics(store, l, sd, "site_a_val")["auroc"], -l))
                srcsel[site][sd] = best
        seeds_ok = [sd for sd in full_seeds if all(store.has(l, sd, site) for l in GRID)]
        if seeds_ok:
            mean_by_l = {l: np.mean([metrics(store, l, sd, site)["auroc"] for sd in seeds_ok]) for l in GRID}
            lam_star = max(GRID, key=lambda l: mean_by_l[l])
            oracle[site] = {sd: lam_star for sd in seeds_ok}
        else:
            oracle[site] = {}
    arms["XGBoost"] = {site: {sd: -1.0 for sd in seeds_all if store.has(-1.0, sd, site)} for site in SITES}
    arms["PCL_srcsel"] = srcsel
    arms["PCL_oracle"] = oracle
    return arms


def boot_diff(store, a, b, site, seeds, n_boot, rng):
    """Two-level bootstrap of mean_seed [AUROC(b) - AUROC(a)] on one site."""
    ids0, y0, _ = store.get(a[seeds[0]], seeds[0], site)
    ys, sa, sb = [], [], []
    for sd in seeds:
        ia, ya, xa = store.get(a[sd], sd, site)
        ib, yb, xb = store.get(b[sd], sd, site)
        assert (ia == ids0).all() and (ib == ids0).all(), "stay order differs between runs"
        ys.append(ya); sa.append(xa); sb.append(xb)
    n = len(y0)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d = np.array([auroc_fast(ys[k][idx], sb[k][idx]) - auroc_fast(ys[k][idx], sa[k][idx])
                      for k in range(len(seeds))])
        pick = rng.integers(0, len(seeds), len(seeds))
        out[i] = np.nanmean(d[pick])
    return np.percentile(out, [2.5, 97.5])


def verdict(lo, hi, n_seeds):
    if hi < 0:
        return "worse" if n_seeds >= 10 else "worse (n<10: claim not eligible)"
    if lo > 0:
        return "better"
    return "no evidence of difference"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    store = Store(args.pred_root)
    seeds_all = sorted({s for (_, s) in store.tags})
    arms = make_arms(store, seeds_all)

    rows = []
    for name, arm in arms.items():
        for site in SITES:
            m = [metrics(store, lam, sd, site) for sd, lam in sorted(arm[site].items())]
            if not m:
                continue
            row = {"arm": name, "site": site, "n_seeds": len(m)}
            for k in ("auroc", "auprc", "brier", "ece10"):
                v = np.array([x[k] for x in m])
                row[f"{k}_mean"], row[f"{k}_sd"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            row["lambda_used"] = json.dumps({str(k): v for k, v in sorted(arm[site].items())})
            rows.append(row)
    write_csv(os.path.join(args.out, "main_comparison.csv"), rows)

    sweep = []
    for lam in GRID:
        for site in SITES + ["site_a_val"]:
            seeds = [sd for sd in seeds_all if store.has(lam, sd, site)]
            if seeds:
                v = np.array([metrics(store, lam, sd, site)["auroc"] for sd in seeds])
                sweep.append({"lambda": lam, "site": site, "n_seeds": len(seeds),
                              "auroc_mean": float(v.mean()), "auroc_sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan")})
    write_csv(os.path.join(args.out, "lambda_sweep.csv"), sweep)

    contrasts = []
    stop_b = []
    pairs = [(o, "ERM") for o in ("PCL_0.5", "PCL_srcsel", "PCL_oracle", "XGBoost")] +             [("XGBoost", "PCL_0.5")]
    for other, ref in pairs:
        for site in SITES:
            a, b = arms[ref][site], arms[other][site]
            seeds = sorted(set(a) & set(b))
            if len(seeds) < 2:
                continue
            d = np.array([metrics(store, b[sd], sd, site)["auroc"] - metrics(store, a[sd], sd, site)["auroc"]
                          for sd in seeds])
            lo_b, hi_b = boot_diff(store, a, b, site, seeds, args.n_boot, rng)
            t = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
            v = verdict(lo_b, hi_b, len(seeds))
            contrasts.append({"contrast": f"{other} - {ref}", "site": site, "n_seeds": len(seeds),
                              "diff_mean": float(d.mean()), "boot_lo": float(lo_b), "boot_hi": float(hi_b),
                              "t_lo": float(t[0]), "t_hi": float(t[1]), "verdict": v})
            if lo_b > 0:
                if other.startswith("PCL") and ref == "ERM":
                    stop_b.append(f"{other} - ERM at {site}")
    write_csv(os.path.join(args.out, "paired_contrasts.csv"), contrasts)
    if stop_b:
        print("STOP B CANDIDATE (paired interval above zero):", "; ".join(stop_b))
    print(f"seeds found: {seeds_all}; wrote tables to {args.out}")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
