"""Readouts for PREREG A8: full-record label variant and the seed grid.

  evlab : PCL (lambda 0.5) minus ERM target AUROC per site (seeds 42-46), two-level bootstrap, own ERM per label definition.
          At Site B the label is the full-record label (positives by onset anywhere in the record); MIMIC-IV and eICU labels are
          unchanged ("ever in the ICU stay").
  sg    : ERM, init seed i x split seed j, i, j in {42, 43, 44}; the diagonal i == j is the main ERM arms lam0_s42..44.
          Variance components of target AUROC from a two-way layout without replication:
          sigma2_init = (MS_init - MS_res) / 3, sigma2_split = (MS_split - MS_res) / 3, residual MS_res (negative estimates set to 0).

    python rebuttal/scripts/analyze_batch2.py --pred-root rebuttal/outputs/predictions --out rebuttal/pack/tables
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an

SITES = ["site_b", "mimic", "eicu"]


class PrefixStore:
    """Adapter so that analyze.boot_diff can read evlab__lam<l>_s<seed> prediction folders."""

    def __init__(self, root, prefix):
        self.root, self.prefix, self.cache = root, prefix, {}

    def get(self, lam, seed, site):
        k = (lam, seed, site)
        if k not in self.cache:
            p = os.path.join(self.root, f"{self.prefix}lam{lam:g}_s{seed}", f"{site}.npz")
            z = np.load(p)
            self.cache[k] = (z["hashed_id"], z["label"].astype(int), z["score"].astype(float))
        return self.cache[k]


def evlab(root, n_boot, rng):
    st = PrefixStore(root, "evlab__")
    rows = []
    for site in SITES:
        seeds = [s for s in range(42, 47)
                 if all(os.path.exists(os.path.join(root, f"evlab__lam{l:g}_s{s}", f"{site}.npz")) for l in (0.0, 0.5))]
        if len(seeds) < 2:
            continue
        e = np.array([an.auroc_fast(st.get(0.0, s, site)[1], st.get(0.0, s, site)[2]) for s in seeds])
        p = np.array([an.auroc_fast(st.get(0.5, s, site)[1], st.get(0.5, s, site)[2]) for s in seeds])
        d = p - e
        lo, hi = an.boot_diff(st, {s: 0.0 for s in seeds}, {s: 0.5 for s in seeds}, site, seeds, n_boot, rng)
        t = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        n_pos = int(st.get(0.0, seeds[0], site)[1].sum())
        rows.append({"site": site, "n_seeds": len(seeds), "positives": n_pos,
                     "erm": e.mean(), "erm_sd": e.std(ddof=1), "pcl": p.mean(), "pcl_sd": p.std(ddof=1),
                     "margin": d.mean(), "boot_lo": lo, "boot_hi": hi, "t_lo": t[0], "t_hi": t[1],
                     "verdict": an.verdict(lo, hi, len(seeds))})
        print(f"evlab {site}: ERM {e.mean():.4f} PCL {p.mean():.4f} margin {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
              f"positives {n_pos}")
    return rows


def seedgrid(root):
    ids = (42, 43, 44)
    rows = []
    for site in SITES:
        A = np.full((3, 3), np.nan)
        for a, i in enumerate(ids):
            for b, j in enumerate(ids):
                tag = f"lam0_s{i}" if i == j else f"sg__lam0_i{i}_j{j}"
                p = os.path.join(root, tag, f"{site}.npz")
                if os.path.exists(p):
                    z = np.load(p)
                    A[a, b] = an.auroc_fast(z["label"].astype(int), z["score"].astype(float))
        if np.isnan(A).any():
            continue
        gm = A.mean()
        ms_init = 3 * ((A.mean(1) - gm) ** 2).sum() / 2
        ms_split = 3 * ((A.mean(0) - gm) ** 2).sum() / 2
        res = A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + gm
        ms_res = (res ** 2).sum() / 4
        s_init = max(0.0, (ms_init - ms_res) / 3)
        s_split = max(0.0, (ms_split - ms_res) / 3)
        rows.append({"site": site, "grid_mean": gm, "grid_sd_all_cells": A.std(ddof=1),
                     "sd_init": s_init ** 0.5, "sd_split": s_split ** 0.5, "sd_residual": ms_res ** 0.5,
                     "cells": "; ".join(f"{i}x{j}={A[a, b]:.3f}" for a, i in enumerate(ids) for b, j in enumerate(ids))})
        print(f"seedgrid {site}: SD all cells {A.std(ddof=1):.4f}; init {s_init**0.5:.4f} split {s_split**0.5:.4f} residual {ms_res**0.5:.4f}")
    return rows


def write(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    os.makedirs(args.out, exist_ok=True)
    write(os.path.join(args.out, "evlab_margin.csv"), evlab(args.pred_root, args.n_boot, rng))
    write(os.path.join(args.out, "seedgrid_components.csv"), seedgrid(args.pred_root))


if __name__ == "__main__":
    main()
