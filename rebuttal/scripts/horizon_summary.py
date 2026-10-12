"""Summarize diagnostic H: AUROC at Site B when the input ends h hours before the first positive label."""
import argparse, csv, os, sys
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an


class HStore:
    def __init__(self, root, h):
        self.root, self.h, self.cache = root, h, {}

    def get(self, lam, seed, site):
        k = (lam, seed)
        if k not in self.cache:
            z = np.load(os.path.join(self.root, f"lam{lam:g}_h{self.h}_s{seed}", f"{site}.npz"))
            self.cache[k] = (z["hashed_id"], z["label"].astype(int), z["score"].astype(float))
        return self.cache[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--full-root", required=True, help="predictions of the uncut models (same root normally)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    seeds = list(range(42, 52))
    rows = []
    full = an.Store(args.full_root)
    for name, lam in (("ERM", 0.0), ("PCL_0.5", 0.5)):
        v = np.array([an.metrics(full, lam, s, "site_b")["auroc"] for s in seeds])
        rows.append({"input": "full 48 h", "arm": name, "n_seeds": 10, "auroc_mean": v.mean(), "auroc_sd": v.std(ddof=1)})
    for h in (0, 6, 12):
        st = HStore(args.pred_root, h)
        per = {}
        for name, lam in (("ERM", 0.0), ("PCL_0.5", 0.5)):
            v = np.array([an.auroc_fast(st.get(lam, s, "site_b")[1], st.get(lam, s, "site_b")[2]) for s in seeds])
            per[name] = v
            rows.append({"input": f"ends {h} h before first label", "arm": name, "n_seeds": 10,
                         "auroc_mean": v.mean(), "auroc_sd": v.std(ddof=1)})
        d = per["PCL_0.5"] - per["ERM"]
        lo, hi = an.boot_diff(st, {s: 0.0 for s in seeds}, {s: 0.5 for s in seeds}, "site_b", seeds, args.n_boot, rng)
        t = stats.t.interval(0.95, 9, loc=d.mean(), scale=stats.sem(d))
        rows.append({"input": f"ends {h} h before first label", "arm": "PCL_0.5 - ERM", "n_seeds": 10,
                     "auroc_mean": d.mean(), "auroc_sd": d.std(ddof=1), "boot_lo": lo, "boot_hi": hi,
                     "t_lo": t[0], "t_hi": t[1]})
    keys = ["input", "arm", "n_seeds", "auroc_mean", "auroc_sd", "boot_lo", "boot_hi", "t_lo", "t_hi"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f'{r["input"]:34s}{r["arm"]:14s} {r["auroc_mean"]:+.4f} +- {r["auroc_sd"]:.4f}' +
              (f' boot[{r["boot_lo"]:+.4f},{r["boot_hi"]:+.4f}]' if "boot_lo" in r else ""))


if __name__ == "__main__":
    main()
