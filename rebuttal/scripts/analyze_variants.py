"""Attribution readout (PREREG A6): does flipping one factor change the paired PCL - ERM margin?

For each variant and site, over seeds 42-46:
  margin_var  = AUROC(PCL variant) - AUROC(ERM variant)            (ERM of the main pipeline for `nosev`)
  margin_main = AUROC(PCL main)    - AUROC(ERM main)
  change      = margin_var - margin_main
Rule fixed in the work order: a factor is "explanatory" at a site if |mean change| exceeds the seed SD of
margin_main and the 95% interval of the change excludes zero. Intervals: two-level percentile bootstrap
(seeds, then stays; stays resampled once per draw and shared by the four prediction sets).
Also reports trigger-B candidates: PCL variant minus ERM variant with interval above zero.

    python rebuttal/scripts/analyze_variants.py --pred-root rebuttal/outputs/predictions --out rebuttal/pack/tables/attribution.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an

SEEDS = list(range(42, 52))   # uses the seeds each variant has (five in the first batch, ten after PREREG A8)
SITES = ["site_b", "mimic", "eicu"]
VARIANT_ARMS = {  # variant -> (tag of its ERM, tag of its PCL); "main" tags have no prefix
    "nosev": (("", 0.0), ("nosev__", 0.5)),
    "tsmask": (("tsmask__", 0.0), ("tsmask__", 0.5)),
    "vars9": (("vars9__", 0.0), ("vars9__", 0.5)),
    "orig_hp": (("orig_hp__", 0.0), ("orig_hp__", 0.5)),
}


def load(root, prefix, lam, seed, site):
    p = os.path.join(root, f"{prefix}lam{lam:g}_s{seed}", f"{site}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["hashed_id"], z["label"].astype(int), z["score"].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for variant, (erm_v, pcl_v) in VARIANT_ARMS.items():
        for site in SITES:
            data, ok = {}, []
            for s in SEEDS:
                sets = {"erm_v": load(args.pred_root, *erm_v, s, site), "pcl_v": load(args.pred_root, *pcl_v, s, site),
                        "erm_m": load(args.pred_root, "", 0.0, s, site), "pcl_m": load(args.pred_root, "", 0.5, s, site)}
                if all(v is not None for v in sets.values()):
                    ids0 = sets["erm_m"][0]
                    assert all((v[0] == ids0).all() for v in sets.values()), "stay order differs"
                    data[s] = sets
                    ok.append(s)
            if len(ok) < 2:
                continue
            y = data[ok[0]]["erm_m"][1]
            n = len(y)

            def stat(idx):
                out = []
                for s in ok:
                    d = data[s]
                    a = {k: an.auroc_fast(d[k][1][idx], d[k][2][idx]) for k in d}
                    out.append((a["pcl_v"] - a["erm_v"], a["pcl_m"] - a["erm_m"]))
                return np.array(out)  # seeds x (margin_var, margin_main)

            base = stat(np.arange(n))
            mv, mm = base[:, 0], base[:, 1]
            change = mv - mm
            boot_c, boot_v = [], []
            for _ in range(args.n_boot):
                idx = rng.integers(0, n, n)
                st = stat(idx)
                pick = rng.integers(0, len(ok), len(ok))
                boot_c.append((st[pick, 0] - st[pick, 1]).mean())
                boot_v.append(st[pick, 0].mean())
            clo, chi = np.percentile(boot_c, [2.5, 97.5])
            vlo, vhi = np.percentile(boot_v, [2.5, 97.5])
            sd_main = float(mm.std(ddof=1))
            explanatory = abs(change.mean()) > sd_main and (clo > 0 or chi < 0)
            rows.append({"variant": variant, "site": site, "n_seeds": len(ok),
                         "margin_variant": mv.mean(), "margin_variant_lo": vlo, "margin_variant_hi": vhi,
                         "margin_main": mm.mean(), "margin_main_seed_sd": sd_main,
                         "change": change.mean(), "change_lo": clo, "change_hi": chi,
                         "explanatory": bool(explanatory), "trigger_B_candidate": bool(vlo > 0)})
            print(f"{variant:8s}{site:7s} n={len(ok)} margin_var {mv.mean():+.4f} [{vlo:+.4f},{vhi:+.4f}] "
                  f"main {mm.mean():+.4f} change {change.mean():+.4f} [{clo:+.4f},{chi:+.4f}] "
                  f"SD_main {sd_main:.4f} explanatory={explanatory} B={vlo > 0}")
    if rows:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
