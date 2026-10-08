"""Build the MIMIC-IV / eICU-CRD caches and cohort aggregates on the machine that holds the data.

The credentialed datasets stay on this machine. Outputs:
  - a preprocessed per-stay cache (RB_WORK/cache/<site>_all.pkl); patient-level, git-ignored
  - rebuttal/audit/outputs/f_external_<site>.json : aggregates only (cohort size, prevalence,
    per-variable and per-constraint availability before and after forward fill)
  - rebuttal/audit/outputs/prep_<site>.log : loader log, with local paths removed

Usage:
    MIMIC_DIR=<.../mimiciv/3.1> EICU_DIR=<.../eicu-crd/2.0> python rebuttal/scripts/prepare_external.py --sites mimic eicu
"""
import argparse
import logging
import os
import pickle
import re
import time

import numpy as np

import rb_common as rb

CONSTRAINTS = ["MAP", "PP", "SI", "HH", "SpO2"]


class _ScrubPaths(logging.Filter):
    def __init__(self, dirs):
        super().__init__()
        self.pats = [re.compile(re.escape(d.replace("\\", "/")), re.I) for d in dirs if d]

    def filter(self, record):
        msg = record.getMessage().replace("\\", "/")
        for p in self.pats:
            msg = p.sub("<data>", msg)
        msg = re.sub(r"[A-Za-z]:/Users/[^/\s]+", "<home>", msg)
        record.msg, record.args = msg, ()
        return True


def aggregates(samples):
    from src.data.variables import CANONICAL_VARIABLES, VAR_TO_IDX
    H = 48
    labels = np.array([float(s["label"].sum() > 0) if hasattr(s["label"], "sum") else float(np.sum(s["label"]) > 0)
                       for s in samples])
    out = {"n_stays": len(samples), "n_pos": int(labels.sum()), "prevalence": float(labels.mean()),
           "hours_total": 0, "raw_obs_hours": {v: 0 for v in CANONICAL_VARIABLES},
           "post_ffill_obs_hours": {v: 0 for v in CANONICAL_VARIABLES},
           "cmask_raw_hours": {c: 0 for c in CONSTRAINTS}, "cmask_post_ffill_hours": {c: 0 for c in CONSTRAINTS},
           "record_len_hist": np.zeros(H + 1, dtype=int)}
    for s in samples:
        raw = np.asarray(s["raw_ts"], dtype=float)
        n = min(H, raw.shape[0])
        out["record_len_hist"][n] += 1
        out["hours_total"] += n
        r = raw[:n]
        for v in CANONICAL_VARIABLES:
            out["raw_obs_hours"][v] += int((~np.isnan(r[:, VAR_TO_IDX[v]])).sum())
        m = np.asarray(s["mask"])[:n]
        for v in CANONICAL_VARIABLES:
            out["post_ffill_obs_hours"][v] += int(m[:, VAR_TO_IDX[v]].sum())

        def present(names):
            ok = np.ones(n, dtype=bool)
            for nm in names:
                ok &= ~np.isnan(r[:, VAR_TO_IDX[nm]])
            return ok
        raw_c = {"MAP": present(["SBP", "DBP", "MAP"]), "PP": present(["SBP", "DBP", "MAP"]),
                 "SI": present(["HR", "SBP"]), "HH": present(["pH", "HCO3", "pCO2"]),
                 "SpO2": present(["SpO2", "PaO2"])}
        cm = np.asarray(s["c_mask"])[:n]
        for ci, c in enumerate(CONSTRAINTS):
            out["cmask_raw_hours"][c] += int(raw_c[c].sum())
            out["cmask_post_ffill_hours"][c] += int(cm[:, ci].sum())
    out["record_len_hist"] = out["record_len_hist"].tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=["mimic", "eicu"], choices=["mimic", "eicu"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from config import MIMIC_DIR, EICU_DIR
    out_dir = os.path.join(rb.RB_ROOT, "audit", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    for site in args.sites:
        cpath = rb.cache_path(site, 1.0, 0)
        if os.path.exists(cpath) and not args.overwrite:
            logging.info(f"{site}: cache exists, skipping (use --overwrite)")
            continue
        fh = logging.FileHandler(os.path.join(out_dir, f"prep_{site}.log"), mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        fh.addFilter(_ScrubPaths([MIMIC_DIR, EICU_DIR]))
        logging.getLogger().addHandler(fh)
        t = time.time()
        if site == "mimic":
            from src.data.mimic4 import load_mimic4
            samples, _ = load_mimic4(MIMIC_DIR, fraction=1.0, seed=0, keep_raw=True)
        else:
            from src.data.eicu import load_eicu
            samples, _ = load_eicu(EICU_DIR, fraction=1.0, seed=0, keep_raw=True)
        load_s = time.time() - t
        agg = aggregates(samples)
        agg["load_seconds"] = load_s
        rb.write_json(os.path.join(out_dir, f"f_external_{site}.json"), agg)
        for s in samples:
            s.pop("raw_ts", None)
        with open(cpath, "wb") as fp:
            pickle.dump(samples, fp, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"{site}: {len(samples)} stays, prevalence {agg['prevalence']:.4f}, "
                     f"load {load_s/60:.1f} min -> cache written")
        logging.getLogger().removeHandler(fh)
        fh.close()
        del samples


if __name__ == "__main__":
    main()
