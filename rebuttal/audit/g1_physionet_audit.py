"""
Phase 0 audit, PhysioNet 2019 side (CPU only, no model, no training).

Re-applies the loader's own filters (src/data/physionet2019.py) to every raw
file and records aggregate counts only. Answers G1 (cohort construction, label
timing relative to the input window, prevalence), and the PhysioNet half of G3
(which constraint inputs exist in the raw release, how PaO2 arises).

Usage:
    PHYSIONET_DIR=/path/to/physionet2019 python g1_physionet_audit.py out.json

Nothing patient-level is written: output is a JSON of counts and histograms.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.environ.setdefault("PCL_TEST_MODE", "0")

from src.data.physionet2019 import load_patient_psv, extract_timeseries  # noqa: E402
from src.data.preprocessing import (  # noqa: E402
    HOURS, MIN_LOS_H, has_hemodynamic_window, preprocess_timeseries, MinMaxNormalizer,
)
from src.data.variables import CANONICAL_VARIABLES, VAR_TO_IDX  # noqa: E402

CONSTRAINTS = ["MAP", "PP", "SI", "HH", "SpO2"]  # column order of c_mask


def audit_site(site_dir):
    files = sorted(f for f in os.listdir(site_dir) if f.endswith(".psv"))
    norm = MinMaxNormalizer()
    out = {
        "n_files": len(files),
        "skipped": {"short_lt24h": 0, "all_nan": 0, "no_hemo_hour": 0},
        "kept": 0,
        "raw_cols_missing_pao2": None,
        # label timing
        "ever_pos_all_files": 0,          # any SepsisLabel=1 anywhere in the record
        "window_pos_all_files": 0,        # any SepsisLabel=1 inside first 48 rows
        "kept_window_pos": 0,             # the per-stay task label (positives kept)
        "kept_ever_pos": 0,
        "kept_late_onset_label_zero": 0,  # septic later than hour 48 -> task label 0
        "first_label_hour_hist": np.zeros(HOURS, dtype=int),  # kept window-positive stays
        "hours_after_label_start": [],    # per kept positive: rows in window at/after first 1-label
        "record_len_hist": np.zeros(HOURS + 1, dtype=int),    # kept stays, min(len,48)
        # availability on raw (pre-ffill) and post-ffill, over real hours (min(len,48))
        "hours_total": 0,
        "raw_obs_hours": {v: 0 for v in CANONICAL_VARIABLES},
        "sao2_raw_hours": 0,
        "pao2_filled_hours_raw": 0,
        "cmask_post_ffill_hours": {c: 0 for c in CONSTRAINTS},
        "cmask_raw_hours": {c: 0 for c in CONSTRAINTS},
    }
    saw_pao2_col = False
    for k, f in enumerate(files):
        df = load_patient_psv(os.path.join(site_dir, f))
        saw_pao2_col = saw_pao2_col or ("PaO2" in df.columns)
        lab_full = df["SepsisLabel"].values.astype(float) if "SepsisLabel" in df.columns else np.zeros(len(df))
        ever = bool((lab_full > 0).any())
        win = bool((lab_full[:HOURS] > 0).any())
        out["ever_pos_all_files"] += int(ever)
        out["window_pos_all_files"] += int(win)
        if len(df) < MIN_LOS_H:
            out["skipped"]["short_lt24h"] += 1
            continue
        raw = extract_timeseries(df)
        if np.all(np.isnan(raw)):
            out["skipped"]["all_nan"] += 1
            continue
        padded = np.full((HOURS, len(CANONICAL_VARIABLES)), np.nan)
        n = min(HOURS, raw.shape[0])
        padded[:n] = raw[:n]
        if not has_hemodynamic_window(padded):
            out["skipped"]["no_hemo_hour"] += 1
            continue

        out["kept"] += 1
        out["kept_ever_pos"] += int(ever)
        out["record_len_hist"][n] += 1
        if win:
            out["kept_window_pos"] += 1
            h1 = int(np.argmax(lab_full[:HOURS] > 0))
            out["first_label_hour_hist"][h1] += 1
            out["hours_after_label_start"].append(n - h1)
        elif ever:
            out["kept_late_onset_label_zero"] += 1

        out["hours_total"] += n
        for v in CANONICAL_VARIABLES:
            out["raw_obs_hours"][v] += int((~np.isnan(padded[:n, VAR_TO_IDX[v]])).sum())
        if "SaO2" in df.columns:
            out["sao2_raw_hours"] += int((~np.isnan(df["SaO2"].values.astype(float)[:n])).sum())
        out["pao2_filled_hours_raw"] += int((~np.isnan(padded[:n, VAR_TO_IDX["PaO2"]])).sum())

        # raw (pre-ffill) constraint computability, same definitions as build_constraint_mask
        def present(names):
            m = np.ones(n, dtype=bool)
            for nm in names:
                m &= ~np.isnan(padded[:n, VAR_TO_IDX[nm]])
            return m
        raw_c = {
            "MAP": present(["SBP", "DBP", "MAP"]), "PP": present(["SBP", "DBP", "MAP"]),
            "SI": present(["HR", "SBP"]), "HH": present(["pH", "HCO3", "pCO2"]),
            "SpO2": present(["SpO2", "PaO2"]),
        }
        pre = preprocess_timeseries(padded, norm)
        cm = pre["c_mask"][:n]
        for ci, c in enumerate(CONSTRAINTS):
            out["cmask_raw_hours"][c] += int(raw_c[c].sum())
            out["cmask_post_ffill_hours"][c] += int(cm[:, ci].sum())
        if (k + 1) % 5000 == 0:
            print(f"  {k + 1}/{len(files)}", flush=True)

    out["raw_cols_missing_pao2"] = (not saw_pao2_col)
    hs = np.array(out["hours_after_label_start"]) if out["hours_after_label_start"] else np.array([0])
    out["hours_after_label_start_summary"] = {
        "n": int(len(out["hours_after_label_start"])),
        "median": float(np.median(hs)), "p25": float(np.percentile(hs, 25)),
        "p75": float(np.percentile(hs, 75)), "mean": float(hs.mean()),
        "frac_ge_12h": float((hs >= 12).mean()), "frac_ge_24h": float((hs >= 24).mean()),
    }
    del out["hours_after_label_start"]
    out["first_label_hour_hist"] = out["first_label_hour_hist"].tolist()
    out["record_len_hist"] = out["record_len_hist"].tolist()
    return out


def main():
    base = os.environ.get("PHYSIONET_DIR", os.path.join(ROOT, "data", "physionet2019"))
    result = {"definition_notes": {
        "task_label": "dataset.py: sepsis = any hourly SepsisLabel > 0 within the first 48 rows",
        "input_window": "same first 48 hours (values, mask) -- encoder causal, head reads last observed hour",
    }}
    for name in ("training_setA", "training_setB"):
        print(f"auditing {name}", flush=True)
        result[name] = audit_site(os.path.join(base, name))
        r = result[name]
        r["prevalence_kept_task_label"] = r["kept_window_pos"] / max(1, r["kept"])
        r["prevalence_all_files_ever"] = r["ever_pos_all_files"] / max(1, r["n_files"])
        r["prevalence_all_files_window"] = r["window_pos_all_files"] / max(1, r["n_files"])
        r["prevalence_kept_ever"] = r["kept_ever_pos"] / max(1, r["kept"])
    out_path = sys.argv[1] if len(sys.argv) > 1 else "g1_physionet_audit.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
