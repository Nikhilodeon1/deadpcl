"""Build the paper-facing tables from saved per-stay predictions and the audit aggregates.

Inputs : rebuttal/outputs/predictions/<arm>_s<seed>/<site>.npz  (hashed id, label, score)
         rebuttal/audit/outputs/*.json, rebuttal/pack/tables/final/*.csv (from analyze.py)
Outputs: rebuttal/pack/tables/*.csv and *.tex
    python rebuttal/scripts/make_tables.py
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "outputs", "predictions")
OUT = os.path.join(ROOT, "pack", "tables")
SITES = ["site_b", "mimic", "eicu"]
SITE_NAMES = {"site_b": "Site B", "mimic": "MIMIC-IV", "eicu": "eICU-CRD"}


def write(name, rows, keys=None, caption=None):
    os.makedirs(OUT, exist_ok=True)
    keys = keys or list(rows[0].keys())
    with open(os.path.join(OUT, name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{" + "l" * len(keys) + "}\n\\hline\n")
        f.write(" & ".join(k.replace("_", "\\_") for k in keys) + " \\\\\n\\hline\n")
        for r in rows:
            f.write(" & ".join(fmt(r.get(k, "")) for k in keys) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v).replace("_", "\\_")


def per_seed():
    st = an.Store(PRED)
    rows = []
    for name, lam in (("ERM", 0.0), ("PCL_0.5", 0.5), ("XGBoost", -1.0)):
        for seed in range(42, 52):
            row = {"arm": name, "seed": seed}
            for site in SITES:
                if st.has(lam, seed, site):
                    row[site] = an.metrics(st, lam, seed, site)["auroc"]
            rows.append(row)
    write("per_seed_auroc", rows, ["arm", "seed"] + SITES)


def metrics_table():
    rows = []
    for r in csv.DictReader(open(os.path.join(OUT, "final", "main_comparison.csv"))):
        rows.append({"arm": r["arm"], "site": SITE_NAMES.get(r["site"], r["site"]), "n_seeds": r["n_seeds"],
                     "auroc": f'{float(r["auroc_mean"]):.3f} ({float(r["auroc_sd"]):.3f})',
                     "auprc": f'{float(r["auprc_mean"]):.3f}', "brier": f'{float(r["brier_mean"]):.3f}',
                     "ece10": f'{float(r["ece10_mean"]):.3f}'})
    write("metrics_auroc_auprc_brier_ece", rows)


def sweep_table():
    """Lambda sweep on the seeds every lambda has (42-46), with ERM = lambda 0 on the same seeds."""
    st = an.Store(PRED)
    seeds = list(range(42, 47))
    rows = []
    for lam in an.GRID:
        row = {"lambda": lam, "n_seeds": len(seeds)}
        for site in ["site_a_val"] + SITES:
            v = [an.metrics(st, lam, s, site)["auroc"] for s in seeds]
            row[site] = f"{np.mean(v):.3f} ({np.std(v, ddof=1):.3f})"
        rows.append(row)
    write("lambda_sweep_seeds42_46", rows, ["lambda", "n_seeds", "site_a_val"] + SITES)


def paired_table():
    rows = []
    for r in csv.DictReader(open(os.path.join(OUT, "final", "paired_contrasts.csv"))):
        rows.append({"contrast": r["contrast"], "site": SITE_NAMES.get(r["site"], r["site"]), "n_seeds": r["n_seeds"],
                     "diff": f'{float(r["diff_mean"]):+.3f}',
                     "boot95": f'[{float(r["boot_lo"]):+.3f}, {float(r["boot_hi"]):+.3f}]',
                     "t95": f'[{float(r["t_lo"]):+.3f}, {float(r["t_hi"]):+.3f}]' if r["t_lo"] not in ("nan", "") else "n/a",
                     "verdict": r["verdict"]})
    write("paired_contrasts_formatted", rows)


def load_json(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def cohort_and_availability():
    a, b = load_json("audit/outputs/g1_physionet_audit.json"), None
    mimic, eicu = load_json("audit/outputs/f_external_mimic.json"), load_json("audit/outputs/f_external_eicu.json")
    rows = []
    for name, d in (("MIMIC-IV", mimic), ("eICU-CRD", eicu)):
        H = d["hours_total"]
        rows.append({"cohort": name, "stays": d["n_stays"], "positives": d["n_pos"],
                     "prevalence": f'{d["prevalence"]*100:.2f}%',
                     "HH_raw": f'{d["cmask_raw_hours"]["HH"]/H*100:.2f}%',
                     "HH_after_fill": f'{d["cmask_post_ffill_hours"]["HH"]/H*100:.2f}%',
                     "Sev_raw": f'{d["cmask_raw_hours"]["SpO2"]/H*100:.2f}%',
                     "Sev_after_fill": f'{d["cmask_post_ffill_hours"]["SpO2"]/H*100:.2f}%',
                     "PaO2_raw": f'{d["raw_obs_hours"]["PaO2"]/H*100:.2f}%'})
    # PhysioNet part: key names follow g1_physionet_audit.json
    for site_key, name in (("training_setA", "Site A"), ("training_setB", "Site B")):
        d = a.get(site_key)
        if not d:
            continue
        H = d["hours_total"]
        rows.insert(0 if name == "Site A" else 1, {
            "cohort": name, "stays": d["kept"], "positives": d["kept_window_pos"],
            "prevalence": f'{d["kept_window_pos"]/d["kept"]*100:.2f}%',
            "HH_raw": f'{d["cmask_raw_hours"]["HH"]/H*100:.2f}%',
            "HH_after_fill": f'{d["cmask_post_ffill_hours"]["HH"]/H*100:.2f}%',
            "Sev_raw": f'{d["cmask_raw_hours"]["SpO2"]/H*100:.2f}%',
            "Sev_after_fill": f'{d["cmask_post_ffill_hours"]["SpO2"]/H*100:.2f}%',
            "PaO2_raw": f'{d["pao2_filled_hours_raw"]/H*100:.2f}% (reconstructed from saturation)'})
    write("cohorts_and_constraint_availability", rows)


def horizon_table():
    src = os.path.join(OUT, "horizon_siteb.csv")
    if os.path.exists(src):
        rows = list(csv.DictReader(open(src)))
        write("horizon_diagnostic_siteb", [{k: (f"{float(v):.3f}" if k.startswith(("auroc", "boot", "t_")) and v not in ("", None) else v)
                                            for k, v in r.items()} for r in rows])


def budget_table():
    runs = [json.load(open(os.path.join(ROOT, "outputs", "runs", f))) for f in sorted(os.listdir(os.path.join(ROOT, "outputs", "runs")))
            if f.startswith("lam")]
    total_s = sum(r["arm_seconds"] for r in runs)
    rows = [{"method": "Neural arms (ERM, PCL, lambda sweep)", "arms": len(runs), "params": "3.28 M",
             "epochs": "30 pretrain + 30 fine-tune", "batch": 64, "gpu": "1 x V100 32 GB",
             "gpu_hours_total": f"{total_s/3600:.2f}", "minutes_per_arm_mean": f"{total_s/60/len(runs):.1f}"},
            {"method": "XGBoost", "arms": 10, "params": "200 trees, depth 4", "epochs": "n/a", "batch": "n/a",
             "gpu": "CPU", "gpu_hours_total": "0", "minutes_per_arm_mean": "under 1"}]
    write("budget", rows)


if __name__ == "__main__":
    per_seed()
    metrics_table()
    sweep_table()
    paired_table()
    cohort_and_availability()
    horizon_table()
    budget_table()
    print("tables written to", OUT)
