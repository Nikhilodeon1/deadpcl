"""Plain matplotlib figures from saved per-stay predictions (single-column width).

    python rebuttal/scripts/make_figures.py
Fig1: target AUROC per arm and site, one dot per seed, bar = mean.
Fig2: lambda sweep, source-validation AUROC vs target AUROC (seeds 42-46, ERM = lambda 0).
Fig5: diagnostic H, Site B AUROC when the input ends h hours before the first positive label.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "pack", "figures")
os.makedirs(OUT, exist_ok=True)
SITES = [("site_b", "Site B"), ("mimic", "MIMIC-IV"), ("eicu", "eICU-CRD")]
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def fig1(st):
    arms = [("ERM", 0.0), ("PCL", 0.5), ("XGBoost", -1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2), sharey=True)
    for ax, (site, name) in zip(axes, SITES):
        for i, (a, lam) in enumerate(arms):
            v = np.array([an.metrics(st, lam, s, site)["auroc"] for s in range(42, 52) if st.has(lam, s, site)])
            ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=8, color="0.45", zorder=2)
            ax.hlines(v.mean(), i - 0.3, i + 0.3, color="k", lw=1.5, zorder=3)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a for a, _ in arms])
        ax.set_title(name)
        ax.axhline(0.5, color="0.7", lw=0.6, ls="--")
    axes[0].set_ylabel("target AUROC")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_auroc_by_arm.pdf"))


def fig2(st):
    seeds = list(range(42, 47))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(an.GRID))
    for site, name in SITES + [("site_a_val", "source validation")]:
        m = np.array([np.mean([an.metrics(st, l, s, site)["auroc"] for s in seeds]) for l in an.GRID])
        sd = np.array([np.std([an.metrics(st, l, s, site)["auroc"] for s in seeds], ddof=1) for l in an.GRID])
        ax.errorbar(x, m, yerr=sd, marker="o", ms=3, lw=1, capsize=2, label=name,
                    ls="--" if site == "site_a_val" else "-")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l:g}" for l in an.GRID])
    ax.set_xlabel(r"constraint weight $\lambda$ ($\lambda=0$ is ERM)")
    ax.set_ylabel("AUROC (mean $\pm$ SD, 5 seeds)")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_lambda_sweep.pdf"))


def fig5():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "pack", "tables", "horizon_siteb.csv"))))
    labels = ["full 48 h", "ends 0 h", "ends 6 h", "ends 12 h"]
    key = ["full 48 h", "ends 0 h before first label", "ends 6 h before first label", "ends 12 h before first label"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for arm, off in (("ERM", -0.08), ("PCL_0.5", 0.08)):
        m = [float(next(r for r in rows if r["input"] == k and r["arm"] == arm)["auroc_mean"]) for k in key]
        s = [float(next(r for r in rows if r["input"] == k and r["arm"] == arm)["auroc_sd"]) for k in key]
        ax.errorbar(np.arange(4) + off, m, yerr=s, marker="o", ms=3, capsize=2, lw=1, label=arm.replace("_", " "))
    ax.axhline(0.5, color="0.7", lw=0.6, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Site B AUROC (10 seeds)")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_horizon_diagnostic.pdf"))


def fig3():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "pack", "tables", "attribution.csv"))))
    variants = ["nosev", "tsmask", "vars9", "orig_hp"]
    labels = ["no Sev.\nterm", "timestep\nmask", "9 vars", "mask .5,\nwarmup .3"]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4), sharey=True)
    for ax, (site, name) in zip(axes, SITES):
        for i, v in enumerate(variants):
            r = next(x for x in rows if x["variant"] == v and x["site"] == site)
            m, lo, hi = (float(r[k]) for k in ("margin_variant", "margin_variant_lo", "margin_variant_hi"))
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=3, capsize=2, color="k", lw=1)
            ax.hlines(float(r["margin_main"]), i - 0.3, i + 0.3, color="0.5", ls="--", lw=1)
        ax.axhline(0, color="0.8", lw=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_title(name)
    axes[0].set_ylabel("PCL $-$ ERM AUROC (5 seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_attribution.pdf"))


if __name__ == "__main__":
    store = an.Store(os.path.join(ROOT, "outputs", "predictions"))
    fig1(store)
    fig2(store)
    fig5()
    fig3()
    print("figures written to", OUT)
