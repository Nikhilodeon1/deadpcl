# Rebuttal materials

Everything here regenerates from saved outputs. Aggregates only; patient-level files (caches, checkpoints,
per-stay predictions) are git-ignored.

Requirements: Python 3.10+, `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `xgboost`
(`requirements.txt`). Seeds are fixed in `scripts/run_arms.py` (neural arms) and `scripts/xgb_local.py`.

## Regenerate tables and figures from saved predictions

```bash
python scripts/analyze.py --pred-root outputs/predictions --out pack/tables/final --n-boot 1000
python scripts/horizon_summary.py --pred-root outputs/predictions --full-root outputs/predictions --out pack/tables/horizon_siteb.csv
python scripts/make_tables.py
python scripts/make_figures.py
```

## Re-run the experiments

1. Data: set `PHYSIONET_DIR` (contains `training_setA`, `training_setB`), `MIMIC_DIR`, `EICU_DIR`.
2. External cohorts (once, on the machine that holds them): `python scripts/prepare_external.py`.
3. Neural arms (GPU): `python scripts/run_arms.py --tier T1 --max-hours 2` then `--tier T2`.
   Output: one result file, one checkpoint and per-stay predictions (hashed id, label, score) per arm.
4. Score external sites from the checkpoints: `python scripts/score_checkpoint.py --ckpt-dir outputs/checkpoints`.
5. XGBoost: `python scripts/xgb_local.py`.
6. Horizon diagnostic: `python scripts/diag_horizon.py --ckpt-dir <folder with ERM and lambda 0.5 checkpoints> --hours 0 6 12`.

Pod helpers for a small persistent disk are in `pod/` (`env.sh`, `setup_venv.sh`, `inventory.sh`).
Pre-registration: `PREREG.md`. Pack: `pack/RESULTS.md`.
