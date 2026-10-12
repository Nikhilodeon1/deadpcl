# Results pack

Aggregates only; no patient-level rows. Every number points to a file under `rebuttal/` or is marked
"carried, not re-verified". Settings, decision rules and the order of work were fixed in `PREREG.md`
(A1-A5) before any run; nulls and failures are reported with the same prominence as the rest.

Seeds: 42-51 (ten; 42-44 are the original three). Target AUROC is the primary metric. "Interval" = two-level
percentile bootstrap (seeds, then stays; 1000 draws) of the paired difference; the t-interval over seeds is
in `tables/paired_contrasts_formatted.csv` as a robustness view. All neural arms: same pipeline, same
schedule (30 pretraining + 30 fine-tuning epochs, warmup fraction 0.4, batch 64), ERM = lambda 0 in the same code path.
Neural arms were trained on one GPU (V100); MIMIC-IV and eICU were scored on the machine that holds them.

## 0. Things that hurt us (read first)

1. **The task is detection of an event inside the input window, not prediction.** The label is "any positive
   hour in the first 48 rows" and the input is the same 48 rows (`audit/AUDIT.md` G1, D1). Cutting each positive
   stay's input so that it ends 0, 6 or 12 hours before the first positive label drops Site B AUROC from 0.689 (ERM)
   and 0.662 (PCL) to 0.54-0.57, close to chance (`tables/horizon_diagnostic_siteb.csv`, Fig. 5). The headline AUROCs
   mostly come from the hours at and after labeled onset. (Diagnostic of the existing models; a model trained on cut
   inputs was not run.)
2. **"PCL falls below ERM at every site" is not supported.** Lambda = 0.5 minus ERM over 10 seeds is -0.026 (Site B),
   +0.001 (MIMIC-IV), -0.003 (eICU); every 95% interval contains zero (`tables/paired_contrasts_formatted.csv`).
   Under our own pre-registered rule the claim is "no evidence of benefit", not "worse".
3. **A target-selected lambda beats ERM at eICU** (trigger B, oracle row only): +0.039, interval [+0.023, +0.057], 5 seeds
   (`STOP_REPORT.md`). Lambda = 5 is the best at eICU; the interval ignores that lambda was chosen on the same target data.
   The deployable selector (lambda by source-validation AUROC) gives +0.016 [0.000, +0.034] at eICU and +0.019 [0.000, +0.041]
   at MIMIC-IV, still "no evidence of a difference" but with lower bounds at zero.
4. **The one-seed lambda sweep we relied on is not reproduced.** The carried seed-42 sweep had lambda 0.5 beating lambda 0 by
   5.7 points at Site B; in our seed 42, lambda 0 is ahead by 0.8 points (`STOP_REPORT.md`). Single cells move by 3-5 points
   across reruns, the size of the seed SD.
5. **How lambda was chosen in the submitted paper cannot be verified.** The code comment says lambda = 0.5 was selected on Site B
   (a target site) (`config.py:41-46`, D6); the paper says source validation. Source validation prefers lambda 0 or small
   lambda in our runs (`tables/lambda_sweep_seeds42_46.csv`: source-validation AUROC falls from 0.817 at lambda 0 to 0.769 at lambda 5).
6. **Sentence in the submitted paper is wrong:** "exceeds 1.0 on four of five site pairs" (the five numbers are detector
   validation scenarios, not site pairs; D9). Also D10 (+4.7 vs +4.2), D11 (XGBoost vs Group DRO in the abstract).
7. **Constraint availability is inflated by forward fill**, contrary to "raw record": Henderson-Hasselbalch is computable in
   3.0% of Site A hours raw vs 27.7% after fill; 1.8% vs 17.0% at MIMIC-IV; 0.55% vs 7.2% at eICU (D4,
   `tables/cohorts_and_constraint_availability.csv`).
8. **PaO2 means different things at source and targets:** reconstructed from saturation (approximate inverse) at the two
   PhysioNet sites, measured at MIMIC-IV and eICU (D8).
9. **Not run, so these reviewer requests are unanswered with new evidence:** reproduction of the original pipeline (A);
   the attribution ladder (B0/B1/B2; C1, Q1, M13); no-Severinghaus variant (M14); timestep masking (M8); constraint activity
   and redundancy with the masked loss (E1/E2: M7, M9); L1 vs squared loss (M10); constraint randomization after correction (C8);
   near-zero-residual / circularity proxy (F2: C4b, C7); Group DRO; seed-variance decomposition (Q3);
   reconstruction-error and constraint-violation selectors (C6).
10. **Residual label-definition shift** between PhysioNet (event hour inside the first 48 h) and MIMIC-IV / eICU (Sepsis-3 "ever in
    the stay", broadcast over 48 h) remains after the labeled "fix" (D2, D3). Prevalence ranges from 2.1% (Site B) to 30.9% (MIMIC-IV).

## 1. Main results (D1-D5-style questions: J2, C2, C3, Q3, Q5, Q8)

Target AUROC, mean (SD over seeds), n = 10 seeds each (`tables/metrics_auroc_auprc_brier_ece.csv`, `tables/per_seed_auroc.csv`):

| Arm | Site B | MIMIC-IV | eICU-CRD |
|---|---|---|---|
| ERM (lambda 0) | 0.689 (0.042) | 0.642 (0.035) | 0.594 (0.042) |
| PCL (lambda 0.5) | 0.662 (0.052) | 0.642 (0.047) | 0.591 (0.050) |
| XGBoost (fixed, 68 aggregate features) | 0.731 (0.012) | 0.646 (0.010) | 0.661 (0.009) |

Paired differences (95% two-level bootstrap):

| Contrast | Site B | MIMIC-IV | eICU-CRD |
|---|---|---|---|
| PCL - ERM | -0.026 [-0.063, +0.013] | +0.001 [-0.021, +0.020] | -0.003 [-0.031, +0.027] |
| XGBoost - ERM | +0.042 [-0.004, +0.090] | +0.004 [-0.019, +0.027] | +0.067 [+0.039, +0.093] |
| XGBoost - PCL | +0.068 [+0.021, +0.115] | +0.004 [-0.024, +0.032] | +0.070 [+0.042, +0.099] |

Seed variability (`tables/per_seed_auroc.csv`): SD of a neural arm across seeds is 3.5-5.2 AUROC points; the min-max range is
11.9-17.1 points. XGBoost: SD 0.9-1.2 points, range 2.9-4.3. SD of the paired PCL-ERM margin across seeds: 5.6 (Site B), 3.6
(MIMIC-IV), 4.9 (eICU) points. Seed enters both model initialisation and the Site-A split. Positives in Site A: 340 of 10,381
(Site B: 304 of 14,779; MIMIC-IV 23,062 of 74,607; eICU 11,536 of 130,446).

Calibration and AUPRC (`tables/metrics_auroc_auprc_brier_ece.csv`; ECE with 10 equal-width bins; AUPRC to be read against prevalence):
at Site B ECE is 0.093 (ERM), 0.174 (PCL), 0.007 (XGBoost); MIMIC-IV 0.146, 0.130, 0.282; eICU 0.113, 0.182, 0.068.
PCL is worse calibrated than ERM at Site B and eICU.

## 2. By reviewer concern

Verdict key: supported / refuted / mixed / not testable / not run. "May claim" = what the paper may say.

### xM3J (R1)

| ID | Verdict | Evidence | May claim |
|---|---|---|---|
| M1 four of five lessons known | not testable | - | Do not claim the lessons are new; claim the quantified audit of one pipeline (D1-D19) and the 10-seed paired result. |
| M2 confound-4 argument | mixed | `audit/AUDIT.md` G3; D8; `tables/cohorts_and_constraint_availability.csv` | PaO2 is an approximate inverse of saturation at the PhysioNet sites (term dropped) and measured elsewhere, so "imputed equals measured" does not hold exactly; the Severinghaus term was not ablated (F6 not run). |
| M3 concurrent submission | not testable | - | Writing decision; nothing in this pack relies on target data except the oracle row, which is labeled non-deployable. |
| M4 constraint novelty | not testable | - | - |
| M5 other settings | not testable | - | One architecture, one task, three external sites (limitation). |
| M7 needs >= 2 masked variables | refuted for the implementation | `audit/AUDIT.md` G4 (`pcl_loss.py:61-79`) | The penalty applies when at least one constraint variable is masked, using model predictions for all variables; redundancy with the masked loss was not measured (E1/E2 not run). |
| M8 timestep masking | not run | - | State as untested. |
| M9 how often constraints trigger | mixed | availability table; G4 | Constraint inputs are jointly present in 1.8-17.0% (HH) and 5.1-23.7% (Severinghaus) of MIMIC-IV hours (raw vs after fill), analogous numbers for the other cohorts; the per-masked-position activity (E1) was not run. |
| M10 L1 vs squared | not testable | `audit/AUDIT.md` G4 | No rationale is recorded in the code or surviving history; L1 for MAP, squared for HH and Severinghaus. |
| M11 final lambda | mixed | `tables/lambda_sweep_seeds42_46.csv`; D6 | "PCL" rows use lambda 0.5 (documented default; provenance unclear). Per-seed source-validation selection picks lambda 0, 0.1, 0, 0, 5 for seeds 42-46. The original paper's per-seed lambda is unresolved. |
| M12 SOFA spelled out | supported (can be written) | `audit/AUDIT.md` G2; `src/data/sofa_sepsis.py` | Labels use single-point SOFA at the suspicion time (mode `single`), "ever in the ICU stay"; suspicion = antibiotic-culture pairing (MIMIC-IV) or culture or infection diagnosis (eICU). |
| M13 why confound 1 favors PCL | not run | - | Attribution experiments not run. |
| M14 fix for confound 4 | not run | - | No-Severinghaus variant not run. |

### JHNu (R2)

| ID | Verdict | Evidence | May claim |
|---|---|---|---|
| J1 one architecture, one task | supported (limitation) | - | State it. |
| J2 only 3 seeds | refuted as a limit of the new results | Section 1 | Ten seeds; the 3-seed differences in the original paper are within seed noise (SD 3.5-5.2 points). |
| J4 PINN framing | not testable | - | - |
| J5 assumptions of PCL | mixed | G3; D4; D8 | Constraint inputs are sparse before forward fill and PaO2 is reconstructed at two sites; MAP-identity residual distribution (E6) was not computed. |
| J6 "only two sites" | refuted (misread) | Section 1 | There are three external sites (Site B, MIMIC-IV, eICU-CRD); the tables show all three. |

### xCda (R3)

| ID | Verdict | Evidence | May claim |
|---|---|---|---|
| C1 attribution | not run | - | The ladder (B0/B1/B2) was not run; the only new evidence on "other things changed" is the 10-seed paired result under one fixed pipeline. |
| C2 statistics | supported | Section 1; `tables/per_seed_auroc.csv`, `paired_contrasts_formatted.csv` | Per-seed values, SD (not range), paired intervals are given; "worse" is not supported, "no evidence of benefit" is. The 3.3-point HH-ablation cost was not re-tested. |
| C3 AUROC for every lambda with ERM | supported | `tables/lambda_sweep_seeds42_46.csv`, Fig. 2 | On seeds 42-46 no lambda separates cleanly from ERM at Site B or MIMIC-IV; at eICU larger lambda (2, 5) is higher (0.605, 0.609 vs 0.570). Lambda 0 and 0.5 have ten seeds (Section 1); other lambdas five. |
| C4a label provenance, prevalence | supported | G1, G2, D16 | Site A/B labels are the challenge's hourly labels cut to the first 48 rows. Prevalence 3.28% / 2.06% reproduced; 8.8% / 5.7% of challenge files are ever septic, 5.6% / 3.7% have a positive hour in the first 48 rows; 24 h and hemodynamic filters remove the rest. |
| C4b HH circularity (calculated HCO3) | not run | - | Near-zero-residual proxy not computed (F2). |
| C4c PaO2 reconstruction | supported | G3, D8 | Raw PhysioNet files have no PaO2; the repository's loader reconstructs it from saturation at 7.3% (Site A) and 1.9% (Site B) of hours with an approximate inverse. The "8.6x" gap is carried, not re-verified (D19). |
| C5 task, DRO, XGBoost, ERM | mixed | D1, D13, D14, `scripts/xgb_local.py` | Task: see Section 0 item 1. XGBoost: fixed 200 trees, depth 4, learning rate 0.05, subsample 0.8, column subsample 0.8, 68 aggregate features (mean, min, max, observation rate), no tuning, same 80% Site-A split. ERM pretraining here is lambda 0 in the same code path (the repository's separate ERM routine differs, D14). Group DRO groups are unresolved and DRO was not run. |
| C6 second test not checkable | mixed | `tables/lambda_sweep_seeds42_46.csv`, `pack/tables/final/main_comparison.csv` | The sweep and source-validation selection are now in the pack; selection by reconstruction error or constraint violation on unlabeled target inputs was not run. |
| C7 circularity frequency across cohorts | mixed | availability table | Constraint availability per cohort is reported; the derived-value proxy was not run. |
| C8 constraint randomization | not run | - | - |
| minor: running head | supported | D12 | Fix. |
| minor: +4.7 vs +4.2 | supported | D10 | Fix. |
| minor: composition-to-gap ratio definition | supported | G5 | Formula in `audit/AUDIT.md` G5. |
| minor: XGBoost vs Group DRO in abstract | mixed | D11, Section 1 | DRO not run here; with ERM and PCL, XGBoost is higher at eICU and not distinguishable at MIMIC-IV. |

### 7MQE (R4)

| ID | Verdict | Evidence | May claim |
|---|---|---|---|
| Q1 per-fix attribution | not run | - | - |
| Q2 why pipeline errors favor PCL | not testable here | Section 1 | Under one fixed pipeline there is no detectable PCL-ERM difference at 10 seeds, so an asymmetric effect can neither be shown nor excluded beyond the width of the intervals (about -6 to +3 points at Site B, -2 to +2 at MIMIC-IV, -3 to +3 at eICU). |
| Q3 seed variance | mixed | Section 1 | SD 3.5-5.2 points (range 12-17 points) for neural arms, about 1 point for XGBoost; the split between initialisation and Site-A split was not decomposed. |
| Q4 label definition and horizon | supported | D1, G1, Fig. 5 | See Section 0 item 1. |
| Q5 training budgets | supported | `tables/budget.csv` | Neural arms: 3.28 M parameters, 30 + 30 epochs, batch 64, 40 arms, 4.57 GPU-hours on one V100 (mean 6.9 min per arm; the timing run measured 4.6 min, pods differ). XGBoost: CPU, under a minute per seed. Group DRO not run. |
| Q6 prior evidence | not testable | - | - |
| Q7 novel vs generic failure modes | not testable | - | - |
| Q8 AUPRC and calibration | supported | Section 1; `tables/metrics_auroc_auprc_brier_ece.csv` | See Section 1. |

### fDtQ (R5)

| ID | Verdict | Evidence | May claim |
|---|---|---|---|
| D1 writing | not testable | - | - |
| D2 no figures | addressed | `figures/fig1_auroc_by_arm.pdf`, `fig2_lambda_sweep.pdf`, `fig5_horizon_diagnostic.pdf` | Three plain figures exist; Figs. 3 (attribution) and 4 (constraint activity) were not produced because those experiments were not run. |
| D3 "autopsy" | not testable | - | - |
| D4 failure modes known | not testable | - | - |
| D5 overclaims from one task | supported (limitation) | - | State the single-task, single-architecture scope. |
| D6 anonymized repository | partly addressed | README, scripts | Regenerate commands in `rebuttal/README.md`; push to the mirror only after the owner confirms. |

## 3. Not run / not resolved

Experiment A (reproduce the original pipeline; needs all-site pretraining and the original drivers, out of budget and not in the
reachable files), B (attribution ladder), C label-free selectors by reconstruction error and constraint violation, Group DRO (D),
seed-variance decomposition, E1-E6, F2, F4. Items blocked by files that were not available are listed in `audit/AUDIT.md`
("Not answered and why") and `audit/DISCREPANCIES.md`. Tier order and the reasons are in `PREREG.md` A2 item 6 and A4.

## 4. File map

`audit/AUDIT.md`, `audit/DISCREPANCIES.md`, `PREREG.md`, `COST_LOG.md`, `STOP_REPORT.md`, `tables/*.csv|tex`, `figures/*.pdf`,
`scripts/` (arm runner `run_arms.py`, scoring `score_checkpoint.py`, statistics `analyze.py`, XGBoost `xgb_local.py`,
diagnostic `diag_horizon.py` + `horizon_summary.py`, `make_tables.py`, `make_figures.py`).
