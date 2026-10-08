# Phase 0 audit

Scope: code reading plus CPU-only scans of the PhysioNet 2019 files. No model
was trained, no pod was started. The archive folder was not opened (work-order
rule). Paths are relative to `deadpcl/`. `src/` and `config.py` here are copies
of the sibling detector project's code; I diffed them against its current
version and they differ only in comments and line endings.

Items I could not answer, and why, are collected at the end.

## G1. Task definition and cohorts

**Label.** One binary label per stay: 1 if any hourly `SepsisLabel` in the first
48 rows is 1 (`src/data/dataset.py:29-30`). The hourly labels come straight
from the challenge files and are cut to 48 rows
(`src/data/physionet2019.py:60-63, 131-134`). Positive stays are therefore
those whose labeled onset falls inside the first 48 hours.

**Input.** The same first 48 hourly bins (`HOURS = 48`,
`src/data/preprocessing.py:7`), values clipped to plausibility bounds,
forward-filled up to 6 hours, min-max scaled (`preprocessing.py:140-175`).
Stays shorter than 48 hours are padded with unobserved rows
(`physionet2019.py:123-125`).

**Model read-out.** The encoder is causal (`src/models/backbone.py:93-99`). The
classification head reads the representation at the last observed hour
(`backbone.py:130-164`; the "observed" test is the post-fill mask, so it is the
last hour with any value including carried-forward ones). The head therefore
sees the whole history, including the label hours.

**Post-onset hours inside the input.** Yes by construction. Measured: for the
340 (Site A) and 304 (Site B) positive kept stays, the number of rows at or
after the first positive label row has median 10, 25th-75th percentile 9-10,
maximum 11 (`audit/outputs/g1_physionet_audit.json`). The first positive label
row is between row 14 and row 47 (histogram in the same file). This is STOP
trigger A; see `STOP_REPORT.md`.

**Septic records are truncated in the files.** Over all 2,932 septic stays
(1,790 at Site A, 1,142 at Site B; `audit/outputs/g1_septic_truncation.json`),
every record ends at most 10 rows after its first positive label row (median 10,
maximum 10), and the label stays 1 from that row to the end of the record in
100% of them. Non-septic records run to discharge (up to 336 rows). Two
consequences, both measured from counts and neither shown to be exploited by the
model:
- The 24-hour filter removes every septic stay whose first label row is before
  row 14 (a record needs 24 rows and a septic record is about 10 rows past its
  first label). That is why the first positive label row among kept stays is 14
  or later.
- For about 75% (Site A, 255 of 340) and 73% (Site B, 221 of 304) of positive
  kept stays the record is shorter than 48 rows, so the head's read-out hour is
  the end of the labeled segment. If the challenge's documented 6-hour label
  shift applies, that is about 4 hours after clinical onset.

**Filters and cohort table (PhysioNet, reproduced from the raw files).**

| Step | Site A | Site B |
|---|---|---|
| Raw files (one stay each) | 20,336 | 20,000 |
| Dropped: fewer than 24 hourly rows (`physionet2019.py:113`) | 4,499 | 5,176 |
| Dropped: all values missing (`:119`) | 0 | 2 |
| Dropped: no single hour with SBP, DBP and MAP all observed, before forward fill (`:127`) | 5,456 | 43 |
| Kept | 10,381 | 14,779 |
| Positive (task label) | 340 | 304 |
| Prevalence, task label | 3.28% | 2.06% |

The same filters in MIMIC-IV and eICU are in `src/data/mimic4.py:17, 43-45,
189` and `src/data/eicu.py:233-239, 307` (adult, ICU stay of at least 24 hours,
one hemodynamic hour). I did not re-derive the 74,607 and 130,446 external
counts: the loaders need more memory than the local machine has free. They are
carried.

**Why 3.3% / 2.1% rather than about 7%.** Prevalence of "ever septic" in the
same files is 8.80% (Site A) and 5.71% (Site B), about 7.3% overall. The task
label counts only onsets inside the first 48 rows: 5.58% (A) and 3.72% (B) of
all files. After the 24-hour and hemodynamic filters the task prevalence is
3.28% and 2.06%. Among kept stays that are ever septic (865 at A, 702 at B),
525 (60.7%) and 398 (56.7%) have onset after hour 48 and are labeled 0.

**Age.** No age filter for PhysioNet; Site B has 22 of 20,000 files under 18
(minimum 14), Site A none.

## G2. Label provenance

- Site A and Site B: the challenge's own hourly `SepsisLabel`, not re-derived
  here (`physionet2019.py:131`).
- MIMIC-IV and eICU: this repo's SOFA-based Sepsis-3 derivation
  (`src/data/sofa_sepsis.py`), called with `mode="single"`
  (`mimic4.py:132`, `eicu.py:253`): SOFA at exactly the suspicion time. The
  label is positive if criteria are met at any time in the stay and is
  broadcast over all 48 hours (`mimic4.py:233-235`, `eicu.py:354-355`;
  `sofa_sepsis.py:205-243`). Definition and simplifications:
  `sofa_sepsis.py:1-36`. Missing SOFA components score 0.
- eICU suspicion differs from MIMIC-IV's: culture OR an infection diagnosis from
  a broad pattern list including pneumonia, UTI, cellulitis
  (`sofa_sepsis.py:68-77`), versus antibiotic-culture pairing for MIMIC-IV
  (`:402-445`).
- The original ICD-derived labels no longer reach the loaders. The code that
  produces them still exists (`src/data/sepsis.py:48-89`, explicit-sepsis ICD-9
  038.x, 995.91, 995.92, 785.52 and ICD-10 A40, A41, R65.20, R65.21) and is used
  only by the sibling detector project. Regenerating them for a re-score needs
  new code outside the shared loaders, keyed by `hadm_id` / `patientunitstayid`.

## G3. Constraint inputs

Raw PhysioNet files (41 columns, checked on files of both sites): HR, O2Sat,
Temp, SBP, MAP, DBP, Resp, EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, ...
**There is no PaO2 column.**

| Variable | PhysioNet | MIMIC-IV | eICU |
|---|---|---|---|
| SpO2 | `O2Sat` (`variables.py:55`) | chart item 220277 (`:76`) | vital `sao2` (`:113`) |
| pH | `pH` | lab 50820, blood gas | lab `pH` |
| pCO2 | `PaCO2` | lab 50818, blood gas | lab `paCO2` |
| HCO3 | `HCO3` (raw column; its measurement source is not documented in the repo) | lab 50882, **Chemistry** panel (verified in `d_labitems`; not the blood-gas "calculated bicarbonate" 50803) | lab `bicarbonate` (chemistry name) |
| PaO2 | **reconstructed by the loader** from SaO2 | lab 50821, blood gas (measured) | lab `paO2` (measured) |

PaO2 reconstruction: `physionet2019.py:47-55`, `cbrt(23400*S/(1-S))` clipped to
[20, 700], applied where SaO2 is present and PaO2 is not. It drops the
`150*PaO2` term of the forward function in `src/losses/pcl_loss.py:10-18`, so it
is an approximate inverse. The reconstruction is not in the released dataset; it
is in this repo. Share of hours with a reconstructed PaO2 (kept stays, first 48
h): 7.29% at Site A, 1.94% at Site B; every PaO2 value at those sites is
reconstructed. HCO3 fallback from base excess (`:41-45`) is never used because
the HCO3 column exists.

Severinghaus term inputs: the model's SpO2 variable (pulse oximetry) and its
PaO2 variable (`pcl_loss.py:169-177`). SaO2 itself is not a model variable.

Not measured: share of MIMIC-IV/eICU observations per variable (needs the large
tables; goes in experiment F on a pod).

## G4. Loss mechanics

- Where the constraint penalty is computed: positions where the constraint's
  variables are all present in the (post-fill) mask AND at least one of them was
  masked in the current draw (`pcl_loss.py:61-79`). One masked variable is
  enough; the reviewer's "at least two" is not what is implemented.
- Inputs to the residual: the model's **predictions** for every variable at the
  position, not observed values (`pcl_loss.py:100-109`; call at
  `train_utils.py:102`). Unmasked variables enter as reconstructions, which the
  masked loss does not supervise at unmasked positions
  (`train_utils.py:55-66`). Whether the penalty is redundant with the masked loss
  when exactly one variable is masked is an empirical question (experiment E2).
- Masking: independent per (hour, variable) among post-fill observed entries,
  30% (`backbone.py:271-283`, `config.py:48`). Whole-timestep masking is not
  implemented.
- L1 for MAP (`pcl_loss.py:56-59, 124`), squared error for Henderson-Hasselbalch
  and Severinghaus (`:51-54, 161, 176`). Pulse-pressure and shock-index terms
  exist but have weight 0 (`:38-45, 184-187`). **No recorded rationale** for L1
  vs squared in the code comments or in the history that survives (the shared
  repository history is gone).
- Not in the paper: a loss balancer rescales the constraint loss by an EMA of
  `min(20, L_mask / L_phys)` (`train_utils.py:12-50, 215`). See D5.
- Warmup: `lam_eff = lam * epoch / warmup_epochs` for the first
  `max(2, int(n_epochs * warmup_fraction))` epochs (`train_utils.py:220-242`).
  Default fraction 0.3 (`:188`).

## G5. Composition-to-gap ratio

Defined in the detector project, `reboot/detectors/checks/
check5_missingness_scale.py:99-134` (`rel(t) = |t0 - t1| / max(|t0|, |t1|)`):

    naive aggregate(site)  = mean over stays of [mean over that stay's computable
                             constraints of the stay's mean residual]
    composition-only(site) = same, but each constraint's residual replaced by
                             its pooled mean over both sites (only the set of
                             computable constraints varies)
    ratio = rel(composition-only) / rel(naive)

Residuals there: MAP `|MAP-(DBP+(SBP-DBP)/3)|/180`; HH `((pH-pred)/1.4)^2`;
and "SpO2" is `((O2Sat-SaO2)/50)^2`, i.e. pulse oximetry against arterial
saturation, not the PaO2-based term used in the loss (`:56-74`). The "8.6x
Severinghaus residual" and the "0.0188 vs 0.0206" blood-pressure residuals are
not in any file I could read; they are carried and will be re-derived on
PhysioNet A vs B in experiment F. The ratio's known instability and the
"four of five" error are in D9.

## G6. Lambda used

Default `LAMBDA_PCL = 0.5` (`config.py:46`), with a comment that it was selected
on Site B (`:41-45`). The grid constant is {0.1, 0.5, 1, 2, 5} (`:68`). The
sweep used for the corrected results had six values including 0; per-lambda
numbers for seed 42 exist only as hardcoded values in a sibling project's
figure script (reproduced in `STOP_REPORT.md`, carried). Per-seed final lambda
for the submitted table: **not determinable** from available files (driver and
per-seed outputs are in the excluded folder or on the pod).

## G7. Inventory

- Local, outside the excluded folder: nothing for the sepsis task. The only
  checkpoints are nine fine-tuned length-of-stay models of a sibling project.
  No per-seed sepsis results.
- Excluded folder: not opened.
- Pod: eight stopped pods are listed in the account (A40 and one MIG-sliced
  B300). None was started or opened. The one pod record I read shows no
  pod-local volume; whether a network volume holds the earlier checkpoints
  and per-seed outputs is not known. Which original artifacts
  (9-variable, all-site pretraining, target-selected lambda, ICD labels)
  survive: unknown.

## G8. Baselines and budgets

| | ERM | PCL | Group DRO | XGBoost |
|---|---|---|---|---|
| Parameters | 3.24M (3.28M with head) | same | same | n/a |
| Pretraining | masked loss only, AdamW 3e-4, wd 1e-4, cosine to 0.1x, clip 0.5, 30 epochs, best val loss (`baselines.py:31-142`) | masked + lambda * balanced constraint, same optimizer, clip 1.0, warmup, best val masked loss (`train_utils.py:176-306`) | uses an ERM- or PCL-pretrained encoder | none |
| Fine-tuning | two phases: first third of epochs head only at 1e-3, then encoder 1e-4 / head 1e-3, positive-class weight (cap 100), clip 0.5, best val AUROC (`evaluate_utils.py:63-270`) | same | all parameters from epoch 1 at 1e-4, group-weighted loss with eta 0.01, same positive-class weight, clip 0.5, best val AUROC (`baselines.py:386-513`) | none |
| Epochs, batch | 30 + 30, batch 64 (`config.py:47, 51-52`) | same | not stated in code; driver unavailable | n/a |
| GPU-hours | not logged | not logged | not logged | CPU |

Group DRO groups are `env_id` = site id (`dataset.py:51`); with source-only
training that is a single group unless a driver overrides it (the helper that
would attach other environments is imported by one script but not defined
anywhere in `src/`). XGBoost: 68 features (mean, min, max, observation rate for
each of 17 variables over the post-fill mask), standard scaling, fixed
settings (200 trees, depth 4, learning rate 0.05, subsample 0.8, column
subsample 0.8, no tuning, no early stopping, no positive-class weight)
(`baselines.py:540-565, 622-632`). Training set and split are whatever the
driver passes.

## G9. Consistency with the submitted paper

See `DISCREPANCIES.md` (19 items, with D10 and D11 the "+4.7 vs +4.2" and the
abstract XGBoost-vs-DRO statements). Tables 2 and 3 values themselves could not
be checked against per-seed source files: carried.

## G10. Seed variance and positives

"4 to 7 AUROC points": the only source is a sentence in a sibling project's
text; it does not say whether it is a range or an SD, over which seeds, methods,
or sites. No per-seed data is available locally. A read-only script to look for
it on the pod exists (`scripts/extract_seed_variance.py`), not run. Positives
per cohort: Site A 340, Site B 304 (this audit); MIMIC-IV and eICU about 23.1k
and 11.5k if the reported 30.9% and 8.8% hold (carried).

## Not answered, and why

| Item | Blocked by |
|---|---|
| G6 per-seed lambda, G8 Group DRO groups and epochs, G9 table values, G10 variance, Experiment A | original drivers are in the excluded folder; per-seed outputs only on the pod |
| G1 external CONSORT, G3 external shares, F | needs a job with more memory than the local machine has free; needs a pod |
| G7 pod contents | needs a pod to be started |
