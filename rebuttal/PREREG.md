# Pre-registration (v0)

Committed before any new training run. This file is append-only after the
commit: corrections go in a dated addendum at the bottom with a reason, never
as an edit to the text above. Nothing here is chosen after seeing a new result.

Status of this version: the audit (rebuttal/audit/AUDIT.md) found a task-
construction issue (STOP trigger A, see STOP_REPORT.md) and several items that
cannot be settled from code alone. Every experiment below therefore has a
**PENDING** field that must be resolved by an addendum before that
experiment's first run. Decision rules, metrics, and the statistics are fixed
now.

## 1. Seeds, metrics, statistics

- Seeds. The original runs used `PCL_SEED` in {42, 43, 44} (config.py:7). These
  three stay in every aggregate. New seeds are 45, 46, 47, 48, 49, 50, 51.
  "Seeds 1-5" in the work order is read as the first five of this list
  (42-46); "10 seeds" is 42-51.
- Primary metric: target AUROC at each external site (PhysioNet Site B,
  MIMIC-IV, eICU-CRD). Secondary: AUPRC (with prevalence baseline), ECE (10
  equal-width bins), Brier.
- All numbers reported to 3 effective digits. Standard deviations are labeled
  SD (sample SD, n-1). Ranges are labeled as ranges.
- Interval estimation, fixed now:
  - Single model, single site: percentile bootstrap, 1000 resamples of stays.
  - Paired difference between two methods on the same stays: resample stays
    jointly for both models, difference of AUROCs, 1000 resamples.
  - Difference aggregated over seeds (primary): two-level percentile
    bootstrap, 1000 resamples: resample seeds with replacement, then stays
    within each drawn seed; statistic is the mean over drawn seeds of the
    paired difference. 95% interval = 2.5th and 97.5th percentiles.
  - Robustness view (reported beside, never instead): t-interval over the
    per-seed paired differences, df = n_seeds - 1.
- Every new run saves per-stay predictions (hashed id, label, score) so that
  every interval, AUPRC, and calibration number can be recomputed.

## 2. Decision rules (fixed)

R1. "PCL is worse than ERM" at a site is claimed only if the paired PCL-ERM
    difference has a 95% interval (primary method above) entirely below zero
    at that site over 10 seeds. Otherwise the claim is "no evidence of
    benefit".
R2. A factor is "explanatory" for the PCL-ERM margin at a site if flipping it
    moves the paired margin by more than the SD of the margin over seeds AND
    the interval of the change excludes zero.
R3. Experiment A gate: the original-pipeline PCL-ERM margin reproduces its
    sign at all three external sites and lies within the originally reported
    mean +/- 2 SD of the margin (carried from the earlier draft: Site B
    +6.4 +/- 1.6, MIMIC-IV +10.1 +/- 2.1, eICU +8.4 +/- 2.3 points; bands
    [3.2, 9.6], [5.9, 14.3], [3.8, 13.0]). Failure triggers STOP C.
R4. Results that hurt the paper's current story get the same prominence and
    placement as results that help it. Seeds 42-44 are never dropped.
R5. No threshold, seed set, or protocol is changed after seeing results.

## 3. Experiments

Primary readout everywhere: target AUROC and paired PCL-ERM margins with
intervals per site.

### A. Reproduce the original pipeline (gate)
Config: 9 variables, 50% masking, lambda grid {0.1, 0.5, 1, 2, 5} with
lambda = 1.0 chosen on a target site, warmup 30%, ICD-derived external labels,
PCL pretrained on all four sites, ERM source-only. Seeds 42-46.
PENDING: the original driver and shell scripts are not available outside the
archive folder, which the work order excludes. Cannot be specified further
(or run) until that is resolved.
Claim if gate passes: the attribution ladder (B) is meaningful. If it fails:
STOP C.

### B. Attribution
Factors: F1 external eval labels (ICD vs Sepsis-3; MIMIC-IV and eICU only);
F2 pretraining data (original asymmetric: PCL all-site / ERM source-only;
source-only for both; all-site for both); F3 lambda selection (target AUROC vs
source validation); F4 variable set (9 vs 17); F5 other hyperparameters (mask
rate 50 vs 30, warmup 30 vs 40, lambda grid); F6 Severinghaus term on vs off.
- B0 (no retraining): rescore corrected models on ICD labels; rescore original
  models on Sepsis-3 labels if checkpoints exist. Readout: change in the paired
  margin. PENDING: needs per-stay predictions or checkpoints (pod inventory).
- B1 leave-one-in: corrected pipeline with one factor flipped back.
- B2 leave-one-out (Tier 2): original pipeline with one factor corrected.
- ERM and PCL in every config, seeds 42-46, three external sites.
- Site B readout: ICD labels were never used there; its margin change is
  explained with F2-F6 only.
- Tier 1 = B0, F2, F3, F6, F4. Tier 2 = F5, B2, only after a cost check.
- Classification of factors (asymmetric: F2, F3, F6; level/measurement: F1 and
  the composition artifact) is a hypothesis; the data decide, and a measurement
  factor that moves the margin is reported as doing so.
PENDING: F3's "target" selection needs the original selection procedure.

### C. Lambda sweep and a self-contained second test
Corrected pipeline, lambda in {0.0, 0.1, 0.5, 1.0, 2.0, 5.0} (the set used by
the sibling sweep; lambda = 0 is the unconstrained model run through the PCL
code path), plus a true ERM run (separate pretraining routine; they differ in
gradient clipping and checkpoint criterion). Seeds 42-46.
Per lambda and seed: source-validation AUROC, target AUROC per site,
reconstruction MSE and mean constraint violation on unlabeled target inputs.
Readouts:
- best-in-hindsight lambda vs ERM, paired, per site;
- selected-lambda vs ERM where the lambda is chosen by each of four selectors:
  source-validation AUROC, unlabeled-target reconstruction error, unlabeled-
  target constraint violation, random;
- label-free selection regret = oracle target AUROC minus selected-config
  target AUROC, averaged over seeds and sites.
The unlabeled-target selectors are transductive (they see target inputs); this
is stated wherever the result is reported.
Pre-registered distinction: hindsight-best lambda uses target labels and is
reported as an oracle bound, not as a method. A paired interval above zero for
the oracle comparison is reported to the project owner before any text is
written (see STOP B handling in the work order).
Overlap handling: if the sweep artifacts coincide with the sibling selection-
criteria project's, reuse them, confirm the regret numbers against that
project's reported values, and write the mapping to OVERLAP_INTERNAL.md only.
PENDING: whether per-(lambda, seed) checkpoints from the existing sweep exist
on the pod (inference-only reuse vs retraining).

### D. Main corrected comparison, 10 seeds
ERM, PCL, Group DRO, XGBoost; seeds 42-51. Per-seed table, mean, SD, range,
paired differences PCL-ERM and DRO-ERM with intervals, per site. Variance
diagnosis: fine-tune seed vs pretrain seed vs split (Tier 2, 3x3 grid). Positives
per cohort and a check of whether small positive counts explain the variance.
PENDING: PCL's lambda for this table must be fixed by a stated rule before the
run; candidate rules are source-validation selection (the paper's stated rule)
and lambda = 0.5. Both will be reported if both are run; the primary is
whichever the addendum names.

### E. Constraint mechanics
E1 per-constraint activity (fraction of masked positions where active; number
of simultaneously masked constraint variables among active positions, in
particular the fraction with exactly one). E2 for exactly-one-masked positions,
redundancy of L_phys with L_mask (per-position loss correlation, gradient
cosine). E3 whole-timestep masking (30% of timesteps), ERM vs PCL, corrected
pipeline, source-only pretraining, lambda by source validation, seeds 42-46;
a test of the reviewer's hypothesis, not a rescue. E4 (Tier 2) all-squared
loss vs mixed L1/squared, 3 seeds. E5 constraint-randomization rerun on the
corrected pipeline (real, shuffled, wrong-equation, noise, ERM), seeds 42-44
minimum. E6 MAP-identity residual distribution by cohort.

### F. Data audits (CPU, no model)
F1 per-constraint availability per cohort with intervals; F2 near-zero-residual
fraction per constraint per cohort as a proxy only (no circularity claim from
it), HH split by bicarbonate source where item ids allow; F3 fraction of PaO2
reconstructed vs measured per cohort; F4 composition audit, recomputed on the
four cohorts, with the formula in AUDIT.md G5.

### G. Metrics
AUPRC with prevalence baseline, ECE, Brier, reliability-curve data for ERM,
PCL, Group DRO, all external sites, from saved predictions.

## 4. What will be claimed under each outcome (summary)

- Margin interval entirely below zero: "worse". Interval includes zero: "no
  evidence of benefit". Entirely above zero: reported as is, with STOP B.
- A factor meeting R2: described as explanatory; not meeting it: described as
  not distinguishable from seed noise, with the interval.
- Anything not run is listed as not run, with the reason; nothing is inferred
  for it.

## 5. Budget

BUDGET_CAP_USD = 15. CPU first. Before any GPU job: one timed run, projection
logged to COST_LOG.md; stop and report if the projection exceeds the cap.
Tiers run in the order above; Tier 2 only after the cost check.

## 6. Open items that block the first run

1. Access to the original pipeline drivers (excluded folder) and to the pod
   volume (inventory of checkpoints, per-seed outputs).
2. Resolution of STOP trigger A (task construction), which changes what the
   AUROC means and therefore what every experiment above measures.
3. Which lambda-selection rule defines "PCL" in the main table (D).
4. Seed naming (42-44 vs "1-3") as assumed in section 1.

## Addenda (append-only)

### A1 - timing-only run and pod inventory (before any experiment run)

Written before the run; nothing below changes any rule above.

- Allowed now: (a) a read-only inventory of the compute pod
  (`rebuttal/pod/inventory.sh`); (b) one dry run at 5% of the data for 1 epoch
  and one full-size source-only cycle (`rebuttal/scripts/time_cycle.py`) with
  seed 9001, to measure wall-clock per stage. Seed 9001 is outside the
  experiment seeds (42-51).
- The metrics of the timing runs are not an experimental result. They are not
  reported, not used to choose any setting, and not pooled with any seed. The
  per-stay predictions are still saved (hashed id, label, score).
- Pod used for these runs: 16 CPU cores, 64 GB RAM, one V100 GPU (owner's
  statement; price per hour to be recorded in COST_LOG.md).
- Not allowed until a further addendum: any run with seeds 42-51, any lambda
  sweep, any DRO / XGBoost run. Open items 1-4 above still block those.


### A2 - decisions on the open items (written 2026-10-09, before any experiment run)

The owner delegated these decisions. They fix the rules for everything that
follows; they cannot be changed after results are seen.

1. Task definition (STOP A). The submitted definition is kept and described
   exactly as implemented: per-stay label = any positive hourly label in the
   first 48 rows; input = the same 48 rows; the head reads the last observed
   hour. All main experiments use it. No horizon-based model is trained
   (retraining on a second task roughly doubles the data work and does not fit
   the budget or the deadline). Instead, one inference-only diagnostic (H):
   the trained models are scored with each positive stay's input cut so that it
   ends h hours before the first positive label (h = 0, 6, 12), with negative
   stays cut to a length drawn from the positives' cut-length distribution.
   Reported as a diagnostic of the existing models, not as a horizon-trained
   model. A horizon-trained comparison is stated as a limitation.
2. Archive folder. Not read. Quantities that need the original drivers
   (per-seed lambda, DRO groups and epochs, table values, seed-variance
   provenance, Experiment A) stay unresolved and are reported as unresolved.
   If the pod inventory shows copies of those drivers or outputs, they may be
   used and are cited by their path on the pod.
3. Oracle lambda (STOP B). The lambda sweep is run and reported with equal
   prominence whatever it shows. The best lambda chosen on target-site AUROC is
   labeled "oracle (target-selected, not deployable)". If any variant beats
   ERM with a paired interval above zero, STOP B is written up in
   STOP_REPORT.md, work continues (owner pre-authorized), and the headline
   claim is restated as "the sign of PCL - ERM depends on how lambda is
   chosen", not as "PCL is worse".
4. Definition of "PCL" in the main table. Three rows, all reported:
   (i) PCL lambda = 0.5 (the only value with documented provenance in the code),
   (ii) PCL with lambda chosen per seed by source-validation AUROC over the
   grid {0, 0.1, 0.5, 1, 2, 5} (label-free with respect to every target),
   (iii) oracle lambda per target (labeled as in item 3). ERM is lambda = 0.
5. Seeds. 42, 43, 44 (original) and 45-51 (new), ten in total. Original seeds
   stay in every aggregate. If the budget allows fewer than ten for some arm,
   the arm is reported with the seeds it has, in the table, with n stated;
   seeds are never dropped after the fact.
6. Priority under the $15 cap (fixed order; later tiers run only if money
   remains after the earlier ones, and an unrun tier is reported as unrun):
   T1 main comparison ERM and PCL lambda = 0.5, seeds 42-51;
   T2 lambda sweep {0.1, 1, 2, 5} on seeds 42-46 (needed for items 3-4 rows
      ii and iii);
   T3 Group DRO and XGBoost on seeds 42-51 (XGBoost is CPU-only and runs
      locally at no cost);
   T4 diagnostic H, calibration, bootstrap statistics (inference / CPU only);
   T5 mechanism and attribution experiments (B, E) as budget allows.
   Experiments that need all-site pretraining (A, confound 2) are out of
   budget and are reported as not run.

### A3 - fixed training settings for all new runs (written 2026-10-09, before any experiment run)

- Constraint warmup fraction = 0.4 for every new run (the function default is
  0.3; the code comment in `config.py` and the corrected paper both say 0.4;
  the original driver that passed it is not available, see DISCREPANCIES D7).
  ERM (lambda = 0) uses the same code path and schedule, with the constraint
  weight at zero. Other settings are the repository defaults: 30 pretraining
  and 30 fine-tuning epochs, batch size 64, learning rate 3e-4 (pretraining),
  mask probability 0.30, 80/20 patient split of Site A seeded by the run seed.
- Seed 9001 (timing) is not used for any reported number.
- Seeds enter the model initialisation and the Site A split; the PhysioNet
  cache itself does not depend on the seed at fraction 1.0.
- XGBoost: the fixed hyperparameters of `src/baselines.py` (200 trees, depth 4,
  learning rate 0.05, subsample 0.8, column subsample 0.8), the same 68
  aggregate features (mean, min, max, observation rate per variable), no
  tuning, trained on the same 80% Site A split as the neural arms for the
  same seed.
- Group DRO is not run from the repository defaults without a decision on its
  groups (with Site A only there is one group, so it reduces to a reweighted
  ERM). If it is run, it is run as the repository's `DROFinetuner` with
  `env_id` groups (a single group) and reported as exactly that, not as a
  multi-site DRO.

### A3 - implementation details fixed before any result (written 2026-10-09)

- Warmup fraction for the constraint weight is 0.4 in every run (the value the
  code comments and the paper's corrected description give; the function
  default of 0.3 is not used). The ERM arm is the same code path with
  lambda = 0, so that the only difference between ERM and PCL is the constraint
  weight. This differs from the ERM routine in `src/baselines.py` (gradient
  clip 0.5 vs 1.0, D14); that routine is not used.
- Epochs: pretraining 30, fine-tuning 30 (config defaults). Batch 64, learning
  rates and fine-tuning schedule as in the code. No setting is tuned.
- Arms and seeds are produced by `scripts/run_arms.py` (T1, T2 lists fixed in
  the file) and `scripts/xgb_local.py` (XGBoost: 200 trees, depth 4, learning
  rate 0.05, subsample 0.8, colsample 0.8, standardized 68 aggregate features,
  no class weighting, the same 80% Site-A training split as the neural arms,
  seeds 42-51).
- Statistics: `scripts/analyze.py`. Paired contrasts on target-site AUROC:
  two-level percentile bootstrap (seeds, then stays; 1000 draws; stays are
  resampled once per draw and shared by both arms), t-interval as a robustness
  view. ECE uses 10 equal-width bins. A "worse" claim requires 10 seeds and an
  interval entirely below zero.
- External sites (MIMIC-IV, eICU) are scored locally from the returned
  checkpoints with `scripts/score_checkpoint.py`; PhysioNet sites are scored on
  the training machine. Per-stay predictions use hashed ids and are aligned by
  those ids.
- The compute cap is enforced by `run_arms.py --max-hours`; it never starts an
  arm that would exceed the stated wall-clock budget. Arms not reached are
  reported as not run.

### A4 - measured cost and run plan (written 2026-10-11, before tier T1 / T2 are run)

- Measured on the pod (16+ cores, one V100 32 GB; seed 9001, not an experiment
  seed): one source-only cycle (30 pretraining + 30 fine-tuning epochs, scoring
  Site B) takes 278 s of compute (pretrain 170 s, fine-tune 102 s, scoring 6 s),
  plus a one-time 157 s data load that is cached. Source: `outputs/timing_timed.json`.
- Projection: T1 (20 arms) about 1.6 h; T2 (20 arms) about 1.6 h; both about
  3.3 h. The hourly price of the pod has not been given to me; the $15 cap
  holds if it is at most about $4.50/h. The wall-clock guard stays at
  `--max-hours 2` per tier.
- Plan: run T1 then T2 unchanged from A2/A3. T3 (Group DRO) follows only after
  T1 and T2 complete and only if time remains; XGBoost (T3, CPU) is already
  complete for seeds 42-51.
- Timing-run metrics are not used or reported.

### A5 - housekeeping note (written 2026-10-11, no change to any rule)

Two sections above are both labeled A3 (both dated 2026-10-09, both written before
any experiment run). They are consistent and both remain in force: warmup
fraction 0.4, 30 pretraining and 30 fine-tuning epochs, ERM as lambda = 0 in the
same code path, XGBoost with the fixed repository hyperparameters, seeds 42-51.
The tier lists T1 and T2 are those in `scripts/run_arms.py`, unchanged since
commit 63399da, which is the code that produced the arms completed so far.
`scripts/aggregate.py` / `run_xgb.py` and `scripts/analyze.py` / `xgb_local.py`
overlap in purpose; the pack states which one produced each table.

### A6 - attribution and mechanism variants (written 2026-10-12, before any variant run)

The GPU turned out to be a free allocation, so the cost cap no longer limits these; wall-clock time does.
Variants of the corrected PCL pipeline, each flipped one factor at a time (leave-one-in), source-only pretraining on
Site A, seeds 42-46 (the five seeds on which the main arms and the lambda sweep already exist), 30 + 30 epochs,
all other settings as in A3:

- `nosev` (F6, M14, C1): PCL lambda 0.5 with the Severinghaus term removed from the constraint loss
  (`rb_variants.NoSeveringhausLoss`; verified numerically to equal the main loss minus the SpO2 term). The ERM reference
  is the main-pipeline ERM (lambda 0 does not depend on the constraint term).
- `tsmask` (E3, M8): pretraining masks whole timesteps (30% of timesteps, all observed variables at that hour), ERM and PCL
  lambda 0.5. Pre-registered as a test of the reviewer's hypothesis, not a rescue attempt.
- `vars9` (F4, C1): the 9 core variables only (first 9 columns of the cached 17-variable arrays; equal to the 9-variable
  pipeline because normalization is a fixed per-variable map and forward fill and masks are per variable), ERM and PCL 0.5.
- `orig_hp` (F5, C1): mask rate 0.5 and warmup fraction 0.3 (original values), 17 variables, ERM and PCL 0.5.

Not run: F1 (external labels: the ICD-derived labels no longer exist), F2 (all-site pretraining needs MIMIC-IV / eICU on the
GPU machine), Experiment A, Group DRO (one source group), E4/E5.

Readout (rule from the work order, unchanged): for each variant and site, margin_variant = AUROC(PCL variant) - AUROC(ERM
variant), margin_main = the same for the main pipeline on the same seeds, change = margin_variant - margin_main. A factor is
"explanatory" for the PCL-ERM margin at a site if |mean change| exceeds the seed SD of margin_main and the 95% two-level
bootstrap interval of the change excludes zero. Otherwise the claim is "no evidence that this factor changes the margin".
Trigger B is evaluated on margin_variant (interval above zero). With five seeds, small effects will not be resolved; a null
here is "not detectable with five seeds", not "no effect". Seeds are never dropped; if time stops a variant early, it is
reported with the seeds it has and marked incomplete. Order: nosev, tsmask, vars9, orig_hp. Scripts: `run_variants.py`,
`score_variants.py`, `analyze_variants.py`.
