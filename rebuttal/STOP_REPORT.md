# Stop report (Phase 0)

Phase 2 (any training) has not started. This report lists each stop trigger
from the work order, what was seen, the evidence, and what I did not do.

## Trigger A: task construction (FIRED)

What I saw. The per-stay label is "any hourly SepsisLabel = 1 inside the first
48 rows", and the model input is the same first 48 hours. The encoder is causal
but the classification head reads the last observed hour, so it sees the whole
history including the label hours.

Evidence.
- `src/data/dataset.py:29-30` (stay label = any positive hour), 
  `src/data/physionet2019.py:131-134` (labels cut to the first 48 rows),
  `src/models/backbone.py:93-99` (causal mask), `:130-164` (head reads the last
  observed hour).
- Raw-data measurement (`audit/outputs/g1_physionet_audit.json`, 340 positive
  kept stays at Site A, 304 at Site B): the number of rows at or after the first
  positive label row has median 10, 25th-75th percentile 9-10, maximum 11. The
  first positive label row is between row 14 and row 47. So for every positive
  stay the input window contains the label hours: roughly the hours leading up
  to and just after the labeled onset.
- Supplementary (`audit/outputs/g1_septic_truncation.json`): all 2,932 septic
  records in the two files end at most 10 rows after the first positive label
  row, and the label is 1 to the end of the record; non-septic records run to
  discharge. For about 75% (Site A) and 73% (Site B) of positive kept stays the
  record is under 48 rows, so the head reads out at the end of the labeled
  segment. The 24-hour filter also removes every septic stay with a first label
  row before row 14.
- What this does not show: I did not measure how much of the ERM/PCL AUROC
  comes from those hours. It is not a label computed from inputs, and it hits
  ERM and PCL identically, so it does not by itself explain the PCL-ERM margin.
  It changes what the AUROC means (detection around onset, not forecasting at a
  horizon) and therefore how Q4 / the task-definition questions must be
  answered.

Related, larger issue (see DISCREPANCIES D2): external labels are "Sepsis-3 at
any time in the ICU stay" (single-point SOFA at the suspicion time), while the
source label is "label hour inside the first 48 rows". Site B shares the
source's definition; MIMIC-IV and eICU do not.

Not done. No variant of the task was built or run. I did not change the shared
loaders.

Decision needed. Either (a) keep the submitted definition and describe it
exactly, running every experiment on it, or (b) additionally define a
horizon-based variant (inputs cut some hours before the labeled onset) and run
the comparison on both. (b) needs a design choice I should not make alone and
roughly doubles the data work.

## Trigger E: projected cost exceeds BUDGET_CAP_USD (FIRED)

Projection in `COST_LOG.md`: about $99 (A40) to $266 (H100) for the full work
order, against a cap of $15. No timed run exists; the per-cycle times are
carried estimates, so the projection could be off by a large factor in either
direction. Nothing was started.

## Trigger F: evidence of a sixth confound (CANDIDATES, not a verdict)

Two candidates with verified evidence, one with partial evidence:
1. Residual label-definition shift (D2): temporal scope differs between the
   source and the two big external sites even after the "fix". Verified in code.
   Effect on the PCL-ERM margin not measured.
2. PaO2 means different things at source and targets (D8): reconstructed with
   an approximate inverse from SaO2 at the PhysioNet sites, measured at
   MIMIC-IV and eICU. This is wider than the "partial circularity" in the
   paper: it is a variable-definition shift on an input and on a constraint
   term. Verified in code and raw data. Effect not measured.
3. Constraint availability is inflated by forward fill (D4); the training
   signal for the acid-base constraints at Site A is 27.7% of hours post-fill
   vs 3.04% raw. Verified. Effect not measured.
I have not decided whether any of these counts as a "sixth confound" in the
paper's sense; that is the owner's call.

## Trigger D: contradictions with the submitted paper (FIRED, as text-vs-code)

No new numeric result contradicts a table. The contradictions are between the
paper's descriptions and the code or raw data; all 19 are in
`audit/DISCREPANCIES.md`. The ones most likely to matter to a reviewer: D1, D2,
D6 (how lambda was chosen), D8, D9 (the "four of five site pairs" sentence is
wrong as worded).

## Trigger B: PCL beats ERM with a paired interval above zero (NOT FIRED, but a warning)

No new run exists, so this did not fire. However, the only per-lambda sweep
data I could find (seed 42, carried from a sibling project's figure script, not
re-verified) shows the hindsight-best lambda beating the unconstrained model by
a wide margin at all three external sites:

| lambda | source val | Site B | MIMIC-IV | eICU |
|---|---|---|---|---|
| 0.0 | 0.8350 | 0.6923 | 0.6006 | 0.5479 |
| 0.1 | 0.8306 | 0.7371 | 0.6908 | 0.6366 |
| 0.5 | 0.8249 | 0.7495 | 0.6952 | 0.6623 |
| 1.0 | 0.8271 | 0.6749 | 0.6451 | 0.5848 |
| 2.0 | 0.8203 | 0.6527 | 0.6326 | 0.5617 |
| 5.0 | 0.8280 | 0.5972 | 0.6076 | 0.5749 |

Hindsight-best vs lambda = 0 in this seed: about +5.7 (Site B), +9.5 (MIMIC-IV),
+11.4 (eICU) AUROC points. If this holds over seeds, the paper's headline
("PCL falls below ERM at every site") depends on how lambda is selected, not
on PCL per se. Experiment C is built to test exactly this, and its oracle
comparison would very likely trigger B. I am flagging it now so that the
handling is decided before any run, not after.

## Trigger C: Experiment A gate

Not evaluated; Experiment A was not run (the original drivers are in the
excluded folder).

## What I did not do

- No training, no pod started, no GPU time, $0 spent.
- No edits to shared code of the sibling projects; no changes to the loaders.
- Did not read the excluded archive folder, which blocks G6, G8 (Group DRO
  groups), G9, G10 and Experiment A.
- Did not push anything; the `rebuttal` branch is local.

## Update 2026-10-11: after the 40 PhysioNet-side arms (Site B only; external sites still being scored)

Source: `outputs/runs/*.json`, per-stay predictions, `pack/tables/siteb_interim/` (interim: MIMIC-IV / eICU
are not yet scored for all arms, ignore those rows).

- Trigger B (PCL beats ERM with a paired interval above zero): NOT fired at Site B. Lambda = 0.5 minus
  ERM over 10 seeds (42-51): mean difference -0.026 AUROC, two-level bootstrap 95% interval
  [-0.063, +0.013], t-interval [-0.066, +0.014]: no evidence of a difference in either direction. The
  seed-to-seed SD of one arm is 4-5 points. Not yet evaluated at MIMIC-IV and eICU.
  The oracle row at Site B is lambda = 0 (best mean Site B AUROC over seeds 42-46), so the oracle
  difference to ERM is exactly 0 there.
- Trigger D (new numbers vs carried numbers): the carried single-seed sweep from the sibling project's
  figure script is NOT reproduced. Seed 42, Site B AUROC, carried vs new:
  lambda 0: 0.6923 vs 0.7219; lambda 0.1: 0.7371 vs 0.7150; lambda 0.5: 0.7495 vs 0.7138;
  lambda 1: 0.6749 vs 0.6321; lambda 2: 0.6527 vs 0.6969; lambda 5: 0.5972 vs 0.7112.
  Source-validation AUROC for lambda 0 is close (0.8350 vs 0.8363). Differences of 3-5 points per cell
  are the size of the seed-to-seed SD (4-5 points), and the hardware, library versions and (possibly)
  pipeline details differ from the original run, which I cannot inspect. The earlier warning that a
  hindsight-best lambda beats lambda = 0 by 5-11 points is therefore not supported by the 10-seed data
  at Site B; a single-seed sweep is not a reliable basis for a claim.
- XGBoost (fixed recipe, 10 seeds) is 0.042 AUROC above ERM at Site B with a bootstrap interval that
  includes zero [-0.005, +0.087] and 0.068 above lambda = 0.5 with an interval above zero
  [+0.023, +0.115].

## Update 2026-10-12: Trigger B fired on the oracle row only (handled as pre-registered in PREREG A2 item 3)

Source: `pack/tables/final/paired_contrasts.csv`, `lambda_sweep.csv`, 40 PhysioNet-side arms scored on all four targets.

- Fired: "PCL_oracle - ERM" at eICU, 5 seeds (42-46), mean difference +0.039 AUROC, two-level bootstrap
  95% interval [+0.023, +0.057]. The oracle picks, per target site, the lambda with the best mean target AUROC;
  at eICU that is lambda = 5 (mean 0.609 over seeds 42-46; ERM on the same five seeds is about 0.570, on all ten seeds 0.594). It is labeled "oracle (target-selected,
  not deployable)": the interval does not account for choosing lambda on the same target data and seeds
  (selection bias), so +0.039 is an upper-end estimate of what a user could obtain.
- Not fired in any deployable variant:
  - lambda = 0.5 minus ERM, 10 seeds: Site B -0.026 [-0.063, +0.013]; MIMIC-IV +0.001 [-0.021, +0.020];
    eICU -0.003 [-0.031, +0.027].
  - lambda chosen per seed by source-validation AUROC minus ERM (5 seeds, lambda grid fully run for seeds 42-46 only):
    Site B +0.008 [-0.003, +0.022]; MIMIC-IV +0.019 [0.000, +0.041]; eICU +0.016 [0.000, +0.034].
    The lower bounds are at zero because the selector picked lambda = 0 for part of the seeds (difference exactly 0);
    verdict "no evidence of difference". This is the closest deployable variant to firing and is stated as such.
- Consequence for the claim (decision recorded in PREREG A2): the sign of PCL - ERM depends on how lambda is
  chosen. With a documented value (0.5) there is no evidence of a difference at any site over 10 seeds.
  With a target-selected value at eICU there is a gain. The paper's sentence "PCL falls below ERM at every site"
  is not supported by the 10-seed paired data in any row.
- Not done: the timestep-masking variant and the no-Severinghaus variant (the two variants the work order names as
  especially likely to fire B) were not run; budget and time went to the main comparison and the sweep.

## Update 2026-10-14: attribution variants (PREREG A6; five seeds, all three external sites)

Source: `pack/tables/attribution.csv`, `attribution_formatted.csv`, Fig. 3.

- Trigger B: NOT fired in any variant. The largest PCL-minus-ERM margins are +0.022 (no Severinghaus term, eICU,
  [-0.015, +0.051]) and +0.014 (MIMIC-IV, [-0.032, +0.044]); no interval lies above zero. The variants the work order named as most
  likely to fire it (timestep masking, no Severinghaus) did not.
- Variants that moved the margin did so in the direction of PCL doing worse: timestep masking at eICU (-0.039, [-0.072, -0.011];
  change vs main -0.061 [-0.116, -0.002]) and the 9-variable pipeline at MIMIC-IV (-0.030, [-0.044, -0.017]; change -0.046
  [-0.079, -0.012]). Two of twelve cells meet the "explanatory" rule; five seeds and twelve comparisons make that fragile.
- Trigger F (sixth confound): no new evidence beyond the Trigger F section above.
- Not run (so no statement): label definition, all-site pretraining, Experiment A.
