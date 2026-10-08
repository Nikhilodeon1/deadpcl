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
