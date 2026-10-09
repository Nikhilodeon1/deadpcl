# Results pack (skeleton, filled as results arrive)

Every number below must point to a file in `rebuttal/` or be marked
"carried, not re-verified". Entries marked PENDING have no result yet.
Aggregates only; no patient-level data.

Common settings: PREREG A1-A3 (decisions fixed 2026-10-09 before any run).

## Evidence available now (Phase 0, verified)

- Task and label construction, cohort funnel, prevalence, constraint-input
  provenance, loss mechanics: `audit/AUDIT.md`, `audit/DISCREPANCIES.md` (D1-D19).
- External cohorts: `audit/AUDIT.md` section "F-external" and
  `audit/outputs/f_external_*.json`.
- Stop triggers: `STOP_REPORT.md`.

## By reviewer concern

IDs follow the reviewers' numbering (xM3J M1-M14, JHNu J1-J6, xCda C1-C8,
7MQE Q1-Q8, fDtQ D1-D6). Fill the one-line finding and the file for each.

| Concern | Finding | Evidence file | Status |
|---|---|---|---|
| Task definition / what is predicted | Label = event inside the 48 h input window; no horizon (D1) | audit/AUDIT.md G1, outputs/g1_*.json | verified |
| Label definition across sites | Residual mismatch remains (D2, D3) | audit/AUDIT.md G2 | verified |
| Constraint availability / forward fill | 3-23 times more hours after fill, at every site (D4) | outputs/g1_physionet_audit.json, f_external_*.json | verified |
| PaO2 meaning differs between sites | reconstructed at PhysioNet, measured at MIMIC-IV / eICU (D8) | audit/AUDIT.md G3 | verified |
| How lambda was chosen | cannot be established from the code (D6) | DISCREPANCIES.md | unresolved |
| Main comparison, 10 seeds, ERM / PCL / XGBoost | PENDING | tables/main | not run |
| Lambda sweep, selection regret | PENDING | tables/sweep | not run |
| Diagnostic H (inputs cut before onset) | PENDING | tables/horizon | not run |
| AUPRC, ECE, Brier | PENDING | tables/main_descriptive.csv | not run |
| Group DRO | PENDING (groups ill-defined with one source site, D13) | | not run |
| Experiment A reproduction, all-site pretraining | out of budget | COST_LOG.md | not run |
| Composition-to-gap ratio | "four of five site pairs" sentence wrong (D9) | DISCREPANCIES.md | verified (log) |

## Not run / not resolved (stated, not hidden)

See PREREG A2 item 6 for the tier order; whatever is not in a table above was
not run.
