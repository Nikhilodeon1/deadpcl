# Dead PCL

Source and supporting code for *Plausible, Scrutinized, Dismantled: A Rigorous
Negative Result for Physiology-Constrained Cross-Hospital Transfer*, submitted
to the ML4H 2026 Proceedings track.

Physiology-constrained pretraining initially appeared to improve cross-hospital
sepsis prediction, beating an unconstrained baseline by 6-10 AUROC points at
every held-out hospital and surviving a full battery of robustness checks.
Systematic re-evaluation traced the entire gain to five evaluation confounds;
correcting them reverses the result. The paper reports both the original and
corrected pipelines in full and states each confound in mechanistic,
generalizable terms.

The manuscript itself is anonymized for double-blind review and is not
included in this repository (see Anonymity below); this repo holds the
supporting reproducibility code.

## Contents

```
jmlr.cls, jmlrutils.sty   official ML4H LaTeX class and utilities
images/                   template assets required by jmlr.cls
scripts/                  reproducibility code (see below)
config.py, src/           vendored model/data/training implementation,
                          used only by scripts/run_confound_ablation.py
```

`config.py` and `src/` are copies of the companion confound-detection
project's code, taken at a fixed point in time (see [VENDORED.md](VENDORED.md)
for provenance) — they are not kept in sync automatically.

## Scripts

- **`scripts/extract_seed_variance.py`** — read-only, no GPU, no training.
  Searches a RunPod volume for existing per-seed AUROC results to source the
  paper's seed-variance figures from raw data rather than a summary.
- **`scripts/run_confound_ablation.py`** — measures each confound's individual
  contribution to the headline result. Verified against the underlying model
  API but not run: at this pipeline's seed-noise floor, a single confound's
  effect is not expected to separate cleanly from noise, so the paper reports
  the five confounds jointly rather than ranked (see the paper's Limitations
  section). Kept here in case that calculus changes.

Both scripts call `pod_monitor.watch_pod()`, which prints an alert if a
RunPod GPU sits idle during data loading (switch to a cheaper pod) or resumes
activity after idle (switch back).

## Build

```
pdflatex pcl_ml4h_draft
bibtex   pcl_ml4h_draft
pdflatex pcl_ml4h_draft
pdflatex pcl_ml4h_draft
```

Compiles against the real ML4H `jmlr.cls`. The submission track is set via
`\mlhtrack{proceedings}` in the preamble; `\iffinal` switches between the
review and camera-ready layouts.

## Anonymity

This is a double-blind submission. The manuscript source, bibliography, and
compiled PDF are excluded from this repository rather than redacted, since a
partial redaction is easy to get wrong. The paper's Data and Code Availability
statement points to a separate anonymized code repository; that link is not
yet live.
