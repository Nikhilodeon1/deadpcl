# Internal only: overlap with the companion selection-criteria work (not for the paper, not for the writing agent)

Status: NOT CHECKED against the companion project's artifacts; only facts available to this audit are listed.

- The lambda grid used here {0, 0.1, 0.5, 1, 2, 5} coincides with the candidate set hardcoded in the companion
  project's figure script (`chat2_papers/paper/make_figure.py:29-37` in the shared workspace). That script also contains
  a seed-42 table of source-validation and target AUROC per lambda. Those numbers were carried here as a warning and
  were NOT reproduced by the new runs (see `STOP_REPORT.md`, 2026-10-11 update).
- The companion's reported selection regrets (0.052 vs 0.012) were not re-derived here. Experiment C's label-free
  selectors (reconstruction error, constraint violation on unlabeled target inputs) were not run, so no mapping
  between the two works' artifacts can be confirmed from this pack.
- Source-validation selection and an oracle selection over this grid exist in the pack
  (`pack/tables/final/main_comparison.csv`, arms PCL_srcsel and PCL_oracle, 5 seeds). A disclosure of the shared grid
  to the venue is a decision for the owner.
- Unlabeled-target selection would be a transductive setting, not zero-shot, if it is added later.
