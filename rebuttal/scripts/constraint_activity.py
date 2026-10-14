"""E1: how often each constraint term is active during pretraining, and with how many masked variables.

CPU only, no model and no outcome. For each cohort, draw one masking pattern exactly as `apply_random_mask`
does (each post-forward-fill observed entry masked independently with probability 0.3) and apply the loss's
activity rule (`pcl_loss.py::_constraint_mask`): a constraint is active at a position when all its variables are
present in the post-fill mask (c_mask) and at least one of them is masked. Count the masked variables at active
positions. 5,000 stays per cohort, fixed seed.

    python rebuttal/scripts/constraint_activity.py
"""
import json
import os

import numpy as np

import rb_common as rb

CONSTRAINTS = {"MAP": (0, ["SBP", "DBP", "MAP"]), "HH": (3, ["pH", "HCO3", "pCO2"]), "Severinghaus": (4, ["SpO2", "PaO2"])}
COHORTS = [("Site A", "physionet_a"), ("Site B", "physionet_b"), ("MIMIC-IV", "mimic"), ("eICU-CRD", "eicu")]


def main():
    from src.data.variables import VAR_TO_IDX
    rng = np.random.default_rng(0)
    out = {}
    for name, key in COHORTS:
        samples = rb.load_site_cached(key, 1.0, 0)
        pick = rng.choice(len(samples), size=min(5000, len(samples)), replace=False)
        cm = np.stack([np.asarray(samples[i]["c_mask"]) for i in pick])        # (N, T, 5)
        ob = np.stack([np.asarray(samples[i]["mask"]) for i in pick])          # (N, T, V)
        draw = (rng.random(ob.shape) < 0.30) & ob
        res = {"stays": int(len(pick)), "stay_hours": int(cm.shape[0] * cm.shape[1])}
        for cname, (ci, vars_) in CONSTRAINTS.items():
            idx = [VAR_TO_IDX[v] for v in vars_]
            n_masked = draw[:, :, idx].sum(-1)
            computable = cm[:, :, ci]
            active = computable & (n_masked >= 1)
            counts = [int((active & (n_masked == k)).sum()) for k in range(1, len(idx) + 1)]
            res[cname] = {"computable_share_of_stay_hours": float(computable.mean()),
                          "active_share_of_stay_hours": float(active.mean()),
                          "active_share_of_computable": float(active.sum() / max(1, computable.sum())),
                          "masked_count_among_active": {str(k + 1): c for k, c in enumerate(counts)},
                          "exactly_one_masked_share_of_active": float(counts[0] / max(1, sum(counts)))}
        out[name] = res
        print(name, {c: round(res[c]["active_share_of_stay_hours"] * 100, 2) for c in CONSTRAINTS},
              {c: round(res[c]["exactly_one_masked_share_of_active"], 3) for c in CONSTRAINTS}, flush=True)
    rb.write_json(os.path.join(rb.RB_ROOT, "audit", "outputs", "e1_constraint_activity.json"), out)


if __name__ == "__main__":
    main()
