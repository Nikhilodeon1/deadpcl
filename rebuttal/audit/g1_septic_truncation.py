"""
Phase 0 audit, quick check: how many hourly rows does a septic PhysioNet record
contain after its first positive SepsisLabel row? Reads only the label column.
Aggregates only. Usage: PHYSIONET_DIR=... python g1_septic_truncation.py out.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

base = os.environ.get("PHYSIONET_DIR", os.path.join("data", "physionet2019"))
out = {}
for site in ("training_setA", "training_setB"):
    rows_after, first, length, stays_one_to_end, n_sept, n_all = [], [], [], 0, 0, 0
    for f in sorted(os.listdir(os.path.join(base, site))):
        if not f.endswith(".psv"):
            continue
        n_all += 1
        lab = pd.read_csv(os.path.join(base, site, f), sep="|", usecols=["SepsisLabel"])["SepsisLabel"].values
        if (lab > 0).any():
            n_sept += 1
            h1 = int(np.argmax(lab > 0))
            first.append(h1)
            length.append(len(lab))
            rows_after.append(len(lab) - h1)
            stays_one_to_end += int((lab[h1:] > 0).all())
    ra, fi, ln = map(np.array, (rows_after, first, length))
    out[site] = {
        "n_files": n_all, "n_septic": n_sept,
        "rows_from_first_label_to_end": {"min": int(ra.min()), "median": float(np.median(ra)),
                                          "p95": float(np.percentile(ra, 95)), "max": int(ra.max())},
        "first_label_row": {"min": int(fi.min()), "median": float(np.median(fi)), "max": int(fi.max())},
        "record_length": {"min": int(ln.min()), "median": float(np.median(ln)), "max": int(ln.max())},
        "frac_label_stays_1_to_end": stays_one_to_end / max(1, n_sept),
    }
    print(site, json.dumps(out[site]), flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "g1_septic_truncation.json", "w"), indent=2)
