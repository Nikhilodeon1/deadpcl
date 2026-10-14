"""Full-record label for the PhysioNet sites (variant `evlab`, PREREG A8).

The corrected pipeline labels a PhysioNet stay positive if any hourly SepsisLabel in the first 48 rows is 1, so the label hours are
inside the model's input for those stays. MIMIC-IV and eICU labels mean "Sepsis-3 at any time in the ICU stay". This module relabels
PhysioNet stays as positive if SepsisLabel is 1 anywhere in the whole record, with inputs unchanged (first 48 hours). That aligns the
definition of "ever septic in the stay" across cohorts and puts onsets after hour 48 into the positive class, with no label hours in
the input for those stays. Cohort filters are untouched (the stays are the ones the standard loader keeps).
"""
import copy
import os

import numpy as np

HOURS = 48


def ever_label(samples, physionet_dir):
    from src.data.physionet2019 import load_patient_psv, get_sepsis_labels
    out = []
    for s in samples:
        folder = "training_setA" if int(np.asarray(s["site_id"]).item()) == 0 else "training_setB"
        df = load_patient_psv(os.path.join(physionet_dir, folder, f"{s['patient_id']}.psv"))
        ever = float(np.any(np.asarray(get_sepsis_labels(df)) > 0))
        t = copy.copy(s)
        t["label"] = np.full(HOURS, ever)
        out.append(t)
    return out
