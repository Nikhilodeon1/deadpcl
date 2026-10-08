"""Shared helpers for the rebuttal drivers.

Everything is path-independent: dataset, cache, checkpoint and output
locations come from environment variables (see rebuttal/pod/env.sh). Nothing
in the shared loader / model / training code is modified; this module only
imports it.
"""
import hashlib
import json
import logging
import os
import pickle
import sys

# Production-scale architecture must be forced before ANY `import config`.
os.environ["PCL_TEST_MODE"] = "0"

_HERE = os.path.dirname(os.path.abspath(__file__))
RB_ROOT = os.path.dirname(_HERE)                 # .../rebuttal
REPO_ROOT = os.path.dirname(RB_ROOT)             # repository root (config.py, src/)
sys.path.insert(0, REPO_ROOT)

RB_OUT = os.environ.get("RB_OUT", os.path.join(RB_ROOT, "outputs"))
RB_WORK = os.environ.get("RB_WORK", os.path.join(RB_ROOT, "work"))
HASH_SALT = os.environ.get("RB_HASH_SALT", "rb")

TASK = "sepsis"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def hash_id(site, patient_id, stay_id):
    """Short stable hash of a stay identifier; the raw identifier is never stored."""
    key = f"{HASH_SALT}|{site}|{patient_id}|{stay_id}".encode()
    return hashlib.sha256(key).hexdigest()[:16]


def load_site_cached(loader_name, fraction, seed):
    """Cached site load (the full read takes minutes). The cache is keyed by
    seed, as in the original driver."""
    from config import CACHE_DIR, PHYSIONET_DIR, MIMIC_DIR, EICU_DIR
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{loader_name}_frac{fraction}_seed{seed}.pkl")
    if os.path.exists(path):
        logging.info(f"[cache] loading {loader_name} from cache")
        with open(path, "rb") as fp:
            return pickle.load(fp)

    from src.data.physionet2019 import load_physionet2019
    from src.data.mimic4 import load_mimic4
    from src.data.eicu import load_eicu
    if loader_name == "physionet_a":
        samples, _ = load_physionet2019(PHYSIONET_DIR, fraction=fraction, sites=[0], seed=seed)
    elif loader_name == "physionet_b":
        samples, _ = load_physionet2019(PHYSIONET_DIR, fraction=fraction, sites=[1], seed=seed)
    elif loader_name == "mimic":
        samples, _ = load_mimic4(MIMIC_DIR, fraction=fraction, seed=seed)
    elif loader_name == "eicu":
        samples, _ = load_eicu(EICU_DIR, fraction=fraction, seed=seed)
    else:
        raise ValueError(loader_name)
    with open(path, "wb") as fp:
        pickle.dump(samples, fp, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"[cache] saved {loader_name}: {len(samples)} stays")
    return samples


def ece(labels, probs, n_bins=10):
    """Expected calibration error, equal-width bins on [0, 1]."""
    import numpy as np
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(total)


def predict_and_dump(model, loader, task, device, site, dump_path=None):
    """Score a loader, save per-stay (hashed id, label, score) to `dump_path`
    (.npz), and return aggregate metrics. Uses the same autocast setting as the
    original evaluation."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from config import USE_AMP

    model.eval()
    ids, ys, ps = [], [], []
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device, non_blocking=True)
            m = b["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                reps = model.encode(x, m)
                logits = model.classify(reps, task, obs_mask=m).squeeze(-1)
            ps.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(b[task].cpu().numpy())
            pid = b["patient_id"]
            sid = b["stay_id"]
            sid = sid.tolist() if hasattr(sid, "tolist") else list(sid)
            ids.extend(hash_id(site, p, s) for p, s in zip(pid, sid))
    probs = np.concatenate(ps)
    labels = np.concatenate(ys)
    if dump_path:
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        np.savez(dump_path, hashed_id=np.array(ids), label=labels, score=probs, site=site)
    out = {"n": int(len(labels)), "n_pos": int(labels.sum())}
    if 0 < out["n_pos"] < out["n"]:
        out.update(auroc=float(roc_auc_score(labels, probs)),
                   auprc=float(average_precision_score(labels, probs)),
                   brier=float(brier_score_loss(labels, probs)),
                   ece10=ece(labels, probs))
    else:
        out.update(auroc=None, auprc=None, brier=None, ece10=None)
    return out


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
