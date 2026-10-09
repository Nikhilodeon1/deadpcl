"""Diagnostic H (PREREG A2 item 1): score already-trained models with inputs cut before the label.

For each positive stay the input is cut so that it ends h hours before the first positive
hourly label (rows from index first_label - h onward are removed: values zero, mask False).
Negative stays are cut to a length drawn from the positives' cut-length distribution
(capped at their observed length). This is a diagnostic of the existing models, not a
horizon-trained model. Only Site B is used here: external sites have a broadcast label with
no onset time. Per-stay predictions go to RB_OUT/predictions/<arm>_h<h>_s<seed>/.

    python rebuttal/scripts/diag_horizon.py --ckpt-dir <folder with *_ft.pt> --hours 0 6 12
"""
import argparse
import copy
import os

import numpy as np

import rb_common as rb


def cut_samples(samples, h, rng):
    first = [int(np.argmax(np.asarray(s["label"]) > 0)) for s in samples if np.sum(s["label"]) > 0]
    cuts = np.array([max(1, f - h) for f in first])
    out = []
    for s in samples:
        n_real = int(np.nonzero(np.asarray(s["mask"]).any(axis=1))[0].max()) + 1
        if np.sum(s["label"]) > 0:
            cut = max(1, int(np.argmax(np.asarray(s["label"]) > 0)) - h)
        else:
            cut = int(min(rng.choice(cuts), n_real))
        t = copy.copy(s)
        v, m = np.array(s["values"], copy=True), np.array(s["mask"], copy=True)
        v[cut:] = 0.0
        m[cut:] = False
        t["values"], t["mask"] = v, m
        out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--hours", type=int, nargs="+", default=[0, 6, 12])
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from config import BATCH_SIZE, TEST_MODE, D_MODEL, N_LAYERS
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset
    from src.training.train_utils import load_state_dict_flexible

    assert not TEST_MODE and D_MODEL == 256 and N_LAYERS == 6
    device = "cuda" if torch.cuda.is_available() else "cpu"
    site_b = rb.load_site_cached("physionet_b", 1.0, 0)
    loaders = {}
    for h in args.hours:
        rng = np.random.default_rng(1234 + h)
        loaders[h] = DataLoader(ICUDataset(cut_samples(site_b, h, rng)), batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=0)
    for f in sorted(os.listdir(args.ckpt_dir)):
        if not f.endswith("_ft.pt"):
            continue
        tag = f[:-len("_ft.pt")]                    # e.g. lam0.5_s42
        arm, seed = tag.rsplit("_s", 1)
        out = os.path.join(rb.RB_OUT, "horizon", f"{tag}.json")
        if os.path.exists(out):
            continue
        model = fresh_model(seed=0)
        model.add_classification_head(rb.TASK)
        model.load_state_dict(load_state_dict_flexible(os.path.join(args.ckpt_dir, f), device))
        model.to(device)
        res = {}
        for h, loader in loaders.items():
            res[f"h{h}"] = rb.predict_and_dump(
                model, loader, rb.TASK, device, "site_b",
                os.path.join(rb.RB_OUT, "predictions", f"{arm}_h{h}_s{seed}", "site_b.npz"))
        rb.write_json(out, {"tag": tag, "metrics": res})
        rb.logging.info(f"{tag}: " + ", ".join(f"{k} {v['auroc']:.4f}" for k, v in res.items()))


if __name__ == "__main__":
    main()
