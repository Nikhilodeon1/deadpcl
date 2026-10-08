"""Score a fine-tuned checkpoint on the cached external sites (run where MIMIC-IV / eICU live).

The pod trains on PhysioNet and returns small checkpoints; the credentialed
cohorts never leave the machine that holds them. Writes per-stay predictions
(hashed id, label, score) under RB_WORK/predictions/<tag>/ and an aggregate JSON.

    python rebuttal/scripts/score_checkpoint.py --ckpt path/to/<tag>_ft.pt --tag <tag>
"""
import argparse
import os

import rb_common as rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sites", nargs="+", default=["mimic", "eicu"], choices=["mimic", "eicu"])
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from config import BATCH_SIZE, TEST_MODE, D_MODEL, N_LAYERS
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset
    from src.training.train_utils import load_state_dict_flexible

    assert not TEST_MODE and D_MODEL == 256 and N_LAYERS == 6, "production architecture required"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = fresh_model(seed=0)
    model.add_classification_head(rb.TASK)
    model.load_state_dict(load_state_dict_flexible(args.ckpt, device))
    model.to(device)

    metrics = {}
    for site in args.sites:
        samples = rb.load_site_cached(site, 1.0, 0)   # cache built by prepare_external.py
        loader = DataLoader(ICUDataset(samples), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        metrics[site] = rb.predict_and_dump(
            model, loader, rb.TASK, device, site,
            os.path.join(rb.RB_WORK, "predictions", args.tag, f"{site}.npz"))
        rb.logging.info(f"{args.tag} {site}: {metrics[site]}")
    rb.write_json(os.path.join(rb.RB_OUT, f"score_{args.tag}.json"), {"tag": args.tag, "metrics": metrics})


if __name__ == "__main__":
    main()
