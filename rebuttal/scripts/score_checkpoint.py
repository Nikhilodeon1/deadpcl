"""Score a fine-tuned checkpoint on the cached external sites (run where MIMIC-IV / eICU live).

The pod trains on PhysioNet and returns small checkpoints; the credentialed
cohorts never leave the machine that holds them. Writes per-stay predictions
(hashed id, label, score) under RB_OUT/predictions/<tag>/ and an aggregate JSON in RB_OUT/scores/.

    python rebuttal/scripts/score_checkpoint.py --ckpt-dir <folder with *_ft.pt files>
"""
import argparse
import os

import rb_common as rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="one checkpoint file")
    ap.add_argument("--tag", help="name for the one checkpoint")
    ap.add_argument("--ckpt-dir", help="score every <tag>_ft.pt in this folder (skips tags already scored)")
    ap.add_argument("--lams", type=float, nargs="+", default=None,
                    help="with --ckpt-dir: only score checkpoints of these lambda values (order of work)")
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
    if args.ckpt_dir:
        jobs = [(f[:-len("_ft.pt")], os.path.join(args.ckpt_dir, f))
                for f in sorted(os.listdir(args.ckpt_dir)) if f.endswith("_ft.pt")]
        if args.lams is not None:
            keep = tuple(f"lam{l:g}_s" for l in args.lams)
            jobs = [j for j in jobs if j[0].startswith(keep)]
    else:
        assert args.ckpt and args.tag, "give --ckpt and --tag, or --ckpt-dir"
        jobs = [(args.tag, args.ckpt)]

    loaders = {}
    for site in args.sites:
        samples = rb.load_site_cached(site, 1.0, 0)   # cache built by prepare_external.py
        loaders[site] = DataLoader(ICUDataset(samples), batch_size=512, shuffle=False, num_workers=0)

    for tag, ckpt in jobs:
        out = os.path.join(rb.RB_OUT, "scores", f"{tag}.json")
        if os.path.exists(out):
            continue
        model = fresh_model(seed=0)
        model.add_classification_head(rb.TASK)
        model.load_state_dict(load_state_dict_flexible(ckpt, device))
        model.to(device)
        metrics = {}
        for site, loader in loaders.items():
            metrics[site] = rb.predict_and_dump(
                model, loader, rb.TASK, device, site,
                os.path.join(rb.RB_OUT, "predictions", tag, f"{site}.npz"))
            rb.logging.info(f"{tag} {site}: {metrics[site]}")
        rb.write_json(out, {"tag": tag, "metrics": metrics})


if __name__ == "__main__":
    main()
