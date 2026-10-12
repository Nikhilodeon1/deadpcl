"""Score variant checkpoints (tags like nosev__lam0.5_s42) on MIMIC-IV and eICU, on the machine that holds them.

    python rebuttal/scripts/score_variants.py --ckpt-dir rebuttal/outputs/checkpoints [--variants nosev tsmask]
Skips tags that already have rebuttal/outputs/scores/<tag>.json.
"""
import argparse
import os

import rb_common as rb
import rb_variants as rv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--variants", nargs="+", default=list(rv.VARIANTS))
    ap.add_argument("--sites", nargs="+", default=["mimic", "eicu"], choices=["mimic", "eicu"])
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from config import TEST_MODE, D_MODEL, N_LAYERS
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset
    from src.training.train_utils import load_state_dict_flexible

    assert not TEST_MODE and D_MODEL == 256 and N_LAYERS == 6
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(f for f in os.listdir(args.ckpt_dir) if f.endswith("_ft.pt") and "__" in f)
    for variant in args.variants:
        cfg = rv.VARIANTS[variant]
        todo = [f for f in files if f.startswith(variant + "__")]
        todo = [f for f in todo if not os.path.exists(os.path.join(rb.RB_OUT, "scores", f[:-len("_ft.pt")] + ".json"))]
        if not todo:
            continue
        loaders = {}
        for site in args.sites:
            samples = rb.load_site_cached(site, 1.0, 0)
            if cfg["n_vars"] == 9:
                samples = rv.core_only(samples)
            loaders[site] = DataLoader(ICUDataset(samples), batch_size=512, shuffle=False, num_workers=0)
        for f in todo:
            tag = f[:-len("_ft.pt")]
            model = fresh_model(n_vars=cfg["n_vars"], seed=0)
            model.add_classification_head(rb.TASK)
            model.load_state_dict(load_state_dict_flexible(os.path.join(args.ckpt_dir, f), device))
            model.to(device)
            metrics = {}
            for site, loader in loaders.items():
                metrics[site] = rb.predict_and_dump(model, loader, rb.TASK, device, site,
                                                    os.path.join(rb.RB_OUT, "predictions", tag, f"{site}.npz"))
                rb.logging.info(f"{tag} {site}: auroc {metrics[site]['auroc']:.4f}")
            rb.write_json(os.path.join(rb.RB_OUT, "scores", f"{tag}.json"), {"tag": tag, "metrics": metrics})
        del loaders


if __name__ == "__main__":
    main()
