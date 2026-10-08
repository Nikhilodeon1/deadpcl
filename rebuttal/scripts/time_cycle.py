"""One timed source-only cycle: pretrain (PCL loss) -> fine-tune -> score 4 sites.

Purpose: measure wall-clock per stage so COST_LOG.md has a real number. The
seed (default 9001) is deliberately outside the experiment seeds (42-51); the
metrics from this run are NOT reported anywhere. Per-stay predictions are still
saved (hashed id, label, score) under RB_WORK.

Dry run first, to catch plumbing errors cheaply:
    python rebuttal/scripts/time_cycle.py --fraction 0.05 --pretrain-epochs 1 --finetune-epochs 1 --tag dry
Then the timed run:
    python rebuttal/scripts/time_cycle.py --tag timed --price-per-hour <USD>
"""
import argparse
import os
import time

import rb_common as rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--lam", type=float, default=None, help="constraint weight (default: config value)")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--finetune-epochs", type=int, default=None)
    ap.add_argument("--tag", default="timed")
    ap.add_argument("--skip-external", action="store_true",
                    help="train/score on PhysioNet only (pod has no MIMIC-IV / eICU); the saved "
                         "checkpoint is scored on those sites locally with score_checkpoint.py")
    ap.add_argument("--price-per-hour", type=float, default=None, help="pod price in USD/h, for the cost projection")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from config import (BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, LAMBDA_PCL, MASK_PROB, D_MODEL, N_LAYERS,
                        PRETRAIN_EPOCHS, FINETUNE_EPOCHS, CHECKPOINT_DIR, TEST_MODE)
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset, make_patient_split_loaders
    from src.eval.evaluate_utils import run_finetuning
    from src.training.train_utils import run_pretraining, load_state_dict_flexible
    from src.losses.pcl_loss import PhysiologicalConstraintLoss

    assert not TEST_MODE and D_MODEL == 256 and N_LAYERS == 6, "production architecture required"
    lam = LAMBDA_PCL if args.lam is None else args.lam
    pt_ep = PRETRAIN_EPOCHS if args.pretrain_epochs is None else args.pretrain_epochs
    ft_ep = FINETUNE_EPOCHS if args.finetune_epochs is None else args.finetune_epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    rb.logging.info(f"timing run tag={args.tag} seed={args.seed} lam={lam} pretrain_ep={pt_ep} "
                    f"finetune_ep={ft_ep} fraction={args.fraction} device={device} ({gpu})")

    stages = {}
    t_all = time.time()

    t = time.time()
    src = rb.load_site_cached("physionet_a", args.fraction, args.seed)
    target_names = (("site_b", "physionet_b"),) if args.skip_external else         (("site_b", "physionet_b"), ("mimic", "mimic"), ("eicu", "eicu"))
    targets = {k: rb.load_site_cached(n, args.fraction, args.seed) for k, n in target_names}
    stages["data_load_s"] = time.time() - t   # includes first-time CSV/PSV parsing if the cache was cold

    ckpt_dir = os.path.join(CHECKPOINT_DIR, "timing")
    os.makedirs(ckpt_dir, exist_ok=True)
    pre_ckpt = os.path.join(ckpt_dir, f"{args.tag}_pre.pt")
    ft_ckpt = os.path.join(ckpt_dir, f"{args.tag}_ft.pt")

    src_ds = ICUDataset(src)
    pt_train, pt_val = make_patient_split_loaders(src_ds, train_frac=0.8, batch_size=BATCH_SIZE,
                                                  seed=args.seed, stratify_key=rb.TASK)
    model = fresh_model(seed=args.seed)

    t = time.time()
    run_pretraining(model, PhysiologicalConstraintLoss(), pt_train, pt_val, n_epochs=pt_ep, lam=lam,
                    mask_prob=MASK_PROB, device=device, save_path=pre_ckpt)
    stages["pretrain_s"] = time.time() - t
    model.load_state_dict(load_state_dict_flexible(pre_ckpt, device="cpu"))

    model.add_classification_head(rb.TASK)
    ft_train, ft_val = make_patient_split_loaders(src_ds, train_frac=0.8, batch_size=BATCH_SIZE,
                                                  seed=args.seed, stratify_key=rb.TASK)
    t = time.time()
    run_finetuning(model, ft_train, ft_val, rb.TASK, n_epochs=ft_ep, device=device, save_path=ft_ckpt)
    stages["finetune_s"] = time.time() - t
    model.load_state_dict(load_state_dict_flexible(ft_ckpt, device))

    t = time.time()
    dump_dir = os.path.join(rb.RB_WORK, "predictions", args.tag)
    metrics = {"source_val": rb.predict_and_dump(model, ft_val, rb.TASK, device, "site_a",
                                                 os.path.join(dump_dir, "site_a_val.npz"))}
    for k, samples in targets.items():
        loader = DataLoader(ICUDataset(samples), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        metrics[k] = rb.predict_and_dump(model, loader, rb.TASK, device, k,
                                         os.path.join(dump_dir, f"{k}.npz"))
    stages["eval_s"] = time.time() - t
    keep = os.path.join(rb.RB_OUT, "checkpoints", f"{args.tag}_ft.pt")   # small (~13 MB); score locally
    os.makedirs(os.path.dirname(keep), exist_ok=True)
    import shutil
    shutil.copy(ft_ckpt, keep)
    stages["total_s"] = time.time() - t_all
    compute_s = stages["pretrain_s"] + stages["finetune_s"] + stages["eval_s"]

    result = {
        "tag": args.tag, "seed": args.seed, "lambda": lam, "fraction": args.fraction,
        "skip_external": args.skip_external, "pretrain_epochs": pt_ep, "finetune_epochs": ft_ep, "device": device, "gpu": gpu,
        "torch": torch.__version__, "stages_s": stages,
        "n_per_site": {"site_a": len(src), **{k: len(v) for k, v in targets.items()}},
        "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else None,
        "note": "timing only; metrics below are not reported anywhere",
        "metrics_not_reported": metrics,
    }
    if args.price_per_hour is not None:
        result["compute_cost_usd_one_cycle"] = compute_s / 3600 * args.price_per_hour
        result["total_cost_usd_incl_data_load"] = stages["total_s"] / 3600 * args.price_per_hour
        result["price_per_hour_usd"] = args.price_per_hour
    out = os.path.join(rb.RB_OUT, f"timing_{args.tag}.json")
    rb.write_json(out, result)
    rb.logging.info(f"stages (s): { {k: round(v, 1) for k, v in stages.items()} }")
    rb.logging.info(f"wrote {out}")


if __name__ == "__main__":
    main()
