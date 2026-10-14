"""Like rb_train.train_and_score, with the variant switches of rb_variants. rb_train.py is left unchanged
(it produced the main arms)."""
import os
import shutil
import time

import rb_common as rb
import rb_variants as rv


def train_and_score_variant(variant, lam, seed, src_samples, target_samples, pt_epochs, ft_epochs, tag, device=None,
                            init_seed=None, split_seed=None):
    import torch
    from torch.utils.data import DataLoader
    from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, CHECKPOINT_DIR
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset, make_patient_split_loaders
    from src.eval.evaluate_utils import run_finetuning
    from src.training.train_utils import run_pretraining, load_state_dict_flexible
    from src.losses.pcl_loss import PhysiologicalConstraintLoss
    import src.models.backbone as backbone

    cfg = rv.VARIANTS[variant]
    init_seed = seed if init_seed is None else init_seed      # defaults reproduce the earlier runs exactly
    split_seed = seed if split_seed is None else split_seed
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(CHECKPOINT_DIR, "runs")
    os.makedirs(ckpt_dir, exist_ok=True)
    pre_ckpt, ft_ckpt = (os.path.join(ckpt_dir, f"{tag}_{k}.pt") for k in ("pre", "ft"))
    prep = rv.core_only if cfg["n_vars"] == 9 else (lambda s: s)
    src_samples = prep(src_samples)
    target_samples = {k: prep(v) for k, v in target_samples.items()}

    original_mask_fn = backbone.apply_random_mask
    if cfg["mask"] == "timestep":
        backbone.apply_random_mask = rv.timestep_mask_fn(cfg["mask_prob"])
    try:
        loss = rv.NoSeveringhausLoss() if cfg["loss"] == "nosev" else PhysiologicalConstraintLoss()
        ds = ICUDataset(src_samples)
        pt_train, pt_val = make_patient_split_loaders(ds, 0.8, BATCH_SIZE, seed=split_seed, stratify_key=rb.TASK)
        model = fresh_model(n_vars=cfg["n_vars"], seed=init_seed)
        stages = {}
        t = time.time()
        run_pretraining(model, loss, pt_train, pt_val, n_epochs=pt_epochs, lam=lam, mask_prob=cfg["mask_prob"],
                        device=device, save_path=pre_ckpt, warmup_fraction=cfg["warmup"])
        stages["pretrain_s"] = time.time() - t
    finally:
        backbone.apply_random_mask = original_mask_fn
    model.load_state_dict(load_state_dict_flexible(pre_ckpt, device="cpu"))
    model.add_classification_head(rb.TASK)
    ft_train, ft_val = make_patient_split_loaders(ds, 0.8, BATCH_SIZE, seed=split_seed, stratify_key=rb.TASK)
    t = time.time()
    run_finetuning(model, ft_train, ft_val, rb.TASK, n_epochs=ft_epochs, device=device, save_path=ft_ckpt)
    stages["finetune_s"] = time.time() - t
    model.load_state_dict(load_state_dict_flexible(ft_ckpt, device))

    t = time.time()
    pred_dir = os.path.join(rb.RB_OUT, "predictions", tag)
    metrics = {"source_val": rb.predict_and_dump(model, ft_val, rb.TASK, device, "site_a",
                                                 os.path.join(pred_dir, "site_a_val.npz"))}
    for key, samples in target_samples.items():
        loader = DataLoader(ICUDataset(samples), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        metrics[key] = rb.predict_and_dump(model, loader, rb.TASK, device, key, os.path.join(pred_dir, f"{key}.npz"))
    stages["eval_s"] = time.time() - t
    keep = os.path.join(rb.RB_OUT, "checkpoints", f"{tag}_ft.pt")
    os.makedirs(os.path.dirname(keep), exist_ok=True)
    shutil.copy(ft_ckpt, keep)
    os.remove(pre_ckpt)
    return {"tag": tag, "variant": variant, "lambda": lam, "seed": seed, "init_seed": init_seed, "split_seed": split_seed, "config": cfg,
            "pretrain_epochs": pt_epochs, "finetune_epochs": ft_epochs, "device": device,
            "stages_s": stages, "metrics": metrics}
