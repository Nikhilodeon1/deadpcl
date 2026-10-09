"""One (lambda, seed) cycle: pretrain with the physiological loss, fine-tune on
the source site, score the source-validation split and the given target sites.

lambda = 0 is the ERM arm: same code path, same data, same schedule, constraint
weight zero. Warmup fraction is fixed at 0.4 (PREREG A3). Nothing in the shared
modules is changed.
"""
import os
import shutil
import time

import rb_common as rb

WARMUP_FRACTION = 0.4


def train_and_score(lam, seed, src_samples, target_samples, pt_epochs, ft_epochs, tag, device=None):
    """target_samples: dict site_key -> list of samples. Returns a result dict."""
    import torch
    from torch.utils.data import DataLoader
    from config import BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, MASK_PROB, CHECKPOINT_DIR
    from src.baselines import fresh_model
    from src.data.dataset import ICUDataset, make_patient_split_loaders
    from src.eval.evaluate_utils import run_finetuning
    from src.training.train_utils import run_pretraining, load_state_dict_flexible
    from src.losses.pcl_loss import PhysiologicalConstraintLoss

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(CHECKPOINT_DIR, "runs")
    os.makedirs(ckpt_dir, exist_ok=True)
    pre_ckpt = os.path.join(ckpt_dir, f"{tag}_pre.pt")
    ft_ckpt = os.path.join(ckpt_dir, f"{tag}_ft.pt")
    stages = {}

    ds = ICUDataset(src_samples)
    pt_train, pt_val = make_patient_split_loaders(ds, train_frac=0.8, batch_size=BATCH_SIZE,
                                                  seed=seed, stratify_key=rb.TASK)
    model = fresh_model(seed=seed)
    t = time.time()
    run_pretraining(model, PhysiologicalConstraintLoss(), pt_train, pt_val, n_epochs=pt_epochs, lam=lam,
                    mask_prob=MASK_PROB, device=device, save_path=pre_ckpt,
                    warmup_fraction=WARMUP_FRACTION)
    stages["pretrain_s"] = time.time() - t
    model.load_state_dict(load_state_dict_flexible(pre_ckpt, device="cpu"))

    model.add_classification_head(rb.TASK)
    ft_train, ft_val = make_patient_split_loaders(ds, train_frac=0.8, batch_size=BATCH_SIZE,
                                                  seed=seed, stratify_key=rb.TASK)
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
        metrics[key] = rb.predict_and_dump(model, loader, rb.TASK, device, key,
                                           os.path.join(pred_dir, f"{key}.npz"))
    stages["eval_s"] = time.time() - t

    keep = os.path.join(rb.RB_OUT, "checkpoints", f"{tag}_ft.pt")
    os.makedirs(os.path.dirname(keep), exist_ok=True)
    shutil.copy(ft_ckpt, keep)
    os.remove(pre_ckpt)   # keep disk small; the fine-tuned checkpoint is the artifact
    return {"tag": tag, "lambda": lam, "seed": seed, "pretrain_epochs": pt_epochs,
            "finetune_epochs": ft_epochs, "warmup_fraction": WARMUP_FRACTION,
            "device": device, "stages_s": stages, "metrics": metrics}
