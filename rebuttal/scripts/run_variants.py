"""Run the attribution variants (PREREG A6) on PhysioNet; external sites are scored locally afterwards.

Order is fixed (nosev, tsmask, vars9, orig_hp), seeds 42-46, lambda 0 (ERM) and 0.5 (PCL) where the variant
needs its own ERM. Resumable; never starts an arm past --max-hours.
    python rebuttal/scripts/run_variants.py --max-hours 6
"""
import argparse
import os
import time

import rb_common as rb
import rb_train2
import rb_variants as rv

ORDER = ["nosev", "tsmask", "vars9", "orig_hp"]
SEEDS = range(42, 47)


def tag_of(variant, lam, seed):
    return f"{variant}__lam{lam:g}_s{seed}"


def jobs():
    return [(v, lam, s) for v in ORDER for s in SEEDS for lam in rv.VARIANTS[v]["arms"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, required=True)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--finetune-epochs", type=int, default=None)
    ap.add_argument("--max-arms", type=int, default=None)
    args = ap.parse_args()
    from config import PRETRAIN_EPOCHS, FINETUNE_EPOCHS, TEST_MODE, D_MODEL, N_LAYERS
    assert not TEST_MODE and D_MODEL == 256 and N_LAYERS == 6
    pt = args.pretrain_epochs or PRETRAIN_EPOCHS
    ft = args.finetune_epochs or FINETUNE_EPOCHS
    src = rb.load_site_cached("physionet_a", args.fraction, 0)
    targets = {"site_b": rb.load_site_cached("physionet_b", args.fraction, 0)}
    run_dir = os.path.join(rb.RB_OUT, "runs")
    t0, done, last = time.time(), 0, None
    for variant, lam, seed in jobs():
        tag = tag_of(variant, lam, seed)
        out = os.path.join(run_dir, f"{tag}.json")
        if os.path.exists(out):
            continue
        if (time.time() - t0) / 3600 + (last or 0) / 3600 > args.max_hours:
            rb.logging.info(f"budget: stopping before {tag}")
            break
        if args.max_arms is not None and done >= args.max_arms:
            break
        t = time.time()
        res = rb_train2.train_and_score_variant(variant, lam, seed, src, targets, pt, ft, tag)
        last = time.time() - t
        res["arm_seconds"] = last
        rb.write_json(out, res)
        done += 1
        rb.logging.info(f"{tag} done in {last/60:.1f} min")
    left = [j for j in jobs() if not os.path.exists(os.path.join(run_dir, tag_of(*j) + ".json"))]
    rb.logging.info(f"finished {done} arms this call; {len(left)} remain")


if __name__ == "__main__":
    main()
