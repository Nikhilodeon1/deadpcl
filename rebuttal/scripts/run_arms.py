"""Run a tier of (lambda, seed) arms on PhysioNet (source = Site A, target = Site B).

Resumable: a finished arm leaves rebuttal/outputs/runs/<tag>.json and is skipped.
Stops before starting a new arm if the wall-clock budget (--max-hours) would be
exceeded, so spend is bounded. MIMIC-IV / eICU are scored later, on the machine
that holds them, from the returned checkpoints (score_checkpoint.py).

    python rebuttal/scripts/run_arms.py --tier T1 --max-hours 2 --price-per-hour <USD>
Tiers are fixed in PREREG A2: T1 = ERM (lambda 0) and lambda 0.5, seeds 42-51;
T2 = lambda {0.1, 1, 2, 5}, seeds 42-46.
"""
import argparse
import os
import time

import rb_common as rb
import rb_train

TIERS = {
    "T1": [(lam, seed) for seed in range(42, 52) for lam in (0.0, 0.5)],
    "T2": [(lam, seed) for seed in range(42, 47) for lam in (0.1, 1.0, 2.0, 5.0)],
}


def tag_of(lam, seed):
    return f"lam{lam:g}_s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=sorted(TIERS))
    ap.add_argument("--max-hours", type=float, required=True, help="do not start an arm past this wall-clock budget")
    ap.add_argument("--price-per-hour", type=float, default=None)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--finetune-epochs", type=int, default=None)
    ap.add_argument("--max-arms", type=int, default=None)
    args = ap.parse_args()

    from config import PRETRAIN_EPOCHS, FINETUNE_EPOCHS, TEST_MODE, D_MODEL, N_LAYERS
    assert not TEST_MODE and D_MODEL == 256 and N_LAYERS == 6, "production architecture required"
    pt = args.pretrain_epochs or PRETRAIN_EPOCHS
    ft = args.finetune_epochs or FINETUNE_EPOCHS

    src = rb.load_site_cached("physionet_a", args.fraction, 0)
    targets = {"site_b": rb.load_site_cached("physionet_b", args.fraction, 0)}

    run_dir = os.path.join(rb.RB_OUT, "runs")
    t_start = time.time()
    done = 0
    last = None
    for lam, seed in TIERS[args.tier]:
        tag = tag_of(lam, seed)
        out = os.path.join(run_dir, f"{tag}.json")
        if os.path.exists(out):
            continue
        elapsed_h = (time.time() - t_start) / 3600
        need_h = (last / 3600) if last else 0.0
        if elapsed_h + need_h > args.max_hours:
            rb.logging.info(f"budget: {elapsed_h:.2f} h used + {need_h:.2f} h for one more arm exceeds "
                            f"{args.max_hours} h; stopping before {tag}")
            break
        if args.max_arms is not None and done >= args.max_arms:
            break
        t = time.time()
        res = rb_train.train_and_score(lam, seed, src, targets, pt, ft, tag)
        last = time.time() - t
        res["arm_seconds"] = last
        if args.price_per_hour is not None:
            res["arm_cost_usd"] = last / 3600 * args.price_per_hour
        rb.write_json(out, res)
        done += 1
        rb.logging.info(f"{tag} done in {last/60:.1f} min")
    left = [t for t in TIERS[args.tier] if not os.path.exists(os.path.join(run_dir, tag_of(*t) + ".json"))]
    rb.logging.info(f"finished {done} arms this call; {len(left)} arms of {args.tier} remain: "
                    f"{[tag_of(*t) for t in left][:6]}{' ...' if len(left) > 6 else ''}")


if __name__ == "__main__":
    main()
