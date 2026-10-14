"""Second batch of PhysioNet-side runs (PREREG A8). Resumable; never starts an arm past --max-hours.

Order is fixed; later groups run only if time remains:
  1. evlab  : ERM and PCL (lambda 0.5), seeds 42-46, label = any SepsisLabel in the whole record (rb_labels.py)
  2. sweep  : lambda in {0.1, 1, 2, 5}, seeds 47-51 (completes the lambda grid to ten seeds)
  3. sg     : ERM seed grid, init seed i and split seed j in {42, 43, 44}, i != j (the diagonal is the main ERM arms)
  4. var    : the four attribution variants, seeds 47-51 (nosev: PCL only)
    python rebuttal/scripts/run_batch2.py --max-hours 10
"""
import argparse
import os
import pickle
import time

import rb_common as rb
import rb_labels
import rb_train
import rb_train2
import rb_variants as rv


def jobs():
    out = []
    for seed in range(42, 47):
        for lam in (0.0, 0.5):
            out.append(("evlab", "evlab", lam, seed, f"evlab__lam{lam:g}_s{seed}", {}))
    for seed in range(47, 52):
        for lam in (0.1, 1.0, 2.0, 5.0):
            out.append(("sweep", None, lam, seed, f"lam{lam:g}_s{seed}", {}))
    for i in (42, 43, 44):
        for j in (42, 43, 44):
            if i != j:
                out.append(("sg", "sg", 0.0, i, f"sg__lam0_i{i}_j{j}", {"init_seed": i, "split_seed": j}))
    for v in ("nosev", "tsmask", "vars9", "orig_hp"):
        for seed in range(47, 52):
            for lam in rv.VARIANTS[v]["arms"]:
                out.append(("var", v, lam, seed, f"{v}__lam{lam:g}_s{seed}", {}))
    return out


def ever_cached(name, samples):
    from config import CACHE_DIR, PHYSIONET_DIR
    path = os.path.join(CACHE_DIR, f"{name}_everlabel_all.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fp:
            return pickle.load(fp)
    out = rb_labels.ever_label(samples, PHYSIONET_DIR)
    with open(path, "wb") as fp:
        pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)
    rb.logging.info(f"[cache] relabeled {name}: {sum(float(s['label'][0]) for s in out):.0f} positives of {len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, required=True)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--pretrain-epochs", type=int, default=None)
    ap.add_argument("--finetune-epochs", type=int, default=None)
    ap.add_argument("--max-arms", type=int, default=None)
    ap.add_argument("--only", choices=["evlab", "sweep", "sg", "var"], default=None)
    args = ap.parse_args()
    from config import PRETRAIN_EPOCHS, FINETUNE_EPOCHS, TEST_MODE, D_MODEL, N_LAYERS
    assert not TEST_MODE and D_MODEL == 256 and N_LAYERS == 6
    pt = args.pretrain_epochs or PRETRAIN_EPOCHS
    ft = args.finetune_epochs or FINETUNE_EPOCHS

    src = rb.load_site_cached("physionet_a", args.fraction, 0)
    tgt = {"site_b": rb.load_site_cached("physionet_b", args.fraction, 0)}
    ever = None
    run_dir = os.path.join(rb.RB_OUT, "runs")
    t0, done, last = time.time(), 0, None
    for group, variant, lam, seed, tag, extra in jobs():
        if args.only and group != args.only:
            continue
        out = os.path.join(run_dir, f"{tag}.json")
        if os.path.exists(out):
            continue
        if (time.time() - t0) / 3600 + (last or 0) / 3600 > args.max_hours:
            rb.logging.info(f"budget: stopping before {tag}")
            break
        if args.max_arms is not None and done >= args.max_arms:
            break
        t = time.time()
        if group == "sweep":
            res = rb_train.train_and_score(lam, seed, src, tgt, pt, ft, tag)
        elif group == "evlab":
            if ever is None:
                ever = (ever_cached(f"physionet_a_f{args.fraction}", src),
                        {"site_b": ever_cached(f"physionet_b_f{args.fraction}", tgt["site_b"])})
            res = rb_train2.train_and_score_variant(variant, lam, seed, ever[0], ever[1], pt, ft, tag)
        else:
            res = rb_train2.train_and_score_variant(variant, lam, seed, src, tgt, pt, ft, tag, **extra)
        last = time.time() - t
        res["arm_seconds"] = last
        rb.write_json(out, res)
        done += 1
        rb.logging.info(f"{tag} done in {last/60:.1f} min")
    left = [j for j in jobs() if not os.path.exists(os.path.join(run_dir, j[4] + ".json"))
            and (args.only is None or j[0] == args.only)]
    rb.logging.info(f"finished {done} arms this call; {len(left)} remain")


if __name__ == "__main__":
    main()
