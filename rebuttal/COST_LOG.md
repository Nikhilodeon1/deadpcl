# Cost log

BUDGET_CAP_USD = 15. Spent so far: **$0.00** (no GPU or pod used; Phase 0 was
CPU-only on the local machine).

## Prices used (not re-queried beyond these)

- A40 pod, 1 GPU, 9 vCPU, 50 GB RAM: $0.49/h (the stored cost of an existing
  stopped pod of this type, read from the pod record).
- H100 SXM: about $3.3/h (carried from the earlier budget discussion; not
  re-queried).
- CPU-only pod: price not queried.

## Time per training cycle: NOT MEASURED

No timed run exists yet. The only numbers available are carried estimates from
earlier planning, not logs:

- source-only pretrain (30 ep) + fine-tune (30 ep) + 4-site evaluation: about
  20 min on an H100 (the earlier estimate was ~6 h for ~30 runs).
- pretraining on all four sites pooled (about 22x more stays): about 2.8 h on
  an H100 (the earlier estimate was ~8-9 h for 3 runs).
- A40 assumed 2.5x slower than H100. This is a guess.

Per cycle, under those assumptions: source-only $1.10 (H100) or $0.41 (A40);
all-site pretraining $9.20 (H100) or $3.40 (A40).

## Projection of the work order (cycles = pretrain + fine-tune + evaluate)

| Experiment | Source-only cycles | All-site cycles |
|---|---|---|
| A reproduce original (ERM + PCL, 5 seeds) | 5 | 5 |
| B Tier 1 (F2, F4, F6; F3 reuses C) | 15 | 10 |
| C lambda sweep (6 lambda + ERM, 5 seeds) | 35 | 0 |
| D main table (ERM, PCL, DRO, 10 seeds; overlaps C) | 30 | 0 |
| E3 timestep masking (ERM + PCL, 5 seeds) | 10 | 0 |
| E4 + E5 | 21 | 0 |
| Total | about 116 | 15 |

Projected cost: about $128 + $138 = $266 on H100; about $48 + $51 = $99 on
A40. Both exceed the $15 cap by a wide margin. This triggers STOP E unless the
plan is cut, and the cut below is a proposal, not a decision.

## What $15 buys (A40, assumptions above)

About 36 source-only cycles, and no all-site cycles at all. A plan that fits:

- Re-score the existing seeds 42-44 sweep from checkpoints (inference only,
  no training) if the checkpoints exist on the pod volume.
- New training only for seeds 45-46 of experiment C (7 configs x 2 seeds = 14
  cycles, about $5.7) and extra seeds for the D table (ERM, PCL, DRO x seeds
  47-51 = 15 cycles, about $6.2).
- Everything needing all-site pretraining (A, F2 variants) is out of budget
  unless a timed run shows the 2.8 h figure is far too high.

## Required before any GPU spend

1. One timed source-only cycle on an A40 (about $0.4), logged here.
2. Pod inventory (read-only, CPU pod or a short A40 session) to find existing
   checkpoints and per-seed outputs; this decides whether C/D are inference-
   only for seeds 42-44.
3. Approval to start any pod.

## Pod in use (stated by the owner, not verified by me)

16 CPU cores, 64 GB RAM, 1 x V100. Hourly price: not given to me, not queried
(the account's stopped pods are A40 / base types; this pod is not visible to
my RunPod CLI, which lists no running pods). Needed from the owner: the price
per hour, so the timed cycle can be turned into dollars
(`time_cycle.py --price-per-hour`). V100 is older than A40 for transformer
training; the A40 x2.5 slowdown assumption above does not transfer, and the
timed cycle replaces it.

## Measured cycle time (V100, source-only, one arm)

278 s compute (pretrain 170 s, fine-tune 102 s, scoring 6 s) + one-time 157 s
data load (cached). Replaces the carried 20-min estimate; about 4x lower.
Projected: T1 about 1.6 h, T2 about 1.6 h. Price per hour: not yet given.

## Log

| Date | Pod | Hours | Cost | Purpose |
|---|---|---|---|---|
| (none) | | | | |
