"""Skeleton-cached labelling — the same code-GT, at a fraction of the model calls.

The per-episode pipeline (``codegen.pipeline.episode``) spends one VLM call per
SLICE per EPISODE, which at corpus scale is far more calls than a router's request
budget supplies. This package spends the calls once per distinct sub-task LABEL
instead, and assembles every episode from the cache.

Three tiers:

  A  GENERATE   one skeleton per label, from a REPRESENTATIVE episode's frames.
                The number of distinct labels grows far slower than the corpus —
                the more you label, the better it amortises.
  B  ASSEMBLE   every episode, from parquet only: no API call, no video decode.
                The skeleton is pasted, then re-grounded against THIS episode's
                own gripper events, then repaired in BOTH directions: missing head/tail
                events inserted, redundant explicit gripper calls commented out.
  C  REFINE     (optional, budget-capped) regenerate the blocks whose events are
                STRUCTURALLY different from the skeleton's — the ones tier B
                cannot rescue.

Why tier B needs BOTH deterministic repairs, measured end to end on one real
task (`fold_blanket_into_square`, 213 episodes, 6,426 blocks, 29 labels):

    reused skeleton matches this block outright     4.1%
    + repair_tail   (skeleton emitted too FEW)     17.8%
    + over-emission lint (too MANY)                32.9%   <- what tier B ships

and the residue that only a fresh look at the video can fix:

    same length, different order                   19.6%
    still short after repair                       ~19%

A single repair direction is not enough: on reused skeletons the two failure
modes are about equally common (29.2% too few events, 24.9% too many, 8.5% the
block has no events at all).

Keying the cache on (label, event-signature) instead would pin everything, and
was measured and rejected: 80% of such pairs occur exactly once, so it amortises
1.6x and collapses back to per-block generation.

The output is the SAME run-dir schema as the per-episode pipeline
(``codegen.pipeline.artifacts``), so the gate, the viz server, the miner and any
downstream cache builder work unchanged. Every slice records where its code came
from (``source``: ``skeleton`` | ``skeleton+repair`` | ``skeleton+lint`` | ``regenerated``), so a
consumer can always tell model-written code from cache-reused code.
"""
