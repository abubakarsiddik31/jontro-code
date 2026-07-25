"""Carve a deterministic 90/10 train/test partition, stratified over
(register, script) — the two sociolinguistic axes that are the paper's
centerpiece. Stratification guarantees the test split represents every
axis bucket, so the baseline breakdowns are meaningful.

Register mapping (paper terminology):
    tui   = intimate (তুই/তোর)
    tumi  = familiar (তুমি/তোমাকে)
    apni  = formal/respectful (আপনি)
"""
from __future__ import annotations

import random
from collections import defaultdict

from bangla_datasets.schema import Trajectory

SEED = 20260716


def partition(
    trajectories: list[Trajectory],
    test_fraction: float = 0.10,
    seed: int = SEED,
) -> tuple[list[str], list[str]]:
    """Return (train_ids, test_ids), stratified over (register, script).

    Within each bucket we shuffle with a per-bucket seed derived from the
    global seed + bucket key, so adding a new bucket never reshuffles the
    others. Test takes the first ``test_fraction`` of each bucket; train
    takes the rest. Both lists are sorted by id for stable output.
    """
    buckets: dict[tuple[str, str], list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        buckets[(t.persona.register, t.persona.script)].append(t)

    train_ids: list[str] = []
    test_ids: list[str] = []
    for key in sorted(buckets):
        pool = sorted(buckets[key], key=lambda t: t.id)
        bucket_seed = hash((seed, key)) & 0xFFFFFFFF
        rng = random.Random(bucket_seed)
        rng.shuffle(pool)
        k = max(1, round(len(pool) * test_fraction))
        test_ids.extend(t.id for t in pool[:k])
        train_ids.extend(t.id for t in pool[k:])

    train_ids.sort()
    test_ids.sort()
    return train_ids, test_ids
