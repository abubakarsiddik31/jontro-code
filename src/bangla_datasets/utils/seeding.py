"""Seeded RNG utilities for reproducible generation."""
import hashlib
import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def derive_seed(base: int, tag: str) -> int:
    """Derive a deterministic sub-seed from a base seed + string tag.

    Ensures persona/tools/simulator streams never collide.
    """
    digest = hashlib.sha256(f"{base}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


class SeededRNG:
    """Reproducible RNG wrapper over random.Random."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def integers(self, low: int, high: int) -> int:
        """Inclusive of low, exclusive of high."""
        return self._rng.randrange(low, high)

    def float(self, low: float, high: float) -> float:
        return self._rng.uniform(low, high)

    def choice(self, seq: Sequence[T]) -> T:
        return self._rng.choice(seq)

    def sample(self, seq: Sequence[T], k: int) -> list[T]:
        return self._rng.sample(seq, k)

    def weighted_choice(self, options: Sequence[T], weights: Sequence[float]) -> T:
        return self._rng.choices(options, weights=weights, k=1)[0]

    @property
    def raw(self) -> random.Random:
        return self._rng
