"""The three regimes, defined once and shared by the attention and GDN benchmarks."""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Optional

MANAGER_BLOCK = 784
WARM_FRACTIONS = (0.0, 0.5, 0.9, 0.99)
RAGGED_DISTRIBUTIONS = ("uniform", "jitter", "lognormal", "bimodal", "mixed")


@dataclass(frozen=True)
class WarmPoint:
    context: int
    requested_fraction: float
    cached: int
    new: int
    manager_block: int = MANAGER_BLOCK

    @property
    def actual_fraction(self) -> float:
        return self.cached / self.context if self.context else 0.0

    @property
    def cached_blocks(self) -> int:
        return self.cached // self.manager_block


def warm_point(context: int, fraction: float, block: int = MANAGER_BLOCK) -> WarmPoint:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("cached fraction must be in [0, 1)")
    cached = int(fraction * context) // block * block
    new = context - cached
    if new <= 0:
        raise ValueError(f"no new tokens left for context {context} at fraction {fraction}")
    return WarmPoint(context, fraction, cached, new, block)


def warm_contexts(blocks_per_context: tuple = (10, 20, 40, 80, 160), block: int = MANAGER_BLOCK) -> list[int]:
    return [count * block for count in blocks_per_context]


@dataclass(frozen=True)
class RaggedBatch:
    distribution: str
    total_tokens: int
    lengths: tuple
    seed: int
    decode_mask: tuple = field(default=())

    @property
    def sequences(self) -> int:
        return len(self.lengths)

    @property
    def length_cv(self) -> float:
        mean = self.total_tokens / self.sequences
        variance = sum((length - mean) ** 2 for length in self.lengths) / self.sequences
        return math.sqrt(variance) / mean if mean else 0.0

    @property
    def max_over_mean(self) -> float:
        return max(self.lengths) / (self.total_tokens / self.sequences)


def _rescale(weights: list[float], total: int, minimum: int = 1) -> list[int]:
    scale = total / sum(weights)
    lengths = [max(minimum, int(round(weight * scale))) for weight in weights]
    diff = total - sum(lengths)
    order = sorted(range(len(lengths)), key=lambda index: -lengths[index])
    for index in itertools.cycle(order):
        if diff == 0:
            break
        step = 1 if diff > 0 else -1
        if lengths[index] + step >= minimum:
            lengths[index] += step
            diff -= step
    return lengths


def ragged_batch(distribution: str, total_tokens: int, sequences: int, seed: int = 0,
                 decode_history: int = 16384) -> RaggedBatch:
    if sequences < 1 or total_tokens < sequences:
        raise ValueError("need at least one token per sequence")
    rng = random.Random(seed)
    decode_mask: tuple = ()
    if distribution == "uniform":
        weights = [1.0] * sequences
    elif distribution == "jitter":
        weights = [rng.uniform(0.8, 1.0) for _ in range(sequences)]
    elif distribution == "lognormal":
        weights = [rng.lognormvariate(0.0, 1.0) for _ in range(sequences)]
    elif distribution == "bimodal":
        weights = [float(sequences - 1)] + [1.0] * (sequences - 1)
    elif distribution == "mixed":
        decodes = sequences // 2
        prefills = sequences - decodes
        prefill_lengths = _rescale([1.0] * prefills, total_tokens - decodes)
        lengths = tuple([1] * decodes + prefill_lengths)
        decode_mask = tuple([True] * decodes + [False] * prefills)
        return RaggedBatch(distribution, total_tokens, lengths, seed, decode_mask)
    else:
        raise ValueError(f"unknown distribution {distribution}")
    return RaggedBatch(distribution, total_tokens, tuple(_rescale(weights, total_tokens)), seed)


def ragged_decode_batch(distribution: str, sequences: int, mean_kv: int, seed: int = 0) -> RaggedBatch:
    """Every sequence has one new token; the KV lengths are what vary."""
    total_kv = sequences * mean_kv
    batch = ragged_batch(distribution, total_kv, sequences, seed)
    return RaggedBatch(distribution, total_kv, batch.lengths, seed, decode_mask=tuple([True] * sequences))
