"""Analytic stand-ins for the GPU so the whole pipeline runs on a machine without one."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

from .roofline import Ceilings
from .timer import Kernel

MOCK_CEILINGS = Ceilings(hbm_gbps=4800.0, bf16_tflops=400.0, fp8_tflops=700.0)


@dataclass
class ModelClock:
    """Returns a modelled latency plus launch overhead and noise instead of timing anything."""

    latency_us: float
    launch_us: float = 25.0
    noise: float = 0.02
    seed: int = 0
    graph_fails: bool = False
    flushes_l2: bool = False
    spreads: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._calls = 0

    def _samples(self, base_us: float, repeats: int) -> list[float]:
        noise = self.noise
        if self.spreads is not None:
            noise = self.spreads[min(self._calls, len(self.spreads) - 1)]
        self._calls += 1
        return [base_us * (1.0 + self._rng.uniform(-noise, noise)) for _ in range(repeats)]

    def measure(self, kernel: Kernel, warmup: int, repeats: int) -> list[float]:
        return self._samples(self.latency_us + self.launch_us, repeats)

    def measure_graph(self, kernel: Kernel, warmup: int, repeats: int) -> list[float]:
        if self.graph_fails:
            raise RuntimeError("kernel cannot be captured in a CUDA graph")
        return self._samples(self.latency_us + 1.0, repeats)
