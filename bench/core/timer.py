"""Kernel timing: CUDA events, warmup, repeats, and graph replay for very short kernels."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Optional, Protocol, Sequence

Kernel = Callable[[], object]


@dataclass(frozen=True)
class TimingConfig:
    warmup: int = 10
    repeats: int = 50
    graph_mode: str = "auto"
    graph_below_us: float = 100.0
    unstable_spread: float = 0.25

    def wants_graph(self, eager_median_us: float) -> bool:
        if self.graph_mode == "always":
            return True
        if self.graph_mode == "never":
            return False
        return eager_median_us < self.graph_below_us


@dataclass(frozen=True)
class Timing:
    median_us: float
    p10_us: float
    p90_us: float
    min_us: float
    mean_us: float
    repeats: int
    warmup: int
    unstable: bool
    graph_median_us: Optional[float]
    graph_error: Optional[str]
    l2_flush: bool

    @property
    def launch_overhead_us(self) -> Optional[float]:
        if self.graph_median_us is None:
            return None
        return self.median_us - self.graph_median_us

    def to_row(self) -> dict:
        row = asdict(self)
        row["eager_median_us"] = self.median_us
        row["launch_overhead_us"] = self.launch_overhead_us
        return row


class Clock(Protocol):
    flushes_l2: bool

    def measure(self, kernel: Kernel, warmup: int, repeats: int) -> Sequence[float]: ...

    def measure_graph(self, kernel: Kernel, warmup: int, repeats: int) -> Sequence[float]: ...


def percentile(sorted_samples: Sequence[float], q: float) -> float:
    if not sorted_samples:
        raise ValueError("no samples")
    rank = (len(sorted_samples) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return sorted_samples[low]
    weight = rank - low
    return sorted_samples[low] * (1 - weight) + sorted_samples[high] * weight


@dataclass(frozen=True)
class _Stats:
    median: float
    p10: float
    p90: float
    low: float
    mean: float

    @property
    def spread(self) -> float:
        return (self.p90 - self.p10) / self.median if self.median > 0 else math.inf


def _summarize(samples: Sequence[float]) -> _Stats:
    ordered = sorted(samples)
    return _Stats(
        median=percentile(ordered, 0.5),
        p10=percentile(ordered, 0.1),
        p90=percentile(ordered, 0.9),
        low=ordered[0],
        mean=sum(ordered) / len(ordered),
    )


def time_kernel(kernel: Kernel, config: TimingConfig, clock: Clock) -> Timing:
    repeats = config.repeats
    stats = _summarize(clock.measure(kernel, config.warmup, repeats))
    if stats.spread > config.unstable_spread:
        repeats *= 2
        stats = _summarize(clock.measure(kernel, config.warmup, repeats))

    graph_median, graph_error = None, None
    if config.wants_graph(stats.median):
        try:
            graph_median = _summarize(clock.measure_graph(kernel, config.warmup, repeats)).median
        except Exception as exc:
            graph_error = f"{type(exc).__name__}: {exc}"

    return Timing(
        median_us=stats.median,
        p10_us=stats.p10,
        p90_us=stats.p90,
        min_us=stats.low,
        mean_us=stats.mean,
        repeats=repeats,
        warmup=config.warmup,
        unstable=stats.spread > config.unstable_spread,
        graph_median_us=graph_median,
        graph_error=graph_error,
        l2_flush=clock.flushes_l2,
    )


class CudaClock:
    """Times with CUDA events and synchronizes once per batch of repeats."""

    def __init__(self, l2_flush_bytes: int = 0):
        import torch

        self._torch = torch
        self.flushes_l2 = l2_flush_bytes > 0
        self._flush = torch.empty(l2_flush_bytes, dtype=torch.uint8, device="cuda") if self.flushes_l2 else None

    def measure(self, kernel: Kernel, warmup: int, repeats: int) -> list[float]:
        torch = self._torch
        for _ in range(warmup):
            kernel()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for start, end in zip(starts, ends):
            if self._flush is not None:
                self._flush.fill_(1)
            start.record()
            kernel()
            end.record()
        torch.cuda.synchronize()
        return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]

    def measure_graph(self, kernel: Kernel, warmup: int, repeats: int) -> list[float]:
        torch = self._torch
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                kernel()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel()
        return self.measure(graph.replay, warmup, repeats)
