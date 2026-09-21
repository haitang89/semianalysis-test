"""Roofline model: the time a kernel needs when only compute or only memory bandwidth limits it."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ceilings:
    hbm_gbps: float
    bf16_tflops: float
    fp8_tflops: float

    def peak_tflops(self, dtype: str) -> float:
        if dtype == "fp8":
            return self.fp8_tflops
        return self.bf16_tflops


def roofline_latency_us(flops: float, bytes_moved: float, peak_tflops: float, peak_gbps: float) -> float:
    compute_s = flops / (peak_tflops * 1e12) if peak_tflops > 0 else 0.0
    memory_s = bytes_moved / (peak_gbps * 1e9) if peak_gbps > 0 else 0.0
    return max(compute_s, memory_s) * 1e6


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    return flops / bytes_moved if bytes_moved > 0 else float("inf")
