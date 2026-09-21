"""Measured ceilings: achieved memory bandwidth and GEMM throughput on this GPU.

Every utilization number in the study divides by these, not by spec sheet values.
A ceiling is the best burst observed, so no later measurement can exceed it by
noise alone. Sustained throughput under the power cap is reported beside it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .ceiling_backends import GEMM_SHAPE, Backend, MockBackend, TorchBackend, Workload
from .env import collect, mock_environment, provenance
from .nvsmi import ClockSample
from .timer import Timing, TimingConfig, time_kernel

GIB = 1 << 30
BANDWIDTH_SIZES_GIB = (1, 4, 16)
CEILING_TIMING = TimingConfig(warmup=5, repeats=20)
VENDOR_HBM_GBPS = {"H200": 4800.0, "B200": 7700.0, "B300": 8000.0}


def _rate(work: float, microseconds: float, unit: float) -> float:
    return work / (microseconds * 1e-6) / unit


def _bandwidth_row(op: str, size_gib: int, workload: Workload, timing: Timing) -> dict:
    return {
        "op": op,
        "size_gib": size_gib,
        "median_us": timing.median_us,
        "min_us": timing.min_us,
        "gbps_peak": _rate(workload.bytes_moved, timing.min_us, 1e9),
        "gbps_median": _rate(workload.bytes_moved, timing.median_us, 1e9),
        "unstable": timing.unstable,
        "path": workload.path,
    }


def _measure_bandwidth(backend: Backend) -> list[dict]:
    rows = []
    for size_gib in BANDWIDTH_SIZES_GIB:
        for op, make in (("copy", backend.copy), ("reduce", backend.reduce)):
            workload = make(size_gib * GIB)
            timing = time_kernel(workload.kernel, CEILING_TIMING, workload.clock)
            rows.append(_bandwidth_row(op, size_gib, workload, timing))
            del workload
            backend.release()
    return rows


def _measure_gemm(backend: Backend, dtype: str) -> dict:
    try:
        workload = backend.gemm(dtype)
        timing = time_kernel(workload.kernel, CEILING_TIMING, workload.clock)
    except Exception as exc:
        backend.release()
        return {"dtype": dtype, "shape": GEMM_SHAPE, "tflops_peak": None, "reason": f"{type(exc).__name__}: {exc}"}
    result = {
        "dtype": dtype,
        "shape": GEMM_SHAPE,
        "median_us": timing.median_us,
        "min_us": timing.min_us,
        "tflops_peak": _rate(workload.flops, timing.min_us, 1e12),
        "tflops_median": _rate(workload.flops, timing.median_us, 1e12),
        "unstable": timing.unstable,
        "path": workload.path,
    }
    del workload
    backend.release()
    return result


def _run_sustained(backend: Backend, seconds: float, interval: float) -> tuple[list[ClockSample], Optional[float]]:
    workload = backend.gemm("bf16")
    samples: list[ClockSample] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            sample = backend.clock_sample()
            if sample:
                samples.append(sample)
            stop.wait(interval)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    kernels, started = 0, time.monotonic()
    while time.monotonic() - started < seconds:
        kernels += backend.sustain(workload)
    elapsed = time.monotonic() - started
    stop.set()
    poller.join()
    if not samples:
        sample = backend.clock_sample()
        if sample:
            samples.append(sample)
    tflops = kernels * workload.flops / elapsed / 1e12 if kernels else None
    del workload
    backend.release()
    return samples, tflops


def _clock_summary(samples: list[ClockSample]) -> dict:
    if not samples:
        return {"samples": 0, "throttled": None, "power_capped": None, "reason": "no clock samples available"}
    sm_clock = statistics.median(sample.sm_clock_mhz for sample in samples)
    sm_clock_max = max(sample.sm_clock_max_mhz for sample in samples)
    return {
        "samples": len(samples),
        "sm_clock_median_mhz": sm_clock,
        "sm_clock_max_mhz": sm_clock_max,
        "sm_clock_ratio": sm_clock / sm_clock_max if sm_clock_max else None,
        "power_median_w": statistics.median(sample.power_w for sample in samples),
        "power_limit_w": max(sample.power_limit_w for sample in samples),
        "temp_median_c": statistics.median(sample.temp_c for sample in samples),
        "power_capped": any(sample.power_capped for sample in samples),
        "throttled": any(sample.slowdown for sample in samples),
    }


def _vendor_hbm_gbps(gpu_name: str) -> Optional[float]:
    return next((gbps for key, gbps in VENDOR_HBM_GBPS.items() if key in gpu_name), None)


def measure_ceilings(
    backend: Backend,
    sustain_seconds: float = 5.0,
    sample_interval: float = 1.0,
    source: Optional[dict] = None,
) -> dict:
    bandwidth = _measure_bandwidth(backend)
    best = max(bandwidth, key=lambda row: row["gbps_peak"])
    bf16, fp8 = _measure_gemm(backend, "bf16"), _measure_gemm(backend, "fp8")
    samples, sustained_tflops = _run_sustained(backend, sustain_seconds, sample_interval)
    clocks = _clock_summary(samples)
    return {
        "gpu_name": backend.gpu_name,
        "hbm_bw_gbps_measured": best["gbps_peak"],
        "hbm_bw_best": {"op": best["op"], "size_gib": best["size_gib"]},
        "hbm_bw_gbps_vendor": _vendor_hbm_gbps(backend.gpu_name),
        "bf16_tflops_measured": bf16["tflops_peak"],
        "bf16_tflops_sustained": sustained_tflops,
        "fp8_tflops_measured": fp8["tflops_peak"],
        "bandwidth": bandwidth,
        "gemm": {"bf16": bf16, "fp8": fp8},
        "clocks": clocks,
        "throttled": clocks["throttled"],
        "power_capped": clocks["power_capped"],
        "timing": {"warmup": CEILING_TIMING.warmup, "repeats": CEILING_TIMING.repeats},
        "source": source,
    }


def _format(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure memory bandwidth and GEMM ceilings")
    parser.add_argument("--out", default="results/env/ceilings.json")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    if args.mock:
        ceilings = measure_ceilings(MockBackend(), sustain_seconds=0.0, source=provenance(mock_environment()))
    else:
        ceilings = measure_ceilings(TorchBackend(), source=provenance(collect()))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ceilings, indent=2) + "\n", encoding="utf-8")

    print(
        f"{ceilings['gpu_name']}: HBM {_format(ceilings['hbm_bw_gbps_measured'])} GB/s, "
        f"BF16 {_format(ceilings['bf16_tflops_measured'])} TFLOPS peak "
        f"({_format(ceilings['bf16_tflops_sustained'])} sustained), "
        f"FP8 {_format(ceilings['fp8_tflops_measured'])} TFLOPS peak, "
        f"throttled={ceilings['throttled']}, power_capped={ceilings['power_capped']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
