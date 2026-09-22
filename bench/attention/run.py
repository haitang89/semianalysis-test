"""Attention kernel sweeps: cold, warm, decode and ragged, from a YAML config.

    python3 -m bench.attention.run -c configs/attention_warm.yaml [--mock] [--dry-run] [--only ...] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from bench.core.env import collect, environment_hash, mock_environment
from bench.core.mock import MOCK_CEILINGS, ModelClock
from bench.core.regimes import ragged_batch, ragged_decode_batch, warm_point
from bench.core.results import Provenance
from bench.core.roofline import roofline_latency_us
from bench.core.sweep import add_sweep_arguments, load_sweep, output_path, run_sweep
from bench.core.timer import TimingConfig, time_kernel

from . import accounting
from .drivers import SequenceSpec

ATTENTION_LAYERS = 16
LAUNCH_US = 8.0
DECODE_HISTORY = 16384


def specs_for(params: dict) -> list[SequenceSpec]:
    kind = params["kind"]
    if kind == "prefill":
        return [SequenceSpec(params["tokens"], 0)] * params["batch"]
    if kind == "decode":
        return [SequenceSpec(1, params["kv"] - 1)] * params["batch"]
    if kind == "warm":
        if "fraction" in params:
            point = warm_point(params["context"], params["fraction"])
            return [SequenceSpec(point.new, point.cached)] * params["batch"]
        return [SequenceSpec(params["new_tokens"], params["cached"])] * params["batch"]
    if kind == "ragged":
        batch = ragged_batch(params["distribution"], params["total_tokens"], params["sequences"], params.get("seed", 0))
        return [SequenceSpec(1, DECODE_HISTORY) if decode else SequenceSpec(length, 0)
                for length, decode in zip(batch.lengths, batch.decode_mask or [False] * batch.sequences)]
    if kind == "ragged_decode":
        batch = ragged_decode_batch(params["distribution"], params["sequences"], params["mean_kv"], params.get("seed", 0))
        return [SequenceSpec(1, length - 1) for length in batch.lengths]
    raise ValueError(f"unknown point kind {kind}")


def ragged_metrics(params: dict) -> dict:
    kind = params["kind"]
    if kind == "ragged":
        batch = ragged_batch(params["distribution"], params["total_tokens"], params["sequences"], params.get("seed", 0))
    elif kind == "ragged_decode":
        batch = ragged_decode_batch(params["distribution"], params["sequences"], params["mean_kv"], params.get("seed", 0))
    else:
        return {}
    return {"length_cv": batch.length_cv, "max_over_mean": batch.max_over_mean, "lengths": list(batch.lengths)}


def extra_params(params: dict) -> dict:
    if params["kind"] == "warm" and "fraction" in params:
        point = warm_point(params["context"], params["fraction"])
        return {"new_tokens": point.new, "cached": point.cached, "actual_fraction": point.actual_fraction}
    return {}


class MockDriver:
    def __init__(self, timing: TimingConfig):
        self.timing = timing

    def __call__(self, params: dict) -> tuple[object, dict]:
        specs = specs_for(params)
        work = accounting.work(specs)
        latency = roofline_latency_us(work.flops, work.bytes, MOCK_CEILINGS.bf16_tflops, MOCK_CEILINGS.hbm_gbps)
        timing = time_kernel(lambda: None, self.timing, ModelClock(latency, launch_us=LAUNCH_US))
        new_tokens = sum(spec.new for spec in specs)
        return timing, {**accounting.metrics(work, timing.median_us, new_tokens, ATTENTION_LAYERS, None),
                        **ragged_metrics(params), **extra_params(params)}


class GpuDriver:
    def __init__(self, timing: TimingConfig, ceilings: Optional[dict]):
        import torch

        from bench.core.timer import CudaClock

        from .drivers import FlashAttentionKernel, make_batch

        self.torch = torch
        self.timing = timing
        self.ceilings = ceilings
        self.clock = CudaClock()
        self.attention = FlashAttentionKernel()
        self.make_batch = make_batch

    def __call__(self, params: dict) -> tuple[object, dict]:
        specs = specs_for(params)
        batch = self.make_batch(specs, params["page_size"])
        timing = time_kernel(self.attention.kernel(batch), self.timing, self.clock)
        work = accounting.work(specs)
        new_tokens = sum(spec.new for spec in specs)
        metrics = {**accounting.metrics(work, timing.median_us, new_tokens, ATTENTION_LAYERS, self.ceilings),
                   **ragged_metrics(params), **extra_params(params)}
        del batch
        self.torch.cuda.empty_cache()
        return timing, metrics


def load_ceilings(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Attention kernel sweep")
    add_sweep_arguments(parser)
    parser.add_argument("--ceilings", default="results/env/ceilings.json")
    args = parser.parse_args(argv)

    config = load_sweep(Path(args.config))
    timing = TimingConfig(**{key: value for key, value in config.timing.items() if key != "point_timeout_s"})
    env = mock_environment() if args.mock else collect()
    provenance = Provenance(environment_hash(env), env.git_sha, env.git_dirty, env.image_digest)
    driver = MockDriver(timing) if args.mock else GpuDriver(timing, load_ceilings(Path(args.ceilings)))
    regime = config.raw.get("regime", config.benchmark)
    outcome = run_sweep(config, driver, output_path(config, args.out, args.mock), provenance, "flash_attn_3", regime,
                        only=args.only, force=args.force, dry_run=args.dry_run)
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
