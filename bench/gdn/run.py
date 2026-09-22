"""GDN kernel sweeps: cold, warm state and ragged, from a YAML config.

    python3 -m bench.gdn.run -c configs/gdn_cold.yaml [--mock] [--dry-run] [--only ...] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from bench.core.env import collect, environment_hash, mock_environment
from bench.core.mock import MOCK_CEILINGS, ModelClock
from bench.core.regimes import ragged_batch, warm_point
from bench.core.results import Provenance
from bench.core.roofline import roofline_latency_us
from bench.core.sweep import add_sweep_arguments, load_sweep, output_path, run_sweep
from bench.core.timer import TimingConfig, time_kernel

from . import accounting
from .reference import GdnShape

GDN_LAYERS = 48
LAUNCH_US = 12.0
TOKEN_LIMIT = 131072


class UnsupportedPoint(ValueError):
    pass


def point_geometry(params: dict) -> tuple[GdnShape, int, int, bool]:
    """Returns the head shape, total tokens, sequence count and whether states are warm."""
    shape = GdnShape()
    kind = params["kind"]
    if kind == "prefill":
        return shape, params["tokens"] * params["batch"], params["batch"], False
    if kind == "warm":
        point = warm_point(params["context"], params["fraction"]) if "fraction" in params else None
        new = point.new if point else params["new_tokens"]
        return shape, new * params["batch"], params["batch"], True
    if kind == "decode":
        return shape, params["batch"], params["batch"], True
    if kind == "ragged":
        batch = ragged_batch(params["distribution"], params["total_tokens"], params["sequences"], params.get("seed", 0))
        return shape, batch.total_tokens, batch.sequences, False
    raise ValueError(f"unknown point kind {kind}")


def lengths_for(params: dict) -> list[int]:
    kind = params["kind"]
    if kind == "prefill":
        return [params["tokens"]] * params["batch"]
    if kind == "warm":
        new = warm_point(params["context"], params["fraction"]).new if "fraction" in params else params["new_tokens"]
        return [new] * params["batch"]
    if kind == "ragged":
        return list(ragged_batch(params["distribution"], params["total_tokens"], params["sequences"], params.get("seed", 0)).lengths)
    raise ValueError(f"no lengths for kind {kind}")


def work_for(params: dict) -> accounting.GdnWork:
    shape, tokens, sequences, warm = point_geometry(params)
    if params["kind"] == "decode":
        return accounting.decode_work(shape, sequences)
    return accounting.prefill_work(shape, tokens, sequences, warm)


class MockDriver:
    def __init__(self, timing: TimingConfig):
        self.timing = timing

    def __call__(self, params: dict) -> tuple[object, dict]:
        work = work_for(params)
        latency = roofline_latency_us(work.flops, work.bytes, MOCK_CEILINGS.bf16_tflops, MOCK_CEILINGS.hbm_gbps)
        timing = time_kernel(lambda: None, self.timing, ModelClock(latency, launch_us=LAUNCH_US))
        _, tokens, _, _ = point_geometry(params)
        return timing, accounting.metrics(work, timing.median_us, tokens, GDN_LAYERS, None)


class GpuDriver:
    def __init__(self, timing: TimingConfig, ceilings: Optional[dict]):
        import torch

        from bench.core.timer import CudaClock

        from .drivers import GdnKernels, make_decode_batch, make_prefill_batch, prefill_history
        from .reference import GdnWeights

        self.torch = torch
        self.timing = timing
        self.ceilings = ceilings
        self.clock = CudaClock()
        self.shape = GdnShape()
        self.kernels = GdnKernels(self.shape, GdnWeights.random(self.shape, "cuda"))
        self.make_prefill_batch = make_prefill_batch
        self.make_decode_batch = make_decode_batch
        self.prefill_history = prefill_history
        self.histories: dict = {}

    def _warm_states(self, history: int, sequences: int):
        key = (history, sequences)
        if key not in self.histories:
            self.histories[key] = self.prefill_history(self.kernels, history, sequences)
        return self.histories[key]

    def __call__(self, params: dict) -> tuple[object, dict]:
        kind, backend = params["kind"], params["backend"]
        shape, tokens, sequences, _ = point_geometry(params)
        if tokens > TOKEN_LIMIT:
            raise UnsupportedPoint(f"{tokens} tokens in one call; the chunked kernels fault past roughly 2^18 tokens and the engine caps a prefill step at max_num_batched_tokens")
        if kind == "decode":
            batch = self.make_decode_batch(sequences, shape)
            if params.get("history"):
                conv_state, ssm_state = self._warm_states(params["history"], sequences)
                batch.conv_state.copy_(conv_state)
                batch.ssm_state.copy_(ssm_state)
            kernel = self.kernels.decode(backend, batch)
        else:
            warm = kind == "warm"
            batch = self.make_prefill_batch(lengths_for(params), shape, warm=warm)
            if warm and params.get("history") is not None:
                conv_state, ssm_state = self._warm_states(params["history"], sequences)
                batch.conv_state.copy_(conv_state)
                batch.ssm_state.copy_(ssm_state)
            kernel = self.kernels.prefill(backend, batch)
        timing = time_kernel(kernel, self.timing, self.clock)
        work = work_for(params)
        metrics = accounting.metrics(work, timing.median_us, tokens, GDN_LAYERS, self.ceilings)
        if kind == "ragged":
            ragged = ragged_batch(params["distribution"], params["total_tokens"], params["sequences"], params.get("seed", 0))
            metrics.update(length_cv=ragged.length_cv, max_over_mean=ragged.max_over_mean, lengths=list(ragged.lengths))
        del batch
        self.torch.cuda.empty_cache()
        return timing, metrics


def load_ceilings(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GDN kernel sweep")
    add_sweep_arguments(parser)
    parser.add_argument("--ceilings", default="results/env/ceilings.json")
    args = parser.parse_args(argv)

    config = load_sweep(Path(args.config))
    timing = TimingConfig(**{key: value for key, value in config.timing.items() if key != "point_timeout_s"})
    env = mock_environment() if args.mock else collect()
    provenance = Provenance(environment_hash(env), env.git_sha, env.git_dirty, env.image_digest)
    driver = MockDriver(timing) if args.mock else GpuDriver(timing, load_ceilings(Path(args.ceilings)))
    regime = config.raw.get("regime", config.benchmark)
    outcome = run_sweep(config, driver, output_path(config, args.out, args.mock), provenance, "gdn", regime,
                        only=args.only, force=args.force, dry_run=args.dry_run)
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
