"""Supporting op sweeps: the projections and elementwise ops at the token counts of a step.

    python3 -m bench.ops.run -c configs/ops_gemm.yaml [--mock] [--dry-run] [--only ...] [--force]

Backend "engine" calls what vLLM calls: the dynamic FP8 block scaled GEMM op for the FP8
projections (activation quantization included), torch.mm for the BF16 ones, and the
vLLM custom ops for norms, activation, quantization and rotary. Backend "bf16" runs every
projection as a plain BF16 torch.mm, as a comparison for the FP8 path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from bench.census.analytic import geometry_from_config
from bench.core.env import collect, environment_hash, mock_environment
from bench.core.mock import MOCK_CEILINGS, ModelClock
from bench.core.results import Provenance
from bench.core.roofline import roofline_latency_us
from bench.core.sweep import add_sweep_arguments, load_sweep, output_path, run_sweep
from bench.core.timer import TimingConfig, time_kernel

from . import catalog

LAUNCH_US = 8.0
LM_HEAD_MAX_ROWS = 2048
MODEL_CONFIG = "configs/qwen3.8-27b-fp8.config.json"


class UnsupportedPoint(ValueError):
    pass


def load_catalog(config_path: str = MODEL_CONFIG) -> dict[str, catalog.OpSpec]:
    return catalog.catalog(geometry_from_config(json.loads(Path(config_path).read_text(encoding="utf-8"))))


class MockDriver:
    def __init__(self, timing: TimingConfig):
        self.timing = timing
        self.specs = load_catalog()

    def __call__(self, params: dict) -> tuple[object, dict]:
        spec = self.specs[params["op"]]
        work = catalog.work(spec, params["tokens"])
        peak = MOCK_CEILINGS.fp8_tflops if spec.dtype == "fp8" else MOCK_CEILINGS.bf16_tflops
        latency = roofline_latency_us(work.flops, work.bytes, peak, MOCK_CEILINGS.hbm_gbps)
        timing = time_kernel(lambda: None, self.timing, ModelClock(latency, launch_us=LAUNCH_US))
        return timing, catalog.metrics(spec, work, timing.median_us, params["tokens"], None)


class GpuDriver:
    def __init__(self, timing: TimingConfig, ceilings: Optional[dict]):
        import torch

        from bench.core.timer import CudaClock

        self.torch = torch
        self.timing = timing
        self.ceilings = ceilings
        self.clock = CudaClock()
        self.specs = load_catalog()
        self.device = torch.device("cuda")
        self.weights: dict = {}

    def fp8_weight(self, spec: catalog.OpSpec):
        torch = self.torch
        key = (spec.name, "fp8")
        if key not in self.weights:
            n, k = spec.out_features, spec.in_features
            weight = torch.randn(n, k, device=self.device, dtype=torch.bfloat16) * 0.02
            blocks = weight.float().view(n // catalog.FP8_BLOCK, catalog.FP8_BLOCK, k // catalog.FP8_BLOCK, catalog.FP8_BLOCK)
            amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-6)
            scale = amax / torch.finfo(torch.float8_e4m3fn).max
            quantized = (blocks / scale).to(torch.float8_e4m3fn).view(n, k)
            self.weights[key] = (quantized, scale.view(n // catalog.FP8_BLOCK, k // catalog.FP8_BLOCK).contiguous())
        return self.weights[key]

    def bf16_weight(self, spec: catalog.OpSpec):
        torch = self.torch
        key = (spec.name, "bf16")
        if key not in self.weights:
            self.weights[key] = torch.randn(spec.in_features, spec.out_features, device=self.device, dtype=torch.bfloat16) * 0.02
        return self.weights[key]

    def gemm_kernel(self, spec: catalog.OpSpec, tokens: int, backend: str):
        torch = self.torch
        x = torch.randn(tokens, spec.in_features, device=self.device, dtype=torch.bfloat16)
        if backend == "engine" and spec.dtype == "fp8":
            import vllm.model_executor.kernels.linear.scaled_mm.flashinfer

            weight, scale = self.fp8_weight(spec)
            op = torch.ops.vllm.dynamic_flashinfer_deepgemm_blockscale_gemm
            return lambda: op(x, weight, scale, catalog.FP8_BLOCK, False)
        weight = self.bf16_weight(spec)
        out = torch.empty(tokens, spec.out_features, device=self.device, dtype=torch.bfloat16)
        return lambda: torch.mm(x, weight, out=out)

    def elementwise_kernel(self, spec: catalog.OpSpec, tokens: int):
        torch = self.torch
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

        x = torch.randn(tokens, spec.in_features, device=self.device, dtype=torch.bfloat16)
        if spec.kind == "rms_norm":
            weight = torch.ones(spec.in_features, device=self.device, dtype=torch.bfloat16)
            out = torch.empty_like(x)
            return lambda: ops.rms_norm(out, x, weight, 1e-6)
        if spec.kind == "fused_add_rms_norm":
            weight = torch.ones(spec.in_features, device=self.device, dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            return lambda: ops.fused_add_rms_norm(x, residual, weight, 1e-6)
        if spec.kind == "silu_and_mul":
            out = torch.empty(tokens, spec.out_features, device=self.device, dtype=torch.bfloat16)
            silu_and_mul = torch.ops._C.silu_and_mul
            return lambda: silu_and_mul(out, x)
        if spec.kind == "quant":
            return lambda: per_token_group_quant_fp8(x, catalog.FP8_BLOCK)
        if spec.kind == "rotary":
            positions = torch.arange(tokens, device=self.device)
            query = torch.randn(tokens, spec.heads * spec.head_dim, device=self.device, dtype=torch.bfloat16)
            cache = torch.randn(max(tokens, 1) + 1, spec.rotary_dim, device=self.device, dtype=torch.bfloat16)
            return lambda: ops.rotary_embedding(positions, query, None, spec.head_dim, cache, True)
        raise UnsupportedPoint(f"no kernel for op kind {spec.kind}")

    def __call__(self, params: dict) -> tuple[object, dict]:
        spec = self.specs[params["op"]]
        tokens, backend = params["tokens"], params["backend"]
        if spec.name == "lm_head" and tokens > LM_HEAD_MAX_ROWS:
            raise UnsupportedPoint(f"lm_head runs on one row per sequence; {tokens} rows is beyond any batch the engine forms")
        if spec.kind == "gemm":
            kernel = self.gemm_kernel(spec, tokens, backend)
        elif backend != "engine":
            raise UnsupportedPoint(f"{spec.name} has only the engine backend")
        else:
            kernel = self.elementwise_kernel(spec, tokens)
        timing = time_kernel(kernel, self.timing, self.clock)
        dtype = spec.dtype if backend == "engine" else "bf16"
        metrics = catalog.metrics(spec, catalog.work(spec, tokens), timing.median_us, tokens, self.ceilings, dtype)
        metrics["dtype"] = dtype
        self.torch.cuda.empty_cache()
        return timing, metrics


def load_ceilings(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Supporting op sweep")
    add_sweep_arguments(parser)
    parser.add_argument("--ceilings", default="results/env/ceilings.json")
    args = parser.parse_args(argv)

    config = load_sweep(Path(args.config))
    timing = TimingConfig(**{key: value for key, value in config.timing.items() if key != "point_timeout_s"})
    env = mock_environment() if args.mock else collect()
    provenance = Provenance(environment_hash(env), env.git_sha, env.git_dirty, env.image_digest)
    driver = MockDriver(timing) if args.mock else GpuDriver(timing, load_ceilings(Path(args.ceilings)))
    regime = config.raw.get("regime", config.benchmark)
    outcome = run_sweep(config, driver, output_path(config, args.out, args.mock), provenance, "ops", regime,
                        only=args.only, force=args.force, dry_run=args.dry_run)
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
