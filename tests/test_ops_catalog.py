import json
from pathlib import Path

import pytest

from bench.census.analytic import geometry_from_config, inventory
from bench.ops import catalog
from bench.ops.run import MockDriver, load_catalog
from bench.core.timer import TimingConfig

GEOMETRY = geometry_from_config(json.loads(Path("configs/qwen3.8-27b-fp8.config.json").read_text(encoding="utf-8")))
SPECS = catalog.catalog(GEOMETRY)


def test_gemm_shapes_and_counts_match_the_analytic_inventory():
    gemms = {op.name: op for op in inventory(GEOMETRY) if op.kind == "gemm" and op.per_step}
    for name, op in gemms.items():
        spec = SPECS[name]
        assert spec.kind == "gemm" and spec.per_step == op.per_step and spec.dtype == op.dtype
        assert op.shape == f"[M, {spec.in_features}] x [{spec.in_features}, {spec.out_features}] -> [M, {spec.out_features}]"
        assert catalog.gemm_weight_bytes(spec) == op.weight_bytes


def test_elementwise_counts_cover_the_step():
    assert SPECS["fused_add_rms_norm"].per_step + SPECS["rms_norm"].per_step == 2 * 64 + 1
    assert SPECS["quant_hidden"].per_step + SPECS["quant_intermediate"].per_step + SPECS["quant_mixer_out"].per_step == 256
    assert SPECS["rotary"].heads == 28 and SPECS["rotary"].rotary_dim == 64 and SPECS["rotary"].per_step == 16


def test_work_counts_fp8_gemm_bytes_and_flops():
    spec = SPECS["gate_up_proj"]
    work = catalog.work(spec, 784)
    assert work.flops == 2 * 784 * 5120 * 34816
    assert work.weight_bytes == 5120 * 34816 + (5120 / 128) * (34816 / 128) * 4
    assert work.bytes == pytest.approx(work.weight_bytes + 784 * 5120 * 2 + 784 * 34816 * 2 + 784 * 5120 * 2 + (784 * 5120 / 128) * 8)
    bf16 = catalog.work(SPECS["lm_head"], 1)
    assert bf16.weight_bytes == 2 * 5120 * 248320 and bf16.flops == 2 * 5120 * 248320


def test_elementwise_work_is_read_plus_write():
    assert catalog.work(SPECS["rms_norm"], 100).bytes == 100 * 5120 * 2 * 2
    assert catalog.work(SPECS["fused_add_rms_norm"], 100).bytes == 100 * 5120 * 2 * 4
    assert catalog.work(SPECS["silu_and_mul"], 10).bytes == 10 * 34816 * 2 + 10 * 17408 * 2
    assert catalog.work(SPECS["rotary"], 10).bytes == 10 * 28 * 64 * 2 * 2
    assert catalog.work(SPECS["quant_hidden"], 10).flops == 0.0


def test_metrics_use_the_ceiling_of_the_dtype():
    ceilings = {"hbm_bw_gbps_measured": 4000.0, "bf16_tflops_measured": 700.0, "fp8_tflops_measured": 1400.0}
    fp8 = catalog.metrics(SPECS["gate_up_proj"], catalog.work(SPECS["gate_up_proj"], 8192), 1000.0, 8192, ceilings)
    assert fp8["compute_util"] == pytest.approx(fp8["achieved_tflops"] / 1400.0)
    assert fp8["per_step_count"] == 64 and fp8["per_step_us"] == 64000.0
    bf16 = catalog.metrics(SPECS["in_proj_ba"], catalog.work(SPECS["in_proj_ba"], 8192), 10.0, 8192, ceilings)
    assert bf16["compute_util"] == pytest.approx(bf16["achieved_tflops"] / 700.0)


def test_mock_driver_returns_timing_and_metrics_for_every_op():
    driver = MockDriver(TimingConfig(warmup=1, repeats=5))
    for name in load_catalog():
        timing, metrics = driver({"backend": "engine", "op": name, "tokens": 64})
        assert timing.median_us > 0 and metrics["bytes_moved"] > 0 and metrics["per_step_count"] == SPECS[name].per_step
