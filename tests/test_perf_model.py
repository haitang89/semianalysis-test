import json
from pathlib import Path

import pytest

from analysis import perf_model as pm
from bench.census.analytic import geometry_from_config, inventory
from bench.core.roofline import Ceilings

CONFIG = json.loads(Path("configs/qwen3.8-27b-fp8.config.json").read_text(encoding="utf-8"))
GEOMETRY = geometry_from_config(CONFIG)
CEILINGS = Ceilings(hbm_gbps=4000.0, bf16_tflops=700.0, fp8_tflops=1400.0)
MODEL = pm.PerfModel(GEOMETRY, CEILINGS)


def test_parse_scenario_expands_groups():
    seqs = pm.parse_scenario("2x(784+7056), 3x(1+4096)")
    assert len(seqs) == 5 and seqs[0] == pm.SequenceState(784, 7056) and seqs[-1].decode
    assert not seqs[0].decode and seqs[0].kv == 7840
    with pytest.raises(ValueError):
        pm.parse_scenario("4 sequences of 1")
    with pytest.raises(ValueError):
        pm.parse_scenario("0x(1+1)")


def test_attention_work_matches_the_benchmark_accounting():
    accounting = pytest.importorskip("bench.attention.accounting")
    from bench.attention.drivers import SequenceSpec

    specs = [SequenceSpec(512, 7840), SequenceSpec(1, 4095)]
    expected = accounting.work(specs)
    flops, bytes_moved = MODEL.attention_work([pm.SequenceState(512, 7840), pm.SequenceState(1, 4095)])
    assert flops == expected.flops and bytes_moved == expected.bytes


def test_gdn_work_matches_the_benchmark_accounting():
    accounting = pytest.importorskip("bench.gdn.accounting")
    from bench.gdn.reference import GdnShape

    shape = GdnShape()
    warm = accounting.prefill_work(shape, 4096, 8, warm=True)
    assert MODEL.gdn_work(4096, 8, 8) == (warm.flops, warm.bytes)
    cold = accounting.prefill_work(shape, 4096, 8, warm=False)
    assert MODEL.gdn_work(4096, 0, 8) == (cold.flops, cold.bytes)
    decode = accounting.decode_work(shape, 64)
    assert MODEL.gdn_work(64, 64, 64) == (decode.flops, decode.bytes)


def test_decode_at_batch_one_is_bound_by_weight_bytes():
    step = MODEL.step(pm.parse_scenario("1x(1+4096)"), "decode 1")
    gemms = [op for op in inventory(GEOMETRY) if op.kind == "gemm"]
    weights = sum(op.weight_bytes * op.per_step for op in gemms)
    gemm_us = sum(op.step_us for op in step.ops if op.name in {op.name for op in gemms})
    assert gemm_us == pytest.approx(weights / CEILINGS.hbm_gbps / 1e3, rel=0.001)
    assert all(op.bound == "memory" for op in step.ops)
    assert step.by_bound()["memory"] == pytest.approx(step.total_us)
    assert {op.name for op in step.ops} >= {"gated_delta_rule_decode", "attention", "lm_head"}
    assert not any(op.name == "gated_delta_rule_prefill" for op in step.ops)


def test_large_prefill_chunk_is_compute_bound_in_the_gemms():
    step = MODEL.step(pm.parse_scenario("1x(8192+0)"), "prefill")
    gate_up = next(op for op in step.ops if op.name == "gate_up_proj")
    assert gate_up.bound == "compute"
    assert gate_up.call_us == pytest.approx(2 * 8192 * 5120 * 34816 / (1400e12) * 1e6)
    assert gate_up.count == 64 and gate_up.step_us == 64 * gate_up.call_us
    assert step.by_block()["mlp"] > step.by_block()["attention"] > 0


def test_mixed_step_has_both_gdn_kernels_and_counts_warm_states_once_each():
    step = MODEL.step(pm.parse_scenario("1x(784+7056),1x(4096+0),32x(1+16384)"), "mixed")
    names = [op.name for op in step.ops]
    assert "gated_delta_rule_prefill" in names and "gated_delta_rule_decode" in names
    prefill = next(op for op in step.ops if op.name == "gated_delta_rule_prefill")
    expected_flops, expected_bytes = MODEL.gdn_work(4880, 1, 2)
    assert (prefill.flops, prefill.bytes) == (expected_flops, expected_bytes)
    assert step.new_tokens == 4912 and step.sequences == 34


def test_kernel_floor_lifts_tiny_ops():
    floored = pm.PerfModel(GEOMETRY, CEILINGS, kernel_floor_us=3.0).step(pm.parse_scenario("1x(1+16)"), "tiny")
    norm = next(op for op in floored.ops if op.name == "input_layernorm")
    assert norm.bound == "floor" and norm.call_us == 3.0


def test_kernel_attainment_uses_the_binding_side_of_the_roofline():
    rows = [
        {"benchmark": "attention_cold", "backend": "fa3", "point_id": "a", "kind": "prefill", "batch": 1, "tokens": 4096,
         "kernel_us": 100.0, "flops": 35e9, "bytes_moved": 1e6},
        {"benchmark": "gdn_decode", "backend": "packed", "point_id": "b", "kind": "decode", "batch": 64,
         "kernel_us": 100.0, "flops": 1e9, "bytes_moved": 200e6},
        {"benchmark": "gdn_decode", "backend": "packed", "point_id": "c", "kind": "decode", "batch": 1, "kernel_us": None,
         "flops": 1e9, "bytes_moved": 1e6},
    ]
    out = pm.kernel_attainment(rows, CEILINGS)
    assert [r["point_id"] for r in out] == ["a", "b"]
    assert out[0]["bound"] == "compute" and out[0]["attainment"] == pytest.approx(50.0 / 100.0)
    assert out[1]["bound"] == "memory" and out[1]["attainment"] == pytest.approx(50.0 / 100.0)
    assert "| gdn_decode | packed | 1 | 50% | 50% | 50% |" in pm.attainment_summary(out)


def test_cli_writes_tables_from_the_committed_inputs(tmp_path):
    out, table = tmp_path / "m.json", tmp_path / "m.md"
    processed = tmp_path / "processed"
    processed.mkdir()
    code = pm.main(["--processed", str(processed), "--scenario", "64x(1+4096)", "--out", str(out), "--table", str(table)])
    assert code == 0
    data = json.loads(out.read_text())
    assert data["steps"][0]["scenario"] == "64x(1+4096)" and data["steps"][0]["sequences"] == 64
    assert data["kernel_attainment"] == []
    assert table.read_text().startswith("| scenario |")
