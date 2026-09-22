import json
from pathlib import Path

import pytest

from analysis import sum_of_parts as sop
from analysis.perf_model import PerfModel
from bench.census.analytic import geometry_from_config
from bench.core.results import write_csv
from bench.core.roofline import Ceilings

GEOMETRY = geometry_from_config(json.loads(Path("configs/qwen3.8-27b-fp8.config.json").read_text(encoding="utf-8")))


def test_log_interp_is_exact_at_grid_points_and_clamped_outside():
    curve = [(1, 10.0), (10, 100.0), (100, 1000.0)]
    assert sop.log_interp(curve, 10) == pytest.approx(100.0)
    assert sop.log_interp(curve, 31.622776601683793) == pytest.approx(316.2277660168, rel=1e-6)
    assert sop.log_interp(curve, 0.5) == 10.0 and sop.log_interp(curve, 1e6) == 1000.0
    assert sop.log_interp([], 5) == 0.0


def test_nearest_picks_in_log_space():
    assert sop.nearest([1, 16, 64, 256], 100) == 64
    assert sop.nearest([1, 16, 64, 256], 130) == 256


@pytest.fixture
def processed(tmp_path):
    attention = [{"benchmark": "attention_cold", "backend": "fa3", "kind": "prefill", "tokens": t, "batch": b, "kernel_us": t * b * 0.01}
                 for t in (128, 1024, 4096) for b in (1, 16)]
    attention += [{"benchmark": "attention_cold", "backend": "fa3", "kind": "decode", "kv": kv, "batch": b, "kernel_us": kv * b * 0.001}
                  for kv in (1024, 4096) for b in (1, 64)]
    write_csv(attention, tmp_path / "attention_cold.csv")
    write_csv([{"benchmark": "gdn_cold", "backend": "flashinfer", "kind": "prefill", "tokens": t, "batch": 1, "kernel_us": 50 + t * 0.07} for t in (128, 1024, 4096)],
              tmp_path / "gdn_cold.csv")
    write_csv([{"benchmark": "gdn_decode", "backend": "packed", "kind": "decode", "batch": b, "history": None, "kernel_us": 8 + b * 1.6} for b in (1, 64)],
              tmp_path / "gdn_decode.csv")
    write_csv([{"benchmark": "ops_gemm", "backend": "engine", "op": "gate_up_proj", "tokens": t, "kernel_us": 40 + t * 0.02, "per_step_count": 64} for t in (1, 64, 1024)]
              + [{"benchmark": "ops_gemm", "backend": "bf16", "op": "gate_up_proj", "tokens": 1, "kernel_us": 999.0, "per_step_count": 64}],
              tmp_path / "ops_gemm.csv")
    return tmp_path


def test_kernel_table_scales_by_sequences_and_ignores_other_backends(processed):
    table = sop.KernelTable(processed)
    assert table.attention_prefill_us(1024, 1) == pytest.approx(10.24)
    assert table.attention_prefill_us(1024, 8) == pytest.approx(10.24 * 8)
    assert table.attention_decode_us(1024, 64) == pytest.approx(1024 * 64 * 0.001)
    assert table.gdn_decode_us(64) == pytest.approx(8 + 64 * 1.6)
    assert table.ops_us(64) == pytest.approx((40 + 64 * 0.02) * 64)


def test_launch_floor_is_taken_off_every_kernel_instance(processed):
    plain = sop.KernelTable(processed)
    floored = sop.KernelTable(processed, launch_floor_us=5.0)
    assert floored.attention_prefill_us(1024, 8) == pytest.approx(plain.attention_prefill_us(1024, 8) - 5.0)
    assert floored.gdn_decode_us(64) == pytest.approx(plain.gdn_decode_us(64) - 5.0)
    assert floored.ops_us(64) == pytest.approx(plain.ops_us(64) - 5.0 * 64)
    tiny = sop.KernelTable(processed, launch_floor_us=1000.0)
    assert tiny.gdn_decode_us(1) == 0.5


def test_predict_sums_layers_and_reports_residual(processed):
    table = sop.KernelTable(processed)
    model = PerfModel(GEOMETRY, Ceilings(4000.0, 700.0, 1400.0))
    step = {"mode": "compiled", "caching": "off", "concurrency": 8, "step": 3, "prefill_sequences": 1, "prefill_tokens": 1024,
            "decode_sequences": 7, "decode_tokens": 7, "tokens": 1031, "duration_us": 20000.0}
    row = sop.predict(step, table, model, decode_kv=1024)
    assert row.attention_us == pytest.approx(16 * (10.24 + table.attention_decode_us(1024, 7)))
    assert row.gdn_us == pytest.approx(48 * ((50 + 1024 * 0.07) + table.gdn_decode_us(7)))
    assert row.kernel_sum_us == pytest.approx(row.attention_us + row.gdn_us + row.ops_us)
    assert row.residual_us == pytest.approx(20000.0 - row.kernel_sum_us)
    assert row.roofline_us > 0
    text = sop.summary([row])
    assert text.startswith("| mode |") and "| compiled | off | 8 | 1 |" in text


def test_cli_writes_json_and_table(processed, tmp_path):
    steps = tmp_path / "steps.csv"
    write_csv([{"mode": "compiled", "caching": "on", "concurrency": 1, "step": 0, "prefill_sequences": 0, "prefill_tokens": 0,
                "decode_sequences": 1, "decode_tokens": 1, "tokens": 1, "duration_us": 9000.0},
               {"mode": "compiled", "caching": "on", "concurrency": 1, "step": 1, "prefill_sequences": 0, "prefill_tokens": 0,
                "decode_sequences": 0, "decode_tokens": 0, "tokens": 0, "duration_us": 100.0}], steps)
    ceilings = tmp_path / "ceilings.json"
    ceilings.write_text('{"hbm_bw_gbps_measured": 4000, "bf16_tflops_measured": 700, "fp8_tflops_measured": 1400}')
    out, table = tmp_path / "s.json", tmp_path / "s.md"
    assert sop.main(["--steps", str(steps), "--processed", str(processed), "--ceilings", str(ceilings), "--out", str(out), "--table", str(table)]) == 0
    data = json.loads(out.read_text())
    assert data["launch_floor_us"] == 0.0 and len(data["steps"]) == 1
    assert data["steps"][0]["decode_sequences"] == 1 and "residual_share" in data["steps"][0]
