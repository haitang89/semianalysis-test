import json

import pytest

from bench.core import ceilings as ceilmod
from bench.core.ceiling_backends import MockBackend
from bench.core.ceilings import measure_ceilings
from bench.core.mock import MOCK_CEILINGS


def mock_ceilings(backend=None, **kwargs):
    return measure_ceilings(backend or MockBackend(), sustain_seconds=0.0, **kwargs)


def test_mock_ceilings_recover_the_modelled_hardware():
    result = mock_ceilings()
    assert result["hbm_bw_gbps_measured"] == pytest.approx(MOCK_CEILINGS.hbm_gbps, rel=0.03)
    assert result["bf16_tflops_measured"] == pytest.approx(MOCK_CEILINGS.bf16_tflops, rel=0.03)
    assert result["fp8_tflops_measured"] == pytest.approx(MOCK_CEILINGS.fp8_tflops, rel=0.03)
    assert result["hbm_bw_gbps_vendor"] == 4800.0
    assert result["throttled"] is False


def test_a_ceiling_is_the_best_burst_so_nothing_can_beat_it_by_noise():
    result = mock_ceilings()
    for row in result["bandwidth"]:
        assert row["gbps_peak"] >= row["gbps_median"]
    assert result["hbm_bw_gbps_measured"] == max(row["gbps_peak"] for row in result["bandwidth"])
    for dtype in ("bf16", "fp8"):
        assert result["gemm"][dtype]["tflops_peak"] >= result["gemm"][dtype]["tflops_median"]


def test_bandwidth_table_covers_every_size_and_op():
    rows = mock_ceilings()["bandwidth"]
    assert {(row["op"], row["size_gib"]) for row in rows} == {
        (op, size) for op in ("copy", "reduce") for size in (1, 4, 16)
    }
    assert all(row["gbps_peak"] > 0 and row["path"] for row in rows)


def test_gemm_records_the_shape_and_path_that_achieved_the_peak():
    gemm = mock_ceilings()["gemm"]
    for dtype in ("bf16", "fp8"):
        assert gemm[dtype]["shape"] == {"tokens": 16384, "out_features": 34816, "in_features": 5120}
        assert gemm[dtype]["path"]


def test_power_capping_is_reported_but_is_not_throttling():
    capped = mock_ceilings(MockBackend(sm_clock_mhz=1440.0, power_capped=True))
    assert capped["power_capped"] is True and capped["throttled"] is False
    assert capped["clocks"]["sm_clock_ratio"] == pytest.approx(1440.0 / 1980.0)


def test_thermal_or_hardware_slowdown_is_throttling():
    assert mock_ceilings(MockBackend(sm_clock_mhz=900.0, slowdown=True))["throttled"] is True


def test_sustained_throughput_counts_the_kernels_that_really_ran():
    class Busy(MockBackend):
        def sustain(self, workload):
            return 10

    result = measure_ceilings(Busy(), sustain_seconds=0.05, sample_interval=0.01)
    assert result["bf16_tflops_sustained"] > 0
    assert mock_ceilings()["bf16_tflops_sustained"] is None


def test_failing_fp8_path_is_null_with_a_reason():
    class NoFp8(MockBackend):
        def gemm(self, dtype):
            if dtype == "fp8":
                raise RuntimeError("fp8 matmul is not supported on this device")
            return super().gemm(dtype)

    result = mock_ceilings(NoFp8())
    assert result["fp8_tflops_measured"] is None
    assert "not supported" in result["gemm"]["fp8"]["reason"]
    assert result["bf16_tflops_measured"] > 0


def test_missing_clock_samples_do_not_break_the_run():
    class NoClocks(MockBackend):
        def clock_sample(self):
            return None

    result = mock_ceilings(NoClocks())
    assert result["throttled"] is None and result["power_capped"] is None
    assert result["clocks"]["samples"] == 0


def test_result_says_which_code_and_environment_produced_it(tmp_path):
    out = tmp_path / "ceilings.json"
    assert ceilmod.main(["--mock", "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["gpu_name"] == "Mock H200"
    assert set(record["source"]) == {"environment_hash", "git_sha", "git_dirty", "image_digest", "collected_at"}
    assert record["source"]["git_dirty"] is False
