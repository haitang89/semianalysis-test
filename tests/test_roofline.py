import pytest

from bench.core.roofline import Ceilings, arithmetic_intensity, roofline_latency_us


def test_latency_is_the_slower_of_compute_and_memory():
    compute_bound = roofline_latency_us(flops=4e12, bytes_moved=1e6, peak_tflops=400.0, peak_gbps=4800.0)
    memory_bound = roofline_latency_us(flops=1e6, bytes_moved=480e9, peak_tflops=400.0, peak_gbps=4800.0)
    assert compute_bound == pytest.approx(10_000.0)
    assert memory_bound == pytest.approx(100_000.0)


def test_zero_peaks_do_not_divide_by_zero():
    assert roofline_latency_us(1e9, 1e9, peak_tflops=0.0, peak_gbps=0.0) == 0.0


def test_ceilings_pick_the_peak_for_the_dtype():
    ceilings = Ceilings(hbm_gbps=4277.0, bf16_tflops=784.0, fp8_tflops=1473.0)
    assert ceilings.peak_tflops("fp8") == 1473.0
    assert ceilings.peak_tflops("bf16") == 784.0


def test_arithmetic_intensity():
    assert arithmetic_intensity(flops=2e9, bytes_moved=1e9) == 2.0
    assert arithmetic_intensity(flops=1.0, bytes_moved=0.0) == float("inf")
