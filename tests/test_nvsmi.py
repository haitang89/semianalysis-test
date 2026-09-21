import pytest

from bench.core.nvsmi import ClockSample, clock_sample, number, query_gpu

LOADED = "1455, 1980, 693.27, 700.00, 58, Not Active, Not Active, Not Active, Active\n"
HOT = "900, 1980, 400.00, 700.00, 91, Not Active, Active, Not Active, Not Active\n"


def test_query_returns_one_clean_value_per_field():
    values = query_gpu(("name", "compute_cap"), runner=lambda command: "NVIDIA H200, 9.0\n")
    assert values == ["NVIDIA H200", "9.0"]


def test_query_rejects_a_field_count_mismatch():
    with pytest.raises(ValueError):
        query_gpu(("name", "compute_cap", "memory.total"), runner=lambda command: "NVIDIA H200, 9.0\n")


def test_number_returns_none_for_not_available():
    assert number("143771") == 143771.0
    assert number("[N/A]") is None


def test_clock_sample_reads_power_cap_and_slowdown_flags():
    capped = clock_sample(runner=lambda command: LOADED)
    assert capped == ClockSample(1455.0, 1980.0, 693.27, 700.0, 58.0, slowdown=False, power_capped=True)
    hot = clock_sample(runner=lambda command: HOT)
    assert hot.slowdown is True and hot.power_capped is False


def test_clock_sample_is_none_when_nvidia_smi_is_missing():
    def missing(command):
        raise FileNotFoundError("nvidia-smi")

    assert clock_sample(runner=missing) is None
