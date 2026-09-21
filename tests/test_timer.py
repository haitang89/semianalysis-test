import pytest

from bench.core.mock import ModelClock
from bench.core.timer import TimingConfig, percentile, time_kernel


def noop():
    return None


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    assert percentile([0.0, 10.0], 0.9) == pytest.approx(9.0)
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_statistics_and_no_graph_for_slow_kernel():
    timing = time_kernel(noop, TimingConfig(), ModelClock(latency_us=5000.0))
    assert timing.repeats == 50 and timing.warmup == 10
    assert timing.p10_us <= timing.median_us <= timing.p90_us
    assert timing.min_us <= timing.p10_us
    assert timing.median_us == pytest.approx(5025.0, rel=0.03)
    assert timing.unstable is False
    assert timing.graph_median_us is None and timing.launch_overhead_us is None


def test_short_kernel_gets_graph_replay_and_launch_overhead():
    timing = time_kernel(noop, TimingConfig(), ModelClock(latency_us=40.0, launch_us=25.0))
    assert timing.median_us < 100.0
    assert timing.graph_median_us == pytest.approx(41.0, rel=0.05)
    assert timing.launch_overhead_us == pytest.approx(24.0, abs=3.0)
    row = timing.to_row()
    assert row["eager_median_us"] == timing.median_us
    assert row["launch_overhead_us"] == timing.launch_overhead_us


def test_graph_capture_failure_is_recorded_not_raised():
    timing = time_kernel(noop, TimingConfig(), ModelClock(latency_us=40.0, graph_fails=True))
    assert timing.graph_median_us is None
    assert "cannot be captured" in timing.graph_error
    assert timing.median_us > 0


def test_unstable_point_is_measured_again_with_double_repeats():
    settles = ModelClock(latency_us=5000.0, spreads=[0.5, 0.01])
    timing = time_kernel(noop, TimingConfig(), settles)
    assert timing.repeats == 100 and timing.unstable is False

    never_settles = ModelClock(latency_us=5000.0, spreads=[0.5, 0.5])
    timing = time_kernel(noop, TimingConfig(), never_settles)
    assert timing.repeats == 100 and timing.unstable is True


def test_flush_mode_is_recorded():
    timing = time_kernel(noop, TimingConfig(), ModelClock(latency_us=5000.0, flushes_l2=True))
    assert timing.l2_flush is True
