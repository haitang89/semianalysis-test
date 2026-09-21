import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

from bench.core.timer import CudaClock, TimingConfig, time_kernel  # noqa: E402


def test_large_matmul_is_timed_without_graph_replay():
    x = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    timing = time_kernel(lambda: torch.mm(x, x), TimingConfig(warmup=3, repeats=10), CudaClock())
    assert timing.median_us > 100.0
    assert timing.graph_median_us is None and timing.graph_error is None


def test_tiny_kernel_is_also_replayed_from_a_graph():
    x = torch.randn(64, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)
    timing = time_kernel(lambda: torch.add(x, x, out=out), TimingConfig(warmup=3, repeats=20), CudaClock())
    assert timing.median_us < 100.0
    assert timing.graph_error is None
    assert 0 < timing.graph_median_us <= timing.median_us * 1.5


def test_flush_buffer_is_kept_out_of_the_timed_region():
    x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config = TimingConfig(warmup=3, repeats=10)
    plain = time_kernel(lambda: torch.mm(x, x), config, CudaClock())
    flushed = time_kernel(lambda: torch.mm(x, x), config, CudaClock(l2_flush_bytes=64 << 20))
    assert flushed.l2_flush is True
    assert flushed.median_us < plain.median_us * 3
