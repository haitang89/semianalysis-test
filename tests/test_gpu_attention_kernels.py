import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
pytest.importorskip("vllm")

from bench.attention.drivers import FlashAttentionKernel, SequenceSpec, make_batch, reference  # noqa: E402

TOLERANCE = dict(rtol=2e-2, atol=2e-2)


@pytest.fixture(scope="module")
def attention():
    return FlashAttentionKernel()


@pytest.mark.parametrize("page_size", [784, 16])
def test_warm_append_matches_reference(attention, page_size):
    batch = make_batch([SequenceSpec(new=37, cached=784), SequenceSpec(new=500, cached=784)], page_size, seed=1)
    out = attention.kernel(batch)()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), reference(batch), **TOLERANCE)


def test_cold_prefill_and_decode_match_reference(attention):
    cold = make_batch([SequenceSpec(new=300, cached=0), SequenceSpec(new=64, cached=0)], 784, seed=2)
    torch.testing.assert_close(attention.kernel(cold)().float(), reference(cold), **TOLERANCE)
    decode = make_batch([SequenceSpec(new=1, cached=1567), SequenceSpec(new=1, cached=99)], 784, seed=3)
    torch.testing.assert_close(attention.kernel(decode)().float(), reference(decode), **TOLERANCE)


def test_scheduler_metadata_does_not_change_the_result(attention):
    batch = make_batch([SequenceSpec(new=128, cached=784)], 784, seed=4)
    with_schedule = attention.kernel(batch)().clone()
    plain = FlashAttentionKernel(use_scheduler_metadata=False).kernel(batch)()
    torch.testing.assert_close(with_schedule.float(), plain.float(), rtol=1e-3, atol=1e-3)
