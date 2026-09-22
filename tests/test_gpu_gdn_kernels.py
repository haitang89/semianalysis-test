import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
pytest.importorskip("vllm")

from bench.gdn import reference as ref  # noqa: E402
from bench.gdn.drivers import GdnKernels, make_decode_batch, make_prefill_batch  # noqa: E402

SHAPE = ref.GdnShape()
TOLERANCE = dict(rtol=5e-2, atol=5e-2)


@pytest.fixture(scope="module")
def kernels():
    return GdnKernels(SHAPE, ref.GdnWeights.random(SHAPE, "cuda"))


def reference_batch(kernels, batch, warm):
    outputs, states = [], []
    for i in range(batch.sequences):
        start, end = batch.cu_seqlens[i].item(), batch.cu_seqlens[i + 1].item()
        slot = batch.state_indices[i].item()
        conv_state = batch.conv_state[slot] if warm else None
        ssm_state = batch.ssm_state[slot] if warm else None
        o, _, state = ref.layer_core(batch.mixed_qkv[start:end], batch.a[start:end], batch.b[start:end],
                                     kernels.weights, kernels.shape, conv_state, ssm_state)
        outputs.append(o)
        states.append(state)
    return torch.cat(outputs), torch.stack(states)


@pytest.mark.parametrize("backend", ["flashinfer", "triton"])
@pytest.mark.parametrize("warm", [False, True])
def test_prefill_chain_matches_the_reference(kernels, backend, warm):
    batch = make_prefill_batch([37, 130], SHAPE, warm=warm, seed=1)
    expected_out, expected_state = reference_batch(kernels, batch, warm)
    out, final_state = kernels.prefill(backend, batch)()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.squeeze(0).float(), expected_out, **TOLERANCE)
    torch.testing.assert_close(final_state.float(), expected_state, **TOLERANCE)


def test_final_state_fed_back_reproduces_one_uninterrupted_pass(kernels):
    whole = make_prefill_batch([200], SHAPE, seed=2)
    out_whole, state_whole = kernels.prefill("flashinfer", whole)()
    first = make_prefill_batch([120], SHAPE, seed=2)
    first.mixed_qkv, first.a, first.b = whole.mixed_qkv[:120].clone(), whole.a[:120].clone(), whole.b[:120].clone()
    _, state_first = kernels.prefill("flashinfer", first)()
    second = make_prefill_batch([80], SHAPE, seed=2, warm=True)
    second.mixed_qkv, second.a, second.b = whole.mixed_qkv[120:].clone(), whole.a[120:].clone(), whole.b[120:].clone()
    second.conv_state.copy_(first.conv_state)
    second.ssm_state[second.state_indices] = state_first.float()
    out_second, state_second = kernels.prefill("flashinfer", second)()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_second.squeeze(0).float(), out_whole.squeeze(0)[120:].float(), **TOLERANCE)
    torch.testing.assert_close(state_second.float(), state_whole.float(), **TOLERANCE)


@pytest.mark.parametrize("backend", ["packed", "sigmoid_gating"])
def test_decode_kernels_match_one_reference_step(kernels, backend):
    batch = make_decode_batch(4, SHAPE, seed=3)
    expected = []
    for i in range(batch.sequences):
        slot = batch.state_indices[i].item()
        o, _, _ = ref.layer_core(batch.mixed_qkv[i:i + 1], batch.a[i:i + 1], batch.b[i:i + 1], kernels.weights,
                                 kernels.shape, batch.conv_state[slot].clone(), batch.ssm_state[slot].clone())
        expected.append(o[0])
    out = kernels.decode(backend, batch)()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.reshape(batch.sequences, SHAPE.v_heads, SHAPE.head_dim).float(), torch.stack(expected), **TOLERANCE)
