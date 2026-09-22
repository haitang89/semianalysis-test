import pytest

torch = pytest.importorskip("torch")

from bench.gdn.reference import GdnShape, GdnWeights, causal_conv, layer_core, recurrent  # noqa: E402

SMALL = GdnShape(k_heads=2, v_heads=6, head_dim=16, conv_kernel=4)


def inputs(tokens, seed=0, shape=SMALL):
    generator = torch.Generator().manual_seed(seed)
    mixed = torch.randn(tokens, shape.conv_dim, generator=generator)
    a = torch.randn(tokens, shape.v_heads, generator=generator)
    b = torch.randn(tokens, shape.v_heads, generator=generator)
    return mixed, a, b


def test_one_pass_equals_two_chained_passes_through_the_states():
    weights = GdnWeights.random(SMALL, "cpu")
    mixed, a, b = inputs(37)
    full, conv_full, ssm_full = layer_core(mixed, a, b, weights, SMALL)
    first, conv_first, ssm_first = layer_core(mixed[:20], a[:20], b[:20], weights, SMALL)
    second, conv_second, ssm_second = layer_core(mixed[20:], a[20:], b[20:], weights, SMALL, conv_first, ssm_first)
    torch.testing.assert_close(torch.cat([first, second]), full, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(conv_second, conv_full, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ssm_second, ssm_full, rtol=1e-4, atol=1e-4)


def test_conv_matches_torch_conv1d_with_zero_history():
    weights = GdnWeights.random(SMALL, "cpu")
    mixed, _, _ = inputs(12)
    out, _ = causal_conv(mixed, weights.conv_weight, weights.conv_bias, None)
    x = mixed.float().T.unsqueeze(0)
    padded = torch.nn.functional.pad(x, (SMALL.conv_kernel - 1, 0))
    expected = torch.nn.functional.conv1d(padded, weights.conv_weight.float().unsqueeze(1), weights.conv_bias.float(), groups=SMALL.conv_dim)
    torch.testing.assert_close(out, torch.nn.functional.silu(expected[0].T), rtol=1e-5, atol=1e-5)


def test_recurrence_with_zero_gates_and_full_beta_is_a_plain_delta_rule_write():
    q = torch.nn.functional.normalize(torch.randn(1, SMALL.k_heads, SMALL.head_dim), dim=-1)
    k = q.clone()
    v = torch.randn(1, SMALL.v_heads, SMALL.head_dim)
    g = torch.zeros(1, SMALL.v_heads)
    beta = torch.ones(1, SMALL.v_heads)
    state = torch.zeros(SMALL.v_heads, SMALL.head_dim, SMALL.head_dim)
    o, new_state = recurrent(q, k, v, g, beta, state, SMALL)
    torch.testing.assert_close(o[0], v[0] * SMALL.head_dim ** -0.5, rtol=1e-5, atol=1e-5)
    assert new_state.abs().sum() > 0


def test_state_decays_with_negative_gate():
    q = torch.nn.functional.normalize(torch.randn(1, SMALL.k_heads, SMALL.head_dim), dim=-1)
    k = q.clone()
    v = torch.zeros(1, SMALL.v_heads, SMALL.head_dim)
    state = torch.ones(SMALL.v_heads, SMALL.head_dim, SMALL.head_dim)
    _, decayed = recurrent(q, k, v, torch.full((1, SMALL.v_heads), -2.0), torch.zeros(1, SMALL.v_heads), state, SMALL)
    assert decayed.mean().item() == pytest.approx(torch.exp(torch.tensor(-2.0)).item(), rel=1e-4)
