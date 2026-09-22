"""Plain PyTorch reference for the GDN layer core, for checking kernels at small shapes.

Follows the Gated DeltaNet recurrence in fp32, one token at a time:
    S <- S * exp(g);  S <- S + k (beta * (v - k^T S));  o = (q * scale)^T S
with grouped heads (several value heads share one key head) and a depthwise
causal conv with SiLU in front, exactly the ops vLLM fuses into its kernels.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GdnShape:
    k_heads: int = 16
    v_heads: int = 48
    head_dim: int = 128
    conv_kernel: int = 4

    @property
    def key_dim(self) -> int:
        return self.k_heads * self.head_dim

    @property
    def value_dim(self) -> int:
        return self.v_heads * self.head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def group(self) -> int:
        return self.v_heads // self.k_heads


@dataclass
class GdnWeights:
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor

    @classmethod
    def random(cls, shape: GdnShape, device: str, seed: int = 0) -> "GdnWeights":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        conv_weight = torch.randn(shape.conv_dim, shape.conv_kernel, generator=generator) * 0.3
        conv_bias = torch.randn(shape.conv_dim, generator=generator) * 0.1
        A_log = torch.log(torch.rand(shape.v_heads, generator=generator) * 15 + 1)
        dt_bias = torch.randn(shape.v_heads, generator=generator) * 0.5
        to = lambda t, dtype: t.to(device=device, dtype=dtype)
        return cls(to(conv_weight, torch.bfloat16), to(conv_bias, torch.bfloat16), to(A_log, torch.float32), to(dt_bias, torch.float32))


def causal_conv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """x [T, C], weight [C, K], state [C, K-1] of previous inputs. Returns SiLU(conv) [T, C] and new state."""
    kernel = weight.shape[1]
    history = state.T.float() if state is not None else torch.zeros(kernel - 1, x.shape[1], device=x.device)
    padded = torch.cat([history, x.float()], dim=0)
    out = torch.stack([(padded[t : t + kernel] * weight.float().T).sum(0) for t in range(x.shape[0])]) + bias.float()
    return torch.nn.functional.silu(out), padded[-(kernel - 1):].T


def gates(a: torch.Tensor, b: torch.Tensor, weights: GdnWeights) -> tuple[torch.Tensor, torch.Tensor]:
    g = -torch.exp(weights.A_log) * torch.nn.functional.softplus(a.float() + weights.dt_bias)
    beta = torch.sigmoid(b.float())
    return g, beta


def split_qkv(conv_out: torch.Tensor, shape: GdnShape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = torch.split(conv_out, [shape.key_dim, shape.key_dim, shape.value_dim], dim=-1)
    q = torch.nn.functional.normalize(q.view(-1, shape.k_heads, shape.head_dim).float(), dim=-1)
    k = torch.nn.functional.normalize(k.view(-1, shape.k_heads, shape.head_dim).float(), dim=-1)
    return q, k, v.view(-1, shape.v_heads, shape.head_dim).float()


def recurrent(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
              state: torch.Tensor, shape: GdnShape) -> tuple[torch.Tensor, torch.Tensor]:
    """One sequence. q, k [T, Hk, D]; v, g, beta [T, Hv, ...]; state [Hv, Dv, Dk] as the kernels store it."""
    scale = shape.head_dim ** -0.5
    state = state.float().clone()
    outputs = []
    for t in range(q.shape[0]):
        k_t = k[t].repeat_interleave(shape.group, dim=0)
        q_t = q[t].repeat_interleave(shape.group, dim=0) * scale
        state = state * torch.exp(g[t])[:, None, None]
        memory = torch.einsum("hvk,hk->hv", state, k_t)
        delta = beta[t][:, None] * (v[t] - memory)
        state = state + torch.einsum("hv,hk->hvk", delta, k_t)
        outputs.append(torch.einsum("hvk,hk->hv", state, q_t))
    return torch.stack(outputs), state


def layer_core(mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor, weights: GdnWeights, shape: GdnShape,
               conv_state: torch.Tensor | None = None, ssm_state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conv, gating, split and recurrence for one sequence [T, conv_dim]. Returns o, conv_state, ssm_state."""
    conv_out, new_conv_state = causal_conv(mixed_qkv, weights.conv_weight, weights.conv_bias, conv_state)
    q, k, v = split_qkv(conv_out, shape)
    g, beta = gates(a, b, weights)
    if ssm_state is None:
        ssm_state = torch.zeros(shape.v_heads, shape.head_dim, shape.head_dim, device=mixed_qkv.device)
    o, new_ssm_state = recurrent(q, k, v, g, beta, ssm_state, shape)
    return o, new_conv_state, new_ssm_state
