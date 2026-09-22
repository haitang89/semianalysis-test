"""GDN kernels exactly as vLLM calls them, plus the inputs and states they need.

Prefill chain: causal_conv1d_fn -> fused_post_conv_prep -> chunked gated delta rule
(FlashInfer or the vendored Triton kernel). Decode chain: causal_conv1d_update ->
fused recurrent kernel (packed Triton, the engine default for plain decode, or the
sigmoid gating variant the engine uses when prefills and decodes share a step).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional

import torch

from .reference import GdnShape, GdnWeights

PREFILL_BACKENDS = ("flashinfer", "triton")
DECODE_BACKENDS = ("packed", "sigmoid_gating")
NULL_SLOT = 1


def _ops():
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as layer
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as conv
    from vllm.third_party.flash_linear_attention import ops as fla
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    return layer, conv, fla, compute_causal_conv1d_metadata


@dataclass
class PrefillBatch:
    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    cu_seqlens: torch.Tensor
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    has_initial_state: torch.Tensor
    state_indices: torch.Tensor
    conv_metadata: Optional[SimpleNamespace] = None

    @property
    def tokens(self) -> int:
        return self.mixed_qkv.shape[0]

    @property
    def sequences(self) -> int:
        return self.cu_seqlens.numel() - 1


@dataclass
class DecodeBatch:
    mixed_qkv: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    state_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    out: torch.Tensor

    @property
    def sequences(self) -> int:
        return self.mixed_qkv.shape[0]


def make_prefill_batch(lengths: list[int], shape: GdnShape, device: str = "cuda", seed: int = 0,
                       warm: bool = False) -> PrefillBatch:
    generator = torch.Generator(device=device).manual_seed(seed)
    tokens = sum(lengths)
    randn = lambda *size: torch.randn(*size, device=device, dtype=torch.bfloat16, generator=generator)
    cu_seqlens = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device=device, dtype=torch.int32)
    n = len(lengths)
    slots = n + NULL_SLOT
    conv_state = torch.zeros(slots, shape.conv_dim, shape.conv_kernel - 1, device=device, dtype=torch.bfloat16)
    ssm_state = torch.zeros(slots, shape.v_heads, shape.head_dim, shape.head_dim, device=device, dtype=torch.float32)
    if warm:
        conv_state = randn(slots, shape.conv_dim, shape.conv_kernel - 1)
        ssm_state = torch.randn(slots, shape.v_heads, shape.head_dim, shape.head_dim, device=device, generator=generator) * 0.05
    return PrefillBatch(
        mixed_qkv=randn(tokens, shape.conv_dim),
        a=randn(tokens, shape.v_heads),
        b=randn(tokens, shape.v_heads),
        cu_seqlens=cu_seqlens,
        conv_state=conv_state,
        ssm_state=ssm_state,
        has_initial_state=torch.full((n,), warm, device=device, dtype=torch.bool),
        state_indices=torch.arange(NULL_SLOT, slots, device=device, dtype=torch.int32),
    )


def make_decode_batch(sequences: int, shape: GdnShape, device: str = "cuda", seed: int = 0) -> DecodeBatch:
    generator = torch.Generator(device=device).manual_seed(seed)
    randn = lambda *size: torch.randn(*size, device=device, dtype=torch.bfloat16, generator=generator)
    slots = sequences + NULL_SLOT
    return DecodeBatch(
        mixed_qkv=randn(sequences, shape.conv_dim),
        a=randn(sequences, shape.v_heads),
        b=randn(sequences, shape.v_heads),
        conv_state=randn(slots, shape.conv_dim, shape.conv_kernel - 1),
        ssm_state=torch.randn(slots, shape.v_heads, shape.head_dim, shape.head_dim, device=device, generator=generator) * 0.05,
        state_indices=torch.arange(NULL_SLOT, slots, device=device, dtype=torch.int32),
        cu_seqlens=torch.arange(sequences + 1, device=device, dtype=torch.int32),
        out=torch.empty(sequences, shape.v_heads, shape.head_dim, device=device, dtype=torch.bfloat16),
    )


class GdnKernels:
    def __init__(self, shape: GdnShape, weights: GdnWeights):
        self.shape = shape
        self.weights = weights
        self.layer, self.conv, self.fla, self._conv_metadata = _ops()
        self.conv_weight = weights.conv_weight.contiguous()

    def prepare(self, batch: PrefillBatch) -> PrefillBatch:
        """The scheduling the engine precomputes once per step, so the kernel call does no host sync."""
        nums_dict, batch_ptr, token_chunk_offset_ptr = self._conv_metadata(
            batch.cu_seqlens.cpu(), device=batch.cu_seqlens.device)
        batch.conv_metadata = SimpleNamespace(nums_dict=nums_dict, batch_ptr=batch_ptr,
                                              token_chunk_offset_ptr=token_chunk_offset_ptr)
        return batch

    def conv_prefill(self, batch: PrefillBatch) -> torch.Tensor:
        out = self.conv.causal_conv1d_fn(
            batch.mixed_qkv.transpose(0, 1),
            self.conv_weight,
            self.weights.conv_bias,
            activation="silu",
            conv_states=batch.conv_state,
            has_initial_state=batch.has_initial_state,
            cache_indices=batch.state_indices,
            query_start_loc=batch.cu_seqlens,
            metadata=batch.conv_metadata,
        )
        return out.transpose(0, 1)

    def prep(self, conv_out: torch.Tensor, batch: PrefillBatch):
        q, k, v, g, beta = self.fla.fused_post_conv_prep(
            conv_output=conv_out,
            a=batch.a,
            b=batch.b,
            A_log=self.weights.A_log,
            dt_bias=self.weights.dt_bias,
            num_k_heads=self.shape.k_heads,
            head_k_dim=self.shape.head_dim,
            head_v_dim=self.shape.head_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        return (t.unsqueeze(0) for t in (q, k, v, g, beta))

    def chunk_rule(self, backend: str, q, k, v, g, beta, batch: PrefillBatch):
        initial_state = batch.ssm_state[batch.state_indices]
        initial_state[~batch.has_initial_state] = 0
        if backend == "flashinfer":
            return self.layer.fi_chunk_gated_delta_rule(
                q, k, v, g, beta, initial_state, True, cu_seqlens=batch.cu_seqlens, use_qk_l2norm_in_kernel=False)
        if backend == "triton":
            return self.fla.chunk_gated_delta_rule(
                q, k, v, g, beta, initial_state=initial_state, output_final_state=True,
                cu_seqlens=batch.cu_seqlens, use_qk_l2norm_in_kernel=False)
        raise ValueError(f"unknown prefill backend {backend}")

    def prefill(self, backend: str, batch: PrefillBatch) -> Callable[[], torch.Tensor]:
        self.prepare(batch)

        def run():
            conv_out = self.conv_prefill(batch)
            q, k, v, g, beta = self.prep(conv_out, batch)
            out, final_state = self.chunk_rule(backend, q, k, v, g, beta, batch)
            return out, final_state

        return run

    def conv_decode(self, batch: DecodeBatch) -> torch.Tensor:
        return self.conv.causal_conv1d_update(
            batch.mixed_qkv,
            batch.conv_state,
            self.conv_weight,
            self.weights.conv_bias,
            "silu",
            conv_state_indices=batch.state_indices,
            validate_data=False,
        )

    def decode(self, backend: str, batch: DecodeBatch) -> Callable[[], torch.Tensor]:
        scale = self.shape.head_dim ** -0.5

        def packed():
            conv_out = self.conv_decode(batch)
            self.fla.fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=conv_out, a=batch.a, b=batch.b, A_log=self.weights.A_log, dt_bias=self.weights.dt_bias,
                scale=scale, initial_state=batch.ssm_state, out=batch.out.unsqueeze(1),
                ssm_state_indices=batch.state_indices, use_qk_l2norm_in_kernel=True)
            return batch.out

        def sigmoid_gating():
            conv_out = self.conv_decode(batch)
            q, k, v = torch.split(conv_out, [self.shape.key_dim, self.shape.key_dim, self.shape.value_dim], dim=-1)
            view = lambda t, heads: t.contiguous().view(1, -1, heads, self.shape.head_dim)
            out, _ = self.fla.fused_sigmoid_gating_delta_rule_update(
                A_log=self.weights.A_log, a=batch.a, b=batch.b, dt_bias=self.weights.dt_bias,
                q=view(q, self.shape.k_heads), k=view(k, self.shape.k_heads), v=view(v, self.shape.v_heads),
                initial_state=batch.ssm_state, inplace_final_state=True, cu_seqlens=batch.cu_seqlens,
                ssm_state_indices=batch.state_indices, use_qk_l2norm_in_kernel=True)
            return out

        if backend == "packed":
            return packed
        if backend == "sigmoid_gating":
            return sigmoid_gating
        raise ValueError(f"unknown decode backend {backend}")


HISTORY_CHUNK = 16384


def prefill_history(kernels: GdnKernels, history_tokens: int, sequences: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Real conv and recurrent states after prefilling `history_tokens`, built once and shared by every sequence."""
    shape = kernels.shape
    slots = sequences + NULL_SLOT
    conv_state = torch.zeros(slots, shape.conv_dim, shape.conv_kernel - 1, device="cuda", dtype=torch.bfloat16)
    ssm_state = torch.zeros(slots, shape.v_heads, shape.head_dim, shape.head_dim, device="cuda")
    if history_tokens == 0:
        return conv_state, ssm_state
    done = 0
    while done < history_tokens:
        chunk = min(HISTORY_CHUNK, history_tokens - done)
        batch = kernels.prepare(make_prefill_batch([chunk], shape, seed=seed + done, warm=done > 0))
        if done > 0:
            batch.conv_state[NULL_SLOT] = conv_state[NULL_SLOT]
            batch.ssm_state[NULL_SLOT] = ssm_state[NULL_SLOT]
        conv_out = kernels.conv_prefill(batch)
        q, k, v, g, beta = kernels.prep(conv_out, batch)
        _, final_state = kernels.chunk_rule("flashinfer", q, k, v, g, beta, batch)
        conv_state[NULL_SLOT] = batch.conv_state[NULL_SLOT]
        ssm_state[NULL_SLOT] = final_state[0].to(torch.float32)
        done += chunk
    conv_state[NULL_SLOT:] = conv_state[NULL_SLOT]
    ssm_state[NULL_SLOT:] = ssm_state[NULL_SLOT]
    torch.cuda.synchronize()
    return conv_state, ssm_state
