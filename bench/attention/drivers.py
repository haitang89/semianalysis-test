"""Paged attention exactly as vLLM's FlashAttention backend calls it on this GPU.

One call covers every regime: each sequence has some new query tokens and some
already cached KV tokens, both living in the paged cache. Cold prefill has no
cached tokens, decode has one new token, warm append has both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

Q_HEADS, KV_HEADS, HEAD_DIM = 24, 4, 256
FA_VERSION = 3


@dataclass(frozen=True)
class SequenceSpec:
    new: int
    cached: int

    @property
    def kv(self) -> int:
        return self.new + self.cached


@dataclass
class AttentionBatch:
    q: torch.Tensor
    kv_cache: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    page_size: int
    specs: tuple
    out: torch.Tensor
    scheduler_metadata: Optional[torch.Tensor] = None

    @property
    def new_tokens(self) -> int:
        return self.q.shape[0]

    @property
    def sequences(self) -> int:
        return len(self.specs)


def make_batch(specs: Sequence[SequenceSpec], page_size: int, device: str = "cuda", seed: int = 0,
               kv_dtype: torch.dtype = torch.bfloat16) -> AttentionBatch:
    generator = torch.Generator(device=device).manual_seed(seed)
    new_total = sum(spec.new for spec in specs)
    blocks_per_seq = [math.ceil(spec.kv / page_size) for spec in specs]
    total_blocks = sum(blocks_per_seq)
    kv_cache = torch.randn(2, total_blocks, page_size, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=generator)
    if kv_dtype != torch.bfloat16:
        kv_cache = kv_cache.to(kv_dtype)
    block_table = torch.zeros(len(specs), max(blocks_per_seq), device=device, dtype=torch.int32)
    next_block = 0
    for row, count in enumerate(blocks_per_seq):
        block_table[row, :count] = torch.arange(next_block, next_block + count, device=device, dtype=torch.int32)
        next_block += count
    cu_seqlens_q = torch.tensor([0] + list(_cumsum(spec.new for spec in specs)), device=device, dtype=torch.int32)
    return AttentionBatch(
        q=torch.randn(new_total, Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=generator),
        kv_cache=kv_cache,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=torch.tensor([spec.kv for spec in specs], device=device, dtype=torch.int32),
        max_seqlen_q=max(spec.new for spec in specs),
        max_seqlen_k=max(spec.kv for spec in specs),
        page_size=page_size,
        specs=tuple(specs),
        out=torch.empty(new_total, Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16),
    )


def _cumsum(values):
    total = 0
    for value in values:
        total += value
        yield total


class FlashAttentionKernel:
    def __init__(self, use_scheduler_metadata: bool = True):
        from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata

        self._attention = flash_attn_varlen_func
        self._schedule = get_scheduler_metadata
        self.use_scheduler_metadata = use_scheduler_metadata
        self.scale = HEAD_DIM ** -0.5

    def prepare(self, batch: AttentionBatch) -> AttentionBatch:
        """The scheduling vLLM's metadata builder computes once per step."""
        if self.use_scheduler_metadata:
            batch.scheduler_metadata = self._schedule(
                batch_size=batch.sequences,
                max_seqlen_q=batch.max_seqlen_q,
                max_seqlen_k=batch.max_seqlen_k,
                num_heads_q=Q_HEADS,
                num_heads_kv=KV_HEADS,
                headdim=HEAD_DIM,
                cache_seqlens=batch.seqused_k,
                qkv_dtype=batch.q.dtype,
                cu_seqlens_q=batch.cu_seqlens_q,
                page_size=batch.page_size,
                causal=True,
            )
        return batch

    def kernel(self, batch: AttentionBatch) -> Callable[[], torch.Tensor]:
        self.prepare(batch)
        key_cache, value_cache = batch.kv_cache[0], batch.kv_cache[1]

        def run():
            self._attention(
                q=batch.q,
                k=key_cache,
                v=value_cache,
                out=batch.out,
                cu_seqlens_q=batch.cu_seqlens_q,
                max_seqlen_q=batch.max_seqlen_q,
                seqused_k=batch.seqused_k,
                max_seqlen_k=batch.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=batch.block_table,
                scheduler_metadata=batch.scheduler_metadata,
                fa_version=FA_VERSION,
            )
            return batch.out

        return run


def reference(batch: AttentionBatch) -> torch.Tensor:
    """Plain attention per sequence in fp32: new queries attend causally over cached plus new keys."""
    group = Q_HEADS // KV_HEADS
    outputs = []
    for row, spec in enumerate(batch.specs):
        start, end = batch.cu_seqlens_q[row].item(), batch.cu_seqlens_q[row + 1].item()
        q = batch.q[start:end].float()
        blocks = batch.block_table[row, : math.ceil(spec.kv / batch.page_size)].long()
        k = batch.kv_cache[0][blocks].reshape(-1, KV_HEADS, HEAD_DIM)[: spec.kv].float()
        v = batch.kv_cache[1][blocks].reshape(-1, KV_HEADS, HEAD_DIM)[: spec.kv].float()
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * HEAD_DIM ** -0.5
        positions = torch.arange(spec.new, device=q.device)[:, None] + spec.cached
        keys = torch.arange(spec.kv, device=q.device)[None, :]
        scores = scores.masked_fill((keys > positions)[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, v))
    return torch.cat(outputs)
