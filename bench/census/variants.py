"""How the operator shapes change under deployment settings that do not change the model.

Tensor parallelism splits the projections and the heads across ranks. The KV cache
dtype changes the manager block size vLLM derives for this hybrid model. Speculative
decoding changes the decode query length and adds the MTP layer. Prefix caching
changes how prefill is chunked. Only the benchmarked configuration was measured;
every other row here is derived and labelled so.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .analytic import BF16_BYTES, FP32_BYTES, Geometry, geometry_from_config

BENCHMARKED = {"tensor_parallel": 1, "kv_dtype": "bf16", "speculative_tokens": 0, "prefix_caching": True}
KV_BYTES = {"bf16": 2, "fp8": 1}
MIN_ATTENTION_BLOCK = 16


@dataclass(frozen=True)
class Variant:
    setting: str
    value: str
    gdn_in_proj_qkvz: str
    gdn_out_proj: str
    attention_qkv_proj: str
    attention_o_proj: str
    mlp_gate_up_proj: str
    mlp_down_proj: str
    q_heads_per_rank: int
    kv_heads_per_rank: int
    gdn_v_heads_per_rank: int
    gdn_state_per_rank: str
    manager_block_tokens: int
    decode_query_len: int
    prefill_chunk_alignment: Optional[int]
    extra_ops: str
    measured: bool


def gemm(rows_in: int, cols_out: int) -> str:
    return f"[M, {rows_in}] x [{rows_in}, {cols_out}]"


def gdn_page_bytes(g: Geometry, per_rank_v_heads: int, speculative_tokens: int) -> float:
    key_dim = g.gdn_k_heads * g.gdn_head_dim
    value_dim = per_rank_v_heads * g.gdn_head_dim
    conv_dim = 2 * key_dim + value_dim
    recurrent = per_rank_v_heads * g.gdn_head_dim * g.gdn_head_dim * FP32_BYTES
    conv = conv_dim * (g.conv_kernel - 1 + speculative_tokens) * BF16_BYTES
    return recurrent + conv


def manager_block(g: Geometry, tp: int, kv_dtype: str, speculative_tokens: int) -> int:
    """vLLM grows the attention block until one attention page is at least one GDN state page."""
    kv_heads = max(1, g.kv_heads // tp)
    attention_bytes_per_token = 2 * kv_heads * g.head_dim * KV_BYTES[kv_dtype]
    gdn_bytes = gdn_page_bytes(g, g.gdn_v_heads // tp, speculative_tokens)
    return MIN_ATTENTION_BLOCK * math.ceil(gdn_bytes / (MIN_ATTENTION_BLOCK * attention_bytes_per_token))


def variant(g: Geometry, setting: str, value: str, tp: int = 1, kv_dtype: str = "bf16",
            speculative_tokens: int = 0, prefix_caching: bool = True, extra_ops: str = "") -> Variant:
    q_heads, kv_heads = g.q_heads // tp, max(1, g.kv_heads // tp)
    gdn_k, gdn_v = g.gdn_k_heads // tp, g.gdn_v_heads // tp
    key_dim, value_dim = gdn_k * g.gdn_head_dim, gdn_v * g.gdn_head_dim
    block = manager_block(g, tp, kv_dtype, speculative_tokens)
    measured = (tp, kv_dtype, speculative_tokens, prefix_caching) == (
        BENCHMARKED["tensor_parallel"], BENCHMARKED["kv_dtype"], BENCHMARKED["speculative_tokens"], BENCHMARKED["prefix_caching"])
    return Variant(
        setting=setting,
        value=value,
        gdn_in_proj_qkvz=gemm(g.hidden, 2 * key_dim + 2 * value_dim),
        gdn_out_proj=gemm(value_dim, g.hidden),
        attention_qkv_proj=gemm(g.hidden, 2 * q_heads * g.head_dim + 2 * kv_heads * g.head_dim),
        attention_o_proj=gemm(q_heads * g.head_dim, g.hidden),
        mlp_gate_up_proj=gemm(g.hidden, 2 * g.intermediate // tp),
        mlp_down_proj=gemm(g.intermediate // tp, g.hidden),
        q_heads_per_rank=q_heads,
        kv_heads_per_rank=kv_heads,
        gdn_v_heads_per_rank=gdn_v,
        gdn_state_per_rank=f"({gdn_v}, {g.gdn_head_dim}, {g.gdn_head_dim}) fp32 + conv ({2 * key_dim + value_dim}, {g.conv_kernel - 1 + speculative_tokens})",
        manager_block_tokens=block,
        decode_query_len=1 + speculative_tokens,
        prefill_chunk_alignment=block if prefix_caching else None,
        extra_ops=extra_ops,
        measured=measured,
    )


def variants(g: Geometry) -> list[Variant]:
    rows = [variant(g, "baseline", "TP 1, bf16 KV, no speculation, prefix caching on")]
    for tp in (2, 4, 8):
        note = "KV heads replicated across ranks" if tp > g.kv_heads else ""
        rows.append(variant(g, "tensor_parallel", str(tp), tp=tp, extra_ops=note or "all reduce after o_proj, out_proj and down_proj"))
    rows.append(variant(g, "kv_dtype", "fp8", kv_dtype="fp8", extra_ops="KV quantization before the cache write; attention kernel reads fp8 KV with scales"))
    rows.append(variant(g, "speculative_tokens", "1", speculative_tokens=1,
                        extra_ops="MTP layer: one attention layer plus MLP plus fc 10240 to 5120; GDN decode through the fused CUDA kernel"))
    rows.append(variant(g, "prefix_caching", "off", prefix_caching=False, extra_ops="prefill chunks fill the full token budget instead of stopping at block multiples"))
    return rows


def markdown(rows: list[Variant]) -> str:
    header = ("| setting | value | measured | GDN in_proj_qkvz | attention qkv_proj | MLP gate_up_proj | heads per rank (q / kv / gdn v) | "
              "GDN state per rank | manager block | decode query len | prefill chunk alignment | other changes |")
    lines = [header, "|" + "---|" * 12]
    for r in rows:
        lines.append(
            f"| {r.setting} | {r.value} | {'yes' if r.measured else 'analytic only'} | {r.gdn_in_proj_qkvz} | {r.attention_qkv_proj} | "
            f"{r.mlp_gate_up_proj} | {r.q_heads_per_rank} / {r.kv_heads_per_rank} / {r.gdn_v_heads_per_rank} | {r.gdn_state_per_rank} | "
            f"{r.manager_block_tokens} | {r.decode_query_len} | {r.prefill_chunk_alignment or 'none'} | {r.extra_ops} |")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Derive shape variants for deployment settings")
    parser.add_argument("--config", default="configs/qwen3.8-27b-fp8.config.json")
    parser.add_argument("--out", default="results/census/shape_variants.json")
    parser.add_argument("--table", default="results/census/shape_variants.md")
    args = parser.parse_args(argv)

    rows = variants(geometry_from_config(json.loads(Path(args.config).read_text(encoding="utf-8"))))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n", encoding="utf-8", newline="\n")
    Path(args.table).write_text(markdown(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rows)} variants, {sum(r.measured for r in rows)} measured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
