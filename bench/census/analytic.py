"""Operator inventory derived from the model config, laid out the way vLLM executes it.

Shapes are given per layer as functions of the tokens in a step (M), the sequences
in the batch (B) and the KV length (L). Weight bytes follow the FP8 checkpoint: the
big projections are FP8 with block scales, everything the checkpoint leaves in BF16
stays BF16.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

FP8_BYTES = 1.0
BF16_BYTES = 2.0
FP32_BYTES = 4.0
FP8_BLOCK = 128


@dataclass(frozen=True)
class Geometry:
    layers: int
    hidden: int
    intermediate: int
    vocab: int
    q_heads: int
    kv_heads: int
    head_dim: int
    rotary_dims: int
    gdn_k_heads: int
    gdn_v_heads: int
    gdn_head_dim: int
    conv_kernel: int
    attention_layers: tuple
    mtp_layers: int
    fp8_weights: bool

    @property
    def gdn_layers(self) -> int:
        return self.layers - len(self.attention_layers)

    @property
    def q_dim(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def gdn_key_dim(self) -> int:
        return self.gdn_k_heads * self.gdn_head_dim

    @property
    def gdn_value_dim(self) -> int:
        return self.gdn_v_heads * self.gdn_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.gdn_key_dim + self.gdn_value_dim


def geometry_from_config(config: dict) -> Geometry:
    text = config.get("text_config", config)
    quant = config.get("quantization_config", {})
    layer_types = text["layer_types"]
    return Geometry(
        layers=text["num_hidden_layers"],
        hidden=text["hidden_size"],
        intermediate=text["intermediate_size"],
        vocab=text["vocab_size"],
        q_heads=text["num_attention_heads"],
        kv_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        rotary_dims=int(text["head_dim"] * text.get("partial_rotary_factor", 1.0)),
        gdn_k_heads=text["linear_num_key_heads"],
        gdn_v_heads=text["linear_num_value_heads"],
        gdn_head_dim=text["linear_key_head_dim"],
        conv_kernel=text["linear_conv_kernel_dim"],
        attention_layers=tuple(i for i, kind in enumerate(layer_types) if kind == "full_attention"),
        mtp_layers=text.get("mtp_num_hidden_layers", 0),
        fp8_weights=quant.get("quant_method") == "fp8",
    )


@dataclass(frozen=True)
class Op:
    block: str
    name: str
    kind: str
    dtype: str
    shape: str
    per_step: int
    weight_bytes: float = 0.0
    flops_per_token: float = 0.0
    note: str = ""


def gemm(block: str, name: str, out_features: int, in_features: int, per_step: int, dtype: str, note: str = "") -> Op:
    weight = out_features * in_features * (FP8_BYTES if dtype == "fp8" else BF16_BYTES)
    if dtype == "fp8":
        weight += (out_features / FP8_BLOCK) * (in_features / FP8_BLOCK) * FP32_BYTES
    return Op(
        block=block,
        name=name,
        kind="gemm",
        dtype=dtype,
        shape=f"[M, {in_features}] x [{in_features}, {out_features}] -> [M, {out_features}]",
        per_step=per_step,
        weight_bytes=weight,
        flops_per_token=2.0 * out_features * in_features,
        note=note,
    )


def inventory(g: Geometry) -> list[Op]:
    proj = "fp8" if g.fp8_weights else "bf16"
    n_attn, n_gdn, n_all = len(g.attention_layers), g.gdn_layers, g.layers
    ops: list[Op] = [
        Op("embedding", "embed_tokens", "gather", "bf16", f"[M] -> [M, {g.hidden}]", 1, g.vocab * g.hidden * BF16_BYTES),
        Op("every_layer", "input_layernorm", "rmsnorm", "bf16", f"[M, {g.hidden}]", n_all, g.hidden * BF16_BYTES),
        Op("every_layer", "post_attention_layernorm", "rmsnorm", "bf16", f"[M, {g.hidden}]", n_all, g.hidden * BF16_BYTES),
        Op("every_layer", "residual_add", "elementwise", "bf16", f"[M, {g.hidden}] x2", 2 * n_all),
        gemm("mlp", "gate_up_proj", 2 * g.intermediate, g.hidden, n_all, proj),
        Op("mlp", "silu_and_mul", "elementwise", "bf16", f"[M, {2 * g.intermediate}] -> [M, {g.intermediate}]", n_all),
        gemm("mlp", "down_proj", g.hidden, g.intermediate, n_all, proj),
        gemm("attention", "qkv_proj", 2 * g.q_dim + 2 * g.kv_dim, g.hidden, n_attn, proj,
             note="q with output gate, k, v fused into one GEMM"),
        Op("attention", "q_norm", "rmsnorm", "bf16", f"[M, {g.q_heads}, {g.head_dim}]", n_attn, g.head_dim * BF16_BYTES),
        Op("attention", "k_norm", "rmsnorm", "bf16", f"[M, {g.kv_heads}, {g.head_dim}]", n_attn, g.head_dim * BF16_BYTES),
        Op("attention", "rotary", "elementwise", "bf16",
           f"[M, {g.q_heads + g.kv_heads}, {g.rotary_dims} of {g.head_dim}]", n_attn),
        Op("attention", "kv_cache_write", "scatter", "bf16", f"[M, 2, {g.kv_heads}, {g.head_dim}] -> paged cache", n_attn),
        Op("attention", "attention", "attention", "bf16",
           f"q [M, {g.q_heads}, {g.head_dim}], kv [L, {g.kv_heads}, {g.head_dim}] per sequence", n_attn,
           flops_per_token=4.0 * g.q_heads * g.head_dim, note="flops per token per KV token; causal halves prefill"),
        Op("attention", "output_gate", "elementwise", "bf16", f"[M, {g.q_dim}] * sigmoid([M, {g.q_dim}])", n_attn),
        gemm("attention", "o_proj", g.hidden, g.q_dim, n_attn, proj),
        gemm("gdn", "in_proj_qkvz", 2 * g.gdn_key_dim + 2 * g.gdn_value_dim, g.hidden, n_gdn, proj,
             note="checkpoint tensors in_proj_qkv and in_proj_z fused"),
        gemm("gdn", "in_proj_ba", 2 * g.gdn_v_heads, g.hidden, n_gdn, "bf16",
             note="checkpoint tensors in_proj_b and in_proj_a fused, kept BF16"),
        Op("gdn", "causal_conv1d", "conv", "bf16", f"[B, {g.conv_dim}, T] kernel {g.conv_kernel}, state [B, {g.conv_dim}, {g.conv_kernel - 1}]",
           n_gdn, g.conv_dim * g.conv_kernel * BF16_BYTES, flops_per_token=2.0 * g.conv_dim * g.conv_kernel),
        Op("gdn", "post_conv_prep", "elementwise", "bf16",
           f"split [M, {g.conv_dim}] -> q,k [M, {g.gdn_k_heads}, {g.gdn_head_dim}], v [M, {g.gdn_v_heads}, {g.gdn_head_dim}]; l2norm; gates from a, b",
           n_gdn, 2 * g.gdn_v_heads * FP32_BYTES),
        Op("gdn", "gated_delta_rule", "linear_attention", "bf16",
           f"q,k [M, {g.gdn_k_heads}, {g.gdn_head_dim}], v [M, {g.gdn_v_heads}, {g.gdn_head_dim}], state [B, {g.gdn_v_heads}, {g.gdn_head_dim}, {g.gdn_head_dim}] fp32",
           n_gdn, flops_per_token=8.0 * g.gdn_v_heads * g.gdn_head_dim * g.gdn_head_dim,
           note="chunked kernel for prefill, recurrent kernel for decode; state is fixed size"),
        Op("gdn", "gated_rmsnorm", "rmsnorm", "bf16", f"[M, {g.gdn_value_dim}] gated by z", n_gdn, g.gdn_value_dim * BF16_BYTES),
        gemm("gdn", "out_proj", g.hidden, g.gdn_value_dim, n_gdn, proj),
        Op("head", "final_norm", "rmsnorm", "bf16", f"[M, {g.hidden}]", 1, g.hidden * BF16_BYTES),
        gemm("head", "lm_head", g.vocab, g.hidden, 1, "bf16", note="only the last token of each sequence"),
    ]
    if g.fp8_weights:
        ops.append(Op("every_layer", "fp8_activation_quant", "quantize", "fp8",
                      "[M, K] bf16 -> fp8 with per token group scales, before each FP8 GEMM",
                      3 * n_gdn + 2 * n_attn + 2 * n_all))
    if g.mtp_layers:
        ops.append(gemm("mtp", "mtp_fc", g.hidden, 2 * g.hidden, 0, "bf16", note="speculative decoding only"))
    return ops


def language_model_weight_bytes(ops: Iterable[Op]) -> float:
    return sum(op.weight_bytes * op.per_step for op in ops if op.block != "mtp")


def gdn_state_bytes(g: Geometry) -> dict:
    recurrent = g.gdn_v_heads * g.gdn_head_dim * g.gdn_head_dim * FP32_BYTES
    conv = g.conv_dim * (g.conv_kernel - 1) * BF16_BYTES
    return {"recurrent_bytes": recurrent, "conv_bytes": conv, "per_layer_bytes": recurrent + conv, "layers": g.gdn_layers}


def kv_bytes_per_token(g: Geometry) -> dict:
    per_layer = 2 * g.kv_heads * g.head_dim * BF16_BYTES
    return {"per_layer_bytes": per_layer, "layers": len(g.attention_layers), "total_bytes": per_layer * len(g.attention_layers)}


def markdown_table(ops: Iterable[Op]) -> str:
    lines = ["| block | op | kind | dtype | shape | per step | weight MiB |", "|---|---|---|---|---|---:|---:|"]
    for op in ops:
        weight = f"{op.weight_bytes / 2**20:.1f}" if op.weight_bytes else ""
        lines.append(f"| {op.block} | {op.name} | {op.kind} | {op.dtype} | {op.shape} | {op.per_step} | {weight} |")
    return "\n".join(lines)


def census(config: dict) -> dict:
    g = geometry_from_config(config)
    ops = inventory(g)
    return {
        "geometry": asdict(g),
        "attention_layers": list(g.attention_layers),
        "ops": [asdict(op) for op in ops],
        "language_model_weight_bytes": language_model_weight_bytes(ops),
        "gdn_state": gdn_state_bytes(g),
        "kv_cache": kv_bytes_per_token(g),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Derive the operator inventory from the model config")
    parser.add_argument("--config", default="configs/qwen3.8-27b-fp8.config.json")
    parser.add_argument("--out", default="results/census/analytic_ops.json")
    parser.add_argument("--table", default="results/census/analytic_ops.md")
    args = parser.parse_args(argv)

    result = census(json.loads(Path(args.config).read_text(encoding="utf-8")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Path(args.table).write_text(markdown_table(Op(**op) for op in result["ops"]) + "\n", encoding="utf-8")
    g = result["geometry"]
    print(
        f"{g['layers']} layers ({len(result['attention_layers'])} attention, {g['layers'] - len(result['attention_layers'])} GDN), "
        f"{len(result['ops'])} op types, language model weights {result['language_model_weight_bytes'] / 1e9:.1f} GB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
