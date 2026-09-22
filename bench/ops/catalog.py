"""The supporting operators of one decoder step: projections and the elementwise ops around the mixers.

Shapes follow the analytic inventory (bench.census.analytic). Work is counted the way
the roofline model counts it: FLOPs of the GEMM, and the bytes an operator has to read
and write once. The FP8 projections take a BF16 activation, quantize it per 128 token
group inside the op the engine calls, and read FP8 weights with 128 x 128 block scales.
"""
from __future__ import annotations

from dataclasses import dataclass

from bench.census.analytic import Geometry

BF16 = 2.0
FP8 = 1.0
FP32 = 4.0
FP8_BLOCK = 128


@dataclass(frozen=True)
class OpSpec:
    name: str
    kind: str
    in_features: int
    out_features: int
    dtype: str
    per_step: int
    heads: int = 0
    head_dim: int = 0
    rotary_dim: int = 0


@dataclass(frozen=True)
class OpWork:
    flops: float
    bytes: float
    weight_bytes: float


def catalog(g: Geometry) -> dict[str, OpSpec]:
    proj = "fp8" if g.fp8_weights else "bf16"
    n_attn, n_gdn, n_all = len(g.attention_layers), g.gdn_layers, g.layers
    specs = [
        OpSpec("gate_up_proj", "gemm", g.hidden, 2 * g.intermediate, proj, n_all),
        OpSpec("down_proj", "gemm", g.intermediate, g.hidden, proj, n_all),
        OpSpec("qkv_proj", "gemm", g.hidden, 2 * g.q_dim + 2 * g.kv_dim, proj, n_attn),
        OpSpec("o_proj", "gemm", g.q_dim, g.hidden, proj, n_attn),
        OpSpec("in_proj_qkvz", "gemm", g.hidden, 2 * g.gdn_key_dim + 2 * g.gdn_value_dim, proj, n_gdn),
        OpSpec("in_proj_ba", "gemm", g.hidden, 2 * g.gdn_v_heads, "bf16", n_gdn),
        OpSpec("out_proj", "gemm", g.gdn_value_dim, g.hidden, proj, n_gdn),
        OpSpec("lm_head", "gemm", g.hidden, g.vocab, "bf16", 1),
        OpSpec("rms_norm", "rms_norm", g.hidden, g.hidden, "bf16", 2),
        OpSpec("fused_add_rms_norm", "fused_add_rms_norm", g.hidden, g.hidden, "bf16", 2 * n_all - 1),
        OpSpec("silu_and_mul", "silu_and_mul", 2 * g.intermediate, g.intermediate, "bf16", n_all),
        OpSpec("quant_hidden", "quant", g.hidden, g.hidden, "fp8", n_all + n_attn + n_gdn),
        OpSpec("quant_intermediate", "quant", g.intermediate, g.intermediate, "fp8", n_all),
        OpSpec("quant_mixer_out", "quant", g.q_dim, g.q_dim, "fp8", n_attn + n_gdn),
        OpSpec("rotary", "rotary", (g.q_heads + g.kv_heads) * g.head_dim, (g.q_heads + g.kv_heads) * g.head_dim, "bf16", n_attn,
               heads=g.q_heads + g.kv_heads, head_dim=g.head_dim, rotary_dim=g.rotary_dims),
    ]
    return {spec.name: spec for spec in specs}


def gemm_weight_bytes(spec: OpSpec) -> float:
    if spec.dtype == "fp8":
        return spec.in_features * spec.out_features * FP8 + (spec.in_features / FP8_BLOCK) * (spec.out_features / FP8_BLOCK) * FP32
    return spec.in_features * spec.out_features * BF16


def work(spec: OpSpec, tokens: int) -> OpWork:
    m, k, n = tokens, spec.in_features, spec.out_features
    if spec.kind == "gemm":
        weight = gemm_weight_bytes(spec)
        activations = m * k * BF16 + m * n * BF16
        if spec.dtype == "fp8":
            activations += m * k * FP8 * 2 + (m * k / FP8_BLOCK) * FP32 * 2
        return OpWork(2.0 * m * k * n, weight + activations, weight)
    if spec.kind == "rms_norm":
        return OpWork(0.0, m * k * BF16 * 2, k * BF16)
    if spec.kind == "fused_add_rms_norm":
        return OpWork(0.0, m * k * BF16 * 4, k * BF16)
    if spec.kind == "silu_and_mul":
        return OpWork(0.0, m * k * BF16 + m * n * BF16, 0.0)
    if spec.kind == "quant":
        return OpWork(0.0, m * k * BF16 + m * k * FP8 + (m * k / FP8_BLOCK) * FP32, 0.0)
    if spec.kind == "rotary":
        touched = m * spec.heads * spec.rotary_dim * BF16
        return OpWork(0.0, touched * 2, 0.0)
    raise ValueError(f"unknown op kind {spec.kind}")


def metrics(spec: OpSpec, work_: OpWork, median_us: float, tokens: int, ceilings: dict | None, dtype: str | None = None) -> dict:
    dtype = dtype or spec.dtype
    seconds = median_us * 1e-6
    result = {
        "flops": work_.flops,
        "bytes_moved": work_.bytes,
        "weight_bytes": work_.weight_bytes,
        "achieved_tflops": work_.flops / seconds / 1e12,
        "achieved_gbps": work_.bytes / seconds / 1e9,
        "us_per_token": median_us / tokens,
        "per_step_us": median_us * spec.per_step,
        "per_step_count": spec.per_step,
    }
    if ceilings:
        peak = ceilings["fp8_tflops_measured"] if dtype == "fp8" else ceilings["bf16_tflops_measured"]
        result["bandwidth_util"] = result["achieved_gbps"] / ceilings["hbm_bw_gbps_measured"]
        result["compute_util"] = result["achieved_tflops"] / peak
    return result
