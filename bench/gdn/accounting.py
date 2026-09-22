"""Bytes and FLOPs of one GDN layer call, for utilization and the roofline model."""
from __future__ import annotations

from dataclasses import dataclass

from .reference import GdnShape

BF16 = 2
FP32 = 4


@dataclass(frozen=True)
class GdnWork:
    flops: float
    activation_bytes: float
    state_bytes: float

    @property
    def bytes(self) -> float:
        return self.activation_bytes + self.state_bytes


def state_bytes(shape: GdnShape, sequences: int) -> float:
    recurrent = shape.v_heads * shape.head_dim * shape.head_dim * FP32
    conv = shape.conv_dim * (shape.conv_kernel - 1) * BF16
    return sequences * (recurrent + conv)


def prefill_work(shape: GdnShape, tokens: int, sequences: int, warm: bool) -> GdnWork:
    conv_flops = 2.0 * shape.conv_dim * shape.conv_kernel * tokens
    delta_flops = 8.0 * shape.v_heads * shape.head_dim * shape.head_dim * tokens
    activations = tokens * (shape.conv_dim * BF16 * 2 + shape.v_heads * 2 * BF16 + shape.value_dim * BF16)
    state = state_bytes(shape, sequences) * (2 if warm else 1)
    return GdnWork(conv_flops + delta_flops, activations, state)


def decode_work(shape: GdnShape, sequences: int) -> GdnWork:
    conv_flops = 2.0 * shape.conv_dim * shape.conv_kernel * sequences
    delta_flops = 8.0 * shape.v_heads * shape.head_dim * shape.head_dim * sequences
    activations = sequences * (shape.conv_dim * BF16 * 2 + shape.v_heads * 2 * BF16 + shape.value_dim * BF16)
    return GdnWork(conv_flops + delta_flops, activations, 2 * state_bytes(shape, sequences))


def metrics(work: GdnWork, median_us: float, tokens: int, layers: int, ceilings: dict | None) -> dict:
    seconds = median_us * 1e-6
    result = {
        "flops": work.flops,
        "bytes_moved": work.bytes,
        "state_bytes": work.state_bytes,
        "achieved_tflops": work.flops / seconds / 1e12,
        "achieved_gbps": work.bytes / seconds / 1e9,
        "us_per_token": median_us / tokens,
        "per_step_us": median_us * layers,
    }
    if ceilings:
        result["bandwidth_util"] = result["achieved_gbps"] / ceilings["hbm_bw_gbps_measured"]
        result["compute_util"] = result["achieved_tflops"] / ceilings["bf16_tflops_measured"]
    return result
