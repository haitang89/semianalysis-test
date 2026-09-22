"""FLOPs and bytes of one attention layer call, causal aware, for utilization and the roofline model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .drivers import HEAD_DIM, KV_HEADS, Q_HEADS, SequenceSpec

BF16 = 2


@dataclass(frozen=True)
class AttentionWork:
    flops: float
    kv_bytes: float
    q_out_bytes: float

    @property
    def bytes(self) -> float:
        return self.kv_bytes + self.q_out_bytes


def work(specs: Sequence[SequenceSpec], kv_bytes_per_element: int = BF16) -> AttentionWork:
    flops = 0.0
    kv_bytes = 0.0
    new_tokens = 0
    for spec in specs:
        attended = spec.new * spec.cached + spec.new * (spec.new + 1) / 2
        flops += 4.0 * Q_HEADS * HEAD_DIM * attended
        kv_bytes += 2.0 * spec.kv * KV_HEADS * HEAD_DIM * kv_bytes_per_element
        new_tokens += spec.new
    q_out = 2.0 * new_tokens * Q_HEADS * HEAD_DIM * BF16
    return AttentionWork(flops, kv_bytes, q_out)


def metrics(work_: AttentionWork, median_us: float, new_tokens: int, layers: int, ceilings: dict | None) -> dict:
    seconds = median_us * 1e-6
    result = {
        "flops": work_.flops,
        "bytes_moved": work_.bytes,
        "kv_bytes": work_.kv_bytes,
        "achieved_tflops": work_.flops / seconds / 1e12,
        "achieved_gbps": work_.bytes / seconds / 1e9,
        "us_per_new_token": median_us / new_tokens,
        "per_step_us": median_us * layers,
    }
    if ceilings:
        result["bandwidth_util"] = result["achieved_gbps"] / ceilings["hbm_bw_gbps_measured"]
        result["compute_util"] = result["achieved_tflops"] / ceilings["bf16_tflops_measured"]
    return result
