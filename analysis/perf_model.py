"""Roofline performance model of one engine step of the hybrid model.

    python3 -m analysis.perf_model [--scenario "64x(1+4096)" ...] [--kernel-floor-us 0]

Each operator of a step gets its FLOPs and the bytes it has to move, from the model
geometry and the step composition (new and cached tokens per sequence). Its roofline
time is the larger of the compute time at the measured GEMM ceiling and the memory
time at the measured HBM ceiling, never less than an optional per op floor for the
launch bound regime. The step prediction is the sum over all layers, so it is a lower
bound for a serial engine, not a fit. Measured kernel rows from the sweeps are put
against the same roofline to give each kernel's attainment.

A scenario is written as groups of "COUNTx(NEW+CACHED)" joined by commas:
"1x(8192+0)" is one cold prefill chunk, "64x(1+4096)" is decode at batch 64 over 4k
tokens, "1x(784+7056),32x(1+16384)" is a warm chunk sharing a step with 32 decodes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from bench.census.analytic import Geometry, geometry_from_config
from bench.core.roofline import Ceilings, roofline_latency_us

from .data import load_csv

BF16 = 2.0
FP8 = 1.0
FP32 = 4.0
FP8_BLOCK = 128

SCENARIOS = {
    "cold prefill chunk 8192": "1x(8192+0)",
    "warm chunk 784 over 7056": "1x(784+7056)",
    "decode batch 1 at 4k": "1x(1+4096)",
    "decode batch 64 at 4k": "64x(1+4096)",
    "decode batch 64 at 64k": "64x(1+65536)",
    "decode batch 256 at 16k": "256x(1+16384)",
    "mixed: 1 chunk 4096 + 32 decodes at 16k": "1x(4096+0),32x(1+16384)",
}
GROUP = re.compile(r"^\s*(\d+)\s*x\s*\(\s*(\d+)\s*\+\s*(\d+)\s*\)\s*$")


@dataclass(frozen=True)
class SequenceState:
    new: int
    cached: int = 0

    @property
    def kv(self) -> int:
        return self.new + self.cached

    @property
    def decode(self) -> bool:
        return self.new == 1 and self.cached > 0


@dataclass(frozen=True)
class OpEstimate:
    block: str
    name: str
    count: int
    flops: float
    bytes: float
    bound: str
    call_us: float

    @property
    def step_us(self) -> float:
        return self.count * self.call_us


@dataclass(frozen=True)
class StepEstimate:
    scenario: str
    new_tokens: int
    sequences: int
    ops: tuple

    @property
    def total_us(self) -> float:
        return sum(op.step_us for op in self.ops)

    def by_block(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for op in self.ops:
            out[op.block] = out.get(op.block, 0.0) + op.step_us
        return out

    def by_bound(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for op in self.ops:
            out[op.bound] = out.get(op.bound, 0.0) + op.step_us
        return out


def parse_scenario(text: str) -> list[SequenceState]:
    sequences: list[SequenceState] = []
    for group in text.split(","):
        match = GROUP.match(group)
        if not match:
            raise ValueError(f"cannot read scenario group {group!r}; expected COUNTx(NEW+CACHED)")
        count, new, cached = (int(match.group(i)) for i in (1, 2, 3))
        if count < 1 or new < 1:
            raise ValueError(f"scenario group {group!r} needs at least one sequence and one new token")
        sequences.extend([SequenceState(new, cached)] * count)
    return sequences


class PerfModel:
    def __init__(self, geometry: Geometry, ceilings: Ceilings, kernel_floor_us: float = 0.0):
        self.g = geometry
        self.ceilings = ceilings
        self.floor_us = kernel_floor_us

    def estimate(self, block: str, name: str, count: int, flops: float, bytes_moved: float, dtype: str = "bf16") -> OpEstimate:
        compute_us = roofline_latency_us(flops, 0.0, self.ceilings.peak_tflops(dtype), self.ceilings.hbm_gbps)
        memory_us = roofline_latency_us(0.0, bytes_moved, self.ceilings.peak_tflops(dtype), self.ceilings.hbm_gbps)
        call_us, bound = max((compute_us, "compute"), (memory_us, "memory"))
        if call_us < self.floor_us:
            call_us, bound = self.floor_us, "floor"
        return OpEstimate(block, name, count, flops, bytes_moved, bound, call_us)

    def gemm(self, block: str, name: str, rows: int, in_features: int, out_features: int, count: int, dtype: str) -> OpEstimate:
        weight = in_features * out_features * (FP8 if dtype == "fp8" else BF16)
        if dtype == "fp8":
            weight += (in_features / FP8_BLOCK) * (out_features / FP8_BLOCK) * FP32
        activations = rows * in_features * (FP8 if dtype == "fp8" else BF16) + rows * out_features * BF16
        return self.estimate(block, name, count, 2.0 * rows * in_features * out_features, weight + activations, dtype)

    def elementwise(self, block: str, name: str, count: int, read_elements: float, write_elements: float,
                    read_bytes: float = BF16, write_bytes: float = BF16) -> OpEstimate:
        return self.estimate(block, name, count, 0.0, read_elements * read_bytes + write_elements * write_bytes)

    def attention_work(self, sequences: Sequence[SequenceState]) -> tuple[float, float]:
        g = self.g
        flops = kv_bytes = 0.0
        new_tokens = 0
        for s in sequences:
            attended = s.new * s.cached + s.new * (s.new + 1) / 2
            flops += 4.0 * g.q_heads * g.head_dim * attended
            kv_bytes += 2.0 * s.kv * g.kv_dim * BF16
            new_tokens += s.new
        return flops, kv_bytes + 2.0 * new_tokens * g.q_dim * BF16

    def gdn_state_bytes(self, sequences: int) -> float:
        g = self.g
        return sequences * (g.gdn_v_heads * g.gdn_head_dim * g.gdn_head_dim * FP32 + g.conv_dim * (g.conv_kernel - 1) * BF16)

    def gdn_work(self, tokens: int, states_read: int, states_written: int) -> tuple[float, float]:
        g = self.g
        flops = 2.0 * g.conv_dim * g.conv_kernel * tokens + 8.0 * g.gdn_v_heads * g.gdn_head_dim * g.gdn_head_dim * tokens
        activations = tokens * (g.conv_dim * BF16 * 2 + g.gdn_v_heads * 2 * BF16 + g.gdn_value_dim * BF16)
        return flops, activations + self.gdn_state_bytes(states_read + states_written)

    def step(self, sequences: Sequence[SequenceState], scenario: str = "") -> StepEstimate:
        g = self.g
        proj = "fp8" if g.fp8_weights else "bf16"
        m = sum(s.new for s in sequences)
        b = len(sequences)
        n_attn, n_gdn, n_all = len(g.attention_layers), g.gdn_layers, g.layers
        prefill = [s for s in sequences if not s.decode]
        decode = [s for s in sequences if s.decode]
        ops: list[OpEstimate] = [
            self.elementwise("embedding", "embed_tokens", 1, m * g.hidden, m * g.hidden),
            self.elementwise("every_layer", "input_layernorm", n_all, m * g.hidden, m * g.hidden),
            self.elementwise("every_layer", "post_attention_layernorm", n_all, m * g.hidden, m * g.hidden),
            self.elementwise("every_layer", "residual_add", 2 * n_all, 2 * m * g.hidden, m * g.hidden),
            self.gemm("mlp", "gate_up_proj", m, g.hidden, 2 * g.intermediate, n_all, proj),
            self.elementwise("mlp", "silu_and_mul", n_all, 2 * m * g.intermediate, m * g.intermediate),
            self.gemm("mlp", "down_proj", m, g.intermediate, g.hidden, n_all, proj),
            self.gemm("attention", "qkv_proj", m, g.hidden, 2 * g.q_dim + 2 * g.kv_dim, n_attn, proj),
            self.elementwise("attention", "q_norm", n_attn, m * g.q_dim, m * g.q_dim),
            self.elementwise("attention", "k_norm", n_attn, m * g.kv_dim, m * g.kv_dim),
            self.elementwise("attention", "rotary", n_attn, m * (g.q_heads + g.kv_heads) * g.rotary_dims,
                             m * (g.q_heads + g.kv_heads) * g.rotary_dims),
            self.elementwise("attention", "kv_cache_write", n_attn, 2 * m * g.kv_dim, 2 * m * g.kv_dim),
            self.estimate("attention", "attention", n_attn, *self.attention_work(sequences)),
            self.elementwise("attention", "output_gate", n_attn, 2 * m * g.q_dim, m * g.q_dim),
            self.gemm("attention", "o_proj", m, g.q_dim, g.hidden, n_attn, proj),
            self.gemm("gdn", "in_proj_qkvz", m, g.hidden, 2 * g.gdn_key_dim + 2 * g.gdn_value_dim, n_gdn, proj),
            self.gemm("gdn", "in_proj_ba", m, g.hidden, 2 * g.gdn_v_heads, n_gdn, "bf16"),
            self.elementwise("gdn", "post_conv_prep", n_gdn, m * g.conv_dim, m * g.conv_dim),
        ]
        if prefill:
            tokens = sum(s.new for s in prefill)
            warm = sum(1 for s in prefill if s.cached > 0)
            ops.append(self.estimate("gdn", "gated_delta_rule_prefill", n_gdn, *self.gdn_work(tokens, warm, len(prefill))))
        if decode:
            ops.append(self.estimate("gdn", "gated_delta_rule_decode", n_gdn, *self.gdn_work(len(decode), len(decode), len(decode))))
        ops += [
            self.elementwise("gdn", "gated_rmsnorm", n_gdn, 2 * m * g.gdn_value_dim, m * g.gdn_value_dim),
            self.gemm("gdn", "out_proj", m, g.gdn_value_dim, g.hidden, n_gdn, proj),
            self.elementwise("head", "final_norm", 1, b * g.hidden, b * g.hidden),
            self.gemm("head", "lm_head", b, g.hidden, g.vocab, 1, "bf16"),
        ]
        if g.fp8_weights:
            quant_elements = (n_all * (g.hidden + g.intermediate) + n_attn * (g.hidden + g.q_dim) + n_gdn * (g.hidden + g.gdn_value_dim)) * m
            ops.append(self.estimate("every_layer", "fp8_activation_quant", 1, 0.0, quant_elements * (BF16 + FP8)))
        return StepEstimate(scenario, m, b, tuple(ops))


def kernel_attainment(rows: Iterable[dict], ceilings: Ceilings, dtype: str = "bf16") -> list[dict]:
    out = []
    for row in rows:
        if not row.get("kernel_us") or row.get("flops") is None or row.get("bytes_moved") is None:
            continue
        compute_us = roofline_latency_us(row["flops"], 0.0, ceilings.peak_tflops(dtype), ceilings.hbm_gbps)
        memory_us = roofline_latency_us(0.0, row["bytes_moved"], ceilings.peak_tflops(dtype), ceilings.hbm_gbps)
        roofline_us, bound = max((compute_us, "compute"), (memory_us, "memory"))
        out.append({
            "benchmark": row["benchmark"], "backend": row["backend"], "point_id": row["point_id"],
            "kind": row.get("kind"), "batch": row.get("batch"), "tokens": row.get("tokens"), "new_tokens": row.get("new_tokens"),
            "cached": row.get("cached"), "kv": row.get("kv"), "history": row.get("history"), "distribution": row.get("distribution"),
            "kernel_us": row["kernel_us"], "roofline_us": roofline_us, "bound": bound,
            "attainment": roofline_us / row["kernel_us"],
        })
    return out


def ceilings_from_file(path: Path) -> Ceilings:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Ceilings(data["hbm_bw_gbps_measured"], data["bf16_tflops_measured"], data["fp8_tflops_measured"])


def step_table(steps: list[StepEstimate]) -> str:
    blocks = ["embedding", "every_layer", "mlp", "attention", "gdn", "head"]
    lines = ["| scenario | new tokens | sequences | " + " | ".join(f"{b} us" for b in blocks) + " | total us | memory bound share |",
             "|---|---:|---:|" + "---:|" * (len(blocks) + 2)]
    for step in steps:
        by_block, by_bound = step.by_block(), step.by_bound()
        cells = [f"{by_block.get(b, 0.0):.0f}" for b in blocks]
        share = by_bound.get("memory", 0.0) / step.total_us if step.total_us else 0.0
        lines.append(f"| {step.scenario} | {step.new_tokens} | {step.sequences} | " + " | ".join(cells)
                     + f" | {step.total_us:.0f} | {100 * share:.0f}% |")
    return "\n".join(lines)


def attainment_summary(rows: list[dict]) -> str:
    lines = ["| benchmark | backend | points | median attainment | min | max |", "|---|---|---:|---:|---:|---:|"]
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        groups.setdefault((row["benchmark"], row["backend"]), []).append(row["attainment"])
    for (benchmark, backend), values in sorted(groups.items()):
        values.sort()
        lines.append(f"| {benchmark} | {backend} | {len(values)} | {100 * values[len(values) // 2]:.0f}% | "
                     f"{100 * values[0]:.0f}% | {100 * values[-1]:.0f}% |")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Roofline model of an engine step and kernel attainment from the sweeps")
    parser.add_argument("--config", default="configs/qwen3.8-27b-fp8.config.json")
    parser.add_argument("--ceilings", default="results/env/ceilings.json")
    parser.add_argument("--processed", default="results/processed")
    parser.add_argument("--scenario", action="append", help="COUNTx(NEW+CACHED) groups joined by commas; repeatable")
    parser.add_argument("--kernel-floor-us", type=float, default=0.0)
    parser.add_argument("--out", default="results/processed/perf_model.json")
    parser.add_argument("--table", default="results/processed/perf_model.md")
    args = parser.parse_args(argv)

    geometry = geometry_from_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    ceilings = ceilings_from_file(Path(args.ceilings))
    model = PerfModel(geometry, ceilings, args.kernel_floor_us)
    scenarios = {s: s for s in args.scenario} if args.scenario else SCENARIOS
    steps = [model.step(parse_scenario(spec), name) for name, spec in scenarios.items()]

    processed = Path(args.processed)
    rows = []
    for path in sorted(processed.glob("*.csv")):
        if path.name.startswith(("attention_", "gdn_")):
            rows += kernel_attainment(load_csv(path), ceilings)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "ceilings": asdict(ceilings), "kernel_floor_us": args.kernel_floor_us,
        "steps": [{"scenario": s.scenario, "new_tokens": s.new_tokens, "sequences": s.sequences, "total_us": s.total_us,
                   "by_block": s.by_block(), "by_bound": s.by_bound(), "ops": [asdict(op) | {"step_us": op.step_us} for op in s.ops]} for s in steps],
        "kernel_attainment": rows,
    }, indent=2) + "\n", encoding="utf-8")
    table = step_table(steps) + ("\n\n" + attainment_summary(rows) if rows else "")
    Path(args.table).write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
