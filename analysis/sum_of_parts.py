"""Measured engine steps against the sum of their parts and against the roofline.

    python3 -m analysis.sum_of_parts [--steps results/census/step_distribution.csv]

Every step the step distribution recorded carries what the engine scheduled (prefill
sequences and tokens, decode sequences) and how long it took. For each step this module
builds two predictions. The kernel sum takes the measured attention and GDN kernel times
at the nearest benchmarked shapes, times the layer counts, plus the supporting ops at the
step's token count. The roofline takes the perf model's bound for the same composition.
The difference between the measured step and the kernel sum is the residual: framework
time, launch gaps and whatever the sweeps did not cover. Prefill tokens of a step are
spread evenly over its prefill sequences, and a decode sequence is given the mean KV
length of the load, because the step annotation does not carry per sequence lengths.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from bench.census.analytic import geometry_from_config

from .data import load_csv, select
from .perf_model import PerfModel, SequenceState, ceilings_from_file

ATTENTION_LAYERS = 16
GDN_LAYERS = 48


@dataclass(frozen=True)
class StepPrediction:
    mode: str
    caching: str
    concurrency: int
    step: int
    prefill_sequences: int
    prefill_tokens: int
    decode_sequences: int
    measured_us: float
    attention_us: float
    gdn_us: float
    ops_us: float
    kernel_sum_us: float
    roofline_us: float

    @property
    def residual_us(self) -> float:
        return self.measured_us - self.kernel_sum_us

    @property
    def residual_share(self) -> float:
        return self.residual_us / self.measured_us if self.measured_us else 0.0


def log_interp(points: list[tuple[float, float]], x: float) -> float:
    """Piecewise log log interpolation over (x, y) points, clamped to the ends."""
    points = sorted(p for p in points if p[0] > 0 and p[1] > 0)
    if not points:
        return 0.0
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))
    return points[-1][1]


def nearest(values: list[int], target: float) -> int:
    return min(values, key=lambda v: abs(math.log(max(v, 1)) - math.log(max(target, 1))))


class KernelTable:
    """Measured kernel times from the processed sweeps, looked up at a step's shapes.

    The sweeps time one kernel per CUDA graph replay, and a replay has a fixed cost of a
    few microseconds that the engine pays once per step, not once per kernel. launch_floor_us
    is taken off every kernel instance before the layers are summed.
    """

    def __init__(self, processed: Path, gdn_backend: str = "flashinfer", launch_floor_us: float = 0.0):
        self.floor = launch_floor_us
        self.attention_cold = load_csv(processed / "attention_cold.csv")
        self.gdn_cold = select(load_csv(processed / "gdn_cold.csv"), backend=gdn_backend)
        self.gdn_decode = select(load_csv(processed / "gdn_decode.csv"), backend="packed", history=None)
        ops_path = processed / "ops_gemm.csv"
        self.ops = select(load_csv(ops_path), backend="engine") if ops_path.exists() else []
        elementwise = processed / "ops_elementwise.csv"
        if elementwise.exists():
            self.ops += select(load_csv(elementwise), backend="engine")

    def once(self, us: float) -> float:
        return max(us - self.floor, 0.5)

    def attention_prefill_us(self, tokens_per_sequence: float, sequences: int) -> float:
        rows = select(self.attention_cold, kind="prefill")
        batch = nearest(sorted({r["batch"] for r in rows}), sequences)
        curve = [(r["tokens"], r["kernel_us"]) for r in rows if r["batch"] == batch]
        return self.once(log_interp(curve, tokens_per_sequence) * sequences / batch)

    def attention_decode_us(self, kv: float, sequences: int) -> float:
        rows = select(self.attention_cold, kind="decode")
        batch = nearest(sorted({r["batch"] for r in rows}), sequences)
        curve = [(r["kv"], r["kernel_us"]) for r in rows if r["batch"] == batch]
        return self.once(log_interp(curve, kv) * sequences / batch)

    def gdn_prefill_us(self, tokens_per_sequence: float, sequences: int) -> float:
        rows = select(self.gdn_cold, kind="prefill")
        batch = nearest(sorted({r["batch"] for r in rows}), sequences)
        curve = [(r["tokens"], r["kernel_us"]) for r in rows if r["batch"] == batch]
        return self.once(log_interp(curve, tokens_per_sequence) * sequences / batch)

    def gdn_decode_us(self, sequences: int) -> float:
        return self.once(log_interp([(r["batch"], r["kernel_us"]) for r in self.gdn_decode], sequences))

    def ops_us(self, tokens: int) -> float:
        total = 0.0
        for op in sorted({r["op"] for r in self.ops}):
            rows = select(self.ops, op=op)
            curve = [(r["tokens"], r["kernel_us"]) for r in rows]
            total += self.once(log_interp(curve, tokens)) * rows[0]["per_step_count"]
        return total


def predict(step: dict, table: KernelTable, model: PerfModel, decode_kv: int) -> StepPrediction:
    ps, pt, ds = int(step["prefill_sequences"]), int(step["prefill_tokens"]), int(step["decode_sequences"])
    per_sequence = pt / ps if ps else 0
    attention = gdn = 0.0
    if ps:
        attention += table.attention_prefill_us(per_sequence, ps)
        gdn += table.gdn_prefill_us(per_sequence, ps)
    if ds:
        attention += table.attention_decode_us(decode_kv, ds)
        gdn += table.gdn_decode_us(ds)
    ops = table.ops_us(pt + ds)
    sequences = [SequenceState(int(round(per_sequence)), 0)] * ps + [SequenceState(1, decode_kv)] * ds
    roofline = model.step(sequences).total_us if sequences else 0.0
    return StepPrediction(step["mode"], step["caching"], int(step["concurrency"]), int(step["step"]), ps, pt, ds,
                          float(step["duration_us"]), ATTENTION_LAYERS * attention, GDN_LAYERS * gdn, ops,
                          ATTENTION_LAYERS * attention + GDN_LAYERS * gdn + ops, roofline)


def summary(rows: list[StepPrediction]) -> str:
    lines = ["| mode | caching | concurrency | steps | measured us p50 | kernel sum us p50 | roofline us p50 | residual share p50 | "
             "attention share of kernel sum | GDN share | ops share |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    groups: dict[tuple, list[StepPrediction]] = {}
    for row in rows:
        groups.setdefault((row.mode, row.caching, row.concurrency), []).append(row)
    for key, group in sorted(groups.items()):
        measured = statistics.median(r.measured_us for r in group)
        kernel = statistics.median(r.kernel_sum_us for r in group)
        roofline = statistics.median(r.roofline_us for r in group)
        residual = statistics.median(r.residual_share for r in group)
        total = sum(r.kernel_sum_us for r in group) or 1.0
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(group)} | {measured:.0f} | {kernel:.0f} | {roofline:.0f} | {100 * residual:.0f}% | "
                     f"{100 * sum(r.attention_us for r in group) / total:.0f}% | {100 * sum(r.gdn_us for r in group) / total:.0f}% | "
                     f"{100 * sum(r.ops_us for r in group) / total:.0f}% |")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sum of parts and roofline against measured engine steps")
    parser.add_argument("--steps", default="results/census/step_distribution.csv")
    parser.add_argument("--processed", default="results/processed")
    parser.add_argument("--config", default="configs/qwen3.8-27b-fp8.config.json")
    parser.add_argument("--ceilings", default="results/env/ceilings.json")
    parser.add_argument("--decode-kv", type=int, default=1100, help="KV length assumed for a decode sequence")
    parser.add_argument("--launch-floor-us", type=float, default=0.0, help="graph replay cost taken off every measured kernel instance")
    parser.add_argument("--out", default="results/processed/sum_of_parts.json")
    parser.add_argument("--table", default="results/processed/sum_of_parts.md")
    args = parser.parse_args(argv)

    steps = load_csv(Path(args.steps))
    table = KernelTable(Path(args.processed), launch_floor_us=args.launch_floor_us)
    model = PerfModel(geometry_from_config(json.loads(Path(args.config).read_text(encoding="utf-8"))), ceilings_from_file(Path(args.ceilings)))
    rows = [predict(step, table, model, args.decode_kv) for step in steps if int(step["tokens"]) > 0]
    if not rows:
        print("no steps to predict", file=sys.stderr)
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"launch_floor_us": args.launch_floor_us, "decode_kv": args.decode_kv,
                                          "steps": [asdict(r) | {"residual_us": r.residual_us, "residual_share": r.residual_share} for r in rows]},
                                         indent=1) + "\n", encoding="utf-8", newline="\n")
    Path(args.table).write_text(summary(rows) + "\n", encoding="utf-8", newline="\n")
    print(summary(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
