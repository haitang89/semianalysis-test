"""Where attention overtakes Gated DeltaNet as the cached history grows.

    python3 -m analysis.crossover [--attention results/processed/attention_warm.csv]
                                  [--gdn results/processed/gdn_warm.csv]

Both warm sweeps compute the same number of new tokens at the same batch. GDN time does
not depend on the history, attention time grows with it. The crossover is the cached
length where one attention layer costs as much as one GDN layer, found by linear
interpolation between measured points. The step crossover weighs the layers as the
model does: 16 attention layers against 48 GDN layers.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

from .data import load_csv, select

ATTENTION_LAYERS = 16
GDN_LAYERS = 48


@dataclass(frozen=True)
class Crossover:
    new_tokens: int
    batch: int
    gdn_backend: str
    gdn_layer_us: float
    attention_layer_us_at_zero: float
    attention_layer_us_at_max: float
    max_cached: int
    layer_crossover_cached: Optional[float]
    step_crossover_cached: Optional[float]
    step_share_attention_at_max: float


def crossing(curve: Sequence[tuple[float, float]], target: float) -> Optional[float]:
    """Cached length where the rising curve first reaches target; None when it stays below."""
    points = sorted(curve)
    if points[0][1] >= target:
        return float(points[0][0])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 < target <= y1:
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return None


def attention_curves(rows: list[dict]) -> dict[tuple[int, int], list[tuple[float, float]]]:
    curves: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for row in select(rows, kind="warm", fraction=None):
        if row.get("cached") is None or row.get("kernel_us") is None:
            continue
        curves.setdefault((row["new_tokens"], row["batch"]), []).append((float(row["cached"]), float(row["kernel_us"])))
    return curves


def gdn_times(rows: list[dict], backend: str, history: int) -> dict[tuple[int, int], float]:
    times = {}
    for row in select(rows, kind="warm", backend=backend, history=history, fraction=None):
        if row.get("kernel_us") is not None:
            times[(row["new_tokens"], row["batch"])] = float(row["kernel_us"])
    return times


def crossovers(attention_rows: list[dict], gdn_rows: list[dict], gdn_backend: str = "flashinfer",
               history: int = 1024) -> list[Crossover]:
    curves = attention_curves(attention_rows)
    gdn = gdn_times(gdn_rows, gdn_backend, history)
    out = []
    for key in sorted(curves.keys() & gdn.keys()):
        curve = sorted(curves[key])
        gdn_us = gdn[key]
        step_curve = [(x, ATTENTION_LAYERS * y) for x, y in curve]
        out.append(Crossover(
            new_tokens=key[0],
            batch=key[1],
            gdn_backend=gdn_backend,
            gdn_layer_us=gdn_us,
            attention_layer_us_at_zero=curve[0][1],
            attention_layer_us_at_max=curve[-1][1],
            max_cached=int(curve[-1][0]),
            layer_crossover_cached=crossing(curve, gdn_us),
            step_crossover_cached=crossing(step_curve, GDN_LAYERS * gdn_us),
            step_share_attention_at_max=ATTENTION_LAYERS * curve[-1][1] / (ATTENTION_LAYERS * curve[-1][1] + GDN_LAYERS * gdn_us),
        ))
    return out


def fmt_tokens(value: Optional[float]) -> str:
    if value is None:
        return "beyond grid"
    return f"{value:,.0f}"


def markdown(rows: list[Crossover]) -> str:
    lines = ["| new tokens | batch | GDN layer us | attention layer us, 0 cached | attention layer us, max cached | "
             "layer crossover (cached tokens) | step crossover (cached tokens) | attention share of step at max cached |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r.new_tokens} | {r.batch} | {r.gdn_layer_us:.0f} | {r.attention_layer_us_at_zero:.0f} | "
                     f"{r.attention_layer_us_at_max:.0f} | {fmt_tokens(r.layer_crossover_cached)} | "
                     f"{fmt_tokens(r.step_crossover_cached)} | {100 * r.step_share_attention_at_max:.0f}% |")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Attention versus GDN crossover from the warm sweeps")
    parser.add_argument("--attention", default="results/processed/attention_warm.csv")
    parser.add_argument("--gdn", default="results/processed/gdn_warm.csv")
    parser.add_argument("--gdn-backend", default="flashinfer")
    parser.add_argument("--history", type=int, default=1024)
    parser.add_argument("--out", default="results/processed/crossover.json")
    parser.add_argument("--table", default="results/processed/crossover.md")
    args = parser.parse_args(argv)

    rows = crossovers(load_csv(Path(args.attention)), load_csv(Path(args.gdn)), args.gdn_backend, args.history)
    if not rows:
        print("no matching warm points", file=sys.stderr)
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n", encoding="utf-8")
    Path(args.table).write_text(markdown(rows) + "\n", encoding="utf-8")
    print(markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
