"""Report figures from the tidy CSVs.

    python3 -m analysis.figures [--processed results/processed] [--out report/figures]

Every figure reads only `results/processed/*.csv` and `results/env/ceilings.json`, so it
can be regenerated from the committed results. A figure whose series are missing is
skipped with a note instead of failing the run. Kernel times are the CUDA graph replay
medians (eager medians where a point has none), the same numbers as in the tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .crossover import ATTENTION_LAYERS, GDN_LAYERS, attention_curves, gdn_times
from .data import load_csv, select
from .perf_model import ceilings_from_file, kernel_attainment

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#F0E442"]
FIGURE_SIZE = (7.0, 4.2)
Data = dict[str, list[dict]]


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    })


def load(processed: Path) -> Data:
    return {path.stem: load_csv(path) for path in sorted(processed.glob("*.csv")) if path.stem != "all_points"}


def series(rows: list[dict], x: str, y: str, **conditions) -> tuple[list, list]:
    points = sorted((r[x], r[y]) for r in select(rows, **conditions) if r.get(x) is not None and r.get(y) is not None)
    return [p[0] for p in points], [p[1] for p in points]


def categorical_x(ax, values: list) -> None:
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([f"{int(v):,}" for v in values])
    ax.grid(axis="x", visible=False)


def new_figure(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def attention_decode_bandwidth(data: Data, ceilings) -> Optional[plt.Figure]:
    rows = select(data.get("attention_cold", []), kind="decode")
    if not rows:
        return None
    fig, ax = new_figure("Attention decode: KV bytes read per second, FlashAttention 3", "KV tokens per sequence", "GB/s")
    for batch in sorted({r["batch"] for r in rows}):
        xs, ys = series(rows, "kv", "kernel_gbps", batch=batch)
        ax.plot(xs, ys, marker="o", label=f"batch {batch}")
    ax.axhline(ceilings.hbm_gbps, color="black", linestyle=":", linewidth=1, label="measured copy bandwidth")
    ax.set_xscale("log", base=2)
    ax.legend()
    return fig


def attention_prefill_tflops(data: Data, ceilings) -> Optional[plt.Figure]:
    rows = select(data.get("attention_cold", []), kind="prefill")
    if not rows:
        return None
    fig, ax = new_figure("Attention cold prefill: achieved TFLOPS, FlashAttention 3", "tokens per sequence", "TFLOPS")
    for batch in sorted({r["batch"] for r in rows}):
        xs, ys = series(rows, "tokens", "kernel_tflops", batch=batch)
        ax.plot(xs, ys, marker="o", label=f"batch {batch}")
    ax.axhline(ceilings.bf16_tflops, color="black", linestyle=":", linewidth=1, label="measured BF16 GEMM peak")
    ax.set_xscale("log", base=2)
    ax.legend()
    return fig


def crossover(data: Data, ceilings, batch: int = 8) -> Optional[plt.Figure]:
    curves = attention_curves(data.get("attention_warm", []))
    gdn = gdn_times(data.get("gdn_warm", []), "flashinfer", 1024)
    keys = sorted(k for k in curves if k[1] == batch and k in gdn)
    if not keys:
        return None
    fig, ax = new_figure(f"One attention layer against one GDN layer, batch {batch}", "cached tokens per sequence", "kernel time, us")
    cached = sorted({x for key in keys for x, _ in curves[key]})
    for color, key in zip(PALETTE, keys):
        points = sorted(curves[key])
        ax.plot([cached.index(p[0]) for p in points], [p[1] for p in points], marker="o", color=color, label=f"attention, {key[0]} new tokens")
        ax.axhline(gdn[key], color=color, linestyle="--", linewidth=1, label=f"GDN, {key[0]} new tokens")
    categorical_x(ax, cached)
    ax.set_yscale("log")
    ax.legend(ncol=2)
    return fig


def step_share(data: Data, ceilings, new_tokens: int = 512, batch: int = 8) -> Optional[plt.Figure]:
    curves = attention_curves(data.get("attention_warm", []))
    gdn = gdn_times(data.get("gdn_warm", []), "flashinfer", 1024)
    key = (new_tokens, batch)
    if key not in curves or key not in gdn:
        return None
    points = sorted(curves[key])
    fig, ax = new_figure(f"Mixer time per step, {new_tokens} new tokens per sequence, batch {batch}",
                         "cached tokens per sequence", "ms per step")
    xs = list(range(len(points)))
    attention_ms = [ATTENTION_LAYERS * p[1] / 1000 for p in points]
    gdn_ms = [GDN_LAYERS * gdn[key] / 1000] * len(points)
    ax.stackplot(xs, gdn_ms, attention_ms, labels=["48 GDN layers", "16 attention layers"], colors=[PALETTE[2], PALETTE[0]], alpha=0.85)
    categorical_x(ax, [p[0] for p in points])
    ax.legend(loc="upper left")
    return fig


def gdn_decode_bandwidth(data: Data, ceilings) -> Optional[plt.Figure]:
    rows = select(data.get("gdn_decode", []), kind="decode", history=None)
    if not rows:
        return None
    fig, ax = new_figure("GDN decode: state bytes moved per second", "sequences in the batch", "GB/s")
    for backend in sorted({r["backend"] for r in rows}):
        xs, ys = series(rows, "batch", "kernel_gbps", backend=backend)
        ax.plot(xs, ys, marker="o", label=backend)
    attention = select(data.get("attention_cold", []), kind="decode", kv=4096)
    if attention:
        xs, ys = series(attention, "batch", "kernel_gbps")
        ax.plot(xs, ys, marker="s", linestyle="-.", label="attention decode, 4k KV")
    ax.axhline(ceilings.hbm_gbps, color="black", linestyle=":", linewidth=1, label="measured copy bandwidth")
    ax.set_xscale("log", base=2)
    ax.legend()
    return fig


def gdn_prefill_per_token(data: Data, ceilings) -> Optional[plt.Figure]:
    rows = select(data.get("gdn_cold", []), kind="prefill", batch=1)
    if not rows:
        return None
    fig, ax = new_figure("GDN cold prefill: kernel time per token, batch 1", "tokens", "us per token per layer")
    for backend in sorted({r["backend"] for r in rows}):
        points = sorted((r["tokens"], r["kernel_us"] / r["tokens"]) for r in select(rows, backend=backend))
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=backend)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend()
    return fig


def ragged_efficiencies(rows: list[dict], total_tokens: int, sequences: int) -> dict[str, dict[str, float]]:
    """Achieved FLOP rate of each ragged batch relative to the uniform batch with the same total tokens.

    Attention work grows with the square of a sequence's length, so a batch with one long
    sequence really has more work than a uniform one; comparing rates instead of times does
    not charge the kernel for that. For GDN the work is linear in tokens and the rate ratio
    equals the time ratio.
    """
    out: dict[str, dict[str, float]] = {}
    for backend in sorted({r["backend"] for r in rows}):
        group = select(rows, kind="ragged", backend=backend, total_tokens=total_tokens, sequences=sequences)
        uniform = [r for r in group if r["distribution"] == "uniform"]
        if not uniform:
            continue
        out[backend] = {r["distribution"]: r["kernel_tflops"] / uniform[0]["kernel_tflops"] for r in group}
    return out


def ragged_efficiency(data: Data, ceilings, total_tokens: int = 16384, sequences: int = 32) -> Optional[plt.Figure]:
    groups = {}
    for name, label in (("attention_ragged", "attention"), ("gdn_ragged", "GDN")):
        for backend, values in ragged_efficiencies(data.get(name, []), total_tokens, sequences).items():
            groups[f"{label} {backend}"] = values
    if not groups:
        return None
    distributions = ["uniform", "jitter", "lognormal", "bimodal", "mixed"]
    fig, ax = new_figure(f"Ragged prefill efficiency, {total_tokens} tokens over {sequences} sequences", "length distribution",
                         "achieved rate / uniform batch rate")
    width = 0.8 / len(groups)
    for i, (label, values) in enumerate(groups.items()):
        xs = [d + (i - (len(groups) - 1) / 2) * width for d in range(len(distributions))]
        ax.bar(xs, [values.get(d, 0.0) for d in distributions], width=width, label=label)
    ax.set_xticks(range(len(distributions)))
    ax.set_xticklabels(distributions)
    ax.set_ylim(0.0, 1.15)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.legend(ncol=2)
    return fig


def page_size(data: Data, ceilings) -> Optional[plt.Figure]:
    rows = data.get("attention_pages", [])
    if not rows:
        return None
    fig, ax = new_figure("Attention kernel time against KV page size, FlashAttention 3", "page size, tokens", "kernel time, us")
    warm = select(rows, kind="warm", new_tokens=512, batch=8)
    for cached in sorted({r["cached"] for r in warm}):
        xs, ys = series(warm, "page_size", "kernel_us", cached=cached)
        ax.plot(xs, ys, marker="o", label=f"512 new over {cached} cached, batch 8")
    decode = select(rows, kind="decode")
    if decode:
        xs, ys = series(decode, "page_size", "kernel_us")
        ax.plot(xs, ys, marker="s", linestyle="-.", label=f"decode, {decode[0]['kv']} KV, batch {decode[0]['batch']}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend()
    return fig


def roofline_attainment(data: Data, ceilings) -> Optional[plt.Figure]:
    rows = []
    for name, table in data.items():
        if name.startswith(("attention_", "gdn_")):
            rows += kernel_attainment(table, ceilings)
    if not rows:
        return None
    fig, ax = new_figure("Measured kernel time against its roofline", "roofline time, us", "measured time, us")
    families = sorted({r["benchmark"].split("_")[0] + " " + r["backend"] for r in rows})
    for color, family in zip(PALETTE, families):
        xs = [r["roofline_us"] for r in rows if r["benchmark"].split("_")[0] + " " + r["backend"] == family]
        ys = [r["kernel_us"] for r in rows if r["benchmark"].split("_")[0] + " " + r["backend"] == family]
        ax.scatter(xs, ys, s=14, alpha=0.75, color=color, label=family)
    lo = min(min(r["roofline_us"] for r in rows), min(r["kernel_us"] for r in rows))
    hi = max(max(r["roofline_us"] for r in rows), max(r["kernel_us"] for r in rows))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, label="measured = roofline")
    ax.plot([lo, hi], [5 * lo, 5 * hi], color="black", linewidth=0.8, linestyle=":", label="5x roofline")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7.5, ncol=2)
    return fig


def ops_gemm_throughput(data: Data, ceilings, op: str = "gate_up_proj") -> Optional[plt.Figure]:
    rows = select(data.get("ops_gemm", []), op=op)
    if not rows:
        return None
    fig, ax = new_figure(f"{op} GEMM: achieved TFLOPS against tokens in the step", "tokens in the step", "TFLOPS")
    labels = {"engine": "FP8 block scaled, the engine's op", "bf16": "BF16 torch.mm"}
    colors = {"engine": PALETTE[0], "bf16": PALETTE[1]}
    for backend in sorted({r["backend"] for r in rows}, reverse=True):
        xs, ys = series(rows, "tokens", "kernel_tflops", backend=backend)
        ax.plot(xs, ys, marker="o", color=colors.get(backend), label=labels.get(backend, backend))
    ax.axhline(ceilings.fp8_tflops, color=PALETTE[0], linestyle=":", linewidth=1, label="measured FP8 GEMM peak")
    ax.axhline(ceilings.bf16_tflops, color=PALETTE[1], linestyle=":", linewidth=1, label="measured BF16 GEMM peak")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend()
    return fig


FIGURES: list[tuple[str, Callable]] = [
    ("attention_decode_bandwidth", attention_decode_bandwidth),
    ("attention_prefill_tflops", attention_prefill_tflops),
    ("crossover", crossover),
    ("step_share", step_share),
    ("gdn_decode_bandwidth", gdn_decode_bandwidth),
    ("gdn_prefill_per_token", gdn_prefill_per_token),
    ("ragged_efficiency", ragged_efficiency),
    ("page_size", page_size),
    ("roofline_attainment", roofline_attainment),
    ("ops_gemm_throughput", ops_gemm_throughput),
]


def render(data: Data, ceilings, out: Path, log=print) -> list[str]:
    style()
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for name, make in FIGURES:
        fig = make(data, ceilings)
        if fig is None:
            log(f"skipped {name}: series missing")
            continue
        fig.tight_layout()
        fig.savefig(out / f"{name}.png")
        fig.savefig(out / f"{name}.svg")
        plt.close(fig)
        made.append(name)
    return made


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render the report figures from the processed CSVs")
    parser.add_argument("--processed", default="results/processed")
    parser.add_argument("--ceilings", default="results/env/ceilings.json")
    parser.add_argument("--out", default="report/figures")
    args = parser.parse_args(argv)
    made = render(load(Path(args.processed)), ceilings_from_file(Path(args.ceilings)), Path(args.out))
    print(f"{len(made)} figures in {args.out}: {', '.join(made)}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
