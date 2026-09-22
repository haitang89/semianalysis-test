"""Observed operator census from a torch profiler trace of the running engine.

    python3 -m bench.census.trace_parser --trace rank0.pt.trace.json.gz --mode eager

Every GPU kernel in the trace is joined to the operator that launched it through the
profiler's External id, which carries the operator's input shapes and dtypes when the
server was profiled with record_shapes. The Python frames open at launch time give the
model block the kernel belongs to. vLLM annotates every engine step with the tokens it
computes, so each kernel is also assigned to a step composition. Kernels replayed from
a CUDA graph have no launching operator; they are kept with their graph id so the
compiled mode census still lists the production kernel set.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Optional

STEP_NAME = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
BLOCK_PATTERNS = (
    ("gdn", re.compile(r"mamba/gdn/|fla/|mamba_mixer|causal_conv1d")),
    ("attention", re.compile(r"attention/backends/|layers/attention/|flash_attn")),
    ("mlp", re.compile(r"qwen2_moe\.py|layers/activation\.py|/mlp\.py")),
    ("norm", re.compile(r"layers/layernorm\.py")),
    ("embedding", re.compile(r"vocab_parallel_embedding\.py")),
    ("logits", re.compile(r"logits_processor\.py")),
    ("sampler", re.compile(r"/sample/|sampler\.py")),
    ("rotary", re.compile(r"rotary_embedding")),
    ("kv_cache", re.compile(r"kv_cache|block_table|slot_mapping")),
)
KERNEL_PATTERNS = (
    ("attention", re.compile(r"flash|fa3|attn|rope|qk_rmsnorm", re.IGNORECASE)),
    ("gdn", re.compile(r"gdn|delta_rule|causal_conv1d|mamba|post_conv|recurrent", re.IGNORECASE)),
    ("sampler", re.compile(r"sampl|topk|top_p|softmax|multinomial|gumbel", re.IGNORECASE)),
    ("kv_cache", re.compile(r"kv_blocks|block_table|slot_mapping|apply_write", re.IGNORECASE)),
)
DECODER_LAYER = re.compile(r"models/qwen3_next\.py\(\d+\): forward")
VLLM_FRAME = re.compile(r"vllm/(.+?\.py)\((\d+)\): (\w+)")


@dataclass(frozen=True)
class Step:
    name: str
    start: float
    end: float
    prefill_sequences: int
    prefill_tokens: int
    decode_sequences: int
    decode_tokens: int

    @property
    def tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens


@dataclass(frozen=True)
class KernelRecord:
    mode: str
    trace: str
    step: str
    step_instances: int
    step_tokens: int
    prefill_sequences: int
    decode_sequences: int
    block: str
    frame: str
    op: str
    input_dims: str
    input_types: str
    kernel: str
    grid: str
    block_dims: str
    graph: bool
    dur_us: float


def load_events(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)["traceEvents"]


def parse_step(name: str, start: float, duration: float) -> Optional[Step]:
    match = STEP_NAME.search(name)
    if not match:
        return None
    ps, pt, ds, dt = (int(match.group(i)) for i in (1, 2, 3, 4))
    return Step(name, start, start + duration, ps, pt, ds, dt)


def step_windows(events: Iterable[dict]) -> list[Step]:
    steps = []
    for event in events:
        if event.get("cat") == "gpu_user_annotation":
            step = parse_step(event["name"], event["ts"], event.get("dur", 0.0))
            if step:
                steps.append(step)
    return sorted(steps, key=lambda s: s.start)


def step_for(steps: list[Step], ts: float) -> Optional[Step]:
    for step in steps:
        if step.start <= ts <= step.end:
            return step
    return None


def label_frames(frames: list[str], kernel: str = "") -> tuple[str, str]:
    """Block from the outermost matching frame, else from the kernel name; innermost vLLM frame as the location."""
    block = "other"
    for frame in frames:
        for name, pattern in BLOCK_PATTERNS:
            if pattern.search(frame):
                block = name
                break
        if block != "other":
            break
    if block == "other" and any(DECODER_LAYER.search(frame) for frame in frames):
        block = "attention"
    if block == "other":
        for name, pattern in KERNEL_PATTERNS:
            if pattern.search(kernel):
                block = name
                break
    location = ""
    for frame in reversed(frames):
        match = VLLM_FRAME.search(frame)
        if match:
            location = f"{match.group(1)}:{match.group(2)} {match.group(3)}"
            break
    return block, location


def op_frames(events: Iterable[dict]) -> dict[int, list[str]]:
    """External id of every cpu op to the Python frames open on its thread when it ran, outermost first."""
    timeline = []
    for event in events:
        cat = event.get("cat")
        if cat in ("python_function", "cpu_op") and event.get("ph") == "X":
            timeline.append((event["ts"], -event.get("dur", 0.0), 0 if cat == "python_function" else 1, event))
    timeline.sort(key=lambda item: item[:3])
    frames: dict[int, list[str]] = {}
    stacks: dict[tuple, list[tuple[float, str]]] = defaultdict(list)
    for ts, neg_dur, _, event in timeline:
        stack = stacks[(event.get("pid"), event.get("tid"))]
        while stack and stack[-1][0] < ts:
            stack.pop()
        if event.get("cat") == "python_function":
            stack.append((ts - neg_dur, event["name"]))
        else:
            ext = (event.get("args") or {}).get("External id")
            if ext is not None and ext not in frames:
                frames[ext] = [name for _, name in stack]
    return frames


def kernel_records(events: list[dict], mode: str, trace: str = "") -> list[KernelRecord]:
    ops = {(e.get("args") or {}).get("External id"): e for e in events if e.get("cat") == "cpu_op"}
    frames = op_frames(events)
    steps = step_windows(events)
    instances = defaultdict(int)
    for step in steps:
        instances[step.name] += 1
    records = []
    for event in events:
        if event.get("cat") != "kernel":
            continue
        args = event.get("args") or {}
        op = ops.get(args.get("External id"))
        op_args = (op or {}).get("args") or {}
        step = step_for(steps, event["ts"])
        block, location = label_frames(frames.get(args.get("External id"), []), event["name"])
        records.append(KernelRecord(
            mode=mode,
            trace=trace,
            step=step.name if step else "outside_step",
            step_instances=instances[step.name] if step else 1,
            step_tokens=step.tokens if step else 0,
            prefill_sequences=step.prefill_sequences if step else 0,
            decode_sequences=step.decode_sequences if step else 0,
            block=block,
            frame=location,
            op=op["name"] if op else "",
            input_dims=json.dumps(op_args.get("Input Dims", []), separators=(",", ":")) if op else "",
            input_types=json.dumps(op_args.get("Input type", []), separators=(",", ":")) if op else "",
            kernel=event["name"],
            grid="x".join(str(v) for v in args.get("grid", [])),
            block_dims="x".join(str(v) for v in args.get("block", [])),
            graph=bool(args.get("graph id")),
            dur_us=float(event.get("dur", 0.0)),
        ))
    return records


GROUP_KEYS = ("mode", "trace", "step", "step_instances", "step_tokens", "prefill_sequences", "decode_sequences", "block", "frame", "op",
              "input_dims", "input_types", "kernel", "grid", "block_dims", "graph")


def aggregate(records: Iterable[KernelRecord]) -> list[dict]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for record in records:
        groups[tuple(getattr(record, key) for key in GROUP_KEYS)].append(record.dur_us)
    rows = []
    for key, durations in groups.items():
        row = dict(zip(GROUP_KEYS, key))
        row.update(count=len(durations), total_us=sum(durations), mean_us=sum(durations) / len(durations),
                   min_us=min(durations), max_us=max(durations))
        rows.append(row)
    rows.sort(key=lambda r: (r["mode"], r["trace"], r["step"], -r["total_us"]))
    return rows


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("step_instances", "step_tokens", "prefill_sequences", "decode_sequences", "count"):
            row[key] = int(row[key])
        for key in ("total_us", "mean_us", "min_us", "max_us"):
            row[key] = float(row[key])
        row["graph"] = row["graph"] == "True"
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(GROUP_KEYS) + ["count", "total_us", "mean_us", "min_us", "max_us"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def block_summary(rows: list[dict]) -> str:
    """GPU kernel time per step, averaged over the instances of each step composition."""
    per_step: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    kernels: dict[tuple, set] = defaultdict(set)
    launches: dict[tuple, float] = defaultdict(float)
    for row in rows:
        key = (row["mode"], row["trace"], row["step"], int(row["step_instances"]))
        per_step[key][row["block"]] += row["total_us"] / key[3]
        launches[key] += row["count"] / key[3]
        kernels[key].add(row["kernel"])
    blocks = [name for name, _ in BLOCK_PATTERNS] + ["other"]
    lines = ["| mode | trace | step | instances | kernel launches | distinct kernels | " + " | ".join(f"{b} us" for b in blocks) + " | total us |",
             "|---|---|---|---:|---:|---:|" + "---:|" * (len(blocks) + 1)]
    for key in sorted(per_step):
        total = sum(per_step[key].values())
        cells = " | ".join(f"{per_step[key].get(b, 0.0):.0f}" for b in blocks)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {launches[key]:.0f} | {len(kernels[key])} | {cells} | {total:.0f} |")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Observed kernels with shapes from a torch profiler trace")
    parser.add_argument("--trace", required=True, action="append", help="profiler trace file; repeatable")
    parser.add_argument("--label", action="append", default=[], help="label per --trace, default the file stem")
    parser.add_argument("--mode", required=True, choices=["eager", "compiled"])
    parser.add_argument("--out", default="results/census/observed_ops.csv")
    parser.add_argument("--summary", default="results/census/observed_summary.md")
    parser.add_argument("--append", action="store_true", help="keep rows already in --out from other modes and traces")
    args = parser.parse_args(argv)

    labels = args.label + [Path(t).name.split(".")[0] for t in args.trace[len(args.label):]]
    rows = []
    for trace, label in zip(args.trace, labels):
        rows += aggregate(kernel_records(load_events(Path(trace)), args.mode, label))
    out = Path(args.out)
    if args.append and out.exists():
        written = {(r["mode"], r["trace"]) for r in rows}
        rows = [r for r in read_rows(out) if (r["mode"], r["trace"]) not in written] + rows
    write_rows(rows, out)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(block_summary(rows) + "\n", encoding="utf-8", newline="\n")
    steps = {r["step"] for r in rows if r["mode"] == args.mode}
    print(f"{args.mode}: {sum(r['count'] for r in rows if r['mode'] == args.mode)} kernels in {len(steps)} steps, "
          f"{len({r['kernel'] for r in rows if r['mode'] == args.mode})} distinct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
