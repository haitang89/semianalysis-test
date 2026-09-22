"""What an engine step really looks like under load: tokens and sequences per step.

    python3 -m bench.census.step_distribution --trace-dir /traces --caching on --concurrency 1 --concurrency 8 ...

For every concurrency level a closed loop of that many clients sends completions with
prompt lengths drawn from a lognormal distribution and a fixed number of output tokens.
Once the loop is in steady state the torch profiler is switched on for a few seconds.
vLLM annotates every engine step with the prefill and decode work it scheduled, so the
step composition is read straight from the step annotations of the trace; no kernel
data is kept. Prompts are random token ids, and with --shared-prefix a share of them
start with the same block aligned prefix so that prefix caching gets hits.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from .profile_driver import TOKEN_HIGH, TOKEN_LOW, get, new_traces, post, send_batch
from .trace_parser import Step, load_events, step_windows

COLUMNS = ("mode", "caching", "concurrency", "step", "prefill_sequences", "prefill_tokens", "decode_sequences",
           "decode_tokens", "tokens", "duration_us")
MANAGER_BLOCK = 784


def prompt_lengths(count: int, mean: int, sigma: float, low: int, high: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    mu = math.log(mean) - sigma * sigma / 2
    return [min(high, max(low, int(rng.lognormvariate(mu, sigma)))) for _ in range(count)]


def make_prompt(length: int, rng: random.Random, prefix: Optional[list[int]]) -> list[int]:
    body = [rng.randint(TOKEN_LOW, TOKEN_HIGH) for _ in range(length)]
    if prefix and length > len(prefix):
        return prefix + body[len(prefix):]
    return body


class ClosedLoop:
    def __init__(self, server: str, model: str, concurrency: int, lengths: list[int], max_tokens: int,
                 prefix: Optional[list[int]], shared_share: float, seed: int):
        self.server, self.model, self.concurrency = server, model, concurrency
        self.lengths, self.max_tokens, self.prefix, self.shared_share = lengths, max_tokens, prefix, shared_share
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.sent = 0
        self.failed = 0

    def next_prompt(self) -> list[int]:
        with self.lock:
            length = self.lengths[self.sent % len(self.lengths)]
            shared = self.rng.random() < self.shared_share
            self.sent += 1
            return make_prompt(length, self.rng, self.prefix if shared else None)

    def worker(self) -> None:
        while not self.stop.is_set():
            result = send_batch(self.server, self.model, [self.next_prompt()], self.max_tokens)
            if result and not result[0]["ok"]:
                with self.lock:
                    self.failed += 1

    def start(self) -> list[threading.Thread]:
        threads = [threading.Thread(target=self.worker, daemon=True) for _ in range(self.concurrency)]
        for thread in threads:
            thread.start()
        return threads


def rows_from_steps(steps: list[Step], mode: str, caching: str, concurrency: int) -> list[dict]:
    return [{"mode": mode, "caching": caching, "concurrency": concurrency, "step": i, "prefill_sequences": s.prefill_sequences,
             "prefill_tokens": s.prefill_tokens, "decode_sequences": s.decode_sequences, "decode_tokens": s.decode_tokens,
             "tokens": s.tokens, "duration_us": round(s.end - s.start, 1)} for i, s in enumerate(steps)]


def percentile(values: list, q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return float(ordered[index])


def summary(rows: list[dict]) -> str:
    lines = ["| mode | caching | concurrency | steps | pure decode | mixed | pure prefill | tokens per step p50 / p90 / p99 | "
             "prefill tokens per step p50 / p90 / max | decode sequences p50 / max | prefill chunks at a 784 multiple | step us p50 |",
             "|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|"]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["mode"], row["caching"], int(row["concurrency"])), []).append(row)
    for (mode, caching, concurrency), group in sorted(groups.items()):
        kinds = Counter("pure decode" if int(r["prefill_sequences"]) == 0 else "pure prefill" if int(r["decode_sequences"]) == 0 else "mixed"
                        for r in group)
        tokens = [int(r["tokens"]) for r in group]
        prefill = [int(r["prefill_tokens"]) for r in group if int(r["prefill_tokens"]) > 0]
        decode = [int(r["decode_sequences"]) for r in group]
        aligned = sum(1 for r in group if int(r["prefill_tokens"]) > 0 and int(r["prefill_tokens"]) % MANAGER_BLOCK == 0)
        lines.append(f"| {mode} | {caching} | {concurrency} | {len(group)} | {kinds['pure decode']} | {kinds['mixed']} | {kinds['pure prefill']} | "
                     f"{percentile(tokens, 0.5):.0f} / {percentile(tokens, 0.9):.0f} / {percentile(tokens, 0.99):.0f} | "
                     f"{percentile(prefill, 0.5):.0f} / {percentile(prefill, 0.9):.0f} / {max(prefill) if prefill else 0} | "
                     f"{percentile(decode, 0.5):.0f} / {max(decode) if decode else 0} | {aligned} of {len(prefill)} | "
                     f"{statistics.median(float(r['duration_us']) for r in group):.0f} |")
    return "\n".join(lines)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def measure(server: str, model: str, mode: str, caching: str, concurrency: int, trace_dir: Path, args, log=print) -> list[dict]:
    lengths = prompt_lengths(4 * concurrency + 8, args.prompt_mean, args.prompt_sigma, args.prompt_min, args.prompt_max, args.seed + concurrency)
    prefix_rng = random.Random(args.seed)
    prefix = [prefix_rng.randint(TOKEN_LOW, TOKEN_HIGH) for _ in range(args.shared_prefix)] if args.shared_prefix else None
    loop = ClosedLoop(server, model, concurrency, lengths, args.output_tokens, prefix, args.shared_share, args.seed + concurrency)
    before = set(trace_dir.glob("rank*.pt.trace.json.gz"))
    threads = loop.start()
    time.sleep(args.warmup_seconds)
    post(server, "/start_profile")
    time.sleep(args.profile_seconds)
    post(server, "/stop_profile")
    loop.stop.set()
    for thread in threads:
        thread.join(timeout=900)
    traces = new_traces(trace_dir, before, wait_s=args.trace_wait)
    if not traces:
        log(f"[warn] concurrency {concurrency}: no trace appeared")
        return []
    steps = step_windows(load_events(traces[-1]))
    rows = rows_from_steps(steps, mode, caching, concurrency)
    log(f"[ok] concurrency {concurrency}: {loop.sent} requests sent ({loop.failed} failed), {len(rows)} steps in the trace")
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tokens and sequences per engine step under a closed loop load")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--mode", default="compiled", choices=["eager", "compiled"])
    parser.add_argument("--caching", required=True, choices=["on", "off"], help="how the server was started; recorded, not set")
    parser.add_argument("--concurrency", type=int, action="append", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--out", default="results/census/step_distribution.csv")
    parser.add_argument("--summary", default="results/census/step_distribution.md")
    parser.add_argument("--prompt-mean", type=int, default=1024)
    parser.add_argument("--prompt-sigma", type=float, default=0.8)
    parser.add_argument("--prompt-min", type=int, default=64)
    parser.add_argument("--prompt-max", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--shared-prefix", type=int, default=0, help="length of a block aligned prefix some prompts share")
    parser.add_argument("--shared-share", type=float, default=0.0, help="share of prompts that start with the shared prefix")
    parser.add_argument("--warmup-seconds", type=float, default=20.0)
    parser.add_argument("--profile-seconds", type=float, default=10.0)
    parser.add_argument("--trace-wait", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args(argv)

    model = get(args.server, "/v1/models")["data"][0]["id"]
    trace_dir = Path(args.trace_dir)
    out = Path(args.out)
    rows = read_rows(out)
    for concurrency in args.concurrency:
        rows = [r for r in rows if not (r["mode"] == args.mode and r["caching"] == args.caching and int(r["concurrency"]) == concurrency)]
        rows += measure(args.server, model, args.mode, args.caching, concurrency, trace_dir, args)
        write_rows(rows, out)
    Path(args.summary).write_text(summary(rows) + "\n", encoding="utf-8", newline="\n")
    print(summary(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
