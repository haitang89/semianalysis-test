"""Drive the live server through a capture grid with the torch profiler on.

    python3 -m bench.census.profile_driver --mode eager --trace-dir /traces [--server http://localhost:8000]

The server must have been started with a torch profiler config. For every grid point
the driver starts a profile, sends the requests of that point at once, waits for all
of them, stops the profile and records which trace file appeared. Prompts are lists of
random token ids, so their length is exact without a tokenizer and no two prompts share
a prefix, which keeps prefix caching out of the picture. Decode points ask for a fixed
number of output tokens with ignore_eos so every request decodes the same steps.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import yaml

TOKEN_LOW, TOKEN_HIGH = 1000, 60000


def post(server: str, path: str, payload: Optional[dict] = None, timeout: float = 3600.0) -> dict:
    data = json.dumps(payload or {}).encode()
    request = urllib.request.Request(f"{server}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else {}


def get(server: str, path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(f"{server}{path}", timeout=timeout) as response:
        return json.loads(response.read())


def prompts_for(point: dict, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    length = point["tokens"] if point["kind"] == "prefill" else point["kv"] - point["max_tokens"]
    if length < 1:
        raise ValueError(f"point {point['label']} leaves no prompt tokens")
    return [[rng.randint(TOKEN_LOW, TOKEN_HIGH) for _ in range(length)] for _ in range(point["batch"])]


def send_batch(server: str, model: str, prompts: list[list[int]], max_tokens: int) -> list[dict]:
    results: list[Optional[dict]] = [None] * len(prompts)

    def one(index: int) -> None:
        payload = {"model": model, "prompt": prompts[index], "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True}
        started = time.perf_counter()
        try:
            body = post(server, "/v1/completions", payload)
            results[index] = {"ok": True, "seconds": time.perf_counter() - started, "usage": body.get("usage")}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            results[index] = {"ok": False, "seconds": time.perf_counter() - started, "error": str(exc)}

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [r for r in results if r is not None]


def new_traces(trace_dir: Path, before: set, wait_s: float = 120.0) -> list[Path]:
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        ranks = [p for p in trace_dir.glob("rank*.pt.trace.json.gz") if p not in before]
        if ranks and all(time.time() - p.stat().st_mtime > 2.0 for p in ranks):
            return sorted(ranks)
        time.sleep(1.0)
    return []


def run_point(server: str, model: str, point: dict, trace_dir: Path, seed: int, log=print) -> dict:
    prompts = prompts_for(point, seed)
    before = set(trace_dir.glob("rank*.pt.trace.json.gz"))
    post(server, "/start_profile")
    started = time.perf_counter()
    results = send_batch(server, model, prompts, point["max_tokens"])
    wall = time.perf_counter() - started
    post(server, "/stop_profile")
    traces = new_traces(trace_dir, before)
    record = {
        "label": point["label"], "params": point, "requests": len(results), "failed": sum(not r["ok"] for r in results),
        "wall_s": round(wall, 3), "traces": [str(p) for p in traces],
    }
    log(f"[{'ok' if traces and not record['failed'] else 'warn'}] {point['label']}: {len(results)} requests in {wall:.1f} s, "
        f"{len(traces)} trace file(s)")
    return record


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Profile the live server over a capture grid")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--mode", required=True, choices=["eager", "compiled"])
    parser.add_argument("--grid", default="configs/profile_grid.yaml")
    parser.add_argument("--trace-dir", required=True, help="directory the server writes traces to, as seen from here")
    parser.add_argument("--manifest", default=None, help="default results/census/profiles_<mode>.json")
    parser.add_argument("--only", default=None, help="comma separated labels")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    grid = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))["points"]
    if args.only:
        wanted = set(args.only.split(","))
        grid = [p for p in grid if p["label"] in wanted]
    model = get(args.server, "/v1/models")["data"][0]["id"]
    trace_dir = Path(args.trace_dir)
    manifest_path = Path(args.manifest or f"results/census/profiles_{args.mode}.json")
    records = []
    if manifest_path.exists():
        records = [r for r in json.loads(manifest_path.read_text(encoding="utf-8"))["points"] if r["label"] not in {p["label"] for p in grid}]
    for point in grid:
        records.append(run_point(args.server, model, point, trace_dir, args.seed))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"mode": args.mode, "server": args.server, "model": model, "points": records}, indent=2) + "\n",
                                 encoding="utf-8")
    missing = [r["label"] for r in records if not r["traces"] or r["failed"]]
    print(f"{len(records)} points, {len(missing)} without a clean trace: {missing or 'none'}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
