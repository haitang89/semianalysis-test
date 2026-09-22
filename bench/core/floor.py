"""The fixed cost of one timed call: an empty kernel through the same clock as every sweep.

    python3 -m bench.core.floor [--out results/env/launch_floor.json]

The sweeps time one kernel per CUDA graph replay. A replay of a one node graph costs a
few microseconds before the kernel does anything, and an eager launch costs more. Both
floors are measured here with a one element add so the sum of parts can take the replay
floor off every isolated kernel time; the engine pays it once per graph, not per kernel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .timer import Clock, TimingConfig, time_kernel


def measure(kernel, clock: Clock, repeats: int = 200) -> dict:
    timing = time_kernel(kernel, TimingConfig(warmup=20, repeats=repeats, graph_mode="always"), clock)
    return {
        "eager_floor_us": timing.median_us,
        "graph_replay_floor_us": timing.graph_median_us,
        "eager_p10_us": timing.p10_us,
        "eager_p90_us": timing.p90_us,
        "repeats": repeats,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the eager launch and graph replay floors")
    parser.add_argument("--out", default="results/env/launch_floor.json")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args(argv)

    import torch

    from .timer import CudaClock

    x = torch.zeros(1, device="cuda")
    result = measure(lambda: x.add_(1), CudaClock(), args.repeats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"eager floor {result['eager_floor_us']:.1f} us, graph replay floor {result['graph_replay_floor_us']:.1f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
