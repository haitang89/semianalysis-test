"""Raw result rows to tidy CSVs, one per benchmark plus one combined table.

    python3 -m analysis.process [--raw results/raw] [--out results/processed]

Failed or unsupported points are listed on stderr and left out of the tables, and mock
sweep outputs (files ending in _mock.jsonl) are skipped.
Kernel time is the graph replay median when the point has one, else the eager median.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from bench.core.results import flatten, read_rows, write_csv


def kernel_time_us(row: dict) -> Optional[float]:
    timing = row.get("timing") or {}
    if timing.get("graph_median_us") is not None:
        return timing["graph_median_us"]
    return timing.get("median_us")


def tidy(row: dict) -> dict:
    record = flatten(row)
    record["kernel_us"] = kernel_time_us(row)
    record["kernel_time_source"] = "graph" if (row.get("timing") or {}).get("graph_median_us") is not None else "eager"
    metrics = row.get("metrics") or {}
    if record["kernel_us"] and metrics.get("bytes_moved"):
        record["kernel_gbps"] = metrics["bytes_moved"] / (record["kernel_us"] * 1e-6) / 1e9
    if record["kernel_us"] and metrics.get("flops"):
        record["kernel_tflops"] = metrics["flops"] / (record["kernel_us"] * 1e-6) / 1e12
    return record


def process(raw_dir: Path, out_dir: Path, err=None) -> dict:
    err = err or sys.stderr
    by_benchmark: dict[str, list[dict]] = defaultdict(list)
    excluded = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        if path.stem.endswith("_mock"):
            continue
        for row in read_rows(path):
            if row.get("status") != "ok":
                excluded.append((row["benchmark"], row["status"], row.get("params"), (row.get("error") or "")[:80]))
                continue
            by_benchmark[row["benchmark"]].append(tidy(row))
    counts = {}
    everything: list[dict] = []
    for benchmark, records in sorted(by_benchmark.items()):
        counts[benchmark] = write_csv(records, out_dir / f"{benchmark}.csv")
        everything.extend(records)
    if everything:
        write_csv(everything, out_dir / "all_points.csv")
    for benchmark, status, params, error in excluded:
        print(f"excluded {benchmark} {status}: {params} {error}", file=err)
    return counts


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert raw result rows into tidy CSVs")
    parser.add_argument("--raw", default="results/raw")
    parser.add_argument("--out", default="results/processed")
    args = parser.parse_args(argv)
    counts = process(Path(args.raw), Path(args.out))
    for benchmark, count in counts.items():
        print(f"{benchmark}: {count} rows")
    return 0 if counts else 1


if __name__ == "__main__":
    sys.exit(main())
