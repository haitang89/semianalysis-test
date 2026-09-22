"""Config driven sweeps: expand a grid, skip finished points, bound each point, record failures."""
from __future__ import annotations

import fnmatch
import itertools
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import yaml

from .results import Provenance, Row, Writer, completed_points, config_hash, point_id

Driver = Callable[[dict], tuple[object, dict]]


@dataclass
class SweepConfig:
    benchmark: str
    backends: list
    points: list
    timing: dict = field(default_factory=dict)
    point_timeout_s: float = 120.0
    geometry: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return config_hash(self.raw)


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    values = [grid[key] if isinstance(grid[key], list) else [grid[key]] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def load_sweep(path: Path) -> SweepConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    points = list(raw.get("points", []))
    if "grid" in raw:
        points.extend(expand_grid(raw["grid"]))
    for extra in raw.get("extra_grids", []):
        points.extend(expand_grid(extra))
    if not points:
        raise ValueError(f"{path} defines no points")
    return SweepConfig(
        benchmark=raw["benchmark"],
        backends=list(raw.get("backends", ["default"])),
        points=points,
        timing=dict(raw.get("timing", {})),
        point_timeout_s=float(raw.get("timing", {}).get("point_timeout_s", 120.0)),
        geometry=dict(raw.get("geometry", {})),
        raw=raw,
    )


@dataclass
class Plan:
    entries: list

    def __len__(self) -> int:
        return len(self.entries)


def plan_points(config: SweepConfig, only: Optional[str] = None) -> Plan:
    entries = []
    for backend in config.backends:
        for params in config.points:
            full = {"backend": backend, **config.geometry, **params}
            pid = point_id({"benchmark": config.benchmark, **full})
            if only and not fnmatch.fnmatch(pid, only) and not fnmatch.fnmatch(_label(full), only):
                continue
            entries.append((pid, backend, full))
    return Plan(entries)


def _label(params: dict) -> str:
    return "_".join(f"{key}={value}" for key, value in params.items() if key != "backend")


class PointTimeout(Exception):
    pass


def _with_timeout(seconds: float, call: Callable[[], object]) -> object:
    if not hasattr(signal, "SIGALRM") or seconds <= 0:
        return call()

    def raise_timeout(signum, frame):
        raise PointTimeout(f"point exceeded {seconds:.0f} s")

    previous = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return call()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class Outcome:
    ok: int = 0
    failed: int = 0
    timed_out: int = 0
    skipped: int = 0
    failed_ids: list = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 1 if self.failed or self.timed_out else 0


def run_sweep(
    config: SweepConfig,
    driver: Driver,
    out: Path,
    provenance: Provenance,
    kernel_name: str,
    regime: str,
    only: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> Outcome:
    provenance.config_hash = config.hash
    plan = plan_points(config, only)
    done = set() if force else completed_points(out)
    writer = Writer(out)
    outcome = Outcome()
    log(f"{config.benchmark}: {len(plan)} points, {len(done & {pid for pid, _, _ in plan.entries})} already done, out={out}")
    started = time.monotonic()
    for pid, backend, params in plan.entries:
        if pid in done:
            outcome.skipped += 1
            continue
        if dry_run:
            log(f"[plan] {pid} {backend} {_label(params)}")
            continue
        try:
            timing, metrics = _with_timeout(config.point_timeout_s, lambda: driver(params))
            writer.append(Row.ok(pid, config.benchmark, kernel_name, backend, regime, params, timing, metrics, provenance))
            outcome.ok += 1
            log(f"[ ok ] {pid} {backend} {_label(params)} {timing.median_us:.1f} us")
        except PointTimeout as exc:
            writer.append(Row.failed(pid, config.benchmark, kernel_name, backend, regime, params, str(exc), provenance, "timeout"))
            outcome.timed_out += 1
            outcome.failed_ids.append(pid)
            log(f"[time] {pid} {backend} {_label(params)}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            writer.append(Row.failed(pid, config.benchmark, kernel_name, backend, regime, params, error, provenance))
            outcome.failed += 1
            outcome.failed_ids.append(pid)
            log(f"[fail] {pid} {backend} {_label(params)} {error[:120]}")
    log(f"done: ok={outcome.ok} failed={outcome.failed} timeout={outcome.timed_out} skipped={outcome.skipped} "
        f"wall={time.monotonic() - started:.0f}s")
    return outcome


def add_sweep_arguments(parser) -> None:
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--out", default=None, help="results file, default results/raw/<benchmark>.jsonl")
    parser.add_argument("--only", default=None, help="glob on point id or on key=value labels")
    parser.add_argument("--force", action="store_true", help="rerun points that already have a result")
    parser.add_argument("--dry-run", action="store_true", help="list the points without running them")
    parser.add_argument("--mock", action="store_true", help="use the analytic latency model instead of the GPU")


def output_path(config: SweepConfig, override: Optional[str], mock: bool) -> Path:
    if override:
        return Path(override)
    suffix = "_mock" if mock else ""
    return Path("results/raw") / f"{config.benchmark}{suffix}.jsonl"
