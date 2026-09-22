"""Result rows: one JSON object per measured point, with everything needed to trace it back."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from .timer import Timing

REQUIRED = ("point_id", "benchmark", "kernel", "backend", "regime", "status")


@dataclass
class Provenance:
    environment_hash: Optional[str] = None
    git_sha: Optional[str] = None
    git_dirty: Optional[bool] = None
    image_digest: Optional[str] = None
    config_hash: Optional[str] = None


@dataclass
class Row:
    point_id: str
    benchmark: str
    kernel: str
    backend: str
    regime: str
    status: str = "ok"
    error: Optional[str] = None
    params: dict = field(default_factory=dict)
    timing: Optional[dict] = None
    metrics: dict = field(default_factory=dict)
    numerics_ok: Optional[bool] = None
    max_abs_err: Optional[float] = None
    provenance: dict = field(default_factory=dict)
    ts_utc: str = ""

    @classmethod
    def ok(cls, point_id: str, benchmark: str, kernel: str, backend: str, regime: str, params: dict,
           timing: Timing, metrics: dict, provenance: Provenance) -> "Row":
        return cls(point_id, benchmark, kernel, backend, regime, params=params, timing=timing.to_row(),
                   metrics=metrics, provenance=asdict(provenance), ts_utc=_now())

    @classmethod
    def failed(cls, point_id: str, benchmark: str, kernel: str, backend: str, regime: str, params: dict,
               error: str, provenance: Provenance, status: str = "failed") -> "Row":
        return cls(point_id, benchmark, kernel, backend, regime, status=status, error=error, params=params,
                   provenance=asdict(provenance), ts_utc=_now())

    def to_dict(self) -> dict:
        return asdict(self)

    def flat(self) -> dict:
        record = {key: value for key, value in self.to_dict().items() if key not in ("params", "timing", "metrics", "provenance")}
        for group in ("params", "timing", "metrics", "provenance"):
            for key, value in (getattr(self, group) or {}).items():
                record[key] = value
        return record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def point_id(params: dict) -> str:
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def validate(record: dict) -> None:
    missing = [key for key in REQUIRED if not record.get(key)]
    if missing:
        raise ValueError(f"result row is missing {missing}")


class Writer:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Row) -> None:
        record = row.to_dict()
        validate(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


def read_rows(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def completed_points(path: Path) -> set[str]:
    return {record["point_id"] for record in read_rows(path) if record.get("status") == "ok"}


def flatten(record: dict) -> dict:
    flat = {key: value for key, value in record.items() if key not in ("params", "timing", "metrics", "provenance")}
    for group in ("params", "timing", "metrics", "provenance"):
        for key, value in (record.get(group) or {}).items():
            flat[key] = value
    return flat


def write_csv(records: Iterable[dict], path: Path) -> int:
    rows = [flatten(record) for record in records]
    if not rows:
        return 0
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in columns})
    return len(rows)


def _cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return "" if value is None else value
