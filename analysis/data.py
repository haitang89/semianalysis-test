"""Reading the tidy CSVs back with numbers as numbers."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional, Union

Value = Union[None, bool, int, float, str]


def parse(cell: str) -> Value:
    if cell == "":
        return None
    if cell in ("True", "False"):
        return cell == "True"
    try:
        number = float(cell)
    except ValueError:
        return cell
    return int(number) if number.is_integer() and "." not in cell and "e" not in cell.lower() else number


def load_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [{key: parse(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def select(rows: Iterable[dict], **conditions: Value) -> list[dict]:
    return [row for row in rows if all(row.get(key) == value for key, value in conditions.items())]


def first(rows: Iterable[dict], **conditions: Value) -> Optional[dict]:
    matches = select(rows, **conditions)
    return matches[0] if matches else None
