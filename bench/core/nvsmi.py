"""Queries to nvidia-smi, so no other module parses its output."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

Runner = Callable[[list[str]], str]

CLOCK_FIELDS = ("clocks.sm", "clocks.max.sm", "power.draw", "power.limit", "temperature.gpu")
SLOWDOWN_FIELDS = (
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
)
POWER_CAP_FIELD = "clocks_event_reasons.sw_power_cap"


def run(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, timeout=30, check=True).stdout


def query_gpu(fields: Sequence[str], runner: Runner = run, index: int = 0) -> list[str]:
    command = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits", "-i", str(index)]
    line = runner(command).strip().splitlines()[0]
    values = [value.strip() for value in line.split(",")]
    if len(values) != len(fields):
        raise ValueError(f"expected {len(fields)} values from nvidia-smi, got {len(values)}")
    return values


def number(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class ClockSample:
    sm_clock_mhz: float
    sm_clock_max_mhz: float
    power_w: float
    power_limit_w: float
    temp_c: float
    slowdown: bool
    power_capped: bool


def clock_sample(runner: Runner = run) -> Optional[ClockSample]:
    fields = CLOCK_FIELDS + SLOWDOWN_FIELDS + (POWER_CAP_FIELD,)
    try:
        values = query_gpu(fields, runner)
        readings = [float(value) for value in values[: len(CLOCK_FIELDS)]]
    except Exception:
        return None
    flags = [value == "Active" for value in values[len(CLOCK_FIELDS):]]
    return ClockSample(*readings, slowdown=any(flags[:-1]), power_capped=flags[-1])
