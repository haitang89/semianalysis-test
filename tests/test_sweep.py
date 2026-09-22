import json
from pathlib import Path

import pytest
import yaml

from bench.core.mock import ModelClock
from bench.core.results import Provenance, read_rows, write_csv
from bench.core.sweep import PointTimeout, expand_grid, load_sweep, plan_points, run_sweep
from bench.core.timer import TimingConfig, time_kernel

SWEEP = {
    "benchmark": "demo",
    "backends": ["fast", "slow"],
    "geometry": {"heads": 24},
    "timing": {"warmup": 2, "repeats": 5, "point_timeout_s": 5},
    "grid": {"context": [1024, 4096], "batch": [1, 8]},
    "points": [{"context": 99, "batch": 1}],
}


def make_driver(fail_on=None, slow_on=None):
    calls = []

    def driver(params):
        calls.append(params)
        if fail_on and params["context"] == fail_on:
            raise RuntimeError("kernel rejected the shape")
        if slow_on and params["context"] == slow_on:
            raise PointTimeout("point exceeded 5 s")
        latency = params["context"] * 0.01 * (2.0 if params["backend"] == "slow" else 1.0)
        timing = time_kernel(lambda: None, TimingConfig(warmup=2, repeats=5), ModelClock(latency_us=latency))
        return timing, {"gbps": 1000.0 / latency}

    driver.calls = calls
    return driver


@pytest.fixture
def sweep_file(tmp_path):
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump(SWEEP), encoding="utf-8")
    return path


def test_grid_expansion_and_point_count(sweep_file):
    assert expand_grid({"a": [1, 2], "b": 3}) == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]
    config = load_sweep(sweep_file)
    assert len(config.points) == 5
    assert len(plan_points(config)) == 10
    assert config.point_timeout_s == 5


def test_dry_run_writes_nothing(sweep_file, tmp_path):
    out = tmp_path / "demo.jsonl"
    outcome = run_sweep(load_sweep(sweep_file), make_driver(), out, Provenance(), "k", "cold", dry_run=True, log=lambda m: None)
    assert outcome.ok == 0 and not out.exists()


def test_resume_skips_finished_points_and_force_reruns(sweep_file, tmp_path):
    out = tmp_path / "demo.jsonl"
    config = load_sweep(sweep_file)
    first = run_sweep(config, make_driver(), out, Provenance(), "k", "cold", log=lambda m: None)
    assert first.ok == 10 and first.exit_code == 0
    again = run_sweep(config, make_driver(), out, Provenance(), "k", "cold", log=lambda m: None)
    assert again.skipped == 10 and again.ok == 0
    forced = run_sweep(config, make_driver(), out, Provenance(), "k", "cold", force=True, log=lambda m: None)
    assert forced.ok == 10
    assert sum(1 for _ in read_rows(out)) == 20


def test_failed_point_is_recorded_and_the_sweep_continues(sweep_file, tmp_path):
    out = tmp_path / "demo.jsonl"
    outcome = run_sweep(load_sweep(sweep_file), make_driver(fail_on=4096), out, Provenance(), "k", "cold", log=lambda m: None)
    assert outcome.failed == 4 and outcome.ok == 6 and outcome.exit_code == 1
    rows = list(read_rows(out))
    failed = [row for row in rows if row["status"] == "failed"]
    assert len(failed) == 4 and "rejected the shape" in failed[0]["error"]
    assert all(row["timing"] is None for row in failed)


def test_timeout_is_its_own_status(sweep_file, tmp_path):
    out = tmp_path / "demo.jsonl"
    outcome = run_sweep(load_sweep(sweep_file), make_driver(slow_on=99), out, Provenance(), "k", "cold", log=lambda m: None)
    assert outcome.timed_out == 2 and outcome.exit_code == 1
    assert {row["status"] for row in read_rows(out)} == {"ok", "timeout"}


def test_failed_points_are_retried_on_the_next_run(sweep_file, tmp_path):
    out = tmp_path / "demo.jsonl"
    config = load_sweep(sweep_file)
    run_sweep(config, make_driver(fail_on=4096), out, Provenance(), "k", "cold", log=lambda m: None)
    retry = run_sweep(config, make_driver(), out, Provenance(), "k", "cold", log=lambda m: None)
    assert retry.skipped == 6 and retry.ok == 4


def test_only_filter_matches_labels(sweep_file, tmp_path):
    out = tmp_path / "demo.jsonl"
    outcome = run_sweep(load_sweep(sweep_file), make_driver(), out, Provenance(), "k", "cold", only="*context=99*", log=lambda m: None)
    assert outcome.ok == 2


def test_rows_carry_provenance_and_flatten_to_csv(sweep_file, tmp_path):
    out = tmp_path / "demo.jsonl"
    provenance = Provenance(environment_hash="env1", git_sha="abc", git_dirty=False, image_digest="sha256:x")
    run_sweep(load_sweep(sweep_file), make_driver(), out, provenance, "torch.mm", "cold", only="*context=99*", log=lambda m: None)
    row = next(read_rows(out))
    assert row["provenance"]["git_sha"] == "abc" and row["provenance"]["config_hash"]
    assert row["params"]["heads"] == 24 and row["timing"]["median_us"] > 0 and row["metrics"]["gbps"] > 0
    csv_path = tmp_path / "demo.csv"
    assert write_csv(read_rows(out), csv_path) == 2
    header = csv_path.read_text().splitlines()[0].split(",")
    assert {"point_id", "backend", "context", "median_us", "gbps", "git_sha"} <= set(header)
