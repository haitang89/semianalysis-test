import csv
import json
from pathlib import Path

from analysis.process import process, tidy


def row(benchmark, status="ok", graph=50.0, eager=80.0, **params):
    return {
        "point_id": "p" + str(abs(hash(str(params))))[:8],
        "benchmark": benchmark,
        "kernel": "k",
        "backend": "b",
        "regime": "cold",
        "status": status,
        "error": None if status == "ok" else "boom",
        "params": params,
        "timing": None if status != "ok" else {"median_us": eager, "graph_median_us": graph, "launch_overhead_us": eager - (graph or eager)},
        "metrics": {} if status != "ok" else {"bytes_moved": 1e9, "flops": 2e12},
        "provenance": {"git_sha": "abc"},
        "ts_utc": "t",
    }


def test_tidy_prefers_graph_time_and_derives_rates():
    record = tidy(row("demo", batch=4))
    assert record["kernel_us"] == 50.0 and record["kernel_time_source"] == "graph"
    assert record["kernel_gbps"] == 1e9 / 50e-6 / 1e9
    assert record["kernel_tflops"] == 2e12 / 50e-6 / 1e12
    assert record["batch"] == 4 and record["git_sha"] == "abc"
    eager_only = tidy(row("demo", graph=None, batch=1))
    assert eager_only["kernel_us"] == 80.0 and eager_only["kernel_time_source"] == "eager"


def test_process_splits_by_benchmark_and_excludes_failures(tmp_path, capsys):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    (raw / "a.jsonl").write_text("\n".join(json.dumps(r) for r in [row("a", batch=1), row("a", batch=2), row("a", status="failed", batch=3)]) + "\n")
    (raw / "b.jsonl").write_text(json.dumps(row("b", batch=8)) + "\n")
    counts = process(raw, out)
    assert counts == {"a": 2, "b": 1}
    with (out / "all_points.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 and {r["benchmark"] for r in rows} == {"a", "b"}
    assert "excluded a failed" in capsys.readouterr().err


def test_mock_outputs_are_left_out_of_the_tables(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    (raw / "a.jsonl").write_text(json.dumps(row("a", batch=1)) + "\n")
    (raw / "a_mock.jsonl").write_text(json.dumps(row("a", batch=2)) + "\n")
    assert process(raw, out) == {"a": 1}


def test_process_twice_is_byte_identical(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    (raw / "a.jsonl").write_text(json.dumps(row("a", batch=1)) + "\n")
    process(raw, out)
    first = (out / "a.csv").read_bytes()
    process(raw, out)
    assert (out / "a.csv").read_bytes() == first
