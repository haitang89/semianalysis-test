import argparse
import csv
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from bench.census import step_distribution as sd
from bench.census.trace_parser import Step


def test_prompt_lengths_are_clipped_and_reproducible():
    lengths = sd.prompt_lengths(200, 1024, 0.8, 64, 8192, seed=3)
    assert len(lengths) == 200 and min(lengths) >= 64 and max(lengths) <= 8192
    assert 400 < sorted(lengths)[100] < 2000
    assert lengths == sd.prompt_lengths(200, 1024, 0.8, 64, 8192, seed=3)


def test_shared_prefix_replaces_the_start_of_long_prompts_only():
    import random

    prefix = [7] * 10
    long = sd.make_prompt(20, random.Random(1), prefix)
    assert long[:10] == prefix and len(long) == 20
    short = sd.make_prompt(5, random.Random(1), prefix)
    assert len(short) == 5 and short[:5] != prefix[:5]


def test_summary_classifies_steps_and_counts_aligned_chunks():
    steps = [Step("a", 0, 900, 1, 784, 3, 3), Step("b", 900, 1400, 0, 0, 4, 4), Step("c", 1400, 3000, 2, 1001, 0, 0)]
    rows = sd.rows_from_steps(steps, "compiled", "on", 4)
    assert rows[2]["tokens"] == 1001 and rows[2]["duration_us"] == 1600
    text = sd.summary(rows)
    assert "| compiled | on | 4 | 3 | 1 | 1 | 1 |" in text
    assert "1 of 2" in text


def test_percentile_handles_empty_and_edges():
    assert sd.percentile([], 0.5) == 0.0
    assert sd.percentile([5], 0.99) == 5.0
    assert sd.percentile([1, 2, 3, 4, 5], 0.0) == 1.0 and sd.percentile([1, 2, 3, 4, 5], 1.0) == 5.0


class FakeServer(BaseHTTPRequestHandler):
    trace_dir: Path = Path(".")
    completions = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        self._reply({"data": [{"id": "fake"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path == "/v1/completions":
            FakeServer.completions += 1
        if self.path == "/stop_profile":
            events = [{"ph": "X", "cat": "gpu_user_annotation", "name": f"execute_context_1(784)_generation_{n}({n})",
                       "pid": 0, "tid": 7, "ts": 1000.0 * n, "dur": 500.0} for n in range(1, 6)]
            with gzip.open(FakeServer.trace_dir / f"rank0.{FakeServer.completions}.pt.trace.json.gz", "wt") as handle:
                json.dump({"traceEvents": events}, handle)
        self._reply({})

    def _reply(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server(tmp_path):
    FakeServer.trace_dir = tmp_path
    FakeServer.completions = 0
    httpd = HTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_main_drives_the_loop_and_writes_rows_and_summary(server, tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "new_traces", lambda trace_dir, before, wait_s=1.0: sorted(p for p in trace_dir.glob("rank*.pt.trace.json.gz") if p not in before))
    out, summary = tmp_path / "steps.csv", tmp_path / "steps.md"
    argv = ["--server", server, "--caching", "on", "--concurrency", "2", "--trace-dir", str(tmp_path), "--out", str(out),
            "--summary", str(summary), "--warmup-seconds", "0.2", "--profile-seconds", "0.2", "--output-tokens", "4"]
    assert sd.main(argv) == 0
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5 and {r["concurrency"] for r in rows} == {"2"} and rows[0]["prefill_tokens"] == "784"
    assert FakeServer.completions >= 2
    assert sd.main(argv + ["--concurrency", "3"]) == 0
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["concurrency"] for r in rows} == {"2", "3"} and len(rows) == 10
    assert summary.read_text().count("| compiled | on |") == 2
