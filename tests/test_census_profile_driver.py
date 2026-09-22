import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from bench.census import profile_driver as pd


def test_prompts_have_exact_lengths_and_no_shared_prefix():
    prefill = pd.prompts_for({"label": "p", "kind": "prefill", "tokens": 128, "batch": 3, "max_tokens": 1}, seed=1)
    assert [len(p) for p in prefill] == [128, 128, 128] and prefill[0][:8] != prefill[1][:8]
    decode = pd.prompts_for({"label": "d", "kind": "decode", "kv": 1024, "batch": 2, "max_tokens": 16}, seed=1)
    assert [len(p) for p in decode] == [1008, 1008]
    assert pd.prompts_for({"label": "p", "kind": "prefill", "tokens": 128, "batch": 1, "max_tokens": 1}, seed=1) == \
        pd.prompts_for({"label": "p", "kind": "prefill", "tokens": 128, "batch": 1, "max_tokens": 1}, seed=1)
    with pytest.raises(ValueError):
        pd.prompts_for({"label": "d", "kind": "decode", "kv": 8, "batch": 1, "max_tokens": 16}, seed=1)


class FakeServer(BaseHTTPRequestHandler):
    calls: list = []
    trace_dir: Path = Path(".")

    def log_message(self, *args):
        pass

    def do_GET(self):
        self._reply({"data": [{"id": "fake-model"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        FakeServer.calls.append((self.path, body))
        if self.path == "/stop_profile":
            (FakeServer.trace_dir / f"rank0.{len(FakeServer.calls)}.pt.trace.json.gz").write_bytes(b"x")
        self._reply({"usage": {"prompt_tokens": len(body.get("prompt", []))}} if self.path == "/v1/completions" else {})

    def _reply(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server(tmp_path):
    FakeServer.calls = []
    FakeServer.trace_dir = tmp_path
    httpd = HTTPServer(("127.0.0.1", 0), FakeServer)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_driver_brackets_each_point_with_profile_calls_and_records_the_trace(server, tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "new_traces", lambda trace_dir, before, wait_s=120.0: sorted(p for p in trace_dir.glob("rank*.pt.trace.json.gz") if p not in before))
    grid = tmp_path / "grid.yaml"
    grid.write_text("points:\n  - {label: p128, kind: prefill, tokens: 128, batch: 2, max_tokens: 1}\n"
                    "  - {label: d1k, kind: decode, kv: 1024, batch: 3, max_tokens: 4}\n")
    manifest = tmp_path / "profiles.json"
    code = pd.main(["--server", server, "--mode", "eager", "--grid", str(grid), "--trace-dir", str(tmp_path), "--manifest", str(manifest)])
    assert code == 0
    paths = [path for path, _ in FakeServer.calls]
    assert paths[0] == "/start_profile" and paths.count("/start_profile") == 2 and paths.count("/stop_profile") == 2
    completions = [body for path, body in FakeServer.calls if path == "/v1/completions"]
    assert len(completions) == 5 and all(body["ignore_eos"] and body["model"] == "fake-model" for body in completions)
    assert {len(body["prompt"]) for body in completions} == {128, 1020}
    data = json.loads(manifest.read_text())
    assert [p["label"] for p in data["points"]] == ["p128", "d1k"]
    assert all(len(p["traces"]) == 1 and p["failed"] == 0 for p in data["points"])


def test_only_reruns_the_named_points_and_keeps_the_rest_of_the_manifest(server, tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "new_traces", lambda trace_dir, before, wait_s=120.0: sorted(p for p in trace_dir.glob("rank*.pt.trace.json.gz") if p not in before))
    grid = tmp_path / "grid.yaml"
    grid.write_text("points:\n  - {label: a, kind: prefill, tokens: 16, batch: 1, max_tokens: 1}\n"
                    "  - {label: b, kind: prefill, tokens: 32, batch: 1, max_tokens: 1}\n")
    manifest = tmp_path / "profiles.json"
    pd.main(["--server", server, "--mode", "eager", "--grid", str(grid), "--trace-dir", str(tmp_path), "--manifest", str(manifest)])
    pd.main(["--server", server, "--mode", "eager", "--grid", str(grid), "--trace-dir", str(tmp_path), "--manifest", str(manifest), "--only", "b"])
    data = json.loads(manifest.read_text())
    assert sorted(p["label"] for p in data["points"]) == ["a", "b"]
    assert [path for path, _ in FakeServer.calls].count("/start_profile") == 3
