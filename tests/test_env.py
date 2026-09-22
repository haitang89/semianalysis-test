import json

import pytest

from bench.core import env as envmod
from bench.core.env import Environment, collect, environment_hash, gdn_prefill_engine_default, toolchain_problems

H200_SMI = "NVIDIA H200, 580.178.04, 143771, 700.00, 1980, 3201, 9.0\n"


def fake_run(smi_line=H200_SMI, git="abc123\n"):
    def run(command):
        if command[0] == "nvidia-smi":
            return smi_line
        if command[0] == "git":
            return git
        raise FileNotFoundError(command[0])

    return run


def h200_torch():
    return {"sm_count": 140, "cuda_runtime": "13.0"}


def test_collect_reads_device_and_launcher_values():
    environ = {"BENCH_IMAGE": "vllm/vllm-openai:v0.29.0", "BENCH_IMAGE_DIGEST": "sha256:c29", "BENCH_GIT_SHA": "deadbeef"}
    env = collect(run=fake_run(), torch_info=h200_torch, environ=environ)
    assert env.gpu_name == "NVIDIA H200"
    assert env.compute_capability == "9.0"
    assert env.memory_total_mib == 143771
    assert env.sm_count == 140
    assert env.cuda_runtime == "13.0"
    assert (env.image, env.image_digest, env.git_sha) == ("vllm/vllm-openai:v0.29.0", "sha256:c29", "deadbeef")
    assert env.gdn_prefill_engine_default == "flashinfer"
    assert toolchain_problems(env) == []


def test_unknown_values_are_null_with_a_reason():
    def broken_run(command):
        raise FileNotFoundError(command[0])

    def no_torch():
        raise ImportError("torch")

    env = collect(run=broken_run, torch_info=no_torch, environ={})
    assert env.gpu_name is None and env.sm_count is None and env.git_sha is None
    for name in ("gpu_name", "sm_count", "cuda_runtime", "image", "image_digest", "git_sha"):
        assert env.missing[name]
    assert toolchain_problems(env) == ["no CUDA device visible"]


def test_gate_rejects_old_gpus_and_blackwell_with_old_cuda():
    old = Environment(compute_capability="7.5", cuda_runtime="12.4")
    assert "below 8.0" in toolchain_problems(old)[0]
    blackwell = Environment(compute_capability="10.3", cuda_runtime="12.8")
    assert "CUDA 12.9" in toolchain_problems(blackwell)[0]
    assert toolchain_problems(Environment(compute_capability="10.3", cuda_runtime="13.0")) == []


@pytest.mark.parametrize(
    "capability,cuda,expected",
    [
        ((9, 0), (12, 8), "flashinfer"),
        ((10, 3), (13, 0), "flashinfer"),
        ((10, 3), (12, 9), "triton"),
        ((12, 0), (13, 0), "triton"),
        ((8, 9), (13, 0), "triton"),
        ((8, 0), (13, 0), "triton"),
    ],
)
def test_gdn_prefill_default_follows_the_engine_rule(capability, cuda, expected):
    assert gdn_prefill_engine_default(capability, cuda) == expected


def test_blackwell_on_cuda_12_warns_about_the_triton_default():
    env = collect(
        run=fake_run("NVIDIA B300, 580.1, 275040, 1100, 2000, 3000, 10.3\n"),
        torch_info=lambda: {"sm_count": 160, "cuda_runtime": "12.9"},
        environ={},
    )
    assert env.gdn_prefill_engine_default == "triton"
    assert any("CUDA 13" in warning for warning in env.warnings)


def test_hash_ignores_the_timestamp_and_the_commit_only():
    first, second = envmod.mock_environment(), envmod.mock_environment()
    second.collected_at = "2026-09-21T00:00:00+00:00"
    second.git_sha = "0" * 40
    second.git_dirty = not first.git_dirty
    assert environment_hash(first) == environment_hash(second)
    second.driver_version = "1.0"
    assert environment_hash(first) != environment_hash(second)


def test_record_has_nothing_that_identifies_the_host(tmp_path):
    env = collect(run=fake_run(), torch_info=h200_torch, environ={"USERNAME": "someone", "COMPUTERNAME": "box"})
    text = json.dumps(env.to_dict())
    assert "someone" not in text and "box" not in text


def test_cli_writes_the_mock_record(tmp_path):
    out = tmp_path / "environment.json"
    assert envmod.main(["--mock", "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["gpu_name"] == "Mock H200" and record["missing"] == {}


def test_dirty_tree_is_recorded_and_warned_about():
    dirty = collect(run=fake_run(), torch_info=h200_torch, environ={"BENCH_GIT_SHA": "abc", "BENCH_GIT_DIRTY": "1"})
    assert dirty.git_dirty is True
    assert any("uncommitted" in warning for warning in dirty.warnings)
    clean = collect(run=fake_run(), torch_info=h200_torch, environ={"BENCH_GIT_SHA": "abc", "BENCH_GIT_DIRTY": "0"})
    assert clean.git_dirty is False and clean.warnings == []


def test_provenance_names_the_code_and_environment():
    env = envmod.mock_environment()
    source = envmod.provenance(env)
    assert source["environment_hash"] == environment_hash(env)
    assert (source["git_sha"], source["git_dirty"], source["image_digest"]) == ("mock", False, "sha256:mock")
