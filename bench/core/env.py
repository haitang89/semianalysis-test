"""Environment record: the hardware and software every result row points back to."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable, Mapping, Optional

from .nvsmi import Runner, number, query_gpu
from .nvsmi import run as run_command

PACKAGES = {
    "vllm": "vllm",
    "flashinfer": "flashinfer-python",
    "triton": "triton",
    "torch": "torch",
    "transformers": "transformers",
}
SMI_FIELDS = (
    "name",
    "driver_version",
    "memory.total",
    "power.limit",
    "clocks.max.sm",
    "clocks.max.memory",
    "compute_cap",
)
MIN_CAPABILITY = (8, 0)
BLACKWELL_MIN_CUDA = (12, 9)
HOPPER = (9, 0)
BLACKWELL_MAJOR = 10
GDN_FLASHINFER_HEAD_DIM = 128


@dataclass
class Environment:
    gpu_name: Optional[str] = None
    compute_capability: Optional[str] = None
    sm_count: Optional[int] = None
    sm_clock_max_mhz: Optional[float] = None
    mem_clock_max_mhz: Optional[float] = None
    memory_total_mib: Optional[int] = None
    power_limit_w: Optional[float] = None
    driver_version: Optional[str] = None
    cuda_runtime: Optional[str] = None
    image: Optional[str] = None
    image_digest: Optional[str] = None
    git_sha: Optional[str] = None
    git_dirty: Optional[bool] = None
    versions: dict = field(default_factory=dict)
    python_version: str = ""
    host_os: str = ""
    gdn_prefill_engine_default: Optional[str] = None
    missing: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    collected_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def parse_version(text: Optional[str]) -> Optional[tuple[int, int]]:
    if not text:
        return None
    parts = text.split(".")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None


def gdn_prefill_engine_default(
    capability: tuple[int, int], cuda: tuple[int, int], head_k_dim: int = GDN_FLASHINFER_HEAD_DIM
) -> str:
    """Mirrors vLLM 0.29 `_resolve_gdn_prefill_backend` for the default "auto" setting."""
    if capability == HOPPER:
        return "flashinfer"
    if capability[0] == BLACKWELL_MAJOR and head_k_dim == GDN_FLASHINFER_HEAD_DIM and cuda[0] >= 13:
        return "flashinfer"
    return "triton"


def toolchain_problems(env: Environment) -> list[str]:
    capability = parse_version(env.compute_capability)
    if capability is None:
        return ["no CUDA device visible"]
    problems = []
    if capability < MIN_CAPABILITY:
        problems.append(f"compute capability {env.compute_capability} is below {MIN_CAPABILITY[0]}.{MIN_CAPABILITY[1]}")
    cuda = parse_version(env.cuda_runtime)
    if capability[0] >= BLACKWELL_MAJOR and cuda is not None and cuda < BLACKWELL_MIN_CUDA:
        problems.append(f"Blackwell needs CUDA 12.9 or newer, container has {env.cuda_runtime}")
    return problems


def _torch_device_info() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch reports no CUDA device")
    props = torch.cuda.get_device_properties(0)
    return {"sm_count": props.multi_processor_count, "cuda_runtime": torch.version.cuda}


def _read_smi(env: Environment, run: Runner) -> None:
    try:
        name, driver, memory, power, sm_clock, mem_clock, capability = query_gpu(SMI_FIELDS, run)
    except Exception as exc:
        reason = f"nvidia-smi failed: {type(exc).__name__}"
        for name in ("gpu_name", "driver_version", "memory_total_mib", "power_limit_w", "compute_capability"):
            env.missing[name] = reason
        return
    memory_mib = number(memory)
    env.gpu_name = name
    env.driver_version = driver
    env.memory_total_mib = int(memory_mib) if memory_mib is not None else None
    env.power_limit_w = number(power)
    env.sm_clock_max_mhz = number(sm_clock)
    env.mem_clock_max_mhz = number(mem_clock)
    env.compute_capability = capability


def _read_torch(env: Environment, torch_info: Callable[[], dict]) -> None:
    try:
        info = torch_info()
    except Exception as exc:
        reason = f"torch unavailable: {type(exc).__name__}"
        env.missing["sm_count"] = reason
        env.missing["cuda_runtime"] = reason
        return
    env.sm_count = info["sm_count"]
    env.cuda_runtime = info["cuda_runtime"]


def _read_versions(env: Environment) -> None:
    for label, package in PACKAGES.items():
        try:
            env.versions[label] = metadata.version(package)
        except metadata.PackageNotFoundError:
            env.versions[label] = None
            env.missing[f"versions.{label}"] = "package not installed"


def _read_git(environ: Mapping[str, str], run: Runner) -> tuple[Optional[str], Optional[bool]]:
    if environ.get("BENCH_GIT_SHA"):
        dirty = environ.get("BENCH_GIT_DIRTY")
        return environ["BENCH_GIT_SHA"], None if dirty is None else dirty == "1"
    try:
        sha = run(["git", "rev-parse", "HEAD"]).strip() or None
        dirty = bool(run(["git", "status", "--porcelain"]).strip())
    except Exception:
        return None, None
    return sha, dirty


def collect(
    run: Runner = run_command,
    torch_info: Callable[[], dict] = _torch_device_info,
    environ: Mapping[str, str] = os.environ,
) -> Environment:
    env = Environment(
        python_version=platform.python_version(),
        host_os=f"{platform.system()} {platform.release()}",
        collected_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    _read_smi(env, run)
    _read_torch(env, torch_info)
    _read_versions(env)

    env.image = environ.get("BENCH_IMAGE") or None
    env.image_digest = environ.get("BENCH_IMAGE_DIGEST") or None
    env.git_sha, env.git_dirty = _read_git(environ, run)
    for name in ("image", "image_digest", "git_sha"):
        if getattr(env, name) is None:
            env.missing[name] = "not provided by the container launcher"
    if env.git_dirty:
        env.warnings.append("results come from uncommitted code")

    capability, cuda = parse_version(env.compute_capability), parse_version(env.cuda_runtime)
    if capability and cuda:
        env.gdn_prefill_engine_default = gdn_prefill_engine_default(capability, cuda)
        if capability[0] == BLACKWELL_MAJOR and env.gdn_prefill_engine_default == "triton":
            env.warnings.append("vLLM selects the FlashInfer GDN prefill kernel on Blackwell only with CUDA 13 or newer")
    return env


def mock_environment() -> Environment:
    return Environment(
        gpu_name="Mock H200",
        compute_capability="9.0",
        sm_count=132,
        sm_clock_max_mhz=1980.0,
        mem_clock_max_mhz=3201.0,
        memory_total_mib=143771,
        power_limit_w=700.0,
        driver_version="0.0",
        cuda_runtime="13.0",
        image="mock",
        image_digest="sha256:mock",
        git_sha="mock",
        git_dirty=False,
        versions={label: "mock" for label in PACKAGES},
        python_version=platform.python_version(),
        host_os="mock",
        gdn_prefill_engine_default="flashinfer",
        collected_at="1970-01-01T00:00:00+00:00",
    )


def environment_hash(env: Environment) -> str:
    stable = {key: value for key, value in env.to_dict().items() if key != "collected_at"}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]


def provenance(env: Environment) -> dict:
    return {
        "environment_hash": environment_hash(env),
        "git_sha": env.git_sha,
        "git_dirty": env.git_dirty,
        "image_digest": env.image_digest,
        "collected_at": env.collected_at,
    }


def write(env: Environment, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(env.to_dict(), indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record the hardware and software environment")
    parser.add_argument("--out", default="results/env/environment.json")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    env = mock_environment() if args.mock else collect()
    write(env, Path(args.out))
    print(f"{env.gpu_name}, capability {env.compute_capability}, CUDA {env.cuda_runtime}, hash {environment_hash(env)}")
    for warning in env.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    problems = toolchain_problems(env)
    for problem in problems:
        print(f"toolchain problem: {problem}", file=sys.stderr)
    return 2 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
