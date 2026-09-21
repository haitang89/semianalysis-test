"""Hardware adapters for the ceiling measurements: the real GPU through torch, or a model of one."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from . import nvsmi
from .mock import MOCK_CEILINGS, ModelClock
from .nvsmi import ClockSample
from .roofline import roofline_latency_us
from .timer import Clock, Kernel

GEMM_TOKENS, GEMM_OUT, GEMM_IN = 16384, 34816, 5120
GEMM_FLOPS = 2.0 * GEMM_TOKENS * GEMM_OUT * GEMM_IN
GEMM_SHAPE = {"tokens": GEMM_TOKENS, "out_features": GEMM_OUT, "in_features": GEMM_IN}
SUSTAIN_BATCH = 20


@dataclass
class Workload:
    kernel: Kernel
    clock: Clock
    bytes_moved: float = 0.0
    flops: float = 0.0
    path: str = ""


class Backend(Protocol):
    gpu_name: str

    def copy(self, nbytes: int) -> Workload: ...

    def reduce(self, nbytes: int) -> Workload: ...

    def gemm(self, dtype: str) -> Workload: ...

    def sustain(self, workload: Workload) -> int: ...

    def release(self) -> None: ...

    def clock_sample(self) -> Optional[ClockSample]: ...


class TorchBackend:
    def __init__(self) -> None:
        import torch

        from .timer import CudaClock

        self._torch = torch
        self._clock = CudaClock()
        self.gpu_name = torch.cuda.get_device_name(0)

    def copy(self, nbytes: int) -> Workload:
        torch = self._torch
        src = torch.ones(nbytes, dtype=torch.uint8, device="cuda")
        dst = torch.empty_like(src)
        return Workload(lambda: dst.copy_(src), self._clock, bytes_moved=2.0 * nbytes, path="tensor.copy_")

    def reduce(self, nbytes: int) -> Workload:
        torch = self._torch
        data = torch.ones(nbytes // 2, dtype=torch.float16, device="cuda")
        return Workload(lambda: torch.amax(data), self._clock, bytes_moved=float(nbytes), path="torch.amax")

    def gemm(self, dtype: str) -> Workload:
        torch = self._torch
        x = torch.randn(GEMM_TOKENS, GEMM_IN, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(GEMM_OUT, GEMM_IN, dtype=torch.bfloat16, device="cuda")
        if dtype == "bf16":
            out = torch.empty(GEMM_TOKENS, GEMM_OUT, dtype=torch.bfloat16, device="cuda")
            return Workload(lambda: torch.mm(x, w.t(), out=out), self._clock, flops=GEMM_FLOPS, path="torch.mm")
        if dtype == "fp8":
            x8, w8 = x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
            scale = torch.ones((), dtype=torch.float32, device="cuda")

            def kernel():
                return torch._scaled_mm(x8, w8.t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)

            return Workload(kernel, self._clock, flops=GEMM_FLOPS, path="torch._scaled_mm, per tensor scale")
        raise ValueError(f"unknown dtype {dtype}")

    def sustain(self, workload: Workload) -> int:
        for _ in range(SUSTAIN_BATCH):
            workload.kernel()
        self._torch.cuda.synchronize()
        return SUSTAIN_BATCH

    def release(self) -> None:
        self._torch.cuda.empty_cache()

    def clock_sample(self) -> Optional[ClockSample]:
        return nvsmi.clock_sample()


class MockBackend:
    gpu_name = "Mock H200"

    def __init__(self, sm_clock_mhz: float = 1980.0, slowdown: bool = False, power_capped: bool = False):
        self._sample = ClockSample(
            sm_clock_mhz=sm_clock_mhz,
            sm_clock_max_mhz=1980.0,
            power_w=650.0,
            power_limit_w=700.0,
            temp_c=60.0,
            slowdown=slowdown,
            power_capped=power_capped,
        )

    def _workload(self, bytes_moved: float, flops: float, peak_tflops: float, path: str) -> Workload:
        latency = roofline_latency_us(flops, bytes_moved, peak_tflops, MOCK_CEILINGS.hbm_gbps)
        return Workload(lambda: None, ModelClock(latency, launch_us=0.0), bytes_moved, flops, path)

    def copy(self, nbytes: int) -> Workload:
        return self._workload(2.0 * nbytes, 0.0, MOCK_CEILINGS.bf16_tflops, "mock copy")

    def reduce(self, nbytes: int) -> Workload:
        return self._workload(float(nbytes), 0.0, MOCK_CEILINGS.bf16_tflops, "mock reduce")

    def gemm(self, dtype: str) -> Workload:
        return self._workload(0.0, GEMM_FLOPS, MOCK_CEILINGS.peak_tflops(dtype), f"mock {dtype} gemm")

    def sustain(self, workload: Workload) -> int:
        return 0

    def release(self) -> None:
        return None

    def clock_sample(self) -> Optional[ClockSample]:
        return self._sample
