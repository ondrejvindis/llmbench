"""Host resource sampling: CPU, RAM, and (best-effort) GPU/VRAM.

GPU support is intentionally soft-fail: on a machine without an NVIDIA
GPU (or without ``nvidia-smi`` on PATH), we simply report ``None`` for
the GPU fields rather than raising — CPU-only benchmarking is a fully
valid use case (llama.cpp on CPU is common) and shouldn't be blocked by
a missing vendor tool.

Apple Silicon unified-memory GPU % is not exposed by any stable public
API without extra native dependencies, so it's out of scope for v0.1;
this module documents that gap rather than faking a number.
"""

from __future__ import annotations

import asyncio
import platform
import shutil

import psutil

from llmbench_tui.models import HostSample

_NVIDIA_SMI = shutil.which("nvidia-smi")


async def sample_host() -> HostSample:
    """Take one point-in-time reading of CPU/RAM/GPU/VRAM."""
    cpu_percent = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_used_gb = (vm.total - vm.available) / (1024**3)
    ram_total_gb = vm.total / (1024**3)
    cpu_name = _sample_cpu_name()

    gpu_percent = vram_used_gb = vram_total_gb = gpu_name = None
    if _NVIDIA_SMI:
        gpu_percent, vram_used_gb, vram_total_gb, gpu_name = await _sample_nvidia()

    return HostSample(
        cpu_percent=cpu_percent,
        ram_used_gb=ram_used_gb,
        ram_total_gb=ram_total_gb,
        cpu_name=cpu_name,
        gpu_percent=gpu_percent,
        vram_used_gb=vram_used_gb,
        vram_total_gb=vram_total_gb,
        gpu_name=gpu_name,
    )


def _sample_cpu_name() -> str | None:
    """Best-effort CPU model name."""
    name = platform.processor() or platform.uname().processor
    return name.strip() or None


async def _sample_nvidia() -> tuple[float | None, float | None, float | None, str | None]:
    """Query nvidia-smi for GPU utilization, VRAM, and model name.

    Uses the CSV query mode, which is stable across driver versions and
    avoids parsing the human-oriented table output.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            _NVIDIA_SMI,
            "--query-gpu=utilization.gpu,memory.used,memory.total,name",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
        line = stdout.decode().strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return None, None, None, None
        util_str, mem_used_str, mem_total_str, gpu_name = parts
        return (
            float(util_str),
            float(mem_used_str) / 1024,  # MiB -> GiB
            float(mem_total_str) / 1024,
            gpu_name or None,
        )
    except Exception:  # noqa: BLE001 - GPU telemetry is best-effort
        return None, None, None, None
