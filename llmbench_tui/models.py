"""Core data models shared across adapters, the runner, and the UI.

Kept dependency-free (stdlib only) so they can be imported anywhere,
including inside unit tests that don't want to spin up Textual.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class RuntimeKind(str, Enum):
    """Which local inference server/engine a backend talks to."""

    OLLAMA = "ollama"
    OPENAI_COMPAT = "openai_compat"  # llama.cpp server, vLLM, LM Studio, etc.


@dataclass(slots=True)
class BackendConfig:
    """One configured target to benchmark (a running server + a model)."""

    name: str  # display label, e.g. "ollama/llama3.1:8b"
    kind: RuntimeKind
    base_url: str
    model: str
    api_key: str | None = None
    timeout_s: float = 120.0


@dataclass(slots=True)
class SampleMetric:
    """A single completed request's timing, ready to feed the live view."""

    backend_name: str
    ok: bool
    ttft_s: float | None  # time to first streamed token, seconds
    total_s: float  # wall-clock time for the whole request
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def tokens_per_second(self) -> float | None:
        """Generation speed, excluding the prefill/TTFT portion when known."""
        if not self.ok or self.completion_tokens <= 0:
            return None
        gen_time = self.total_s - (self.ttft_s or 0.0)
        if gen_time <= 0:
            return None
        return self.completion_tokens / gen_time


@dataclass(slots=True)
class RunningStats:
    """Incrementally-updated aggregate stats for one backend's samples.

    Deliberately simple (no numpy dependency) — this is a CLI tool that
    should install in under a second.
    """

    count: int = 0
    errors: int = 0
    ttft_sum: float = 0.0
    ttft_n: int = 0
    tps_sum: float = 0.0
    tps_n: int = 0
    tps_values: list[float] = field(default_factory=list)
    last_tps: float | None = None
    last_ttft: float | None = None

    def add(self, sample: SampleMetric) -> None:
        self.count += 1
        if not sample.ok:
            self.errors += 1
            return
        if sample.ttft_s is not None:
            self.ttft_sum += sample.ttft_s
            self.ttft_n += 1
            self.last_ttft = sample.ttft_s
        tps = sample.tokens_per_second
        if tps is not None:
            self.tps_sum += tps
            self.tps_n += 1
            self.tps_values.append(tps)
            self.last_tps = tps

    @property
    def avg_ttft(self) -> float | None:
        return self.ttft_sum / self.ttft_n if self.ttft_n else None

    @property
    def avg_tps(self) -> float | None:
        return self.tps_sum / self.tps_n if self.tps_n else None

    @property
    def p95_tps(self) -> float | None:
        if not self.tps_values:
            return None
        ordered = sorted(self.tps_values)
        idx = max(0, int(len(ordered) * 0.95) - 1)
        return ordered[idx]


@dataclass(slots=True)
class HostSample:
    """One point-in-time reading of host resource usage."""

    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    cpu_name: str | None = None
    gpu_percent: float | None = None
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None
    gpu_name: str | None = None
    timestamp: float = field(default_factory=time.time)
