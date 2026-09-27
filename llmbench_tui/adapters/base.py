"""Abstract adapter interface every runtime backend must implement."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import httpx

from llmbench_tui.models import BackendConfig, SampleMetric


class BackendAdapter(ABC):
    """Talks to one local inference runtime and turns a streamed response
    into a single :class:`SampleMetric`.

    Subclasses only need to implement :meth:`stream_completion`, which must
    yield token *text* chunks as they arrive. Timing (TTFT, total time) and
    token counting are handled uniformly here so every runtime is measured
    the same way — this is what makes cross-runtime comparison meaningful.
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    async def stream_completion(
        self, client: httpx.AsyncClient, prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        """Yield completion text chunks as they stream in from the server."""
        raise NotImplementedError

    @abstractmethod
    async def is_reachable(self, client: httpx.AsyncClient) -> tuple[bool, str]:
        """Cheap health check. Returns (ok, message-for-humans)."""
        raise NotImplementedError

    async def run_once(
        self, client: httpx.AsyncClient, prompt: str, max_tokens: int
    ) -> SampleMetric:
        """Run one timed request and produce a SampleMetric.

        Never raises: adapter/network errors are captured into the
        SampleMetric so a single flaky backend can't crash the whole
        live dashboard.
        """
        start = time.perf_counter()
        ttft: float | None = None
        completion_chars = 0
        completion_tokens_est = 0
        try:
            async for chunk in self.stream_completion(client, prompt, max_tokens):
                if not chunk:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - start
                completion_chars += len(chunk)
                # Rough, runtime-agnostic token estimate (~4 chars/token for
                # English). Adapters that receive an exact usage count from
                # the server overwrite this before returning — see
                # OllamaAdapter/OpenAICompatAdapter.
                completion_tokens_est = max(1, completion_chars // 4)
            total = time.perf_counter() - start
            return SampleMetric(
                backend_name=self.name,
                ok=True,
                ttft_s=ttft,
                total_s=total,
                prompt_tokens=max(1, len(prompt) // 4),
                completion_tokens=completion_tokens_est,
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            total = time.perf_counter() - start
            return SampleMetric(
                backend_name=self.name,
                ok=False,
                ttft_s=ttft,
                total_s=total,
                prompt_tokens=0,
                completion_tokens=0,
                error=f"{type(exc).__name__}: {exc}",
            )
