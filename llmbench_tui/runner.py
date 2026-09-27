"""Orchestrates repeated benchmark requests across all configured backends
and pushes results onto an asyncio.Queue that the UI (or headless mode)
consumes from.

Design choice: backends run *concurrently*, each on its own loop, rather
than round-robin sequentially. This matches how someone would actually
use multiple runtimes side by side, and means a slow backend doesn't
block sampling of a fast one.
"""

from __future__ import annotations

import asyncio
import itertools

import httpx

from llmbench_tui.adapters import make_adapter
from llmbench_tui.models import BackendConfig, SampleMetric

DEFAULT_PROMPTS = [
    "Explain the difference between TCP and UDP in three sentences.",
    "Write a short Python function that reverses a linked list.",
    "Summarize the causes of the French Revolution in one paragraph.",
    "What are the main trade-offs between microservices and a monolith?",
    "Describe how a hash table resolves collisions.",
]


class BenchmarkRunner:
    """Drives one adapter in a loop, emitting a SampleMetric per request."""

    def __init__(
        self,
        config: BackendConfig,
        result_queue: asyncio.Queue[SampleMetric],
        prompts: list[str] | None = None,
        max_tokens: int = 256,
        interval_s: float = 0.5,
    ) -> None:
        self.adapter = make_adapter(config)
        self.result_queue = result_queue
        self.prompts = prompts or DEFAULT_PROMPTS
        self.max_tokens = max_tokens
        self.interval_s = interval_s
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        """Loop until stop() is called, cycling through the prompt set.

        A fresh httpx.AsyncClient per runner keeps connection pools
        isolated between backends, so one backend's slow/hanging
        connection can't starve another's pool.
        """
        async with httpx.AsyncClient() as client:
            for prompt in itertools.cycle(self.prompts):
                if self._stop.is_set():
                    return
                sample = await self.adapter.run_once(client, prompt, self.max_tokens)
                await self.result_queue.put(sample)
                if self._stop.is_set():
                    return
                await asyncio.sleep(self.interval_s)

    async def run_n(self, n: int) -> list[SampleMetric]:
        """Run exactly n requests and return the samples (headless/batch mode)."""
        samples: list[SampleMetric] = []
        async with httpx.AsyncClient() as client:
            for i in range(n):
                prompt = self.prompts[i % len(self.prompts)]
                sample = await self.adapter.run_once(client, prompt, self.max_tokens)
                samples.append(sample)
                await self.result_queue.put(sample)
        return samples

    async def check_reachable(self) -> tuple[bool, str]:
        async with httpx.AsyncClient() as client:
            return await self.adapter.is_reachable(client)
