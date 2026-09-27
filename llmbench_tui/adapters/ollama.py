"""Adapter for Ollama's native streaming API (``/api/generate``).

Ollama also exposes an OpenAI-compatible endpoint, but the native API
returns exact ``eval_count``/``prompt_eval_count`` token numbers in the
final streamed object, which is more accurate than estimating from
character counts — so we use it directly instead of going through the
generic OpenAI-compat adapter.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from llmbench_tui.adapters.base import BackendAdapter
from llmbench_tui.models import SampleMetric


class OllamaAdapter(BackendAdapter):
    async def is_reachable(self, client: httpx.AsyncClient) -> tuple[bool, str]:
        try:
            resp = await client.get(
                f"{self.config.base_url}/api/tags", timeout=5.0
            )
            resp.raise_for_status()
            tags = [m.get("name") for m in resp.json().get("models", [])]
            if self.config.model not in tags and tags:
                return True, f"reachable (model '{self.config.model}' not yet pulled?)"
            return True, "reachable"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    async def stream_completion(
        self, client: httpx.AsyncClient, prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens},
        }
        async with client.stream(
            "POST",
            f"{self.config.base_url}/api/generate",
            json=payload,
            timeout=self.config.timeout_s,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("response"):
                    yield obj["response"]
                if obj.get("done"):
                    # Stash exact counters for run_once() to prefer over
                    # its character-based estimate.
                    self._last_prompt_eval = obj.get("prompt_eval_count")
                    self._last_eval = obj.get("eval_count")

    async def run_once(
        self, client: httpx.AsyncClient, prompt: str, max_tokens: int
    ) -> SampleMetric:
        self._last_prompt_eval = None
        self._last_eval = None
        sample = await super().run_once(client, prompt, max_tokens)
        # Overwrite the rough estimate with Ollama's exact token counts
        # when the request succeeded and the server reported them.
        if sample.ok:
            if self._last_eval:
                sample.completion_tokens = self._last_eval
            if self._last_prompt_eval:
                sample.prompt_tokens = self._last_prompt_eval
        return sample
