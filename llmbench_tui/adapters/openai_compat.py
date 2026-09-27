"""Adapter for any server exposing the OpenAI-compatible ``/v1/completions``
(or ``/v1/chat/completions``) streaming API — this covers llama.cpp's
built-in server, vLLM, and LM Studio without needing separate adapters
for each, since they all converged on the same wire format.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from llmbench_tui.adapters.base import BackendAdapter
from llmbench_tui.models import SampleMetric


class OpenAICompatAdapter(BackendAdapter):
    async def is_reachable(self, client: httpx.AsyncClient) -> tuple[bool, str]:
        headers = self._auth_headers()
        for path in ("/v1/models", "/models"):
            try:
                resp = await client.get(
                    f"{self.config.base_url}{path}", headers=headers, timeout=5.0
                )
                if resp.status_code < 500:
                    return True, "reachable"
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
        return False, last_err if "last_err" in dir() else "unreachable"

    def _auth_headers(self) -> dict[str, str]:
        if self.config.api_key:
            return {"Authorization": f"Bearer {self.config.api_key}"}
        return {}

    async def stream_completion(
        self, client: httpx.AsyncClient, prompt: str, max_tokens: int
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"Content-Type": "application/json", **self._auth_headers()}
        self._last_usage: dict | None = None
        async with client.stream(
            "POST",
            f"{self.config.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.config.timeout_s,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                usage = obj.get("usage")
                if usage:
                    self._last_usage = usage
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text

    async def run_once(
        self, client: httpx.AsyncClient, prompt: str, max_tokens: int
    ) -> SampleMetric:
        sample = await super().run_once(client, prompt, max_tokens)
        usage = getattr(self, "_last_usage", None)
        if sample.ok and usage:
            sample.completion_tokens = usage.get(
                "completion_tokens", sample.completion_tokens
            )
            sample.prompt_tokens = usage.get("prompt_tokens", sample.prompt_tokens)
        return sample
