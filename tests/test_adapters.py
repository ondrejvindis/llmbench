"""Integration tests: real HTTP round-trips against in-process fake
servers (via aiohttp's test utilities), so the adapters are exercised
against actual streamed bytes rather than mocked objects.
"""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from llmbench_tui.adapters.ollama import OllamaAdapter
from llmbench_tui.adapters.openai_compat import OpenAICompatAdapter
from llmbench_tui.models import BackendConfig, RuntimeKind

# Reuse the fake server app factories from the manual test scripts.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fake_ollama_server import make_app as make_ollama_app  # noqa: E402
from fake_openai_server import make_app as make_openai_app  # noqa: E402


@pytest.mark.asyncio
async def test_ollama_adapter_streams_and_counts_tokens():
    app = make_ollama_app()
    async with TestServer(app) as server, TestClient(server) as aio_client:
        base_url = str(server.make_url(""))
        cfg = BackendConfig(
            name="ollama/test", kind=RuntimeKind.OLLAMA,
            base_url=base_url.rstrip("/"), model="llama3.1:8b",
        )
        adapter = OllamaAdapter(cfg)

        import httpx
        # aiohttp's TestServer binds a real localhost socket, so a plain
        # httpx client connects to it exactly like it would to a real
        # Ollama/llama.cpp server — no ASGI transport shim needed.
        async with httpx.AsyncClient() as client:
            sample = await adapter.run_once(client, "hi", 50)

        assert sample.ok, sample.error
        assert sample.completion_tokens > 0
        assert sample.ttft_s is not None
        assert sample.tokens_per_second is not None


@pytest.mark.asyncio
async def test_openai_compat_adapter_streams_and_counts_tokens():
    app = make_openai_app()
    async with TestServer(app) as server:
        base_url = str(server.make_url("")).rstrip("/")
        cfg = BackendConfig(
            name="llama.cpp/test", kind=RuntimeKind.OPENAI_COMPAT,
            base_url=base_url, model="local-model",
        )
        adapter = OpenAICompatAdapter(cfg)

        import httpx
        async with httpx.AsyncClient() as client:
            ok, _ = await adapter.is_reachable(client)
            assert ok
            sample = await adapter.run_once(client, "hi", 50)

        assert sample.ok, sample.error
        assert sample.completion_tokens > 0
        assert sample.prompt_tokens == 15  # from the fake server's usage block
