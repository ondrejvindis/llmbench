"""Minimal fake OpenAI-compatible streaming server (mimics llama.cpp
server / vLLM's /v1/chat/completions SSE wire format) for adapter tests."""

from __future__ import annotations

import asyncio
import json

from aiohttp import web

FAKE_TOKENS = ["Hello", " there", ",", " this", " is", " a", " test", " response", "."] * 4


async def models(request: web.Request) -> web.Response:
    return web.json_response({"data": [{"id": "local-model"}]})


async def chat_completions(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await asyncio.sleep(0.03)
    count = 0
    for tok in FAKE_TOKENS:
        await asyncio.sleep(0.008)
        count += 1
        chunk = {"choices": [{"delta": {"content": tok}}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
    usage_chunk = {
        "choices": [],
        "usage": {"prompt_tokens": 15, "completion_tokens": count},
    }
    await resp.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat_completions)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), port=8080)
