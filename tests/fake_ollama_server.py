"""Minimal fake Ollama server for end-to-end testing without a real GPU
or downloaded model. Streams a handful of fake tokens with small delays
so TTFT/tok-s math has something real to compute against.
"""

from __future__ import annotations

import asyncio
import json

from aiohttp import web

FAKE_TOKENS = ["The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog. "] * 3


async def tags(request: web.Request) -> web.Response:
    return web.json_response({"models": [{"name": "llama3.1:8b"}]})


async def generate(request: web.Request) -> web.StreamResponse:
    resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
    await resp.prepare(request)
    await asyncio.sleep(0.05)  # simulate prefill / TTFT
    count = 0
    for tok in FAKE_TOKENS:
        await asyncio.sleep(0.01)  # simulate per-token generation time
        count += 1
        await resp.write((json.dumps({"response": tok, "done": False}) + "\n").encode())
    await resp.write(
        (json.dumps({
            "response": "", "done": True,
            "prompt_eval_count": 12, "eval_count": count,
        }) + "\n").encode()
    )
    await resp.write_eof()
    return resp


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/api/tags", tags)
    app.router.add_post("/api/generate", generate)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), port=11434)
