# llmbench-tui

**Live terminal dashboard that benchmarks Ollama, llama.cpp, vLLM, and LM Studio side by side — in real time.**

Stop guessing which local LLM runtime is faster on your hardware. Point `llmbench-tui` at one or more running servers and watch tokens/sec, time-to-first-token, and host resource usage (CPU/RAM/GPU/VRAM) update live, in one screen.


┌─ ollama/llama3.1:8b ──────────┐  ┌─ llama.cpp/local-7b ──────────┐
│ tok/s: 42.3                   │  │ tok/s: 68.1                   │
│ avg tok/s: 40.8               │  │ avg tok/s: 65.2               │
│ TTFT: 0.31s                   │  │ TTFT: 0.09s                   │
│ avg TTFT: 0.28s               │  │ avg TTFT: 0.11s               │
│ requests: 14  errors: 0       │  │ requests: 14  errors: 0       │
│ ▁▂▃▅▆▇█▇▆▅▃▂▁▂▃▄▅▆▇█      │  │ ▅▆▇█▇▆▅▄▃▂▁▂▃▅▆▇█▇▆       │
└───────────────────────────────┘  └───────────────────────────────┘

Recent requests
 12:04:01  ollama/llama3.1:8b     ok    41.2   0.29   1.84
 12:04:01  llama.cpp/local-7b     ok    67.9   0.10   0.73
 ...

CPU: 38.2%  |  RAM: 11.4/32 GB  |  GPU: 71.0%  |  VRAM: 9.8/24 GB
```

## Install

```bash
pip install llmbench-tui
```

Or from source:

```bash
git clone https://github.com/ondrej/llmbench-tui.git
cd llmbench-tui
pip install -e .
```

## Quickstart

Benchmark a single running Ollama server:

```bash
llmbench-tui -b ollama:llama3.1:8b
```

Compare Ollama against a llama.cpp server running side by side:

```bash
llmbench-tui -b ollama:llama3.1:8b -b llama.cpp:local-7b@http://localhost:8080
```

Just check that your backends are reachable, without launching the dashboard:

```bash
llmbench-tui -b ollama:llama3.1:8b --check
```

Press **`e`** inside the dashboard at any time to export every collected sample to CSV. Press **`q`** to quit.

## Supported runtimes

| Runtime | `kind` value | Notes |
|---|---|---|
| Ollama | `ollama` | Uses the native `/api/generate` streaming endpoint for exact token counts. |
| llama.cpp server | `llama.cpp` | OpenAI-compatible `/v1/chat/completions` streaming endpoint. |
| vLLM | `vllm` | Same OpenAI-compatible endpoint. |
| LM Studio | `lmstudio` | Same OpenAI-compatible endpoint (enable the local server in LM Studio's settings first). |
| Any OpenAI-compatible server | `openai` | Generic fallback — works with anything speaking the same wire format. |

Backend spec syntax: `kind:model[@base_url]`. The `base_url` defaults to `http://localhost:11434` for Ollama and `http://localhost:8080` for everything else.

## Config file (for more than 2–3 backends)

```json
{
  "backends": [
    { "name": "ollama-8b", "kind": "ollama", "base_url": "http://localhost:11434", "model": "llama3.1:8b" },
    { "name": "llamacpp-7b", "kind": "llama.cpp", "base_url": "http://localhost:8080", "model": "local-7b" },
    { "name": "vllm-remote", "kind": "vllm", "base_url": "http://192.168.1.50:8000", "model": "Qwen2.5-7B" }
  ]
}
```

```bash
llmbench-tui -c config.json
```

## GPU monitoring

GPU utilization and VRAM usage are read via `nvidia-smi` when it's present on `PATH`. Without an NVIDIA GPU (or on Apple Silicon), the dashboard runs fine in CPU-only mode — the GPU/VRAM fields just show `n/a`.

## How the numbers are measured

- **TTFT** (time-to-first-token): wall-clock time from sending the request to the first non-empty streamed chunk.
- **tok/s**: completion tokens divided by the *generation* time only (total time minus TTFT), so a slow prefill doesn't get counted as slow generation. Token counts come from each server's own usage/eval fields when available (exact), falling back to a ~4-chars-per-token estimate otherwise.
- Every backend runs on its own connection pool and its own request loop, so a slow/hanging backend can't stall the others.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite includes minimal fake Ollama/OpenAI-compatible servers (`tests/fake_*_server.py`) so adapter behavior is verified against real streamed HTTP responses, not mocks.

## Roadmap / contributions welcome

- [ ] Apple Silicon GPU utilization (no stable public API yet — pointers welcome)
- [ ] Batch/headless mode with a summary table export
- [ ] Per-backend custom prompt sets via config file
- [ ] Docker Compose example for multi-runtime local setups

Issues and PRs welcome — this is a young project and early feedback shapes it a lot.

## License

MIT — see [LICENSE](LICENSE).
