# Contributing

Thanks for considering a contribution — this is a young project and early feedback matters a lot.

## Setup

```bash
git clone https://github.com/ondrej/llmbench-tui.git
cd llmbench-tui
pip install -e ".[dev]"
pytest
```

## Adding a new runtime adapter

If a runtime speaks the OpenAI-compatible `/v1/chat/completions` streaming
format, you likely don't need a new adapter at all — add it as a `kind`
alias in `llmbench_tui/config.py` pointing at `RuntimeKind.OPENAI_COMPAT`.

A genuinely new adapter is only needed for a different wire protocol.
To add one:

1. Create `llmbench_tui/adapters/your_runtime.py`, subclassing `BackendAdapter`
   (see `llmbench_tui/adapters/base.py` for the contract).
2. Implement `stream_completion()` (yield text chunks) and `is_reachable()`.
3. Register it in `llmbench_tui/adapters/__init__.py`'s `make_adapter()`.
4. Add a fake test server under `tests/` (see `tests/fake_ollama_server.py`
   for the pattern) and an integration test in `tests/test_adapters.py`.

## Reporting bugs

Please include: your OS, Python version, the runtime(s) you were
benchmarking, and the exact `llmbench-tui` command you ran. A `--check`
run's output is very helpful for connectivity issues.

## Code style

Plain, readable Python. Type hints on public functions. No new
dependencies without a good reason — this tool should stay quick to
install.
