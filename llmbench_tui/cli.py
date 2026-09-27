"""Command-line entry point: ``llmbench-tui``."""

from __future__ import annotations

import asyncio
import sys

import click

from llmbench_tui.config import backend_from_flag, load_config_file
from llmbench_tui.models import BackendConfig


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    "-b",
    "backend_specs",
    multiple=True,
    help=(
        "Backend to benchmark, as kind:model[@base_url]. Repeatable. "
        "kind is one of: ollama, llama.cpp, vllm, lmstudio, openai. "
        "Example: -b ollama:llama3.1:8b -b llama.cpp:local-7b@http://localhost:8080"
    ),
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a JSON config file with a top-level 'backends' list "
         "(see README for the schema). Overrides --backend if both given.",
)
@click.option(
    "--max-tokens",
    default=256,
    show_default=True,
    help="Max tokens to request per benchmark call.",
)
@click.option(
    "--interval",
    "interval_s",
    default=0.5,
    show_default=True,
    help="Seconds to wait between requests to the same backend.",
)
@click.option(
    "--export",
    "export_path",
    default=None,
    help="CSV path used when pressing 'e' inside the dashboard "
         "(default: llmbench-results.csv in the current directory).",
)
@click.option(
    "--check",
    is_flag=True,
    help="Just verify all backends are reachable, then exit (no dashboard).",
)
def main(
    backend_specs: tuple[str, ...],
    config_path: str | None,
    max_tokens: int,
    interval_s: float,
    export_path: str | None,
    check: bool,
) -> None:
    """Live terminal dashboard for benchmarking local LLM runtimes.

    Point it at one or more running Ollama / llama.cpp / vLLM / LM Studio
    servers and it streams tok/s, time-to-first-token, and host resource
    usage side by side, live, in your terminal.
    """
    configs: list[BackendConfig]
    if config_path:
        configs = load_config_file(config_path)
    elif backend_specs:
        configs = [backend_from_flag(spec) for spec in backend_specs]
    else:
        click.echo(
            "No backends given. Use --backend kind:model[@base_url] "
            "(repeatable) or --config path/to/config.json.\n\n"
            "Example:\n"
            "  llmbench-tui -b ollama:llama3.1:8b\n"
            "  llmbench-tui -b ollama:llama3.1:8b -b llama.cpp:local-7b@http://localhost:8080",
            err=True,
        )
        sys.exit(1)

    if check:
        sys.exit(asyncio.run(_run_check(configs)))

    from llmbench_tui.app import LLMBenchApp  # deferred: Textual import is heavier

    app = LLMBenchApp(
        configs, max_tokens=max_tokens, interval_s=interval_s, export_path=export_path
    )
    app.run()


async def _run_check(configs: list[BackendConfig]) -> int:
    from llmbench_tui.runner import BenchmarkRunner

    exit_code = 0
    for cfg in configs:
        runner = BenchmarkRunner(cfg, asyncio.Queue())
        ok, msg = await runner.check_reachable()
        status = "OK" if ok else "FAIL"
        click.echo(f"[{status}] {cfg.name} ({cfg.base_url}): {msg}")
        if not ok:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    main()
