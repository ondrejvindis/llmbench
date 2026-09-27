"""The live TUI dashboard, built with Textual.

Layout: one row of "sparkline scoreboard" cards per backend (tok/s, TTFT,
error count) up top, a host-resource strip (CPU/RAM/GPU/VRAM) below it,
and a scrolling log of the most recent samples at the bottom. Everything
updates in place — nothing scrolls the log-spam way a plain print() loop
would.
"""

from __future__ import annotations

import asyncio
import csv
import time
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Sparkline, Static

from llmbench_tui.host_monitor import sample_host
from llmbench_tui.models import BackendConfig, RunningStats, SampleMetric
from llmbench_tui.runner import BenchmarkRunner

HOST_POLL_INTERVAL_S = 1.0


class BackendCard(Vertical):
    """One scoreboard card: name, live tok/s sparkline, and key stats."""

    def __init__(self, backend_name: str) -> None:
        super().__init__(id=f"card-{_safe_id(backend_name)}")
        self.backend_name = backend_name
        self.stats = RunningStats()
        self.border_title = backend_name

    def compose(self) -> ComposeResult:
        yield Static("waiting for first sample…", id=f"summary-{_safe_id(self.backend_name)}")
        yield Sparkline([], id=f"spark-{_safe_id(self.backend_name)}")

    def update_with(self, sample: SampleMetric) -> None:
        self.stats.add(sample)
        summary = self.query_one(f"#summary-{_safe_id(self.backend_name)}", Static)
        spark = self.query_one(f"#spark-{_safe_id(self.backend_name)}", Sparkline)

        if sample.ok:
            tps = sample.tokens_per_second
            spark_data = list(spark.data) if spark.data else []
            if tps is not None:
                spark_data.append(tps)
                spark.data = spark_data[-60:]  # keep last 60 points

            avg_tps = self.stats.avg_tps
            avg_ttft = self.stats.avg_ttft
            text = Text()
            text.append(f"tok/s: ", style="dim")
            text.append(f"{tps:.1f}\n" if tps is not None else "n/a\n", style="bold green")
            text.append(f"avg tok/s: {avg_tps:.1f}\n" if avg_tps else "avg tok/s: n/a\n")
            text.append(f"TTFT: {sample.ttft_s:.2f}s\n" if sample.ttft_s is not None else "TTFT: n/a\n")
            text.append(f"avg TTFT: {avg_ttft:.2f}s\n" if avg_ttft else "avg TTFT: n/a\n")
            text.append(f"requests: {self.stats.count}  errors: {self.stats.errors}", style="dim")
            summary.update(text)
        else:
            text = Text()
            text.append("REQUEST FAILED\n", style="bold red")
            text.append(f"{sample.error}\n", style="red")
            text.append(f"requests: {self.stats.count}  errors: {self.stats.errors}", style="dim")
            summary.update(text)


class HostStrip(Static):
    """Bottom strip showing CPU/RAM/GPU/VRAM as the host runs the models."""

    def on_mount(self) -> None:
        self.update("host: sampling…")

    def update_sample(self, s) -> None:  # HostSample, avoiding import cycle noise
        cpu_desc = s.cpu_name or "CPU"
        parts = [
            f"CPU: {s.cpu_percent:5.1f}% ({cpu_desc})",
            f"RAM: {s.ram_used_gb:5.1f}/{s.ram_total_gb:.0f} GB",
        ]
        if s.gpu_percent is not None:
            gpu_desc = s.gpu_name or "GPU"
            parts.append(f"GPU: {s.gpu_percent:5.1f}% ({gpu_desc})")
            parts.append(f"VRAM: {s.vram_used_gb:5.1f}/{s.vram_total_gb:.0f} GB")
        else:
            parts.append("GPU: n/a (no NVIDIA GPU detected)")
        self.update("  |  ".join(parts))


def _safe_id(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name)


class LLMBenchApp(App):
    """Top-level Textual app wiring runners, host monitor, and widgets together."""

    CSS = """
    #cards {
        height: auto;
        margin: 1 0;
    }
    BackendCard {
        border: round $accent;
        padding: 0 1;
        width: 1fr;
        height: auto;
        margin: 0 1 0 0;
    }
    HostStrip {
        dock: bottom;
        height: 1;
        background: $panel;
        padding: 0 1;
    }
    #log-title {
        margin-top: 1;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("e", "export_csv", "Export CSV"),
    ]

    def __init__(
        self,
        configs: list[BackendConfig],
        max_tokens: int = 256,
        interval_s: float = 0.5,
        export_path: str | None = None,
    ) -> None:
        super().__init__()
        self.configs = configs
        self.max_tokens = max_tokens
        self.interval_s = interval_s
        self.export_path = export_path
        self.result_queue: asyncio.Queue[SampleMetric] = asyncio.Queue()
        self.runners = [
            BenchmarkRunner(
                cfg, self.result_queue, max_tokens=max_tokens, interval_s=interval_s
            )
            for cfg in configs
        ]
        self.all_samples: list[SampleMetric] = []
        self._cards: dict[str, BackendCard] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="cards"):
            for cfg in self.configs:
                card = BackendCard(cfg.name)
                self._cards[cfg.name] = card
                yield card
        yield Static("Recent requests", id="log-title")
        with VerticalScroll():
            yield DataTable(id="log-table")
        yield HostStrip()
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#log-table", DataTable)
        table.add_columns("time", "backend", "status", "tok/s", "TTFT (s)", "total (s)")
        table.zebra_stripes = True

        for runner in self.runners:
            self.run_worker(runner.run_forever(), exclusive=False)
        self.run_worker(self._consume_results(), exclusive=False)
        self.run_worker(self._poll_host(), exclusive=False)

    async def _consume_results(self) -> None:
        while True:
            sample = await self.result_queue.get()
            self.all_samples.append(sample)
            card = self._cards.get(sample.backend_name)
            if card is not None:
                card.update_with(sample)
            self._append_log_row(sample)

    def _append_log_row(self, sample: SampleMetric) -> None:
        table = self.query_one("#log-table", DataTable)
        ts = time.strftime("%H:%M:%S", time.localtime(sample.timestamp))
        status = Text("ok", style="green") if sample.ok else Text("error", style="bold red")
        tps = f"{sample.tokens_per_second:.1f}" if sample.tokens_per_second else "-"
        ttft = f"{sample.ttft_s:.2f}" if sample.ttft_s is not None else "-"
        total = f"{sample.total_s:.2f}"
        table.add_row(ts, sample.backend_name, status, tps, ttft, total)
        if table.row_count > 200:
            # Keep the visible table bounded so the dashboard stays snappy
            # on long runs; the CSV export still has everything via
            # self.all_samples regardless of what's trimmed here.
            oldest_key = next(iter(table.rows))
            table.remove_row(oldest_key)

    async def _poll_host(self) -> None:
        strip = self.query_one(HostStrip)
        while True:
            sample = await sample_host()
            strip.update_sample(sample)
            await asyncio.sleep(HOST_POLL_INTERVAL_S)

    def action_export_csv(self) -> None:
        path = Path(self.export_path or "llmbench-results.csv")
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "backend", "ok", "ttft_s", "total_s", "tokens_per_second",
                 "prompt_tokens", "completion_tokens", "error"]
            )
            for s in self.all_samples:
                writer.writerow(
                    [s.timestamp, s.backend_name, s.ok, s.ttft_s, s.total_s,
                     s.tokens_per_second, s.prompt_tokens, s.completion_tokens, s.error]
                )
        self.notify(f"Exported {len(self.all_samples)} samples to {path}")

    def action_quit(self) -> None:
        for runner in self.runners:
            runner.stop()
        self.exit()
