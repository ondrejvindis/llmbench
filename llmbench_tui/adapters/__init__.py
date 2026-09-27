"""Backend adapters, one per runtime wire protocol.

New runtimes are added here, not scattered across the UI/runner —
anything speaking either Ollama's native API or the OpenAI-compatible
streaming format needs zero new code, just a BackendConfig entry.
"""

from __future__ import annotations

from llmbench_tui.adapters.base import BackendAdapter
from llmbench_tui.adapters.ollama import OllamaAdapter
from llmbench_tui.adapters.openai_compat import OpenAICompatAdapter
from llmbench_tui.models import BackendConfig, RuntimeKind


def make_adapter(config: BackendConfig) -> BackendAdapter:
    """Construct the right adapter for a given backend's runtime kind."""
    if config.kind is RuntimeKind.OLLAMA:
        return OllamaAdapter(config)
    if config.kind is RuntimeKind.OPENAI_COMPAT:
        return OpenAICompatAdapter(config)
    raise ValueError(f"Unknown runtime kind: {config.kind!r}")


__all__ = [
    "BackendAdapter",
    "OllamaAdapter",
    "OpenAICompatAdapter",
    "make_adapter",
]
