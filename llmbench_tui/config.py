"""Loads backend definitions from a small YAML/JSON config file, or builds
them from CLI flags for the common single/dual-backend case where writing
a config file would be overkill.
"""

from __future__ import annotations

import json
from pathlib import Path

from llmbench_tui.models import BackendConfig, RuntimeKind

_KIND_ALIASES = {
    "ollama": RuntimeKind.OLLAMA,
    "openai": RuntimeKind.OPENAI_COMPAT,
    "openai_compat": RuntimeKind.OPENAI_COMPAT,
    "llama.cpp": RuntimeKind.OPENAI_COMPAT,
    "llamacpp": RuntimeKind.OPENAI_COMPAT,
    "vllm": RuntimeKind.OPENAI_COMPAT,
    "lmstudio": RuntimeKind.OPENAI_COMPAT,
}


def _parse_kind(raw: str) -> RuntimeKind:
    try:
        return _KIND_ALIASES[raw.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(_KIND_ALIASES))
        raise ValueError(f"Unknown backend kind '{raw}'. Valid: {valid}") from exc


def load_config_file(path: str | Path) -> list[BackendConfig]:
    """Load backend definitions from a JSON (or YAML-as-JSON-subset) file.

    Kept to plain JSON parsing (stdlib-only) to avoid a PyYAML dependency
    for a config format simple enough not to need it; ``.yaml``/``.yml``
    files are accepted too as long as they're written as flow-style JSON,
    which is valid YAML.
    """
    p = Path(path)
    data = json.loads(p.read_text())
    backends = []
    for entry in data.get("backends", []):
        backends.append(
            BackendConfig(
                name=entry.get("name") or f"{entry['kind']}/{entry['model']}",
                kind=_parse_kind(entry["kind"]),
                base_url=entry["base_url"].rstrip("/"),
                model=entry["model"],
                api_key=entry.get("api_key"),
                timeout_s=float(entry.get("timeout_s", 120.0)),
            )
        )
    if not backends:
        raise ValueError(f"No backends defined in {p}")
    return backends


def backend_from_flag(spec: str) -> BackendConfig:
    """Parse a compact CLI spec: ``kind:model@base_url``.

    Example: ``ollama:llama3.1:8b@http://localhost:11434``
             ``llama.cpp:local-model@http://localhost:8080``
    The base_url defaults per-kind if omitted, since Ollama and llama.cpp
    servers both have well-known default ports.
    """
    if "@" in spec:
        left, base_url = spec.rsplit("@", 1)
    else:
        left, base_url = spec, ""

    if ":" not in left:
        raise ValueError(
            f"Invalid backend spec '{spec}'. Expected kind:model[@base_url]"
        )
    kind_str, model = left.split(":", 1)
    kind = _parse_kind(kind_str)

    if not base_url:
        base_url = (
            "http://localhost:11434" if kind is RuntimeKind.OLLAMA
            else "http://localhost:8080"
        )

    return BackendConfig(
        name=f"{kind_str}/{model}",
        kind=kind,
        base_url=base_url.rstrip("/"),
        model=model,
    )
