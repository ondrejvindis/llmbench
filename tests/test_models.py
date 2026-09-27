from llmbench_tui.config import backend_from_flag
from llmbench_tui.models import HostSample, RuntimeKind, SampleMetric, RunningStats


def test_host_sample_keeps_device_names():
    sample = HostSample(
        cpu_percent=42.0,
        ram_used_gb=11.4,
        ram_total_gb=32.0,
        cpu_name="Intel Core i7-12700K",
        gpu_name="NVIDIA GeForce RTX 4070",
    )
    assert sample.cpu_name == "Intel Core i7-12700K"
    assert sample.gpu_name == "NVIDIA GeForce RTX 4070"


def test_tokens_per_second_excludes_ttft():
    s = SampleMetric(
        backend_name="x", ok=True, ttft_s=0.5, total_s=2.5,
        prompt_tokens=10, completion_tokens=80,
    )
    # generation-only time is 2.0s -> 80/2.0 = 40 tok/s
    assert s.tokens_per_second == 40.0


def test_tokens_per_second_none_when_failed():
    s = SampleMetric(
        backend_name="x", ok=False, ttft_s=None, total_s=1.0,
        prompt_tokens=0, completion_tokens=0, error="boom",
    )
    assert s.tokens_per_second is None


def test_running_stats_aggregates_and_ignores_errors():
    stats = RunningStats()
    stats.add(SampleMetric(backend_name="x", ok=True, ttft_s=0.1, total_s=1.1,
                            prompt_tokens=5, completion_tokens=100))
    stats.add(SampleMetric(backend_name="x", ok=True, ttft_s=0.3, total_s=1.3,
                            prompt_tokens=5, completion_tokens=100))
    stats.add(SampleMetric(backend_name="x", ok=False, ttft_s=None, total_s=0.2,
                            prompt_tokens=0, completion_tokens=0, error="timeout"))

    assert stats.count == 3
    assert stats.errors == 1
    assert stats.ttft_n == 2
    assert abs(stats.avg_ttft - 0.2) < 1e-9


def test_backend_from_flag_ollama_default_url():
    cfg = backend_from_flag("ollama:llama3.1:8b")
    assert cfg.kind is RuntimeKind.OLLAMA
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.model == "llama3.1:8b"


def test_backend_from_flag_llamacpp_explicit_url():
    cfg = backend_from_flag("llama.cpp:local-7b@http://localhost:8080")
    assert cfg.kind is RuntimeKind.OPENAI_COMPAT
    assert cfg.base_url == "http://localhost:8080"
    assert cfg.model == "local-7b"


def test_backend_from_flag_invalid_spec_raises():
    import pytest
    with pytest.raises(ValueError):
        backend_from_flag("no-colon-here")


def test_backend_from_flag_unknown_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        backend_from_flag("unknownkind:model")
