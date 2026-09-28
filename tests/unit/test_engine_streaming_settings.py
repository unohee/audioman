# tests/unit/test_engine_streaming_settings.py — core/engine.py settings wiring.
#
# Verifies that large_file_threshold_mb / auto_stream / default_chunk_size are
# actually read (they used to be declared but never consumed), using an injected
# settings stub and a fake wrapper so no VST3 plugin is needed.

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from audioman.core import engine
from audioman.core.audio_file import AudioStats
from audioman.plugins.parameter import PluginMeta


def _settings(**overrides):
    values = {
        "large_file_threshold_mb": 500,
        "auto_stream": True,
        "default_chunk_size": 441000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeWrapper:
    def __init__(self, path):
        self.path = path
        self.params = None

    def load(self):
        pass

    def set_parameters(self, params):
        self.params = params

    def process(self, audio, sample_rate, reset=True):
        return audio


def _stats(audio, sample_rate):
    return AudioStats(
        duration=1.0, sample_rate=sample_rate, channels=2, frames=100,
        peak=0.0, rms=0.0, format="float32",
    )


@pytest.fixture
def engine_env(monkeypatch):
    """Patch out the registry, wrapper, file IO and record stream_process calls."""
    meta = PluginMeta(name="Fake", short_name="fake", path="/tmp/Fake.vst3", format="vst3")
    calls: dict = {}

    def fake_stream_process(input_path, output_path, process_fn, chunk_seconds=10.0, subtype="PCM_24"):
        calls["chunk_seconds"] = chunk_seconds
        return {"frames_processed": 100, "duration": 0.1, "chunks": 2}

    monkeypatch.setattr(engine, "get_registry", lambda: SimpleNamespace(get=lambda name: meta))
    monkeypatch.setattr(engine, "VST3PluginWrapper", _FakeWrapper)
    monkeypatch.setattr(engine, "stream_process", fake_stream_process)
    monkeypatch.setattr(engine, "read_audio", lambda p: (np.zeros((2, 100), np.float32), 44100))
    monkeypatch.setattr(engine, "get_audio_stats", _stats)
    monkeypatch.setattr(engine, "write_audio", lambda p, a, sr: None)
    monkeypatch.setattr(engine, "get_file_info", lambda p: {
        "file_size_mb": 900.0, "duration": 1.0, "sample_rate": 44100,
        "channels": 2, "frames": 100,
    })
    return SimpleNamespace(calls=calls, monkeypatch=monkeypatch)


def _run(engine_env, **settings_overrides):
    with patch.object(engine, "get_settings", lambda: _settings(**settings_overrides)):
        return engine.process_file("/tmp/in.wav", "/tmp/out.wav", "fake")


class TestStreamThresholdHelper:
    def test_reads_setting(self):
        with patch.object(engine, "get_settings", lambda: _settings(large_file_threshold_mb=123)):
            assert engine._stream_threshold_mb() == 123

    def test_falls_back_when_settings_broken(self):
        def boom():
            raise RuntimeError("no settings")

        with patch.object(engine, "get_settings", boom):
            assert engine._stream_threshold_mb() == 500


class TestChunkSecondsHelper:
    def test_frame_count_converted_per_sample_rate(self):
        with patch.object(engine, "get_settings", lambda: _settings(default_chunk_size=441000)):
            assert engine._stream_chunk_seconds(44100) == pytest.approx(10.0)
            assert engine._stream_chunk_seconds(48000) == pytest.approx(441000 / 48000)

    @pytest.mark.parametrize("sr", [0, -1])
    def test_invalid_sample_rate_falls_back(self, sr):
        with patch.object(engine, "get_settings", lambda: _settings()):
            assert engine._stream_chunk_seconds(sr) == 10.0


class TestAutoStreamWiring:
    def test_large_file_streams_by_default(self, engine_env):
        result = _run(engine_env)
        assert result.output_stats["streamed"] is True
        assert engine_env.calls["chunk_seconds"] == pytest.approx(10.0)

    def test_auto_stream_false_forces_offline(self, engine_env):
        result = _run(engine_env, auto_stream=False)
        assert "streamed" not in result.output_stats
        assert "chunk_seconds" not in engine_env.calls

    def test_threshold_above_file_size_forces_offline(self, engine_env):
        result = _run(engine_env, large_file_threshold_mb=1000)
        assert "streamed" not in result.output_stats

    def test_chunk_size_setting_reaches_stream_process(self, engine_env):
        result = _run(engine_env, default_chunk_size=220500)
        assert result.output_stats["streamed"] is True
        assert engine_env.calls["chunk_seconds"] == pytest.approx(5.0)
