# tests/unit/test_engine.py — core/engine.py process_file: param parsing, offline and
# streaming plugin paths.
#
# No registrable VST3 plugins on this host (AUD-1857): registry and wrapper are faked.
# The streaming-threshold wiring itself is covered in test_engine_streaming_settings.py.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from audioman.core import engine
from audioman.core.audio_file import AudioStats
from audioman.plugins.parameter import PluginMeta

SR = 48000


class TestParseParams:
    def test_float_and_bool_and_string(self):
        params = engine.parse_params(["threshold=-20", "enabled=true", "name=vocal"])
        assert params == {"threshold": -20.0, "enabled": True, "name": "vocal"}

    def test_false_bool(self):
        assert engine.parse_params(["lookahead=false"]) == {"lookahead": False}

    def test_quoted_values_keep_string(self):
        assert engine.parse_params(['ratio="4.00"', "mode='0.97'"]) == {
            "ratio": "4.00", "mode": "0.97",
        }

    def test_string_with_equals_in_value(self):
        assert engine.parse_params(["expr=a=b"]) == {"expr": "a=b"}

    def test_key_whitespace_stripped(self):
        assert engine.parse_params(["  gain = -3"]) == {"gain": -3.0}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="key=value"):
            engine.parse_params(["threshold-20"])

    def test_empty_list(self):
        assert engine.parse_params([]) == {}


class TestProcessResultDict:
    def test_to_dict(self):
        r = engine.ProcessResult("/in.wav", "/out.wav", "p", {"a": 1}, {"peak": 0.5},
                                 {"peak": 0.4}, 1.25)
        d = r.to_dict()
        assert d["plugin_name"] == "p"
        assert d["duration_seconds"] == 1.25


class _FakeWrapper:
    """Deterministic half-gain plugin, no VST3 needed."""

    instances: list["_FakeWrapper"] = []

    def __init__(self, path):
        self.path = path
        self.params = None
        self.processed = 0
        _FakeWrapper.instances.append(self)

    def load(self):
        pass

    def set_parameters(self, params):
        self.params = params

    def process(self, audio, sample_rate, reset=True):
        self.processed += 1
        return (audio * 0.5).astype(np.float32)


def _stats(audio, sample_rate):
    frames = audio.shape[-1] if audio.ndim == 2 else len(audio)
    return AudioStats(
        duration=frames / sample_rate, sample_rate=sample_rate,
        channels=1 if audio.ndim == 1 else audio.shape[0], frames=frames,
        peak=float(np.max(np.abs(audio))), rms=float(np.sqrt(np.mean(audio ** 2))),
        format="float32",
    )


@pytest.fixture
def offline_env(monkeypatch):
    _FakeWrapper.instances = []
    meta = PluginMeta(name="Fake", short_name="fake", path="/tmp/Fake.vst3", format="vst3")
    monkeypatch.setattr(engine, "get_registry", lambda: SimpleNamespace(get=lambda n: meta))
    monkeypatch.setattr(engine, "VST3PluginWrapper", _FakeWrapper)
    monkeypatch.setattr(engine, "get_audio_stats", _stats)
    monkeypatch.setattr(engine, "get_file_info", lambda p: {
        "file_size_mb": 1.0, "duration": 1.0, "sample_rate": SR,
        "channels": 2, "frames": 100,
    })
    # force offline regardless of host settings
    monkeypatch.setattr(engine, "_auto_stream_enabled", lambda: False)
    return SimpleNamespace(monkeypatch=monkeypatch)


class TestProcessFileOffline:
    def test_applies_plugin_and_writes(self, offline_env, tmp_path):
        src = tmp_path / "in.wav"
        mono = np.full(200, 0.4, dtype=np.float32)
        sf.write(str(src), mono, SR, subtype="FLOAT")
        out = tmp_path / "out.wav"

        result = engine.process_file(src, out, "fake")
        assert out.exists()
        assert result.plugin_name == "fake"
        rendered, _ = sf.read(str(out))
        np.testing.assert_allclose(rendered, 0.2, atol=1e-5)

    def test_params_forwarded(self, offline_env, tmp_path):
        src = tmp_path / "in.wav"
        sf.write(str(src), np.full(100, 0.1, dtype=np.float32), SR, subtype="FLOAT")
        engine.process_file(src, tmp_path / "o.wav", "fake", params={"gain": -3.0})
        assert _FakeWrapper.instances[-1].params == {"gain": -3.0}

    def test_multi_pass_processes_repeatedly(self, offline_env, tmp_path):
        src = tmp_path / "in.wav"
        sf.write(str(src), np.full(100, 0.8, dtype=np.float32), SR, subtype="FLOAT")
        out = tmp_path / "o.wav"
        result = engine.process_file(src, out, "fake", passes=3)
        # each pass re-processes the original input; only the last pass is written
        assert _FakeWrapper.instances[-1].processed == 3
        rendered, _ = sf.read(str(out))
        np.testing.assert_allclose(rendered, 0.8 * 0.5, atol=1e-5)
        assert result.params_applied == {}

    def test_unknown_plugin_raises(self, offline_env, tmp_path):
        offline_env.monkeypatch.setattr(
            engine, "get_registry", lambda: SimpleNamespace(get=lambda n: None)
        )
        with pytest.raises(ValueError, match="플러그인을 찾을 수 없습니다"):
            engine.process_file(tmp_path / "in.wav", tmp_path / "o.wav", "missing")

    def test_file_info_failure_falls_back_offline(self, offline_env, tmp_path):
        def _boom(p):
            raise OSError("no stat")

        offline_env.monkeypatch.setattr(engine, "get_file_info", _boom)
        offline_env.monkeypatch.setattr(engine, "_auto_stream_enabled", lambda: True)
        src = tmp_path / "in.wav"
        sf.write(str(src), np.full(100, 0.5, dtype=np.float32), SR, subtype="FLOAT")
        result = engine.process_file(src, tmp_path / "o.wav", "fake")
        assert "streamed" not in result.output_stats


class TestProcessFileStreaming:
    def test_stream_true_uses_stream_process(self, monkeypatch, tmp_path):
        _FakeWrapper.instances = []
        meta = PluginMeta(name="Fake", short_name="fake", path="/tmp/Fake.vst3", format="vst3")
        calls = {}

        def fake_stream_process(input_path, output_path, process_fn, chunk_seconds=10.0,
                                subtype="PCM_24"):
            calls["chunk_seconds"] = chunk_seconds
            # exercise the chunk callback so the wrapper's process path is covered
            out = process_fn(np.full((2, 64), 0.4, dtype=np.float32), SR)
            calls["chunk_out"] = out
            return {"frames_processed": 100, "duration": 1.0, "chunks": 2}

        monkeypatch.setattr(engine, "get_registry", lambda: SimpleNamespace(get=lambda n: meta))
        monkeypatch.setattr(engine, "VST3PluginWrapper", _FakeWrapper)
        monkeypatch.setattr(engine, "stream_process", fake_stream_process)
        monkeypatch.setattr(engine, "get_file_info", lambda p: {
            "file_size_mb": 900.0, "duration": 2.0, "sample_rate": SR,
            "channels": 2, "frames": SR * 2,
        })

        result = engine.process_file(tmp_path / "in.wav", tmp_path / "o.wav", "fake",
                                     params={"x": 1}, stream=True)
        assert result.output_stats["streamed"] is True
        assert result.output_stats["frames_processed"] == 100
        assert calls["chunk_seconds"] == pytest.approx(441000 / SR)
        assert _FakeWrapper.instances[-1].params == {"x": 1}
        np.testing.assert_allclose(calls["chunk_out"], 0.2, atol=1e-5)
        assert result.input_stats["sample_rate"] == SR


class TestAutoStreamDecision:
    """stream=None → decide from file size vs settings threshold."""

    def _env(self, monkeypatch, file_mb, auto=True, threshold=500):
        _FakeWrapper.instances = []
        meta = PluginMeta(name="Fake", short_name="fake", path="/tmp/Fake.vst3", format="vst3")
        stream_calls = {}

        def fake_stream_process(input_path, output_path, process_fn, chunk_seconds=10.0,
                                subtype="PCM_24"):
            stream_calls["hit"] = True
            return {"frames_processed": 10, "duration": 1.0, "chunks": 1}

        monkeypatch.setattr(engine, "get_registry", lambda: SimpleNamespace(get=lambda n: meta))
        monkeypatch.setattr(engine, "VST3PluginWrapper", _FakeWrapper)
        monkeypatch.setattr(engine, "stream_process", fake_stream_process)
        monkeypatch.setattr(engine, "get_audio_stats", _stats)
        monkeypatch.setattr(engine, "get_file_info", lambda p: {
            "file_size_mb": file_mb, "duration": 1.0, "sample_rate": SR,
            "channels": 2, "frames": 100,
        })
        monkeypatch.setattr(engine, "_auto_stream_enabled", lambda: auto)
        monkeypatch.setattr(engine, "_stream_threshold_mb", lambda: threshold)
        return stream_calls

    def test_large_file_streams_when_auto_enabled(self, monkeypatch, tmp_path):
        calls = self._env(monkeypatch, file_mb=900.0)
        src = tmp_path / "in.wav"
        sf.write(str(src), np.full(100, 0.5, dtype=np.float32), SR, subtype="FLOAT")
        result = engine.process_file(src, tmp_path / "o.wav", "fake")
        assert calls.get("hit") is True
        assert result.output_stats["streamed"] is True

    def test_small_file_stays_offline(self, monkeypatch, tmp_path):
        calls = self._env(monkeypatch, file_mb=10.0)
        src = tmp_path / "in.wav"
        sf.write(str(src), np.full(100, 0.5, dtype=np.float32), SR, subtype="FLOAT")
        result = engine.process_file(src, tmp_path / "o.wav", "fake")
        assert "hit" not in calls
        assert "streamed" not in result.output_stats

    def test_auto_disabled_stays_offline(self, monkeypatch, tmp_path):
        calls = self._env(monkeypatch, file_mb=900.0, auto=False)
        src = tmp_path / "in.wav"
        sf.write(str(src), np.full(100, 0.5, dtype=np.float32), SR, subtype="FLOAT")
        result = engine.process_file(src, tmp_path / "o.wav", "fake")
        assert "hit" not in calls
        assert "streamed" not in result.output_stats

    def test_threshold_from_settings(self, monkeypatch, tmp_path):
        calls = self._env(monkeypatch, file_mb=200.0, threshold=100)
        src = tmp_path / "in.wav"
        sf.write(str(src), np.full(100, 0.5, dtype=np.float32), SR, subtype="FLOAT")
        result = engine.process_file(src, tmp_path / "o.wav", "fake")
        assert result.output_stats["streamed"] is True


class TestSettingsHelperFallbacks:
    def test_helpers_fall_back_on_broken_settings(self, monkeypatch):
        def boom():
            raise RuntimeError("no settings")

        monkeypatch.setattr(engine, "get_settings", boom)
        assert engine._stream_threshold_mb() == 500
        assert engine._auto_stream_enabled() is True
        assert engine._stream_chunk_seconds(48000) == 10.0

    def test_chunk_seconds_nonpositive_frames(self, monkeypatch):
        monkeypatch.setattr(
            engine, "get_settings",
            lambda: SimpleNamespace(default_chunk_size=0, large_file_threshold_mb=500,
                                    auto_stream=True),
        )
        assert engine._stream_chunk_seconds(SR) == 10.0
