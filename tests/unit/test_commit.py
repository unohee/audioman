# tests/unit/test_commit.py — destructive commit: plugin chain + delay compensation.
#
# This host has no registrable VST3 plugins (AUD-1857), so the registry and the
# plugin wrapper are replaced by fakes. The fake wrapper is a deterministic
# integer delay line, which lets the delay-compensation and tail-trim logic be
# asserted on exact sample indices.

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from audioman.core import commit
from audioman.core.latency import LatencyMeasurement
from audioman.core.pipeline import PipelineStep
from audioman.plugins.parameter import PluginMeta

SR = 48000


class _FakeWrapper:
    """Deterministic delay line: out[n] = in[n - delay] * gain."""

    def __init__(self, path, delay=0, gain=1.0, tail=0, reported=None):
        self.path = path
        self._delay = delay
        self._gain = gain
        self._tail = tail
        self._plugin = type("P", (), {"latency_samples": reported})() if reported is not None else object()
        self.params = None
        self.loaded = False

    def load(self):
        self.loaded = True

    def set_parameters(self, params):
        self.params = params

    def process(self, audio, sample_rate, reset=True):
        out = np.roll(audio, self._delay, axis=-1)
        if self._delay > 0:
            if audio.ndim == 1:
                out[: self._delay] = 0.0
            else:
                out[:, : self._delay] = 0.0
        out = out * self._gain
        if self._tail > 0:
            pad_shape = (out.shape[0], self._tail) if out.ndim == 2 else (self._tail,)
            out = np.concatenate([out, np.zeros(pad_shape, dtype=out.dtype)], axis=-1)
        return out.astype(np.float32)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Patch registry/wrapper IO; record wrapper construction order."""
    made: list[_FakeWrapper] = []
    meta_by_name = {
        "delay": PluginMeta(name="Delay", short_name="delay", path="/tmp/Delay.vst3", format="vst3"),
        "tail": PluginMeta(name="Tail", short_name="tail", path="/tmp/Tail.vst3", format="vst3"),
    }
    delays = {"delay": 100, "tail": 0}

    class _Registry:
        def get(self, name):
            return meta_by_name.get(name)

    def _wrapper(path):
        name = path.split("/")[-1].split(".")[0].lower()
        w = _FakeWrapper(path, delay=delays.get(name, 0))
        made.append(w)
        return w

    monkeypatch.setattr(commit, "get_registry", lambda: _Registry())
    monkeypatch.setattr(commit, "VST3PluginWrapper", _wrapper)

    def _fake_measure(steps, sample_rate=48000):
        total = 0
        ms = []
        for s in steps:
            n = s.plugin_name if isinstance(s, PipelineStep) else s.get("plugin", "")
            ms.append(LatencyMeasurement(n, 0, delays.get(n, 0), 1.0, delays.get(n, 0)))
            total += delays.get(n, 0)
        return ms, total

    monkeypatch.setattr(commit, "measure_chain_latency", _fake_measure)
    return type("Env", (), {"made": made, "delays": delays, "monkeypatch": monkeypatch})


def _write(path, audio, sr=SR):
    sf.write(str(path), audio.T if audio.ndim == 2 else audio, sr, subtype="FLOAT")
    return path


class TestCommitFile:
    def test_applies_chain_and_compensates_delay(self, env, tmp_path):
        # impulse at sample 10 so delay compensation is unambiguous
        audio = np.zeros((2, 1000), dtype=np.float32)
        audio[:, 10] = 1.0
        src = _write(tmp_path / "in.wav", audio)
        out = tmp_path / "out.wav"

        result = commit.commit_file(src, out, [PipelineStep("delay", {})])

        assert result.total_latency_samples == 100
        assert len(result.latency_compensation) == 1
        assert result.output_path == str(out)
        rendered, _ = sf.read(str(out), always_2d=True)
        # 100-sample latency compensated → impulse back at sample 10, same length
        assert rendered.shape[0] == 1000
        assert int(np.argmax(np.abs(rendered[:, 0]))) == 10

    def test_compensate_false_leaves_latency(self, env, tmp_path):
        audio = np.zeros((2, 1000), dtype=np.float32)
        audio[:, 10] = 1.0
        src = _write(tmp_path / "in.wav", audio)
        out = tmp_path / "out.wav"

        result = commit.commit_file(src, out, [PipelineStep("delay", {})], compensate_latency=False)

        assert result.total_latency_samples == 0
        assert result.latency_compensation == []
        rendered, _ = sf.read(str(out), always_2d=True)
        assert int(np.argmax(np.abs(rendered[:, 0]))) == 110

    def test_tail_trim_restores_original_length(self, env, tmp_path):
        audio = np.ones((2, 500), dtype=np.float32) * 0.1
        src = _write(tmp_path / "in.wav", audio)
        out = tmp_path / "out.wav"

        result = commit.commit_file(src, out, [PipelineStep("tail", {})], compensate_latency=False)

        rendered, _ = sf.read(str(out), always_2d=True)
        assert rendered.shape[0] == 500  # tail trimmed back to source length

    def test_tail_trim_false_keeps_grown_length(self, env, tmp_path):
        audio = np.ones((2, 500), dtype=np.float32) * 0.1
        src = _write(tmp_path / "in.wav", audio)
        out = tmp_path / "out.wav"

        # give the chain a tail by injecting extra samples through the wrapper
        env.monkeypatch.setattr(
            commit, "VST3PluginWrapper",
            lambda path: _FakeWrapper(path, delay=0, tail=200),
        )
        commit.commit_file(src, out, [PipelineStep("tail", {})], compensate_latency=False, tail_trim=False)

        rendered, _ = sf.read(str(out), always_2d=True)
        assert rendered.shape[0] == 700

    def test_mono_audio_tail_trim(self, env, tmp_path):
        audio = np.ones(500, dtype=np.float32) * 0.1
        src = _write(tmp_path / "in.wav", audio)
        out = tmp_path / "out.wav"
        env.monkeypatch.setattr(
            commit, "VST3PluginWrapper",
            lambda path: _FakeWrapper(path, delay=0, tail=200),
        )
        commit.commit_file(src, out, [PipelineStep("tail", {})], compensate_latency=False)
        rendered, _ = sf.read(str(out))
        assert rendered.shape[0] == 500

    def test_params_forwarded_to_wrapper(self, env, tmp_path):
        audio = np.ones((2, 200), dtype=np.float32) * 0.1
        src = _write(tmp_path / "in.wav", audio)
        commit.commit_file(src, tmp_path / "out.wav",
                           [PipelineStep("delay", {"mix": 0.5})], compensate_latency=False)
        assert env.made[0].params == {"mix": 0.5}
        assert env.made[0].loaded

    def test_unknown_plugin_raises(self, env, tmp_path):
        src = _write(tmp_path / "in.wav", np.ones((2, 100), dtype=np.float32) * 0.1)
        with pytest.raises(ValueError, match="플러그인을 찾을 수 없습니다"):
            commit.commit_file(src, tmp_path / "out.wav", [PipelineStep("missing", {})])

    def test_result_to_dict(self, env, tmp_path):
        src = _write(tmp_path / "in.wav", np.ones((2, 100), dtype=np.float32) * 0.1)
        result = commit.commit_file(src, tmp_path / "out.wav", [PipelineStep("delay", {})])
        d = result.to_dict()
        assert d["input_path"] == str(src)
        assert d["total_latency_samples"] == 100
        assert isinstance(d["steps"], list)
        assert isinstance(d["output_stats"], dict)


class TestDryRunCommit:
    def test_measures_without_processing(self, env, tmp_path):
        steps = [PipelineStep("delay", {}), PipelineStep("tail", {})]
        measurements, total = commit.dry_run_commit(steps, sample_rate=44100)
        assert total == 100
        assert [m.plugin_name for m in measurements] == ["delay", "tail"]


class TestMonoPath:
    def test_mono_tail_trim(self, env, monkeypatch, tmp_path):
        """1D in-memory audio takes the mono slice branch of tail trim."""
        env.monkeypatch.setattr(
            commit, "VST3PluginWrapper", lambda path: _FakeWrapper(path, tail=200)
        )
        env.monkeypatch.setattr(
            commit, "read_audio", lambda p: (np.ones(500, dtype=np.float32) * 0.1, SR)
        )
        result = commit.commit_file(
            tmp_path / "in.wav", tmp_path / "out.wav",
            [PipelineStep("tail", {})], compensate_latency=False,
        )
        assert result.output_stats["frames"] == 500
