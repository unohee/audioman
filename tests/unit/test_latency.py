# tests/unit/test_latency.py — 레이턴시 측정 + compensation 테스트

import numpy as np
import pytest

from audioman.core.latency import apply_delay_compensation


class TestApplyDelayCompensation:
    """delay compensation 알고리즘 단위 테스트"""

    def test_stereo_compensation(self):
        """스테레오 오디오에서 레이턴시만큼 앞부분 제거 + zero-pad"""
        sr = 48000
        n = sr  # 1초
        audio = np.random.randn(2, n).astype(np.float32)
        latency = 100

        result = apply_delay_compensation(audio, latency)

        assert result.shape == audio.shape
        # 앞부분은 원래 latency 이후의 데이터
        np.testing.assert_array_equal(result[:, :n - latency], audio[:, latency:])
        # 뒤에 zero-pad
        np.testing.assert_array_equal(result[:, n - latency:], 0.0)

    def test_mono_compensation(self):
        """모노 오디오에서 compensation"""
        n = 1000
        audio = np.random.randn(n).astype(np.float32)
        latency = 50

        result = apply_delay_compensation(audio, latency)

        assert result.shape == audio.shape
        np.testing.assert_array_equal(result[:n - latency], audio[latency:])
        np.testing.assert_array_equal(result[n - latency:], 0.0)

    def test_zero_latency(self):
        """레이턴시 0이면 원본 그대로 반환"""
        audio = np.random.randn(2, 1000).astype(np.float32)

        result = apply_delay_compensation(audio, 0)

        np.testing.assert_array_equal(result, audio)

    def test_negative_latency(self):
        """음수 레이턴시도 원본 그대로"""
        audio = np.random.randn(2, 1000).astype(np.float32)

        result = apply_delay_compensation(audio, -10)

        np.testing.assert_array_equal(result, audio)

    def test_latency_exceeds_length(self):
        """레이턴시가 오디오 길이보다 크면 전체 silence"""
        audio = np.ones((2, 100), dtype=np.float32)

        result = apply_delay_compensation(audio, 200)

        assert result.shape == audio.shape
        np.testing.assert_array_equal(result, 0.0)

    def test_impulse_alignment(self):
        """임펄스가 delay된 신호를 compensation으로 원위치 복원"""
        n = 48000
        latency = 256
        # 원본: sample 0에 임펄스
        original = np.zeros((2, n), dtype=np.float32)
        original[:, 0] = 1.0

        # 지연된 신호: sample latency에 임펄스
        delayed = np.zeros((2, n), dtype=np.float32)
        delayed[:, latency] = 1.0

        compensated = apply_delay_compensation(delayed, latency)

        # 임펄스가 sample 0으로 복원되어야 함
        assert compensated[0, 0] == pytest.approx(1.0)
        assert compensated[1, 0] == pytest.approx(1.0)
        # 나머지는 0
        assert np.max(np.abs(compensated[:, 1:])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Plugin-hosted latency measurement (fake wrapper — no VST3 on this host)
# ---------------------------------------------------------------------------


class _FakePlugin:
    def __init__(self, latency_samples=None, raise_on_get=False):
        if latency_samples is not None:
            self.latency_samples = latency_samples
        if raise_on_get:
            self._raise = True


class _FakeWrapper:
    """Deterministic delay line: output[n] = input[n - delay]."""

    def __init__(self, name="Fake", path="/tmp/Fake.vst3", delay=0,
                 reported=None, silent=False, gain=1.0):
        self.name = name
        self.path = path
        self._delay = delay
        self._plugin = _FakePlugin(latency_samples=reported)
        self._silent = silent
        self._gain = gain
        self.loaded = False
        self.reset_called = False
        self.params_set = None

    def load(self):
        self.loaded = True

    def reset(self):
        self.reset_called = True

    def set_parameters(self, params):
        self.params_set = params

    def process(self, audio, sample_rate, reset=True):
        if self._silent:
            return np.zeros_like(audio)
        shifted = np.roll(audio, self._delay, axis=-1)
        if self._delay > 0:
            if audio.ndim == 1:
                shifted[: self._delay] = 0.0
            else:
                shifted[:, : self._delay] = 0.0
        return (shifted * self._gain).astype(np.float32)


class TestGetReportedLatency:
    def test_returns_reported_value(self):
        from audioman.core.latency import _get_reported_latency
        w = _FakeWrapper(reported=128)
        assert _get_reported_latency(w) == 128

    def test_missing_attribute_returns_zero(self):
        from audioman.core.latency import _get_reported_latency
        w = _FakeWrapper(reported=None)
        assert _get_reported_latency(w) == 0

    def test_non_convertible_returns_zero(self):
        from audioman.core.latency import _get_reported_latency

        class _Bad:
            latency_samples = object()

        w = _FakeWrapper()
        w._plugin = _Bad()
        assert _get_reported_latency(w) == 0


class TestMeasurePluginLatency:
    def test_measures_impulse_delay(self):
        from audioman.core.latency import measure_plugin_latency
        w = _FakeWrapper(delay=64)
        m = measure_plugin_latency(w, sample_rate=48000)
        assert w.loaded and w.reset_called
        assert m.measured_latency == 64
        assert m.used_latency == 64
        assert m.confidence == pytest.approx(1.0)
        assert m.plugin_name == "Fake"

    def test_mono_output_path(self):
        from audioman.core.latency import measure_plugin_latency

        class _MonoWrapper(_FakeWrapper):
            def process(self, audio, sample_rate, reset=True):
                shifted = np.roll(audio[0], self._delay)
                shifted[: self._delay] = 0.0
                return shifted.astype(np.float32)

        w = _MonoWrapper(delay=32)
        m = measure_plugin_latency(w, sample_rate=48000)
        assert m.measured_latency == 32

    def test_silent_plugin_low_confidence_falls_back_to_reported(self):
        from audioman.core.latency import measure_plugin_latency
        w = _FakeWrapper(silent=True, reported=256)
        m = measure_plugin_latency(w, sample_rate=48000)
        assert m.confidence == 0.0
        assert m.used_latency == 256  # reported preferred when measurement unusable

    def test_silent_plugin_no_reported_uses_measurement(self):
        from audioman.core.latency import measure_plugin_latency
        w = _FakeWrapper(silent=True, reported=None)
        m = measure_plugin_latency(w, sample_rate=48000)
        assert m.used_latency == m.measured_latency

    def test_reported_mismatch_still_uses_measured_when_confident(self):
        from audioman.core.latency import measure_plugin_latency
        w = _FakeWrapper(delay=64, reported=999)
        m = measure_plugin_latency(w, sample_rate=48000)
        assert m.reported_latency == 999
        assert m.used_latency == m.measured_latency == 64

    def test_to_dict_roundtrip(self):
        from audioman.core.latency import LatencyMeasurement
        m = LatencyMeasurement("p", 1, 2, 0.5, 2)
        assert m.to_dict()["used_latency"] == 2


class TestMeasureChainLatency:
    def test_pipeline_step_objects(self, monkeypatch):
        from audioman.core import latency as lat
        from audioman.core.pipeline import PipelineStep
        from audioman.plugins.parameter import PluginMeta

        meta = PluginMeta(name="Fake", short_name="fake", path="/tmp/Fake.vst3", format="vst3")
        made = []

        def _wrapper(path):
            w = _FakeWrapper(name="Fake", path=path, delay=48)
            made.append(w)
            return w

        monkeypatch.setattr(lat, "get_registry", lambda: type("R", (), {"get": lambda s, n: meta})())
        monkeypatch.setattr(lat, "VST3PluginWrapper", _wrapper)

        steps = [PipelineStep("fake", {"a": 1}), PipelineStep("fake", {})]
        measurements, total = lat.measure_chain_latency(steps, sample_rate=48000)
        assert len(measurements) == 2
        assert total == 96
        assert measurements[0].plugin_name == "fake"
        assert made[0].params_set == {"a": 1}
        assert made[1].params_set is None

    def test_dict_steps(self, monkeypatch):
        from audioman.core import latency as lat
        from audioman.plugins.parameter import PluginMeta

        meta = PluginMeta(name="F", short_name="f", path="/p.vst3", format="vst3")
        calls = []

        def _wrapper(path):
            w = _FakeWrapper(name="F", path=path, delay=16)
            calls.append(w)
            return w

        monkeypatch.setattr(lat, "get_registry", lambda: type("R", (), {"get": lambda s, n: meta})())
        monkeypatch.setattr(lat, "VST3PluginWrapper", _wrapper)

        measurements, total = lat.measure_chain_latency(
            [{"plugin": "f", "params": {"x": 2}}], sample_rate=48000
        )
        assert total == 16
        assert calls[0].params_set == {"x": 2}

    def test_unknown_plugin_raises(self, monkeypatch):
        from audioman.core import latency as lat
        monkeypatch.setattr(lat, "get_registry", lambda: type("R", (), {"get": lambda s, n: None})())
        with pytest.raises(ValueError, match="플러그인을 찾을 수 없습니다"):
            lat.measure_chain_latency([{"plugin": "nope"}], sample_rate=48000)


class TestMeasureChainViaCommitModule:
    def test_dry_run_commit_delegates(self, monkeypatch):
        from audioman.core import commit
        from audioman.core.pipeline import PipelineStep
        captured = {}

        def fake_measure(steps, sample_rate=48000):
            captured["steps"] = steps
            captured["sr"] = sample_rate
            return (["m1"], 32)

        monkeypatch.setattr(commit, "measure_chain_latency", fake_measure)
        steps = [PipelineStep("p", {})]
        measurements, total = commit.dry_run_commit(steps, sample_rate=44100)
        assert total == 32
        assert captured["sr"] == 44100
        assert captured["steps"] == steps


class TestApplyDelayCompensationMonoExceeds:
    def test_mono_latency_exceeds_length(self):
        audio = np.ones(50, dtype=np.float32)
        result = apply_delay_compensation(audio, 100)
        assert result.shape == (50,)
        np.testing.assert_array_equal(result, 0.0)


class TestLatencyConfidenceBranches:
    def test_short_capture_leaves_no_noise_floor(self):
        """Signal shorter than the ±100 exclusion window → mask empty → snr path skipped."""
        from audioman.core.latency import measure_plugin_latency
        w = _FakeWrapper(delay=4)
        m = measure_plugin_latency(w, sample_rate=48000, test_duration_sec=0.001)
        assert m.confidence == pytest.approx(1.0)
        assert m.measured_latency == 4

    def test_noise_floor_yields_snr_confidence(self):
        from audioman.core.latency import measure_plugin_latency

        class _NoisyWrapper(_FakeWrapper):
            def process(self, audio, sample_rate, reset=True):
                out = np.full_like(audio, 1e-3)
                out[..., self._delay] = 1.0
                return out.astype(np.float32)

        w = _NoisyWrapper(delay=200)
        m = measure_plugin_latency(w, sample_rate=48000)
        assert 0.0 < m.confidence <= 1.0
