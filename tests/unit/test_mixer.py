# tests/unit/test_mixer.py — 멀티트랙 믹싱 테스트

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from audioman.core.mixer import (
    TrackConfig,
    apply_pan,
    mix_tracks,
    bounce,
    mixdown,
    _ensure_stereo,
)


class TestApplyPan:
    """Equal Power Pan Law 검증"""

    def test_center_pan(self):
        """pan=0 (center) → 양 채널 동일 레벨"""
        audio = np.ones((2, 100), dtype=np.float32)
        result = apply_pan(audio, 0.0)
        # center에서 L/R gain은 cos(pi/4) = sin(pi/4) ≈ 0.707
        expected_gain = np.cos(np.pi / 4)
        np.testing.assert_allclose(result[0], expected_gain, atol=1e-6)
        np.testing.assert_allclose(result[1], expected_gain, atol=1e-6)

    def test_hard_left(self):
        """pan=-1 → L만 출력"""
        audio = np.ones((2, 100), dtype=np.float32)
        result = apply_pan(audio, -1.0)
        # hard left: L=cos(0)=1.0, R=sin(0)=0.0
        np.testing.assert_allclose(result[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(result[1], 0.0, atol=1e-6)

    def test_hard_right(self):
        """pan=1 → R만 출력"""
        audio = np.ones((2, 100), dtype=np.float32)
        result = apply_pan(audio, 1.0)
        # hard right: L=cos(pi/2)=0.0, R=sin(pi/2)=1.0
        np.testing.assert_allclose(result[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[1], 1.0, atol=1e-6)

    def test_equal_power_law(self):
        """L^2 + R^2 = 1 (모든 pan 위치에서 에너지 보존)"""
        audio = np.ones((2, 100), dtype=np.float32)
        for pan in np.linspace(-1.0, 1.0, 21):
            result = apply_pan(audio, pan)
            power = result[0, 0] ** 2 + result[1, 0] ** 2
            assert power == pytest.approx(1.0, abs=1e-5), f"pan={pan}: power={power}"


class TestEnsureStereo:
    def test_mono_to_stereo(self):
        mono = np.ones(100, dtype=np.float32)
        stereo = _ensure_stereo(mono)
        assert stereo.shape == (2, 100)
        np.testing.assert_array_equal(stereo[0], mono)
        np.testing.assert_array_equal(stereo[1], mono)

    def test_already_stereo(self):
        stereo = np.ones((2, 100), dtype=np.float32)
        result = _ensure_stereo(stereo)
        assert result.shape == (2, 100)

    def test_single_channel_2d(self):
        mono2d = np.ones((1, 100), dtype=np.float32)
        stereo = _ensure_stereo(mono2d)
        assert stereo.shape == (2, 100)


class TestMixTracks:
    """mix_tracks 통합 테스트 (파일 I/O 필요)"""

    @pytest.fixture
    def two_tracks(self, tmp_path, sample_rate):
        """2개 트랙 WAV 파일 생성"""
        # 트랙1: L채널에 0.5, R채널에 0
        t1 = np.zeros((2, sample_rate), dtype=np.float32)
        t1[0, :] = 0.5

        # 트랙2: R채널에 0.3, L채널에 0
        t2 = np.zeros((2, sample_rate), dtype=np.float32)
        t2[1, :] = 0.3

        p1 = tmp_path / "track1.wav"
        p2 = tmp_path / "track2.wav"
        sf.write(str(p1), t1.T, sample_rate, subtype="PCM_24")
        sf.write(str(p2), t2.T, sample_rate, subtype="PCM_24")
        return p1, p2

    def test_basic_mix(self, two_tracks, sample_rate):
        """2트랙 합산 — center pan, 0dB gain"""
        p1, p2 = two_tracks
        tracks = [
            TrackConfig(path=str(p1)),
            TrackConfig(path=str(p2)),
        ]
        mix, sr = mix_tracks(tracks, apply_chain=False)
        assert sr == sample_rate
        assert mix.shape[0] == 2
        # center pan의 gain ≈ 0.707
        pan_gain = np.cos(np.pi / 4)
        # L = t1_L*pan_gain + t2_L*pan_gain = 0.5*0.707 + 0*0.707
        np.testing.assert_allclose(mix[0, 0], 0.5 * pan_gain, atol=1e-3)
        # R = t1_R*pan_gain + t2_R*pan_gain = 0*0.707 + 0.3*0.707
        np.testing.assert_allclose(mix[1, 0], 0.3 * pan_gain, atol=1e-3)

    def test_gain_applied(self, two_tracks, sample_rate):
        """게인 적용 확인"""
        p1, p2 = two_tracks
        tracks = [
            TrackConfig(path=str(p1), gain_db=-6.0),
            TrackConfig(path=str(p2), gain_db=0.0),
        ]
        mix, sr = mix_tracks(tracks, apply_chain=False)
        # -6dB ≈ 0.501 배
        gain_factor = 10 ** (-6.0 / 20.0)
        pan_gain = np.cos(np.pi / 4)
        np.testing.assert_allclose(mix[0, 0], 0.5 * gain_factor * pan_gain, atol=1e-3)

    def test_mute_track(self, two_tracks, sample_rate):
        """뮤트된 트랙 제외"""
        p1, p2 = two_tracks
        tracks = [
            TrackConfig(path=str(p1), mute=True),
            TrackConfig(path=str(p2)),
        ]
        mix, sr = mix_tracks(tracks, apply_chain=False)
        # 트랙1 뮤트 → L채널은 0
        pan_gain = np.cos(np.pi / 4)
        np.testing.assert_allclose(mix[0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(mix[1, 0], 0.3 * pan_gain, atol=1e-3)

    def test_solo_track(self, two_tracks, sample_rate):
        """솔로 트랙만 재생"""
        p1, p2 = two_tracks
        tracks = [
            TrackConfig(path=str(p1), solo=True),
            TrackConfig(path=str(p2)),
        ]
        mix, sr = mix_tracks(tracks, apply_chain=False)
        # 트랙1만 솔로 → R채널의 0.3 신호는 없어야 함
        pan_gain = np.cos(np.pi / 4)
        np.testing.assert_allclose(mix[0, 0], 0.5 * pan_gain, atol=1e-3)
        # 트랙2의 R 기여분 없음
        np.testing.assert_allclose(mix[1, 0], 0.0, atol=1e-3)

    def test_different_lengths(self, tmp_path, sample_rate):
        """길이가 다른 트랙 → 짧은 쪽 zero-pad"""
        t1 = np.ones((2, sample_rate), dtype=np.float32) * 0.5      # 1초
        t2 = np.ones((2, sample_rate // 2), dtype=np.float32) * 0.3  # 0.5초

        p1 = tmp_path / "long.wav"
        p2 = tmp_path / "short.wav"
        sf.write(str(p1), t1.T, sample_rate, subtype="PCM_24")
        sf.write(str(p2), t2.T, sample_rate, subtype="PCM_24")

        tracks = [
            TrackConfig(path=str(p1)),
            TrackConfig(path=str(p2)),
        ]
        mix, sr = mix_tracks(tracks, apply_chain=False)
        # 결과는 긴 트랙 길이
        assert mix.shape[1] == sample_rate


class TestBounce:
    def test_bounce_writes_file(self, tmp_path, sample_rate):
        """bounce() → 파일 정상 생성"""
        t1 = np.ones((2, sample_rate), dtype=np.float32) * 0.3
        p1 = tmp_path / "t1.wav"
        sf.write(str(p1), t1.T, sample_rate, subtype="PCM_24")

        out = tmp_path / "bounced.wav"
        result = bounce([TrackConfig(path=str(p1))], out)

        assert Path(result.output_path).exists()
        assert result.track_count == 1
        assert result.sample_rate == sample_rate


# ---------------------------------------------------------------------------
# TrackConfig.to_dict / result dicts
# ---------------------------------------------------------------------------


class TestTrackConfigDict:
    def test_minimal_dict(self):
        d = TrackConfig(path="/a.wav").to_dict()
        assert d == {"path": "/a.wav", "gain_db": 0.0, "pan": 0.0}

    def test_optional_fields_included_only_when_set(self):
        from audioman.core.pipeline import PipelineStep
        cfg = TrackConfig(
            path="/a.wav", gain_db=-3.0, pan=0.5, mute=True, solo=True,
            chain=[PipelineStep("p", {"x": 1})], offset_samples=100,
        )
        d = cfg.to_dict()
        assert d["gain_db"] == -3.0
        assert d["pan"] == 0.5
        assert d["mute"] is True
        assert d["solo"] is True
        assert d["chain"] == [{"plugin": "p", "params": {"x": 1}}]
        assert d["offset_samples"] == 100


class TestResultDataclasses:
    def test_bounce_result_to_dict(self):
        from audioman.core.mixer import BounceResult
        r = BounceResult("/o.wav", 2, [{"path": "a"}], {"peak": 0.5}, 48000, 1.0, False)
        d = r.to_dict()
        assert d["output_path"] == "/o.wav"
        assert d["track_count"] == 2
        assert d["clipping_detected"] is False

    def test_mixdown_result_to_dict(self):
        from audioman.core.mixer import MixdownResult
        r = MixdownResult("/o.wav", 1, [], None, 0, {"peak": 0.5}, 48000, 1.0, True)
        d = r.to_dict()
        assert d["master_chain"] is None
        assert d["clipping_detected"] is True


class TestEnsureStereoExtra:
    def test_three_channel_uses_first_two(self):
        audio = np.stack([np.ones(50), np.full(50, 2.0), np.full(50, 3.0)]).astype(np.float32)
        out = _ensure_stereo(audio)
        assert out.shape == (2, 50)
        np.testing.assert_array_equal(out[0], 1.0)
        np.testing.assert_array_equal(out[1], 2.0)

    def test_single_channel_2d(self):
        audio = np.ones((1, 50), dtype=np.float32)
        out = _ensure_stereo(audio)
        assert out.shape == (2, 50)


class TestResample:
    def test_same_rate_passthrough(self):
        from audioman.core.mixer import _resample_if_needed
        audio = np.ones((2, 100), dtype=np.float32)
        out = _resample_if_needed(audio, 48000, 48000)
        assert out is audio

    def test_stereo_resample(self):
        from audioman.core.mixer import _resample_if_needed
        audio = np.ones((2, 480), dtype=np.float32)
        out = _resample_if_needed(audio, 48000, 24000)
        assert out.shape == (2, 240)

    def test_mono_resample(self):
        from audioman.core.mixer import _resample_if_needed
        audio = np.ones(480, dtype=np.float32)
        out = _resample_if_needed(audio, 48000, 24000)
        assert out.shape == (240,)


class TestApplyTrackChain:
    def test_unknown_plugin_raises(self, monkeypatch):
        from audioman.core import mixer
        from audioman.core.pipeline import PipelineStep
        monkeypatch.setattr(mixer, "get_registry", lambda: type("R", (), {"get": lambda s, n: None})())
        with pytest.raises(ValueError, match="플러그인을 찾을 수 없습니다"):
            mixer._apply_track_chain(np.ones((2, 10), dtype=np.float32), 48000,
                                     [PipelineStep("nope", {})])

    def test_applies_delay_wrapper(self, monkeypatch):
        from audioman.core import mixer
        from audioman.core.pipeline import PipelineStep
        from audioman.plugins.parameter import PluginMeta
        meta = PluginMeta(name="D", short_name="d", path="/d.vst3", format="vst3")

        made = []

        class _W:
            def __init__(self, path):
                self.path = path
                self.params = None
                made.append(self)

            def load(self):
                pass

            def set_parameters(self, p):
                self.params = p

            def process(self, audio, sr):
                return audio * 0.5

        monkeypatch.setattr(mixer, "get_registry", lambda: type("R", (), {"get": lambda s, n: meta})())
        monkeypatch.setattr(mixer, "VST3PluginWrapper", _W)
        out = mixer._apply_track_chain(np.ones((2, 10), dtype=np.float32), 48000,
                                       [PipelineStep("d", {"g": 1})])
        np.testing.assert_allclose(out, 0.5)
        assert made[0].params == {"g": 1}


class TestMixTracksBranches:
    def _wav(self, tmp_path, name, audio, sr=48000):
        sf.write(str(tmp_path / name), audio.T, sr, subtype="FLOAT")
        return str(tmp_path / name)

    def test_no_tracks_raises(self):
        with pytest.raises(ValueError, match="트랙이 없습니다"):
            mix_tracks([])

    def test_all_muted_returns_silence(self, tmp_path):
        src = self._wav(tmp_path, "a.wav", np.ones((2, 100), dtype=np.float32))
        out, sr = mix_tracks([TrackConfig(path=src, mute=True)], sample_rate=48000)
        assert out.shape == (2, 48000)
        assert sr == 48000
        assert np.max(np.abs(out)) == 0.0

    def test_all_muted_default_rate(self, tmp_path):
        src = self._wav(tmp_path, "a.wav", np.ones((2, 100), dtype=np.float32))
        out, sr = mix_tracks([TrackConfig(path=src, mute=True)])
        assert sr == 48000  # falls back to 48k when no sample_rate given

    def test_solo_excludes_others(self, tmp_path):
        a = self._wav(tmp_path, "a.wav", np.full((2, 100), 1.0, dtype=np.float32))
        b = self._wav(tmp_path, "b.wav", np.full((2, 100), 2.0, dtype=np.float32))
        out, _ = mix_tracks([TrackConfig(path=a), TrackConfig(path=b, solo=True)], sample_rate=48000)
        # equal-power center pan = 0.7071; only b survives
        np.testing.assert_allclose(out[0], 2.0 * np.cos(np.pi / 4), atol=1e-4)

    def test_offset_pads_front(self, tmp_path):
        a = self._wav(tmp_path, "a.wav", np.ones((2, 100), dtype=np.float32))
        out, _ = mix_tracks([TrackConfig(path=a, offset_samples=50)], sample_rate=48000)
        assert out.shape[1] == 150
        assert np.max(np.abs(out[:, :50])) == 0.0
        assert np.max(np.abs(out[:, 50:])) > 0.0

    def test_unequal_lengths_zero_padded(self, tmp_path):
        a = self._wav(tmp_path, "a.wav", np.ones((2, 100), dtype=np.float32))
        b = self._wav(tmp_path, "b.wav", np.ones((2, 300), dtype=np.float32))
        out, _ = mix_tracks([TrackConfig(path=a), TrackConfig(path=b)], sample_rate=48000)
        assert out.shape == (2, 300)

    def test_resamples_second_track(self, tmp_path):
        a = self._wav(tmp_path, "a.wav", np.ones((2, 480), dtype=np.float32), sr=48000)
        b = self._wav(tmp_path, "b.wav", np.ones((2, 240), dtype=np.float32), sr=24000)
        out, sr = mix_tracks([TrackConfig(path=a), TrackConfig(path=b)], sample_rate=48000)
        assert sr == 48000
        assert out.shape[1] == 480

    def test_clipping_warning_detected(self, caplog, tmp_path):
        a = self._wav(tmp_path, "a.wav", np.full((2, 100), 0.9, dtype=np.float32))
        b = self._wav(tmp_path, "b.wav", np.full((2, 100), 0.9, dtype=np.float32))
        with caplog.at_level("WARNING"):
            mix_tracks([TrackConfig(path=a, pan=-1.0), TrackConfig(path=b, pan=-1.0)],
                       sample_rate=48000)
        assert any("클리핑" in r.message for r in caplog.records)

    def test_gain_applied(self, tmp_path):
        a = self._wav(tmp_path, "a.wav", np.full((2, 100), 0.5, dtype=np.float32))
        out, _ = mix_tracks([TrackConfig(path=a, gain_db=-6.0, pan=-1.0)], sample_rate=48000)
        # hard left: L gets cos(0)=1.0, -6dB = 0.5012
        assert out[0, 0] == pytest.approx(0.5 * 10 ** (-6 / 20), abs=1e-4)

    def test_chain_applied_when_requested(self, monkeypatch, tmp_path):
        from audioman.core import mixer
        from audioman.core.pipeline import PipelineStep
        from audioman.plugins.parameter import PluginMeta
        meta = PluginMeta(name="D", short_name="d", path="/d.vst3", format="vst3")

        class _W:
            def __init__(self, path):
                pass

            def load(self):
                pass

            def set_parameters(self, p):
                pass

            def process(self, audio, sr):
                return audio * 0.25

        monkeypatch.setattr(mixer, "get_registry", lambda: type("R", (), {"get": lambda s, n: meta})())
        monkeypatch.setattr(mixer, "VST3PluginWrapper", _W)
        a = self._wav(tmp_path, "a.wav", np.full((2, 100), 1.0, dtype=np.float32))
        out, _ = mix_tracks(
            [TrackConfig(path=a, chain=[PipelineStep("d", {})], pan=-1.0)],
            sample_rate=48000,
        )
        assert out[0, 0] == pytest.approx(0.25, abs=1e-4)

    def test_chain_skipped_when_apply_chain_false(self, tmp_path):
        from audioman.core.pipeline import PipelineStep
        a = self._wav(tmp_path, "a.wav", np.full((2, 100), 1.0, dtype=np.float32))
        out, _ = mix_tracks(
            [TrackConfig(path=a, chain=[PipelineStep("does-not-exist", {})], pan=-1.0)],
            sample_rate=48000, apply_chain=False,
        )
        assert out[0, 0] == pytest.approx(1.0, abs=1e-4)


class TestBounce:
    def test_bounce_writes_and_reports(self, tmp_path):
        sf.write(str(tmp_path / "a.wav"), np.full((100, 2), 0.25, dtype=np.float32),
                 48000, subtype="FLOAT")
        out_path = tmp_path / "bounce.wav"
        result = bounce([TrackConfig(path=str(tmp_path / "a.wav"), pan=-1.0)], out_path,
                        sample_rate=48000)
        assert out_path.exists()
        assert result.track_count == 1
        assert result.sample_rate == 48000
        assert result.clipping_detected is False
        assert result.tracks[0]["path"].endswith("a.wav")

    def test_bounce_detects_clipping(self, tmp_path):
        sf.write(str(tmp_path / "a.wav"), np.full((100, 2), 0.9, dtype=np.float32),
                 48000, subtype="FLOAT")
        result = bounce([TrackConfig(path=str(tmp_path / "a.wav"), gain_db=6.0, pan=-1.0)],
                        tmp_path / "b.wav", sample_rate=48000)
        assert result.clipping_detected is True


class TestMixdown:
    def _src(self, tmp_path):
        sf.write(str(tmp_path / "a.wav"), np.full((100, 2), 0.3, dtype=np.float32),
                 48000, subtype="FLOAT")
        return str(tmp_path / "a.wav")

    def test_mixdown_without_master_chain(self, tmp_path):
        out_path = tmp_path / "mix.wav"
        result = mixdown([TrackConfig(path=self._src(tmp_path))], out_path, sample_rate=48000)
        assert out_path.exists()
        assert result.master_chain is None
        assert result.master_latency_samples == 0
        assert result.duration_seconds >= 0.0

    def test_mixdown_with_master_chain_latency_compensation(self, monkeypatch, tmp_path):
        from audioman.core import mixer
        from audioman.core.pipeline import PipelineStep
        from audioman.plugins.parameter import PluginMeta
        meta = PluginMeta(name="D", short_name="d", path="/d.vst3", format="vst3")

        class _W:
            def __init__(self, path):
                pass

            def load(self):
                pass

            def set_parameters(self, p):
                pass

            def process(self, audio, sr):
                shifted = np.roll(audio, 10, axis=-1)
                shifted[:, :10] = 0.0
                return shifted

        monkeypatch.setattr(mixer, "get_registry", lambda: type("R", (), {"get": lambda s, n: meta})())
        monkeypatch.setattr(mixer, "VST3PluginWrapper", _W)
        from audioman.core import latency as lat
        monkeypatch.setattr(lat, "measure_chain_latency",
                            lambda steps, sample_rate=48000: ([], 10))
        monkeypatch.setattr(lat, "apply_delay_compensation", lambda audio, n: audio)

        out_path = tmp_path / "mix.wav"
        result = mixdown(
            [TrackConfig(path=self._src(tmp_path))], out_path,
            master_chain=[PipelineStep("d", {})], sample_rate=48000,
        )
        assert result.master_latency_samples == 10
        assert result.master_chain == [{"plugin": "d", "params": {}}]

    def test_mixdown_compensate_false_skips_measurement(self, monkeypatch, tmp_path):
        from audioman.core import mixer
        from audioman.core.pipeline import PipelineStep
        from audioman.plugins.parameter import PluginMeta
        meta = PluginMeta(name="D", short_name="d", path="/d.vst3", format="vst3")
        called = {"n": 0}

        class _W:
            def __init__(self, path):
                pass

            def load(self):
                pass

            def set_parameters(self, p):
                pass

            def process(self, audio, sr):
                return audio

        from audioman.core import latency as lat

        def _measure(*a, **k):
            called["n"] += 1
            return [], 0

        monkeypatch.setattr(mixer, "get_registry", lambda: type("R", (), {"get": lambda s, n: meta})())
        monkeypatch.setattr(mixer, "VST3PluginWrapper", _W)
        monkeypatch.setattr(lat, "measure_chain_latency", _measure)

        result = mixdown(
            [TrackConfig(path=self._src(tmp_path))], tmp_path / "m.wav",
            master_chain=[PipelineStep("d", {})], sample_rate=48000,
            compensate_latency=False,
        )
        assert called["n"] == 0
        assert result.master_latency_samples == 0
