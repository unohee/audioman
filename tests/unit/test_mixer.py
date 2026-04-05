# tests/unit/test_mixer.py — 멀티트랙 믹싱 테스트

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from audioman.core.mixer import TrackConfig, apply_pan, mix_tracks, bounce, _ensure_stereo


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
