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
