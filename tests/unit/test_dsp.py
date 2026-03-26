# tests/unit/test_dsp.py

import numpy as np
import pytest
from audioman.core.dsp import normalize, gate, trim_silence, fade_in, fade_out, gain, trim


class TestNormalize:
    def test_peak_normalize(self, test_audio):
        result = normalize(test_audio, peak_db=0.0)
        peak = np.max(np.abs(result))
        assert abs(peak - 1.0) < 0.01

    def test_peak_normalize_minus6(self, test_audio):
        result = normalize(test_audio, peak_db=-6.0)
        peak = np.max(np.abs(result))
        expected = 10 ** (-6.0 / 20.0)
        assert abs(peak - expected) < 0.02

    def test_rms_normalize(self, test_audio):
        result = normalize(test_audio, target_rms_db=-20.0)
        rms = np.sqrt(np.mean(result**2))
        expected = 10 ** (-20.0 / 20.0)
        assert abs(rms - expected) < 0.02

    def test_silent_unchanged(self, silent_audio):
        result = normalize(silent_audio, peak_db=0.0)
        assert np.max(np.abs(result)) == 0.0


class TestGate:
    def test_gate_removes_silence(self, sample_rate):
        audio = np.zeros((2, sample_rate), dtype=np.float32)
        t = np.linspace(0, 0.5, sample_rate // 2, dtype=np.float32)
        audio[:, sample_rate // 2:] = 0.5 * np.sin(2 * np.pi * 440 * t)
        result = gate(audio, sample_rate, threshold_db=-40.0)
        # 앞 무음 구간의 RMS가 낮아야 함
        rms_first = np.sqrt(np.mean(result[:, :sample_rate // 4]**2))
        assert rms_first < 0.01


class TestFade:
    def test_fade_in(self, test_audio, sample_rate):
        fade_samples = sample_rate // 10  # 0.1초
        result = fade_in(test_audio, fade_samples)
        assert abs(result[0, 0]) < 0.01
        np.testing.assert_allclose(result[:, -1000:], test_audio[:, -1000:], atol=1e-6)

    def test_fade_out(self, test_audio, sample_rate):
        fade_samples = sample_rate // 10
        result = fade_out(test_audio, fade_samples)
        assert abs(result[0, -1]) < 0.01
        np.testing.assert_allclose(result[:, :1000], test_audio[:, :1000], atol=1e-6)


class TestGain:
    def test_gain_6db(self, test_audio):
        result = gain(test_audio, db=6.0)
        ratio = np.max(np.abs(result)) / np.max(np.abs(test_audio))
        expected = 10 ** (6.0 / 20.0)
        assert abs(ratio - expected) < 0.1

    def test_gain_zero(self, test_audio):
        result = gain(test_audio, db=0.0)
        np.testing.assert_allclose(result, test_audio, atol=1e-6)

    def test_gain_negative(self, test_audio):
        result = gain(test_audio, db=-6.0)
        assert np.max(np.abs(result)) < np.max(np.abs(test_audio))


class TestTrim:
    def test_trim_basic(self, test_audio):
        result = trim(test_audio, start=100, end=200)
        assert result.shape[1] == 100

    def test_trim_silence(self, sample_rate):
        # 0.2초 무음 + 0.6초 톤 + 0.2초 무음
        audio = np.zeros((2, sample_rate), dtype=np.float32)
        start = int(0.2 * sample_rate)
        end = int(0.8 * sample_rate)
        t = np.linspace(0, 0.6, end - start, dtype=np.float32)
        audio[:, start:end] = 0.5 * np.sin(2 * np.pi * 440 * t)
        result = trim_silence(audio, sample_rate, threshold_db=-40.0)
        # 트리밍 후 길이가 줄어야 함
        assert result.shape[1] < audio.shape[1]
        # 시작 부분에 소리가 있어야 함
        assert np.max(np.abs(result[:, :100])) > 0.01
