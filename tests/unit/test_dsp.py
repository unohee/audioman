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


class TestFadeCurves:
    def test_linear_curve_endpoints(self, test_audio, sample_rate):
        from audioman.core.dsp import fade_in, fade_out
        n = sample_rate // 10
        fi = fade_in(test_audio, n, curve="linear")
        fo = fade_out(test_audio, n, curve="linear")
        assert abs(fi[0, 0]) < 1e-6
        assert abs(fo[0, -1]) < 1e-6

    def test_cosine_curve_smooth(self, test_audio, sample_rate):
        from audioman.core.dsp import fade_in
        n = sample_rate // 10
        fi = fade_in(test_audio, n, curve="cosine")
        # cosine S-curve: 시작과 끝 미분이 0이어야 부드럽다
        # 첫 두 샘플의 차이 < 중간 두 샘플의 차이
        diff_start = abs(fi[0, 1] - fi[0, 0])
        diff_mid = abs(fi[0, n // 2 + 1] - fi[0, n // 2])
        assert diff_start < diff_mid

    def test_equal_power_midpoint(self, test_audio, sample_rate):
        from audioman.core.dsp import fade_in
        n = sample_rate // 10
        fi = fade_in(test_audio, n, curve="equal_power")
        # equal_power 중간점: sqrt(0.5) ≈ 0.707 (linear는 0.5)
        original_mid = test_audio[0, n // 2]
        if abs(original_mid) > 0.01:  # silence 회피
            ratio = fi[0, n // 2] / original_mid
            assert 0.65 < ratio < 0.75

    def test_unknown_curve_raises(self, test_audio):
        from audioman.core.dsp import fade_in
        with pytest.raises(ValueError, match="알 수 없는 fade curve"):
            fade_in(test_audio, 100, curve="bouncy")

    def test_all_curves_accept(self, test_audio, sample_rate):
        from audioman.core.dsp import fade_in, fade_out, FADE_CURVES
        n = sample_rate // 100
        for curve in FADE_CURVES:
            fade_in(test_audio, n, curve=curve)
            fade_out(test_audio, n, curve=curve)


class TestPad:
    def test_pad_head_only(self, test_audio, sample_rate):
        from audioman.core.dsp import pad
        original_len = test_audio.shape[1]
        result = pad(test_audio, head_samples=sample_rate // 2)  # 0.5초
        assert result.shape[1] == original_len + sample_rate // 2
        # 헤드 패딩은 무음
        assert np.max(np.abs(result[:, :sample_rate // 2])) == 0.0
        # 원본이 그 뒤에 보존됨
        np.testing.assert_allclose(result[:, sample_rate // 2:], test_audio, atol=1e-6)

    def test_pad_tail_only(self, test_audio, sample_rate):
        from audioman.core.dsp import pad
        result = pad(test_audio, tail_samples=sample_rate)  # 1초
        assert result.shape[1] == test_audio.shape[1] + sample_rate
        assert np.max(np.abs(result[:, -sample_rate:])) == 0.0

    def test_pad_both(self, test_audio, sample_rate):
        from audioman.core.dsp import pad
        head = sample_rate // 4
        tail = sample_rate // 2
        result = pad(test_audio, head_samples=head, tail_samples=tail)
        assert result.shape[1] == test_audio.shape[1] + head + tail
        assert np.max(np.abs(result[:, :head])) == 0.0
        assert np.max(np.abs(result[:, -tail:])) == 0.0

    def test_pad_mono(self, test_audio_mono, sample_rate):
        from audioman.core.dsp import pad
        result = pad(test_audio_mono, head_samples=100, tail_samples=200)
        assert result.shape == (test_audio_mono.shape[0] + 300,)

    def test_pad_zero_returns_copy(self, test_audio):
        from audioman.core.dsp import pad
        result = pad(test_audio, head_samples=0, tail_samples=0)
        np.testing.assert_array_equal(result, test_audio)
        assert result is not test_audio  # copy, not same object

    def test_pad_negative_raises(self, test_audio):
        from audioman.core.dsp import pad
        with pytest.raises(ValueError, match="음수일 수 없"):
            pad(test_audio, head_samples=-1)


class TestRemoveDC:
    def test_remove_dc_offset(self, sample_rate):
        from audioman.core.dsp import remove_dc, measure_dc_offset
        # 채널별로 다른 DC bias
        t = np.linspace(0, 1, sample_rate, dtype=np.float32)
        sine = 0.3 * np.sin(2 * np.pi * 440 * t)
        ch_l = sine + 0.1   # +0.1 DC
        ch_r = sine - 0.05  # -0.05 DC
        stereo = np.stack([ch_l, ch_r])

        before = measure_dc_offset(stereo)
        assert abs(before[0] - 0.1) < 0.01
        assert abs(before[1] - (-0.05)) < 0.01

        cleaned = remove_dc(stereo)
        after = measure_dc_offset(cleaned)
        assert abs(after[0]) < 1e-6
        assert abs(after[1]) < 1e-6

    def test_remove_dc_preserves_signal_shape(self, test_audio):
        from audioman.core.dsp import remove_dc
        result = remove_dc(test_audio)
        assert result.shape == test_audio.shape
        # 원래 DC가 거의 0이므로 신호는 거의 그대로
        np.testing.assert_allclose(result, test_audio, atol=1e-3)

    def test_remove_dc_mono(self, test_audio_mono):
        from audioman.core.dsp import remove_dc
        biased = test_audio_mono + 0.2
        cleaned = remove_dc(biased)
        assert abs(np.mean(cleaned)) < 1e-6


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
