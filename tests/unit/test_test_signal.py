# tests/unit/test_test_signal.py — 테스트 신호 생성기 단위 테스트

import numpy as np
import pytest

from audioman.core.test_signal import (
    generate_impulse,
    generate_sine,
    generate_two_tone,
    generate_white_noise,
    generate_sweep,
    generate_dynamics_ramp,
    generate_dynamics_attack_release,
    generate_log_sweep_deconv,
    generate_multitone,
    generate_pink_noise,
    generate_band_limited_noise,
    generate_impulse_train,
    to_mid_side,
    from_mid_side,
)


class TestGenerateImpulse:
    def test_shape_stereo(self):
        audio = generate_impulse(sample_rate=44100, duration_sec=1.0, channels=2)
        assert audio.shape == (2, 44100)
        assert audio.dtype == np.float32

    def test_shape_mono(self):
        audio = generate_impulse(channels=1, duration_sec=0.5, sample_rate=48000)
        assert audio.shape == (1, 24000)

    def test_impulse_at_sample_zero(self):
        audio = generate_impulse(level_db=0.0)
        # sample 0에 1.0 (0dB)
        assert audio[0, 0] == pytest.approx(1.0, abs=1e-6)
        # 나머지는 0
        assert np.all(audio[:, 1:] == 0.0)

    def test_level_db(self):
        audio = generate_impulse(level_db=-6.0)
        expected_amp = 10 ** (-6.0 / 20.0)
        assert audio[0, 0] == pytest.approx(expected_amp, rel=1e-4)

    def test_multichannel(self):
        audio = generate_impulse(channels=5)
        assert audio.shape[0] == 5
        # 모든 채널에 동일한 임펄스
        for ch in range(5):
            assert audio[ch, 0] == pytest.approx(1.0, abs=1e-6)


class TestGenerateSine:
    def test_shape_and_dtype(self):
        audio = generate_sine(frequency=440.0, sample_rate=44100, duration_sec=1.0)
        assert audio.shape == (2, 44100)
        assert audio.dtype == np.float32

    def test_peak_amplitude(self):
        audio = generate_sine(level_db=0.0)
        peak = np.max(np.abs(audio))
        assert peak == pytest.approx(1.0, abs=0.01)

    def test_frequency_content(self):
        """FFT 피크가 목표 주파수 근처에 위치"""
        sr = 44100
        audio = generate_sine(frequency=1000.0, sample_rate=sr, duration_sec=1.0)
        spectrum = np.abs(np.fft.rfft(audio[0]))
        freqs = np.fft.rfftfreq(sr, 1.0 / sr)
        peak_freq = freqs[np.argmax(spectrum)]
        assert abs(peak_freq - 1000.0) < 2.0

    def test_level_db_negative(self):
        audio = generate_sine(level_db=-20.0)
        expected = 10 ** (-20.0 / 20.0)
        peak = np.max(np.abs(audio))
        assert peak == pytest.approx(expected, rel=0.02)


class TestGenerateTwoTone:
    def test_shape(self):
        audio = generate_two_tone()
        assert audio.shape == (2, 44100)
        assert audio.dtype == np.float32

    def test_two_frequency_peaks(self):
        """FFT에 60Hz, 7kHz 두 피크가 존재"""
        sr = 44100
        audio = generate_two_tone(
            freq_low=60.0, freq_high=7000.0, sample_rate=sr, duration_sec=1.0,
        )
        spectrum = np.abs(np.fft.rfft(audio[0]))
        freqs = np.fft.rfftfreq(sr, 1.0 / sr)

        # 각 주파수 주변에서 피크 확인
        low_band = spectrum[(freqs > 50) & (freqs < 70)]
        high_band = spectrum[(freqs > 6900) & (freqs < 7100)]
        noise_band = spectrum[(freqs > 2000) & (freqs < 3000)]

        assert np.max(low_band) > 10 * np.mean(noise_band)
        assert np.max(high_band) > 10 * np.mean(noise_band)


class TestGenerateWhiteNoise:
    def test_shape_and_dtype(self):
        audio = generate_white_noise(sample_rate=44100, duration_sec=2.0)
        assert audio.shape == (2, 88200)
        assert audio.dtype == np.float32

    def test_deterministic_with_seed(self):
        a = generate_white_noise(seed=42)
        b = generate_white_noise(seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self):
        a = generate_white_noise(seed=42)
        b = generate_white_noise(seed=99)
        assert not np.allclose(a, b)

    def test_peak_normalized(self):
        audio = generate_white_noise(level_db=0.0)
        peak = np.max(np.abs(audio))
        assert peak == pytest.approx(1.0, abs=0.01)


class TestGenerateSweep:
    def test_shape(self):
        audio = generate_sweep(sample_rate=44100, duration_sec=2.0)
        assert audio.shape == (2, 88200)

    def test_exponential_vs_linear(self):
        exp = generate_sweep(exponential=True, duration_sec=1.0)
        lin = generate_sweep(exponential=False, duration_sec=1.0)
        # 같은 shape이지만 다른 내용
        assert exp.shape == lin.shape
        assert not np.allclose(exp, lin)

    def test_mono(self):
        audio = generate_sweep(channels=1)
        assert audio.shape[0] == 1


class TestGenerateDynamicsRamp:
    def test_returns_tuple(self):
        audio, levels = generate_dynamics_ramp()
        assert isinstance(audio, np.ndarray)
        assert isinstance(levels, list)

    def test_level_count(self):
        audio, levels = generate_dynamics_ramp(
            level_start_db=-10.0, level_end_db=0.0, step_db=2.0,
        )
        # -10, -8, -6, -4, -2, 0 → 6 steps
        assert len(levels) == 6

    def test_audio_length_matches_steps(self):
        sr = 44100
        step_dur = 0.5
        audio, levels = generate_dynamics_ramp(
            sample_rate=sr, step_duration_sec=step_dur,
            level_start_db=-6.0, level_end_db=0.0, step_db=2.0,
        )
        step_samples = int(step_dur * sr)
        expected_length = step_samples * len(levels)
        assert audio.shape == (2, expected_length)

    def test_amplitude_increases(self):
        """레벨이 올라가면 RMS도 증가"""
        sr = 44100
        audio, levels = generate_dynamics_ramp(
            sample_rate=sr, step_duration_sec=0.5,
            level_start_db=-60.0, level_end_db=0.0, step_db=20.0,
        )
        step_samples = int(0.5 * sr)
        rms_values = []
        for i in range(len(levels)):
            start = i * step_samples
            segment = audio[0, start:start + step_samples]
            rms_values.append(float(np.sqrt(np.mean(segment ** 2))))
        # 각 단계의 RMS가 단조 증가
        for i in range(1, len(rms_values)):
            assert rms_values[i] > rms_values[i - 1]


class TestGenerateDynamicsAttackRelease:
    def test_shape(self):
        audio = generate_dynamics_attack_release(sample_rate=44100)
        expected_n = int((0.5 + 1.0 + 0.5) * 44100)
        assert audio.shape == (2, expected_n)
        assert audio.dtype == np.float32

    def test_three_segments(self):
        """중간 구간(above)의 RMS가 앞뒤(below)보다 높음"""
        sr = 44100
        audio = generate_dynamics_attack_release(
            sample_rate=sr, t1_sec=0.5, t2_sec=1.0, t3_sec=0.5,
            level_below_db=-30.0, level_above_db=0.0,
        )
        n1, n2 = int(0.5 * sr), int(1.0 * sr)
        seg1 = audio[0, :n1]
        seg2 = audio[0, n1:n1 + n2]
        seg3 = audio[0, n1 + n2:]
        rms1 = float(np.sqrt(np.mean(seg1 ** 2)))
        rms2 = float(np.sqrt(np.mean(seg2 ** 2)))
        rms3 = float(np.sqrt(np.mean(seg3 ** 2)))
        assert rms2 > rms1 * 5
        assert rms2 > rms3 * 5


class TestGenerateLogSweepDeconv:
    def test_returns_tuple(self):
        sweep, inverse = generate_log_sweep_deconv(duration_sec=1.0)
        assert isinstance(sweep, np.ndarray)
        assert isinstance(inverse, np.ndarray)

    def test_sweep_shape(self):
        sweep, inverse = generate_log_sweep_deconv(
            sample_rate=44100, duration_sec=1.0, channels=2,
        )
        assert sweep.shape == (2, 44100)
        assert inverse.shape == (1, 44100)

    def test_inverse_filter_different(self):
        """역필터는 sweep의 단순 반전이 아님 (envelope 보상 포함)"""
        sweep, inverse = generate_log_sweep_deconv(duration_sec=1.0, channels=1)
        reversed_sweep = sweep[0, ::-1].copy()
        assert not np.allclose(inverse[0], reversed_sweep)


class TestGenerateMultitone:
    def test_shape(self):
        audio = generate_multitone(n_tones=32, sample_rate=44100, duration_sec=1.0)
        assert audio.shape == (2, 44100)

    def test_multiple_spectral_peaks(self):
        """여러 주파수에 에너지 분포"""
        sr = 44100
        audio = generate_multitone(
            n_tones=16, freq_start=100.0, freq_end=10000.0,
            sample_rate=sr, duration_sec=1.0,
        )
        spectrum = np.abs(np.fft.rfft(audio[0]))
        freqs = np.fft.rfftfreq(sr, 1.0 / sr)
        # 100-10000Hz 범위에 에너지가 DC보다 훨씬 큼
        in_band = spectrum[(freqs >= 100) & (freqs <= 10000)]
        dc_region = spectrum[freqs < 10]
        assert np.mean(in_band) > 5 * np.mean(dc_region)


class TestGeneratePinkNoise:
    def test_shape(self):
        audio = generate_pink_noise(sample_rate=44100, duration_sec=1.0)
        assert audio.shape == (2, 44100)

    def test_spectral_slope(self):
        """핑크 노이즈: 저주파 에너지 > 고주파 에너지"""
        sr = 44100
        audio = generate_pink_noise(sample_rate=sr, duration_sec=3.0)
        spectrum = np.abs(np.fft.rfft(audio[0]))
        freqs = np.fft.rfftfreq(len(audio[0]), 1.0 / sr)
        low = np.mean(spectrum[(freqs > 100) & (freqs < 500)])
        high = np.mean(spectrum[(freqs > 5000) & (freqs < 10000)])
        assert low > high

    def test_deterministic(self):
        a = generate_pink_noise(seed=42)
        b = generate_pink_noise(seed=42)
        np.testing.assert_array_equal(a, b)


class TestGenerateBandLimitedNoise:
    def test_shape(self):
        audio = generate_band_limited_noise(sample_rate=44100, duration_sec=1.0)
        assert audio.shape == (2, 44100)

    def test_energy_in_band(self):
        """밴드 내 에너지 > 밴드 외 에너지"""
        sr = 44100
        audio = generate_band_limited_noise(
            freq_low=1000.0, freq_high=3000.0,
            sample_rate=sr, duration_sec=3.0,
        )
        spectrum = np.abs(np.fft.rfft(audio[0]))
        freqs = np.fft.rfftfreq(len(audio[0]), 1.0 / sr)
        in_band = np.mean(spectrum[(freqs >= 1000) & (freqs <= 3000)])
        out_band_low = np.mean(spectrum[(freqs > 100) & (freqs < 500)])
        out_band_high = np.mean(spectrum[(freqs > 5000) & (freqs < 10000)])
        assert in_band > 10 * out_band_low
        assert in_band > 10 * out_band_high


class TestGenerateImpulseTrain:
    def test_shape(self):
        audio = generate_impulse_train(sample_rate=44100, duration_sec=1.0)
        assert audio.shape == (2, 44100)

    def test_impulse_spacing(self):
        """rate_hz에 맞는 간격으로 임펄스 배치"""
        sr = 44100
        rate = 10.0
        audio = generate_impulse_train(rate_hz=rate, sample_rate=sr, duration_sec=1.0)
        mono = audio[0]
        nonzero = np.nonzero(mono)[0]
        # 10Hz → 4410 샘플 간격
        expected_period = int(sr / rate)
        for i in range(1, len(nonzero)):
            assert nonzero[i] - nonzero[i - 1] == expected_period


class TestMidSide:
    def test_round_trip(self):
        """L/R → M/S → L/R 왕복"""
        original = np.random.randn(2, 1000).astype(np.float32)
        ms = to_mid_side(original)
        restored = from_mid_side(ms)
        np.testing.assert_allclose(restored, original, atol=1e-6)

    def test_mid_is_sum(self):
        audio = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]], dtype=np.float32)
        ms = to_mid_side(audio)
        expected_mid = (audio[0] + audio[1]) * 0.5
        np.testing.assert_allclose(ms[0], expected_mid)

    def test_side_is_difference(self):
        audio = np.array([[1.0, 2.0], [0.5, 1.0]], dtype=np.float32)
        ms = to_mid_side(audio)
        expected_side = (audio[0] - audio[1]) * 0.5
        np.testing.assert_allclose(ms[0 + 1], expected_side)

    def test_mono_raises(self):
        mono = np.random.randn(1, 100).astype(np.float32)
        with pytest.raises(ValueError, match="스테레오"):
            to_mid_side(mono)

    def test_from_mid_side_mono_raises(self):
        mono = np.random.randn(1, 100).astype(np.float32)
        with pytest.raises(ValueError, match="스테레오"):
            from_mid_side(mono)
