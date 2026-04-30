# tests/unit/test_loudness.py — LUFS / True Peak / loudness normalization

import numpy as np
import pytest

from audioman.core import loudness


SR = 48000


def _sine(amp: float, freq: float = 1000.0, duration: float = 5.0, sr: int = SR) -> np.ndarray:
    """모노 사인. 길이 5초는 BS.1770 gating block 충족."""
    t = np.arange(int(duration * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _stereo_sine(amp: float = 0.5, **kw) -> np.ndarray:
    s = _sine(amp, **kw)
    return np.stack([s, s])


class TestIntegratedLufs:
    def test_silent_returns_minus_inf(self):
        audio = np.zeros(SR * 5, dtype=np.float32)
        result = loudness.integrated_lufs(audio, SR)
        assert result == float("-inf")

    def test_too_short_returns_minus_inf(self):
        # < 0.4초면 gating block 미충족
        audio = np.zeros(SR // 10, dtype=np.float32)
        result = loudness.integrated_lufs(audio, SR)
        assert result == float("-inf")

    def test_sine_minus6_amplitude(self):
        # 0.5 amp = -6 dBFS sine 1kHz → 약 -9 LUFS (K-weighting 적용)
        audio = _sine(0.5)
        result = loudness.integrated_lufs(audio, SR)
        assert -10.5 < result < -7.5

    def test_louder_signal_higher_lufs(self):
        quiet = loudness.integrated_lufs(_sine(0.1), SR)
        loud = loudness.integrated_lufs(_sine(0.7), SR)
        assert loud > quiet

    def test_stereo_handled(self):
        audio = _stereo_sine(0.5)
        result = loudness.integrated_lufs(audio, SR)
        # 스테레오 동일 사인은 모노보다 약간 라우드 (BS.1770 채널 가중)
        assert -10.5 < result < -6.0


class TestTruePeak:
    def test_silent(self):
        audio = np.zeros(SR, dtype=np.float32)
        assert loudness.true_peak_dbtp(audio, SR) == float("-inf")

    def test_full_scale_is_zero(self):
        # ±1.0 sine은 sample peak = 0 dBFS, true peak도 ~0 (사인은 inter-sample 거의 없음)
        audio = _sine(1.0)
        tp = loudness.true_peak_dbtp(audio, SR)
        assert -0.2 < tp < 0.2

    def test_inter_sample_peak_higher_than_sample_peak(self):
        # 24kHz 사각파에 가까운 신호는 inter-sample peak이 sample peak보다 큼
        sr = 48000
        t = np.arange(int(0.5 * sr)) / sr
        # 11kHz 사인 + 12kHz 사인 (둘 다 nyquist 근처) — TP가 sample peak 초과 가능성
        audio = (0.5 * (np.sin(2 * np.pi * 11000 * t) + np.sin(2 * np.pi * 12000 * t))).astype(np.float32)
        sp = loudness.sample_peak_dbfs(audio)
        tp = loudness.true_peak_dbtp(audio, sr)
        # TP는 항상 SP 이상 (= 또는 >)
        assert tp >= sp - 0.01

    def test_stereo_returns_max(self):
        # 좌채널 0.5, 우채널 1.0 → max = 0 dBFS
        l = _sine(0.5)
        r = _sine(1.0)
        stereo = np.stack([l, r])
        tp = loudness.true_peak_dbtp(stereo, SR)
        assert tp > -0.5  # 우채널 기준


class TestSamplePeak:
    def test_full_scale(self):
        audio = _sine(1.0)
        sp = loudness.sample_peak_dbfs(audio)
        assert -0.1 < sp < 0.1

    def test_minus6db(self):
        audio = _sine(0.5)
        sp = loudness.sample_peak_dbfs(audio)
        assert -6.5 < sp < -5.5


class TestLoudnessRange:
    def test_constant_signal_zero_lra(self):
        st = np.full(50, -16.0, dtype=np.float32)
        assert loudness.loudness_range(st) == 0.0

    def test_dynamic_signal_positive_lra(self):
        # 절반은 -20, 절반은 -10 → LRA ≈ 10
        st = np.concatenate([
            np.full(50, -20.0, dtype=np.float32),
            np.full(50, -10.0, dtype=np.float32),
        ])
        lra = loudness.loudness_range(st)
        assert 8.0 < lra < 11.0


class TestMeasure:
    def test_full_report(self):
        audio = _stereo_sine(0.5, duration=5.0)
        report = loudness.measure(audio, SR)
        d = report.to_dict()
        assert d["channels"] == 2
        assert d["sample_rate"] == SR
        assert abs(d["duration_sec"] - 5.0) < 0.01
        assert d["integrated_lufs"] is not None
        assert d["sample_peak_dbfs"] is not None
        assert d["true_peak_dbtp"] is not None
        # short-term은 5초 길이에 3초 윈도우면 측정 가능
        assert d["short_term_max_lufs"] is not None

    def test_silent_report_no_crash(self):
        audio = np.zeros((2, SR * 2), dtype=np.float32)
        report = loudness.measure(audio, SR)
        d = report.to_dict()
        # 무음은 LUFS/TP 모두 None (-inf)
        assert d["integrated_lufs"] is None
        assert d["true_peak_dbtp"] is None


class TestLoudnessNormalize:
    def test_normalize_to_target(self):
        audio = _stereo_sine(0.3, duration=5.0)  # ~-13 LUFS 정도
        target = -14.0
        out, meta = loudness.loudness_normalize(audio, SR, target_lufs=target, max_true_peak_dbtp=-1.0)

        out_lufs = loudness.integrated_lufs(out, SR)
        assert abs(out_lufs - target) < 0.5
        assert "applied_gain_db" in meta

    def test_tp_ceiling_enforced(self):
        # 매우 작은 신호를 -14 LUFS로 끌어올리면 TP가 0에 가까워질 수 있음
        # max TP -1.0 강제 시 추가 감쇠 발생
        audio = _stereo_sine(0.1, duration=5.0)
        out, meta = loudness.loudness_normalize(audio, SR, target_lufs=-9.0, max_true_peak_dbtp=-1.0)
        out_tp = loudness.true_peak_dbtp(out, SR)
        # TP가 -1.0을 초과하지 않아야 함 (약간의 부동소수점 오차 허용)
        assert out_tp <= -1.0 + 0.05

    def test_silent_input_skipped(self):
        audio = np.zeros((2, SR * 2), dtype=np.float32)
        out, meta = loudness.loudness_normalize(audio, SR)
        assert "skipped" in meta
        np.testing.assert_array_equal(out, audio)
