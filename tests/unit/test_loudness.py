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


# ---------------------------------------------------------------------------
# Additional numeric / boundary coverage
# ---------------------------------------------------------------------------


class TestConversionHelpers:
    def test_mono_roundtrip(self):
        mono = np.ones(10, dtype=np.float32)
        assert loudness._to_pyln(mono).ndim == 1
        assert loudness._to_audioman(mono, original_ndim=1).ndim == 1

    def test_stereo_roundtrip_transposes(self):
        stereo = np.ones((2, 10), dtype=np.float32)
        pyln = loudness._to_pyln(stereo)
        assert pyln.shape == (10, 2)
        back = loudness._to_audioman(pyln, original_ndim=2)
        assert back.shape == (2, 10)

    def test_to_audioman_2d_output_from_non_2d_input_kept(self):
        # original 1D → output passed through even if it came back 2D
        arr = np.ones((5, 2), dtype=np.float32)
        assert loudness._to_audioman(arr, original_ndim=1).shape == (5, 2)


class TestShortTermLufs:
    def test_too_short_returns_empty(self):
        audio = np.zeros(SR, dtype=np.float32)  # 1s < 3s window
        out = loudness.short_term_lufs(audio, SR)
        assert out.shape == (0,)

    def test_window_slides_over_long_signal(self):
        audio = _stereo_sine(0.5, duration=5.0)
        out = loudness.short_term_lufs(audio, SR, window_sec=3.0, hop_sec=0.5)
        assert len(out) > 0
        finite = out[np.isfinite(out)]
        assert len(finite) > 0
        # 0.5 amplitude stereo 1kHz sine sits in the expected LUFS band
        assert -12.0 < float(finite.mean()) < -5.0

    def test_silent_windows_are_minus_inf(self):
        audio = np.zeros((2, SR * 6), dtype=np.float32)
        out = loudness.short_term_lufs(audio, SR)
        assert len(out) > 0
        assert not np.any(np.isfinite(out))

    def test_mono_input(self):
        out = loudness.short_term_lufs(_sine(0.5, duration=5.0), SR)
        assert len(out) > 0


class TestLoudnessRangeEdges:
    def test_all_neg_inf_returns_zero(self):
        st = np.full(10, float("-inf"), dtype=np.float32)
        assert loudness.loudness_range(st) == 0.0

    def test_single_finite_sample_returns_zero(self):
        st = np.array([-16.0], dtype=np.float32)
        assert loudness.loudness_range(st) == 0.0

    def test_inf_entries_are_ignored(self):
        st = np.concatenate([
            np.array([float("-inf")] * 20, dtype=np.float32),
            np.full(50, -20.0, dtype=np.float32),
            np.full(50, -10.0, dtype=np.float32),
        ])
        lra = loudness.loudness_range(st)
        assert 8.0 < lra < 11.0


class TestTruePeakOversample:
    def test_oversample_one_uses_raw_samples(self):
        audio = _sine(0.5)
        tp_raw = loudness.true_peak_dbtp(audio, SR, oversample=1)
        assert tp_raw == pytest.approx(loudness.sample_peak_dbfs(audio), abs=1e-6)

    def test_higher_oversample_never_lower(self):
        audio = _sine(0.5)
        tp1 = loudness.true_peak_dbtp(audio, SR, oversample=1)
        tp4 = loudness.true_peak_dbtp(audio, SR, oversample=4)
        assert tp4 >= tp1 - 1e-6

    def test_silent_returns_minus_inf(self):
        assert loudness.true_peak_dbtp(np.zeros(1000, dtype=np.float32), SR) == float("-inf")

    def test_sample_peak_silent_returns_minus_inf(self):
        assert loudness.sample_peak_dbfs(np.zeros(100, dtype=np.float32)) == float("-inf")

    def test_true_peak_exceeds_sample_peak_for_nyquist_adjacent_tone(self):
        # two tones straddling Nyquist/2 create inter-sample overshoot
        sr = 48000
        t = np.arange(int(0.25 * sr)) / sr
        audio = (0.45 * (np.sin(2 * np.pi * 11500 * t) + np.sin(2 * np.pi * 12500 * t))).astype(np.float32)
        sp = loudness.sample_peak_dbfs(audio)
        tp = loudness.true_peak_dbtp(audio, sr, oversample=4)
        assert tp > sp


class TestLoudnessNormalizeTpAttenuation:
    def test_tp_attenuation_recorded_when_ceiling_exceeded(self):
        # tiny signal pushed hard to a loud target → TP ceiling must attenuate
        audio = _stereo_sine(0.08, duration=5.0)
        out, meta = loudness.loudness_normalize(
            audio, SR, target_lufs=-2.0, max_true_peak_dbtp=-3.0
        )
        assert meta["tp_limit_attenuation_db"] < 0.0
        assert loudness.true_peak_dbtp(out, SR) <= -3.0 + 0.05

    def test_no_attenuation_when_under_ceiling(self):
        audio = _stereo_sine(0.05, duration=5.0)
        _out, meta = loudness.loudness_normalize(
            audio, SR, target_lufs=-30.0, max_true_peak_dbtp=-1.0
        )
        assert meta["tp_limit_attenuation_db"] == 0.0

    def test_metadata_reports_targets(self):
        _out, meta = loudness.loudness_normalize(_stereo_sine(0.3, duration=5.0), SR,
                                                 target_lufs=-18.0, max_true_peak_dbtp=-2.0)
        assert meta["target_lufs"] == -18.0
        assert meta["max_true_peak_dbtp"] == -2.0
        assert "measured_in" in meta and "measured_out" in meta


class TestLevelUtterances:
    class _Seg:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    def _audio(self, sr=SR, duration=4.0):
        # loud first half, quiet second half — leveling should bring both toward target
        n = int(sr * duration)
        t = np.arange(n) / sr
        sig = np.zeros(n, dtype=np.float32)
        sig[: n // 2] = 0.6 * np.sin(2 * np.pi * 300 * t[: n // 2])
        sig[n // 2:] = 0.05 * np.sin(2 * np.pi * 300 * t[n // 2:])
        return np.stack([sig, sig])

    def test_length_preserved_and_segments_gained(self):
        audio = self._audio()
        segs = [self._Seg(0, SR * 2), self._Seg(SR * 2, SR * 4)]
        out, meta = loudness.level_utterances(audio, SR, speech_segments=segs, target_lufs=-20.0)
        assert out.shape == audio.shape
        assert meta["n_speech_segments"] == 2
        assert len(meta["per_segment"]) == 2

    def test_quiet_segment_gains_more_than_loud_segment(self):
        audio = self._audio()
        segs = [self._Seg(0, SR * 2), self._Seg(SR * 2, SR * 4)]
        _out, meta = loudness.level_utterances(audio, SR, speech_segments=segs, target_lufs=-20.0)
        loud_gain = meta["per_segment"][0]["applied_gain_db"]
        quiet_gain = meta["per_segment"][1]["applied_gain_db"]
        assert quiet_gain > loud_gain  # quieter source needs more boost

    def test_short_segment_skipped(self):
        audio = self._audio()
        segs = [self._Seg(0, 100), self._Seg(SR * 2, SR * 4)]  # 100 samples < min
        _out, meta = loudness.level_utterances(audio, SR, speech_segments=segs, min_segment_ms=200.0)
        assert meta["per_segment"][0]["skipped"] == "too_short_or_silent"
        assert meta["per_segment"][0]["applied_gain_db"] == 0.0

    def test_noise_gap_attenuated(self):
        audio = self._audio()
        # speech only in the middle; gaps are noise and get attenuated
        segs = [self._Seg(SR, SR * 2)]
        out, meta = loudness.level_utterances(
            audio, SR, speech_segments=segs, noise_attenuation_db=-20.0
        )
        assert meta["noise_attenuation_db"] == -20.0
        # head region (noise) quieter after attenuation
        head_rms = float(np.sqrt(np.mean(out[:, : SR // 2] ** 2)))
        orig_head_rms = float(np.sqrt(np.mean(audio[:, : SR // 2] ** 2)))
        assert head_rms < orig_head_rms

    def test_trailing_noise_after_last_segment(self):
        audio = self._audio()
        segs = [self._Seg(0, SR)]  # trailing 3s is noise
        _out, meta = loudness.level_utterances(audio, SR, speech_segments=segs)
        assert meta["n_speech_segments"] == 1

    def test_empty_segments_all_noise(self):
        audio = self._audio()
        out, meta = loudness.level_utterances(audio, SR, speech_segments=[])
        assert meta["n_speech_segments"] == 0
        assert out.shape == audio.shape

    def test_mono_segment_slicing(self):
        audio = self._audio()[0]
        out, _meta = loudness.level_utterances(audio, SR, speech_segments=[self._Seg(0, SR * 2)])
        assert out.shape == audio.shape

    def test_tp_ceiling_applied(self):
        # boost to a very loud target so the true-peak ceiling forces attenuation
        t = np.arange(int(2.0 * SR)) / SR
        audio = np.stack([0.5 * np.sin(2 * np.pi * 300 * t)] * 2).astype(np.float32)
        segs = [self._Seg(0, SR * 2)]
        _out, meta = loudness.level_utterances(
            audio, SR, speech_segments=segs, target_lufs=0.0, max_true_peak_dbtp=-1.0
        )
        assert meta["tp_limit_attenuation_db"] < 0.0

    def test_silent_segment_skipped_not_gained(self):
        t = np.arange(int(4.0 * SR)) / SR
        audio = np.stack([0.5 * np.sin(2 * np.pi * 300 * t)] * 2).astype(np.float32)
        audio[:, SR * 2:SR * 3] = 0.0  # silent, long enough segment
        _out, meta = loudness.level_utterances(
            audio, SR, speech_segments=[self._Seg(SR * 2, SR * 3)]
        )
        assert meta["per_segment"][0]["skipped"] == "too_short_or_silent"
        assert meta["per_segment"][0]["input_lufs"] is None

    def test_segments_sorted_by_start(self):
        audio = self._audio()
        segs = [self._Seg(SR * 2, SR * 4), self._Seg(0, SR * 2)]  # reversed order
        _out, meta = loudness.level_utterances(audio, SR, speech_segments=segs)
        starts = [s["start"] for s in meta["per_segment"]]
        assert starts == sorted(starts)


class TestShortTermErrorHandling:
    def test_short_block_value_error_becomes_minus_inf(self, monkeypatch):
        """pyloudnorm raises ValueError for blocks below its gating size."""
        class _RaisingMeter:
            def __init__(self, *a, **k):
                pass

            def integrated_loudness(self, block):
                raise ValueError("block too short")

        monkeypatch.setattr(loudness.pyloudnorm, "Meter", _RaisingMeter)
        out = loudness.short_term_lufs(_stereo_sine(0.5, duration=5.0), SR)
        assert len(out) > 0
        assert not np.any(np.isfinite(out))


class TestLevelUtterancesZeroLength:
    class _Seg:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    def test_zero_length_segment_is_ignored(self):
        t = np.arange(int(2.0 * SR)) / SR
        audio = np.stack([0.4 * np.sin(2 * np.pi * 300 * t)] * 2).astype(np.float32)
        # start == end → gain ramp helper must short-circuit
        segs = [self._Seg(SR, SR), self._Seg(SR, SR * 2)]
        out, meta = loudness.level_utterances(audio, SR, speech_segments=segs)
        assert out.shape == audio.shape
        assert len(meta["per_segment"]) == 2
