# tests/unit/test_analysis.py

import numpy as np
import pytest
from audioman.core.analysis import compute_frame_metrics, compute_summary, detect_silence


class TestFrameMetrics:
    def test_returns_metrics(self, test_audio, sample_rate):
        metrics = compute_frame_metrics(test_audio, sample_rate)
        assert len(metrics.rms) > 0
        assert len(metrics.peak) > 0
        assert len(metrics.spectral_centroid) > 0
        assert len(metrics.spectral_entropy) > 0
        assert len(metrics.zero_crossing_rate) > 0

    def test_sine_centroid_near_440(self, test_audio, sample_rate):
        metrics = compute_frame_metrics(test_audio, sample_rate)
        avg_centroid = np.mean(metrics.spectral_centroid)
        # 440Hz 사인파 → centroid ~440Hz
        assert 400 < avg_centroid < 500

    def test_silent_audio_zero_rms(self, silent_audio, sample_rate):
        metrics = compute_frame_metrics(silent_audio, sample_rate)
        assert all(r < 1e-10 for r in metrics.rms)

    def test_frame_count(self, test_audio, sample_rate):
        metrics = compute_frame_metrics(test_audio, sample_rate, frame_size=2048, hop_size=512)
        expected = (sample_rate - 2048) // 512 + 1
        assert abs(len(metrics.rms) - expected) <= 1


class TestComputeSummary:
    def test_summary_keys(self, test_audio, sample_rate):
        metrics = compute_frame_metrics(test_audio, sample_rate)
        summary = compute_summary(metrics)
        for key in ["rms", "peak", "spectral_centroid", "spectral_entropy", "zero_crossing_rate"]:
            assert key in summary
            assert "mean" in summary[key]
            assert "min" in summary[key]
            assert "max" in summary[key]
            assert "std" in summary[key]

    def test_summary_values_positive(self, test_audio, sample_rate):
        metrics = compute_frame_metrics(test_audio, sample_rate)
        summary = compute_summary(metrics)
        assert summary["rms"]["mean"] > 0
        assert summary["peak"]["max"] > 0


class TestDetectSilence:
    def test_no_silence_in_tone(self, test_audio, sample_rate):
        regions = detect_silence(test_audio, sample_rate, threshold_db=-60.0)
        assert len(regions) == 0

    def test_detect_silence_in_silent(self, silent_audio, sample_rate):
        regions = detect_silence(silent_audio, sample_rate, threshold_db=-40.0)
        assert len(regions) > 0
        assert regions[0].duration_sec > 0.5

    def test_detect_silence_mixed(self, sample_rate):
        # 0.5초 무음 + 0.5초 톤
        audio = np.zeros((2, sample_rate), dtype=np.float32)
        t = np.linspace(0, 0.5, sample_rate // 2, dtype=np.float32)
        audio[:, sample_rate // 2 :] = 0.5 * np.sin(2 * np.pi * 440 * t)
        regions = detect_silence(audio, sample_rate, threshold_db=-40.0, min_duration_sec=0.1)
        assert len(regions) >= 1
        assert regions[0].start_sample == 0

    def test_silence_region_to_dict(self):
        region = detect_silence.__wrapped__ if hasattr(detect_silence, '__wrapped__') else None
        from audioman.core.analysis import SilenceRegion
        r = SilenceRegion(start_sample=0, end_sample=44100, duration_sec=1.0)
        d = r.to_dict()
        assert d["start_sample"] == 0
        assert d["duration_sec"] == 1.0
