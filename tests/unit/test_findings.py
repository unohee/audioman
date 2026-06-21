# Created: 2026-05-11
# Purpose: Finding 스키마 + signal/spectral detector 회귀.

import numpy as np
import pytest

from audioman.core.detectors import (
    detect_channel_imbalance,
    detect_clipping,
    detect_dc_offset,
    detect_signal_findings,
    silence_to_findings,
    spectrum_to_findings,
)
from audioman.core.analysis import SilenceRegion
from audioman.core.findings import (
    Category,
    Code,
    Finding,
    Severity,
    SCHEMA_URI,
    envelope,
    filter_findings,
)


SR = 48000


def _sine(freq=440.0, duration=1.0, amp=0.5, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return amp * np.sin(2 * np.pi * freq * t)


class TestFindingSerialization:
    def test_to_dict_has_required_fields(self):
        f = Finding(
            code=Code.CLIP_SAMPLE_PEAK_EXCEEDED,
            category=Category.SIGNAL,
            severity=Severity.CRITICAL,
            hint="x",
        )
        d = f.to_dict()
        assert d["code"] == "CLIP_SAMPLE_PEAK_EXCEEDED"
        assert d["category"] == "signal"
        assert d["severity"] == "critical"
        assert "id" in d
        assert "where" in d
        assert "measurement" in d

    def test_severity_rank(self):
        assert Severity.INFO.rank < Severity.WARN.rank < Severity.CRITICAL.rank

    def test_filter_findings_by_severity(self):
        items = [
            Finding(Code.SILENCE_LEADING, Category.SIGNAL, Severity.INFO),
            Finding(Code.DC_OFFSET_DETECTED, Category.SIGNAL, Severity.WARN),
            Finding(Code.CLIP_SAMPLE_PEAK_EXCEEDED, Category.SIGNAL, Severity.CRITICAL),
        ]
        warn = filter_findings(items, min_severity=Severity.WARN)
        assert len(warn) == 2
        crit = filter_findings(items, min_severity=Severity.CRITICAL)
        assert len(crit) == 1

    def test_filter_findings_by_category(self):
        items = [
            Finding(Code.SILENCE_LEADING, Category.SIGNAL, Severity.INFO),
            Finding(Code.MAINS_HUM, Category.SPECTRAL, Severity.WARN),
        ]
        signal = filter_findings(items, categories={"signal"})
        assert len(signal) == 1
        assert signal[0].category is Category.SIGNAL

    def test_envelope_has_schema_uri(self):
        env = envelope([])
        assert env["$schema"] == SCHEMA_URI
        assert "audioman_version" in env
        assert env["summary"]["total"] == 0


class TestClipDetector:
    def test_clip_finding_emitted(self):
        sig = np.clip(2.0 * _sine(), -1.0, 1.0)
        findings = detect_clipping(sig, SR, file="x.wav")
        assert len(findings) == 1
        f = findings[0]
        assert f.code is Code.CLIP_SAMPLE_PEAK_EXCEEDED
        assert f.measurement["samples_clipped"] > 0

    def test_no_clip_no_finding(self):
        sig = 0.5 * _sine()
        assert detect_clipping(sig, SR) == []


class TestDcOffsetDetector:
    def test_offset_above_threshold(self):
        sig = _sine() + 0.01
        findings = detect_dc_offset(sig, SR, file="x.wav")
        assert len(findings) == 1
        assert findings[0].code is Code.DC_OFFSET_DETECTED

    def test_clean_no_offset(self):
        sig = _sine()
        assert detect_dc_offset(sig, SR) == []


class TestChannelImbalance:
    def test_stereo_imbalance(self):
        left = _sine(amp=0.5)
        right = _sine(amp=0.1)
        stereo = np.stack([left, right])
        findings = detect_channel_imbalance(stereo, SR)
        assert len(findings) == 1
        assert findings[0].code is Code.CHANNEL_IMBALANCE

    def test_balanced_stereo_no_finding(self):
        sig = _sine()
        stereo = np.stack([sig, sig])
        assert detect_channel_imbalance(stereo, SR) == []

    def test_mono_skipped(self):
        assert detect_channel_imbalance(_sine(), SR) == []


class TestSilenceToFindings:
    def test_leading_and_trailing_classified(self):
        regions = [
            SilenceRegion(start_sample=0, end_sample=int(0.3 * SR), duration_sec=0.3),
            SilenceRegion(start_sample=int(0.9 * SR), end_sample=SR, duration_sec=0.1),
        ]
        findings = silence_to_findings(regions, SR, SR)
        codes = {f.code for f in findings}
        assert Code.SILENCE_LEADING in codes
        assert Code.SILENCE_TRAILING in codes

    def test_inner_long_silence_warns(self):
        regions = [
            SilenceRegion(start_sample=int(0.4 * SR), end_sample=int(0.41 * SR + SR), duration_sec=1.01),
        ]
        findings = silence_to_findings(regions, SR * 3, SR)
        assert len(findings) == 1
        assert findings[0].code is Code.SILENCE_INNER
        assert findings[0].severity is Severity.WARN


class TestSpectrumToFindings:
    def test_hum_finding(self):
        spec = {
            "hum_check": [{"frequency_hz": 60, "snr_db": 30.0, "is_hum": True}],
            "hf_slope": {"mid_db": -20.0, "high_db": -40.0, "slope_db": -20.0},
        }
        findings = spectrum_to_findings(spec)
        assert any(f.code is Code.MAINS_HUM for f in findings)


class TestSignalFindingsCombined:
    def test_clean_signal_emits_no_findings(self):
        sig = _sine()
        findings = detect_signal_findings(sig, SR)
        assert findings == []
