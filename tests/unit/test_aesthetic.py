# tests/unit/test_aesthetic.py — audio aesthetic issue screening

import numpy as np
import pytest
import soundfile as sf

from audioman.core import aesthetic


SR = 48000


def _sine(amp: float = 0.2, freq: float = 1000.0, duration: float = 2.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    mono = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono, mono])


def _band_noise(
    low_hz: float,
    high_hz: float,
    *,
    duration: float = 1.0,
    amp: float = 0.05,
    sr: int = SR,
    seed: int = 1234,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    white = rng.standard_normal(n).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0
    noise = np.fft.irfft(spectrum, n=n).astype(np.float32)
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise = noise / peak * amp
    return noise


def test_screen_audio_detects_click_with_fallback():
    audio = _sine(duration=2.0)
    audio[:, SR] = 0.95

    report = aesthetic.screen_audio(audio, SR, issues=["click"], backend="fallback")

    assert report["backends"]["click"] == "fallback"
    assert report["summary"]["click"] >= 1
    assert any(abs(event["start_sec"] - 1.0) < 0.02 for event in report["events"])


def test_screen_audio_detects_hum_with_fallback():
    duration = 3.0
    t = np.arange(int(duration * SR)) / SR
    hum = 0.05 * np.sin(2 * np.pi * 60.0 * t)
    body = 0.01 * np.sin(2 * np.pi * 1000.0 * t)
    mono = (hum + body).astype(np.float32)
    audio = np.stack([mono, mono])

    report = aesthetic.screen_audio(audio, SR, issues=["hum"], backend="fallback")

    assert report["backends"]["hum"] == "fallback"
    assert report["summary"]["hum"] >= 1
    assert any(event.get("frequency_hz") == 60 for event in report["events"])


def test_screen_file_adds_file_path(tmp_path):
    path = tmp_path / "click.wav"
    audio = _sine(duration=1.0)
    audio[:, SR // 2] = 0.95
    sf.write(str(path), audio.T, SR, subtype="PCM_24")

    report = aesthetic.screen_file(path, issues=["click"], backend="fallback")

    assert report["file"] == str(path)
    assert report["sample_rate"] == SR
    assert report["summary"]["click"] >= 1


def test_screen_audio_detects_sibilance():
    audio = np.zeros((2, SR), dtype=np.float32)
    hiss = _band_noise(5000.0, 9000.0, duration=0.25, amp=0.25)
    start = int(0.4 * SR)
    audio[:, start : start + len(hiss)] = hiss

    report = aesthetic.screen_audio(audio, SR, issues=["de_ess"], backend="fallback")

    assert report["summary"]["sibilance"] >= 1
    assert report["backends"]["sibilance"] == "heuristic"


def test_screen_audio_detects_breath():
    audio = np.zeros((2, SR), dtype=np.float32)
    breath = _band_noise(1200.0, 7000.0, duration=0.35, amp=0.03)
    start = int(0.25 * SR)
    audio[:, start : start + len(breath)] = breath

    report = aesthetic.screen_audio(audio, SR, issues=["breath"], backend="fallback")

    assert report["summary"]["breath"] >= 1


def test_screen_audio_detects_background_noise():
    noise = _band_noise(80.0, 12000.0, duration=1.0, amp=0.015)
    audio = np.stack([noise, noise])

    report = aesthetic.screen_audio(audio, SR, issues=["background_noise"], backend="fallback")

    assert report["summary"]["background_noise"] >= 1


def test_screen_audio_detects_rf_noise():
    t = np.arange(SR * 2) / SR
    mono = (
        0.02 * np.sin(2 * np.pi * 8000.0 * t)
        + 0.018 * np.sin(2 * np.pi * 12300.0 * t)
        + 0.002 * np.sin(2 * np.pi * 500.0 * t)
    ).astype(np.float32)
    audio = np.stack([mono, mono])

    report = aesthetic.screen_audio(audio, SR, issues=["rf_noise"], backend="fallback")

    assert report["summary"]["rf_noise"] == 1
    assert len(report["events"][0]["tones"]) >= 2


def test_screen_audio_detects_mouth_click_with_fallback():
    audio = _sine(amp=0.02, freq=180.0, duration=1.0)
    click = np.array([0.0, 0.35, -0.28, 0.16, -0.05, 0.0], dtype=np.float32)
    pos = SR // 2
    audio[:, pos : pos + len(click)] += click

    report = aesthetic.screen_audio(audio, SR, issues=["mouth-click"], backend="fallback")

    assert report["summary"]["mouth_click"] >= 1


def test_unsupported_issue_is_reported():
    report = aesthetic.screen_audio(_sine(), SR, issues=["click", "unknown_issue"], backend="fallback")

    assert "unknown_issue" in report["unsupported_issues"]
    assert report["summary"]["unknown_issue"] == 0


# ---------------------------------------------------------------------------
# Helper functions, merge logic, essentia paths, and screening branches
# ---------------------------------------------------------------------------

from audioman.core.aesthetic import (  # noqa: E402
    AestheticEvent,
    _band_power,
    _db,
    _events_from_mask,
    _frame_bounds,
    _merge_events,
    _normalize_issue,
    _power_db,
    _spectral_flatness,
    _to_mono_float32,
    _zero_crossing_rate,
    essentia_available,
    detect_click_events,
    detect_hum_events,
    detect_mouth_click_events,
    detect_rf_noise_events,
    screen_audio,
)


def test_essentia_available_returns_bool():
    assert isinstance(essentia_available(), bool)


def test_event_to_dict_omits_none_confidence():
    e = AestheticEvent("click", 0.1, 0.2, confidence=None, detail=None)
    d = e.to_dict()
    assert "confidence" not in d
    assert d["type"] == "click"


def test_event_to_dict_merges_detail():
    e = AestheticEvent("hum", 0.0, 1.0, confidence=0.5, detail={"frequency_hz": 60})
    d = e.to_dict()
    assert d["confidence"] == 0.5
    assert d["frequency_hz"] == 60


def test_to_mono_float32_stereo_and_mono():
    stereo = np.stack([np.ones(4), np.zeros(4)]).astype(np.float32)
    assert _to_mono_float32(stereo).shape == (4,)
    assert _to_mono_float32(np.ones(4, dtype=np.float32)).shape == (4,)


@pytest.mark.parametrize("raw,expected", [
    ("de-ess", "sibilance"),
    ("De Ess", "sibilance"),
    ("mouth-click", "mouth_click"),
    ("rf", "rf_noise"),
    ("click", "click"),
])
def test_normalize_issue_aliases(raw, expected):
    assert _normalize_issue(raw) == expected


def test_db_floors_low_values():
    assert _db(0.0) == 20.0 * np.log10(1e-12)
    assert _power_db(0.0) == 10.0 * np.log10(1e-24)
    assert _db(1.0) == pytest.approx(0.0)
    assert _power_db(1.0) == pytest.approx(0.0)


class TestFrameBounds:
    def test_short_signal_returns_single_frame(self):
        assert _frame_bounds(100, 48000, 25.0, 10.0) == [(0, 100)]

    def test_empty_signal_returns_empty(self):
        assert _frame_bounds(0, 48000, 25.0, 10.0) == []

    def test_long_signal_slides_with_hop(self):
        bounds = _frame_bounds(48000, 48000, 25.0, 10.0)
        assert len(bounds) > 1
        assert bounds[0] == (0, 1200)
        assert bounds[1][0] == 480  # 10 ms hop @ 48 kHz


class TestFrameHelpers:
    def test_band_power_short_frame_zero(self):
        assert _band_power(np.ones(1, dtype=np.float32), 48000, (100.0, 1000.0)) == 0.0

    def test_band_power_out_of_range_band_zero(self):
        # band entirely above Nyquist → no bins match
        frame = np.ones(512, dtype=np.float32)
        assert _band_power(frame, 48000, (40000.0, 50000.0)) == 0.0

    def test_band_power_tone_energy_in_band(self):
        t = np.arange(1024) / 48000
        frame = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        in_band = _band_power(frame, 48000, (900.0, 1100.0))
        out_band = _band_power(frame, 48000, (5000.0, 6000.0))
        assert in_band > out_band

    def test_spectral_flatness_short_frame(self):
        assert _spectral_flatness(np.ones(1, dtype=np.float32)) == 0.0

    def test_spectral_flatness_noise_higher_than_tone(self):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(1024).astype(np.float32)
        t = np.arange(1024) / 48000
        tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        assert _spectral_flatness(noise) > _spectral_flatness(tone)

    def test_zero_crossing_rate_short_frame(self):
        assert _zero_crossing_rate(np.ones(1, dtype=np.float32)) == 0.0

    def test_zero_crossing_rate_alternating_signal_is_one(self):
        alt = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        assert _zero_crossing_rate(alt) == pytest.approx(1.0)


class TestMergeEvents:
    def test_empty_returns_empty(self):
        assert _merge_events([], max_gap_sec=0.1) == []

    def test_same_type_within_gap_merges(self):
        events = [
            AestheticEvent("click", 0.0, 0.1, confidence=0.3),
            AestheticEvent("click", 0.15, 0.2, confidence=0.9),
        ]
        merged = _merge_events(events, max_gap_sec=0.1, same_type="click")
        assert len(merged) == 1
        assert merged[0].end_sec == 0.2
        assert merged[0].confidence == 0.9

    def test_gap_too_large_keeps_separate(self):
        events = [
            AestheticEvent("click", 0.0, 0.1),
            AestheticEvent("click", 5.0, 5.1),
        ]
        assert len(_merge_events(events, max_gap_sec=0.1, same_type="click")) == 2

    def test_different_types_not_merged(self):
        events = [
            AestheticEvent("click", 0.0, 0.1),
            AestheticEvent("hum", 0.05, 0.2),
        ]
        merged = _merge_events(events, max_gap_sec=1.0)
        assert len(merged) == 2

    def test_max_gap_without_same_type_merges_only_matching_types(self):
        events = [
            AestheticEvent("click", 0.0, 0.1),
            AestheticEvent("click", 0.15, 0.2),
            AestheticEvent("hum", 0.16, 0.25),
        ]
        merged = _merge_events(events, max_gap_sec=0.1, same_type=None)
        # clicks merge (same type), hum stays separate despite overlapping time
        assert len(merged) == 2
        assert [m.type for m in merged] == ["click", "hum"]

    def test_confidence_none_kept_from_prev(self):
        events = [
            AestheticEvent("click", 0.0, 0.1, confidence=None),
            AestheticEvent("click", 0.12, 0.2, confidence=None),
        ]
        merged = _merge_events(events, max_gap_sec=0.1, same_type="click")
        assert merged[0].confidence is None


class TestEventsFromMaskBranches:
    def test_empty_mask_returns_empty(self):
        assert _events_from_mask([], [], 48000, event_type="click",
                                 min_duration_sec=0.1, max_gap_sec=0.1,
                                 severity="warn", backend="x") == []


class TestEssentiaFallbackPaths:
    def test_click_essentia_backend_reraises(self, monkeypatch):
        # backend="essentia" and essentia missing → ImportError propagates
        audio = np.zeros((2, 4800), dtype=np.float32)
        with pytest.raises(Exception):
            detect_click_events(audio, 48000, backend="essentia")

    def test_hum_essentia_backend_reraises(self):
        audio = np.zeros((2, 4800), dtype=np.float32)
        with pytest.raises(Exception):
            detect_hum_events(audio, 48000, backend="essentia")

    def test_auto_backend_falls_back_on_import_error(self):
        audio = np.zeros((2, 4800), dtype=np.float32)
        _events, used = detect_click_events(audio, 48000, backend="auto")
        assert used == "fallback"


class TestRFNoiseBranches:
    def test_short_signal_returns_empty(self):
        events, backend = detect_rf_noise_events(np.zeros(1024, dtype=np.float32), 48000)
        assert events == []
        assert backend == "heuristic"

    def test_explicit_max_frequency_respected(self):
        t = np.arange(48000) / 48000
        mono = (0.02 * np.sin(2 * np.pi * 8000 * t)).astype(np.float32)
        events, backend = detect_rf_noise_events(
            np.stack([mono, mono]), 48000, min_tones=1,
            min_frequency=1000.0, max_frequency=10000.0,
        )
        assert backend == "heuristic"
        for e in events:
            for tone in e.detail["tones"]:
                assert 1000.0 <= tone["frequency_hz"] <= 10000.0


class TestMouthClickBranches:
    def test_quiet_click_filtered_by_rms_floor(self):
        # a click inside a very quiet frame is below the -65 dB gate
        audio = _sine(amp=0.00005, freq=180.0, duration=1.0)
        click = np.array([0.0, 0.0001, -0.0001, 0.0], dtype=np.float32)
        pos = SR // 2
        audio[:, pos:pos + len(click)] += click
        events, _used = detect_mouth_click_events(audio, SR, backend="fallback")
        assert events == []

    def test_no_click_events_empty(self):
        events, _used = detect_mouth_click_events(_sine(amp=0.2), SR, backend="fallback")
        assert events == []


class TestScreenAudioBranches:
    def test_events_sorted_and_summarized(self):
        audio = _sine(duration=2.0)
        audio[:, SR] = 0.95
        report = screen_audio(audio, SR, issues=["click", "hum"], backend="fallback")
        assert set(report["issues"]) == {"click", "hum"}
        assert report["summary"]["click"] >= 1
        starts = [e["start_sec"] for e in report["events"]]
        assert starts == sorted(starts)

    def test_mono_channel_count(self):
        report = screen_audio(_sine()[:, 0], SR, issues=["click"], backend="fallback")
        assert report["channels"] == 1

    def test_essentia_available_none_for_fallback(self):
        report = screen_audio(_sine(), SR, issues=["click"], backend="fallback")
        assert report["essentia_available"] is None

    def test_non_fallback_reports_availability_bool(self):
        report = screen_audio(_sine(), SR, issues=["click"], backend="auto")
        assert isinstance(report["essentia_available"], bool)

    def test_duplicate_issues_deduped(self):
        report = screen_audio(_sine(), SR, issues=["click", "click"], backend="fallback")
        assert report["issues"] == ["click"]

    def test_blank_issue_ignored(self):
        report = screen_audio(_sine(), SR, issues=["click", "  "], backend="fallback")
        assert report["issues"] == ["click"]

    def test_duration_and_all_issue_routes(self):
        # every supported issue routes through its detector without crashing
        report = screen_audio(_sine(), SR, backend="fallback")
        assert report["duration"] == pytest.approx(2.0, abs=0.01)
        assert set(report["backends"].keys()) <= set(report["issues"])


# ---------------------------------------------------------------------------
# Essentia-backed detectors: exercised with an injected fake `essentia.standard`.
# ---------------------------------------------------------------------------

import sys  # noqa: E402
import types  # noqa: E402


class _FakeClickDetector:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def reset(self):
        pass

    def __call__(self, frame):
        return [0.25], [0.26]


def _fake_frame_generator(mono, frameSize=512, hopSize=256, startFromZero=True):
    for start in range(0, max(1, len(mono) - frameSize + 1), hopSize):
        yield mono[start:start + frameSize]


class _FakeHumDetector:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, mono):
        import numpy as np
        return (
            None,
            np.array([60.0, 120.0]),
            np.array([0.9, 0.4]),
            np.array([0.0, 2.0]),
            np.array([0.4, 2.4]),
        )


@pytest.fixture
def fake_essentia(monkeypatch):
    std = types.ModuleType("essentia.standard")
    std.ClickDetector = _FakeClickDetector
    std.FrameGenerator = _fake_frame_generator
    std.HumDetector = _FakeHumDetector
    essentia = types.ModuleType("essentia")
    essentia.standard = std
    monkeypatch.setitem(sys.modules, "essentia", essentia)
    monkeypatch.setitem(sys.modules, "essentia.standard", std)
    return std


def test_essentia_available_true_with_fake_module(fake_essentia):
    assert essentia_available() is True


def test_click_events_via_essentia_backend(fake_essentia):
    audio = _sine(duration=1.0)
    events, used = detect_click_events(audio, SR, backend="essentia")
    assert used == "essentia"
    assert len(events) >= 1
    assert events[0].backend == "essentia"
    assert events[0].detail["detector"] == "essentia.ClickDetector"


def test_hum_events_via_essentia_backend(fake_essentia):
    audio = _sine(duration=1.0)
    events, used = detect_hum_events(audio, SR, backend="essentia")
    assert used == "essentia"
    assert len(events) == 2  # 1.6 s apart → above the 0.25 s merge gap
    assert events[0].detail["frequency_hz"] == 60.0
    assert events[0].severity == "fail"      # salience 0.9 >= 0.75
    assert events[1].severity == "warn"      # salience 0.4 < 0.75


def test_auto_backend_uses_essentia_when_available(fake_essentia):
    _events, used = detect_click_events(_sine(duration=1.0), SR, backend="auto")
    assert used == "essentia"


class TestRFNoiseSingleTone:
    def test_fewer_than_min_tones_returns_empty(self):
        t = np.arange(48000) / 48000
        # single strong 8 kHz tone; default min_tones=2 must reject it
        mono = (0.05 * np.sin(2 * np.pi * 8000 * t)).astype(np.float32)
        events, backend = detect_rf_noise_events(np.stack([mono, mono]), 48000)
        assert events == []
        assert backend == "heuristic"


class TestMouthClickTinyFrame:
    def test_frame_below_16_samples_skipped(self, monkeypatch):
        from audioman.core import aesthetic as aes
        monkeypatch.setattr(
            aes, "detect_click_events",
            lambda audio, sample_rate, **kw: ([
                aes.AestheticEvent("click", 0.25, 0.26, backend="fallback"),
            ], "fallback"),
        )
        # pad = max(1, 0.012 * 400) = 4 → frame at most 8 samples < 16
        audio = np.zeros(400, dtype=np.float32)
        events, _used = aes.detect_mouth_click_events(audio, 400, backend="fallback")
        assert events == []
