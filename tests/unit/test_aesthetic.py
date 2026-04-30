# tests/unit/test_aesthetic.py — audio aesthetic issue screening

import numpy as np
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
