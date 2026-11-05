"""Texture descriptors: per-segment values, repeat floors, and direction on synthetic signals."""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf
from scipy import signal as ss

from audioman.core.texture import FRAME, MIN_FLOOR, PITCH_DESCRIPTORS, describe, to_measurements
from audioman.core.texture import spectral as sp

SR = 48000
N = SR * 4
_PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
_PINK_A = [1, -2.494956002, 2.017265875, -0.522189400]


def _pink(seed: int = 0, n: int = N) -> np.ndarray:
    x = ss.lfilter(_PINK_B, _PINK_A, np.random.default_rng(seed).standard_normal(n))
    return 0.5 * x / np.max(np.abs(x))


def _sine(freq: float = 220.0, amp: float = 0.3, n: int = N) -> np.ndarray:
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / SR)


def _lowpass(x: np.ndarray, hz: float = 2000.0) -> np.ndarray:
    return ss.sosfilt(ss.butter(6, hz, fs=SR, output="sos"), x)


def _clicks(per_sec: float, n: int = N, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train = np.zeros(n)
    train[:: int(SR / per_sec)] = 1.0
    burst = 0.5 * np.exp(-np.arange(400) / 40.0) * rng.standard_normal(400)
    return np.convolve(train, burst)[:n]


def _d(audio) -> dict:
    return describe(audio, SR)["descriptors"]


def _moved(a: dict, b: dict, name: str, direction: int) -> None:
    """b moved from a in `direction` by more than three times the larger repeat floor."""
    va, vb = a[name]["value"], b[name]["value"]
    floor = max(a[name]["floor"] or 0.0, b[name]["floor"] or 0.0)
    delta = vb - va
    assert np.sign(delta) == direction, f"{name}: {va} -> {vb}"
    assert abs(delta) > 3 * floor, f"{name}: |{delta}| within 3x floor {floor}"


# ---------------------------------------------------------------------------
# Direction on synthetic processing
# ---------------------------------------------------------------------------


def test_lowpass_darkens():
    a, b = _d(_pink()), _d(_lowpass(_pink()))
    for name in ("spectral_centroid", "spectral_rolloff", "spectral_bandwidth", "hf_ratio", "sharpness",
                 "spectral_flatness"):
        _moved(a, b, name, -1)
    _moved(a, b, "spectral_contrast", +1)


def test_saturation_adds_harmonics_and_compresses_peaks():
    clean = _sine()
    driven = np.tanh(6 * clean) / np.tanh(6 * 0.3) * 0.3
    a, b = _d(clean), _d(driven)
    for name in ("spectral_centroid", "spectral_bandwidth", "hf_ratio", "sharpness", "spectral_flatness"):
        _moved(a, b, name, +1)
    _moved(a, b, "crest_factor", -1)


def test_added_noise_makes_it_noisier():
    tone = _sine()
    noisy = tone + 0.03 * np.random.default_rng(2).standard_normal(N)
    a, b = _d(tone), _d(noisy)
    for name in ("spectral_flatness", "spectral_flux", "hf_ratio", "spectral_bandwidth", "spectral_centroid"):
        _moved(a, b, name, +1)
    _moved(a, b, "spectral_contrast", -1)


def test_decorrelated_channels_widen_the_image():
    narrow = np.stack([_pink(0), _pink(0)])
    wide = np.stack([_pink(0), _pink(7)])
    a, b = _d(narrow), _d(wide)
    assert a["stereo_width"]["value"] == -60.0 and a["lr_correlation"]["value"] == 1.0
    _moved(a, b, "stereo_width", +1)
    _moved(a, b, "lr_correlation", -1)


def test_more_transients_raise_onset_density():
    a, b = _d(_clicks(2)), _d(_clicks(8))
    _moved(a, b, "onset_density", +1)
    assert 7.0 <= b["onset_density"]["value"] <= 9.0


def test_erb_profile_shows_where_the_energy_went():
    full = describe(_pink(), SR)["erb_profile"]
    dark = describe(_lowpass(_pink()), SR)["erb_profile"]
    centers = np.array(full["centers_hz"])
    high = centers > 6000
    low = centers < 1000
    drop = np.array(full["level_db"]) - np.array(dark["level_db"])
    assert drop[high].min() > 20.0
    assert abs(drop[low]).max() < 3.0
    assert all(f is not None and f >= 0 for f in full["floor_db"])


# ---------------------------------------------------------------------------
# Segments and floors
# ---------------------------------------------------------------------------


def test_segments_follow_a_change_over_time():
    x = np.concatenate([_lowpass(_pink(0, SR * 2)), _pink(1, SR * 2)])
    out = describe(x, SR, segment_sec=1.0)
    assert [s["start_sec"] for s in out["segments"]] == [0.0, 1.0, 2.0, 3.0]
    centroid = out["descriptors"]["spectral_centroid"]["segments"]
    assert len(centroid) == 4
    assert max(centroid[:2]) < min(centroid[2:])  # darker first half, brighter second half


def test_every_descriptor_carries_a_non_negative_floor():
    out = describe(np.stack([_pink(0), _pink(3)]), SR)
    for name, d in out["descriptors"].items():
        if d["value"] is None:  # pitch on noise: undefined, so no floor either
            assert name in PITCH_DESCRIPTORS and d["floor"] is None, name
            continue
        # never below the reporting resolution: a floor never rounds away to 0
        assert d["floor"] is not None and d["floor"] >= MIN_FLOOR, name
        assert d["segment_floor"] is not None and d["segment_floor"] >= MIN_FLOOR, name
        assert len(d["segments"]) == len(out["segments"])
    assert out["floor_method"].startswith("re-measured")


def test_rolloff_floor_is_at_least_one_bin():
    out = describe(_sine(), SR)  # stationary tone: re-measuring changes nothing
    assert out["descriptors"]["spectral_rolloff"]["floor"] == pytest.approx(SR / FRAME, abs=1e-4)
    assert out["descriptors"]["spectral_centroid"]["floor"] < 0.01


def test_floor_is_larger_for_unstable_material():
    steady = _d(_sine())["crest_factor"]["floor"]
    clicks = _d(_clicks(2))["rms"]["floor"]
    assert clicks > steady


def test_mono_silence_and_short_inputs():
    mono = _d(_pink())
    assert mono["stereo_width"]["value"] is None and mono["lr_correlation"]["value"] is None

    silent = describe(np.zeros((2, SR)), SR)
    for name in ("spectral_centroid", "rms", "crest_factor", "lr_correlation"):
        assert silent["descriptors"][name]["value"] is None, name
    assert silent["descriptors"]["stereo_width"]["value"] is None
    assert silent["descriptors"]["onset_density"]["value"] == 0.0

    short = describe(_sine(n=1000), SR)  # shorter than one analysis frame
    assert len(short["segments"]) == 1
    assert short["descriptors"]["spectral_centroid"]["value"] == pytest.approx(220.0, rel=0.2)

    nan = _pink()
    nan[100] = np.nan
    assert describe(nan, SR)["descriptors"]["rms"]["value"] is not None


def test_side_only_signal_is_widest_and_one_channel_silent_has_no_correlation():
    p = _pink()
    out = _d(np.stack([p, -p]))
    assert out["stereo_width"]["value"] == -60.0 or out["stereo_width"]["value"] is None
    left_only = _d(np.stack([p, np.zeros_like(p)]))
    assert left_only["lr_correlation"]["value"] is None
    assert left_only["stereo_width"]["value"] == pytest.approx(0.0, abs=1e-6)


def test_spectral_helpers_edge_cases():
    freqs = np.fft.rfftfreq(64, 1 / SR)
    P = np.ones((3, len(freqs)))
    assert np.all(sp.contrast_db(P, freqs, low_hz=20000.0) == 0.0)  # no octave band at all
    assert np.all(sp.contrast_db(P, freqs) == 0.0)  # bands exist but hold < 5 bins at 750 Hz/bin
    assert len(sp.onset_times(P[:2], 512, SR)) == 0  # too few frames
    coarse = np.fft.rfftfreq(16, 1 / SR)
    narrow = sp.erb_profile_db(np.ones((3, len(coarse))), coarse, np.array([60.0, 100.0]))
    assert np.all(np.isfinite(narrow))  # bands narrower than a bin fall back to the nearest bin


def test_to_measurements_uses_repeat_floors():
    tex = describe(np.stack([_pink(0), _pink(4)]), SR)
    ms = to_measurements(tex)
    assert {m.floor_kind for m in ms} == {"repeat"}
    assert {m.name for m in ms} >= {"texture.spectral_centroid", "texture.stereo_width"}
    tex["descriptors"]["rms"]["floor"] = None
    tex["descriptors"]["sharpness"]["value"] = None
    ms = {m.name: m for m in to_measurements(tex)}
    assert ms["texture.rms"].floor_kind == "unknown" and "texture.sharpness" not in ms


# ---------------------------------------------------------------------------
# CLI: analyze --texture, and report.v2 carries the floors
# ---------------------------------------------------------------------------


def _run(argv):
    from audioman.cli import app

    out = io.StringIO()
    real = sys.stdout
    sys.stdout = out
    plain = os.environ.get("AUDIOMAN_PLAIN")
    try:
        app.main(argv)
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdout = real
        if plain is None:
            os.environ.pop("AUDIOMAN_PLAIN", None)
    return code, out.getvalue()


def test_analyze_texture_flag(tmp_path, monkeypatch):
    from audioman.config import settings as settings_module
    from test_json_contract import _resolve_ref, validate_against

    monkeypatch.setenv("AUDIOMAN_CACHE_DIR", str(tmp_path / "cache"))
    settings_module.reset_settings()
    path = tmp_path / "pink.wav"
    sf.write(path, np.stack([_pink(0), _pink(5)]).T.astype(np.float32), SR, subtype="FLOAT")
    try:
        code, text = _run(["--json", "analyze", str(path), "--texture", "--segment-sec", "2", "--detail", "full"])
        assert code == 0
        payload = json.loads(text)
        tex = payload["texture"]
        assert tex["segment_sec"] == 2.0 and len(tex["segments"]) == 2
        report = payload["report"]
        validate_against(report, _resolve_ref("audioman://schema/report.v2.json"))
        centroid = next(m for m in report["measurements"] if m["name"] == "texture.spectral_centroid")
        assert centroid["floor_kind"] == "repeat" and centroid["floor"] == tex["descriptors"]["spectral_centroid"]["floor"]

        code, text = _run(["--json", "analyze", str(path)])
        assert code == 0 and "texture" not in json.loads(text)
    finally:
        settings_module.reset_settings()
