"""Pitch descriptors (YIN f0 track), the envelope alignment fallback, perceptual minimums and figure states."""

from __future__ import annotations

import numpy as np
import pytest

from audioman.core.diff import _align_reason, _envelope_lag, align, diff_audio
from audioman.core.diff_figure import _state, render
from audioman.core.digest import digest_diff, digest_texture
from audioman.core.pdc import LagEstimate
from audioman.core.texture import describe
from audioman.core.texture import pitch as pt
from audioman.core.vocabulary import propose_terms

SR = 48000


def _tone(f0, seconds=3.0, sr=SR, harmonics=5):
    t = np.arange(int(seconds * sr)) / sr
    phase = 2 * np.pi * np.cumsum(np.broadcast_to(f0, t.shape)) / sr
    return 0.3 * sum(np.sin(k * phase) / k for k in range(1, harmonics + 1))


def _cents(f):
    return 1200 * np.log2(f / 440.0)


@pytest.mark.parametrize("f0", [110.0, 220.0, 440.0])
def test_known_f0_is_tracked_to_within_5_cents_with_a_small_floor(f0):
    d = describe(np.stack([_tone(f0)] * 2), SR)["descriptors"]
    assert abs(d["pitch"]["value"] - _cents(f0)) < 5
    assert d["pitch"]["floor"] < 2
    assert d["voiced_fraction"]["value"] > 0.9
    assert d["pitch_deviation"]["value"] < 2


def test_vibrato_widens_spread_and_detuning_shows_as_deviation():
    t = np.arange(3 * SR) / SR
    vibrato = describe(_tone(440 * 2 ** (50 / 1200 * np.sin(2 * np.pi * 6 * t))), SR)["descriptors"]
    steady = describe(_tone(440.0), SR)["descriptors"]
    assert vibrato["pitch_spread"]["value"] > 40 > steady["pitch_spread"]["value"]
    detuned = describe(_tone(440 * 2 ** (30 / 1200)), SR)["descriptors"]
    assert abs(detuned["pitch_deviation"]["value"] - 30) < 2


def test_noise_has_no_pitch():
    noise = 0.1 * np.random.default_rng(0).standard_normal(3 * SR)
    d = describe(noise, SR)["descriptors"]
    assert d["pitch"]["value"] is None and d["pitch"]["floor"] is None
    assert d["voiced_fraction"]["value"] < pt.MIN_VOICED_FRACTION


def test_a_chord_reports_the_common_subharmonic():
    # documented limit: a chord is periodic at the subharmonic its notes share
    chord = _tone(261.63) + _tone(329.63) + _tone(392.0)
    d = describe(chord, SR)["descriptors"]
    assert d["pitch"]["value"] is not None and d["pitch"]["value"] < _cents(80.0)
    assert "subharmonic" in describe(chord, SR)["pitch_scope"]


def test_track_edge_cases():
    short = pt.track(np.zeros(100), SR)
    assert len(short.times_sec) == 0
    assert pt.descriptors(short, 0, 1) == {"pitch": None, "pitch_spread": None, "pitch_deviation": None,
                                           "voiced_fraction": None}
    silent = pt.track(np.zeros(SR), SR)
    assert silent.silent.all() and not silent.voiced.any()
    native = pt.track(_tone(220.0, sr=pt.TRACK_RATE, seconds=1.0), pt.TRACK_RATE)  # no resampling
    assert abs(_cents(np.nanmedian(native.f0_hz)) - _cents(220.0)) < 5
    # two voiced frames are not enough to state a pitch
    two = pt.PitchTrack(np.array([0.1, 0.2]), np.array([220.0, 220.0]), np.zeros(2), np.ones(2, bool),
                        np.zeros(2, bool))
    assert pt.descriptors(two, 0, 1) == {"pitch": None, "pitch_spread": None, "pitch_deviation": None,
                                         "voiced_fraction": 1.0}


def _sung(cents_offset, seconds=4.0):
    t = np.arange(int(seconds * SR)) / SR
    notes = np.array([-12.0, -8.0, -5.0, -10.0])
    semis = notes[np.minimum((t / (seconds / 4)).astype(int), 3)]
    f0 = 440 * 2 ** ((100 * semis + cents_offset) / 1200)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    env = np.minimum(1, 20 * (t % 1.0)) * np.minimum(1, 20 * (1 - t % 1.0))
    return 0.2 * env * sum(np.sin(k * phase) / k for k in range(1, 7))


def test_envelope_fallback_recovers_the_lag_when_the_waveform_was_rewritten():
    a = _sung(0.0)
    lag = 480
    b = np.concatenate([np.zeros(lag), _sung(30.0)])[: len(a)]  # retuned and late by 10 ms
    a2, b2, info = align(np.stack([a, a]), np.stack([b, b]), SR)
    assert info["method"] == "envelope" and info["applied"]
    assert abs(info["lag_samples"] - lag) <= SR // 1000  # within 1 ms
    assert "lag from the RMS envelopes" in info["reason"]
    result = diff_audio(np.stack([a, a]), np.stack([b, b]), SR)
    assert "from the envelopes" in digest_diff(result)
    assert result["descriptors"]["pitch"]["significant"] and result["descriptors"]["pitch"]["delta"] > 20
    assert "pitched up" in {t["term"] for t in result["vocabulary"]["terms"]}


def test_envelope_fallback_declines_without_a_shared_envelope():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((2, 2 * SR)) * 0.1
    _, _, info = align(a, b, SR)
    assert info["method"] is None and not info["applied"] and info["envelope_ncc"] is not None
    assert _envelope_lag(np.zeros((1, 100)), np.zeros((1, 100)), SR, 500.0) == (None, None)
    edge = LagEstimate(lag_samples=10, ncc=0.9, at_edge=True, max_lag_samples=10, settle_samples=0)
    assert _align_reason(edge, None, None) == "lag on the search edge"


def test_pitch_terms_need_a_perceptible_change():
    def diff(**d):
        return {"descriptors": {k: {"delta": v, "significant": True, "floor": 0.1, "unit": "cents", "a": 0.0}
                                for k, v in d.items()}, "named_bands": []}

    corrected = propose_terms(diff(pitch_deviation=-20.0, pitch_spread=-30.0))
    assert [t["term"] for t in corrected["terms"]] == ["pitch-corrected"]
    tiny = propose_terms(diff(pitch=4.0))
    assert tiny["terms"] == [] and tiny["below_perceptual_minimum"][0]["minimum"] == "10 cents"
    # a relative minimum needs A; without it the change is not shown to be perceptible
    no_a = {"descriptors": {"spectral_centroid": {"delta": -500.0, "significant": True, "floor": 0.1,
                                                  "unit": "Hz", "a": None}}, "named_bands": []}
    assert propose_terms(no_a)["below_perceptual_minimum"][0]["minimum"] == "3% of A"


def test_figure_marks_changes_below_the_per_unit_minimum(tmp_path):
    assert _state({"significant": True, "delta": 1.0, "floor": 0.1}) == "significant"
    assert _state({"significant": False, "delta": 0.04, "floor": 0.001}) == "small"
    assert _state({"significant": False, "delta": 0.001, "floor": 0.01}) == "noise"
    assert _state({"significant": False, "delta": None, "floor": None}) == "noise"
    a = _tone(220.0, seconds=2.0)
    out = render(diff_audio(np.stack([a, a]), np.stack([a * 1.001, a * 1.001]), SR), tmp_path / "f.png")
    assert out.stat().st_size > 0


def test_floors_never_round_to_zero_and_are_always_printed():
    d = describe(np.stack([_tone(220.0)] * 2), SR)
    assert all(e["floor"] is None or e["floor"] >= 1e-4 for e in d["descriptors"].values())
    assert all(f >= 1e-4 for f in d["erb_profile"]["floor_db"])
    assert "(+/-" in digest_texture(d) and "pitch " in digest_texture(d)


def test_figure_handles_unknown_floors(tmp_path):
    # floors are None (0 after summing) when neither file defines one: drawn at the clip edge
    result = {"erb_bands": [{"center_hz": 1000.0, "delta": 0.1, "floor": 0.01, "significant": True}], "descriptors": {
        "up": {"delta": 1.0, "floor": 0.0, "significant": True},
        "down": {"delta": -1.0, "floor": 0.0, "significant": True},
        "flat": {"delta": 0.0, "floor": 0.0, "significant": False},
        "undefined": {"delta": None, "floor": None, "significant": False}}}
    assert render(result, tmp_path / "g.png").stat().st_size > 0
