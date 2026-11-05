"""Experimental descriptor-to-word mapping: terms only from significant evidence, with evidence attached."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as ss

from audioman.core.diff import diff_audio
from audioman.core.vocabulary import EXPERIMENTAL_NOTE, RULES, propose_terms

SR = 48000


def _desc(delta, significant=True, floor=0.1, unit="Hz", a=1000.0):
    return {"delta": delta, "significant": significant, "floor": floor, "unit": unit, "a": a}


def _diff(**descriptors):
    return {"descriptors": descriptors, "named_bands": []}


def test_term_needs_a_significant_primary_cue_in_its_direction():
    assert propose_terms(_diff(spectral_centroid=_desc(-500, significant=False)))["terms"] == []
    assert propose_terms(_diff(spectral_centroid=_desc(-500)))["terms"][0]["term"] == "darker"
    assert [t["term"] for t in propose_terms(_diff(spectral_centroid=_desc(+500)))["terms"]] == ["brighter"]


def test_confidence_counts_agreeing_and_contradicting_evidence():
    agree = propose_terms(_diff(spectral_centroid=_desc(-500), hf_ratio=_desc(-6, unit="dB"),
                                sharpness=_desc(-0.5, unit="acum"), spectral_rolloff=_desc(-900)))
    darker = agree["terms"][0]
    mixed = propose_terms(_diff(spectral_centroid=_desc(-500), hf_ratio=_desc(+6, unit="dB"),
                                sharpness=_desc(-0.5, unit="acum"), spectral_rolloff=_desc(-900)))["terms"][0]
    assert darker["confidence"] > mixed["confidence"] > 0
    hf = next(e for e in mixed["evidence"] if e["cue"] == "hf_ratio")
    assert hf["contradicts"] and not hf["supports"]
    assert agree["contradicted"] == []


def test_terms_the_evidence_argues_against_are_not_proposed():
    # the primary cue moved, but as many significant cues point the other way
    out = propose_terms(_diff(spectral_centroid=_desc(-500), hf_ratio=_desc(+6, unit="dB")))
    assert out["terms"] == []
    [entry] = out["contradicted"]
    assert entry["term"] == "darker" and "confidence" not in entry
    assert {e["cue"]: e["contradicts"] for e in entry["evidence"]} == {"spectral_centroid": False, "hf_ratio": True}


def test_weak_primary_lowers_confidence_and_zero_floor_counts_as_strong():
    weak = propose_terms(_diff(rms=_desc(-0.6, floor=0.1, unit="dBFS")))["terms"][0]
    strong = propose_terms(_diff(rms=_desc(-6.0, floor=0.1, unit="dBFS")))["terms"][0]
    exact = propose_terms(_diff(rms=_desc(-6.0, floor=0.0, unit="dBFS")))["terms"][0]
    assert weak["confidence"] < strong["confidence"] == exact["confidence"] == 1.0


def test_named_band_cues_and_missing_bands():
    with_band = {"descriptors": {"spectral_centroid": _desc(-500)},
                 "named_bands": [{"band": "air", "delta_db": -20.0, "significant": True, "floor_db": 0.1}]}
    evidence = {e["cue"]: e for e in propose_terms(with_band)["terms"][0]["evidence"]}
    assert evidence["band:air"]["supports"]
    without = propose_terms({"descriptors": {"spectral_centroid": _desc(-500)}, "named_bands": []})
    assert "band:air" not in {e["cue"] for e in without["terms"][0]["evidence"]}


def test_undefined_deltas_are_ignored():
    assert propose_terms(_diff(stereo_width={"delta": None, "significant": False}))["terms"] == []


def test_every_rule_has_its_opposite_rule():
    terms = {r.term for r in RULES}
    for rule in RULES:
        assert rule.opposite in terms or rule.opposite == "purer"
    assert propose_terms(_diff())["note"] == EXPERIMENTAL_NOTE


def test_diff_results_carry_vocabulary_end_to_end():
    rng = np.random.default_rng(0)
    x = np.stack([rng.standard_normal(SR * 3), rng.standard_normal(SR * 3)]) * 0.1
    dark = ss.sosfilt(ss.butter(6, 2000, fs=SR, output="sos"), x, axis=-1)
    result = diff_audio(x, dark, SR)
    terms = [t["term"] for t in result["vocabulary"]["terms"]]
    assert "darker" in terms and "brighter" not in terms
    assert diff_audio(x, x.copy(), SR)["vocabulary"]["terms"] == []


@pytest.mark.parametrize("seconds", [0.05, 2.0])
def test_dynamic_range_descriptor(seconds):
    from audioman.core.texture import describe

    t = np.arange(int(SR * seconds)) / SR
    steady = 0.3 * np.sin(2 * np.pi * 440 * t)
    gated = steady * (np.sin(2 * np.pi * 2 * t) > 0) + steady * 0.05
    a = describe(steady, SR)["descriptors"]["dynamic_range"]["value"]
    b = describe(gated, SR)["descriptors"]["dynamic_range"]["value"]
    if seconds < 0.08:  # fewer than four 20 ms blocks
        assert a is None and b is None
    else:
        assert a < 1.0 and b > 20.0
