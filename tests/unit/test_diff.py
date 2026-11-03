"""audioman diff: alignment, per-band / per-segment deltas, floor significance."""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf
from scipy import signal as ss

from audioman.core.diff import NAMED_BANDS, align, diff_audio
from audioman.core.digest import digest_diff
from audioman.core.report_v2 import build_report
from test_json_contract import _resolve_ref, validate_against

SR = 48000
N = SR * 4
_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
_A = [1, -2.494956002, 2.017265875, -0.522189400]


def _pink(seed: int, n: int = N) -> np.ndarray:
    x = ss.lfilter(_B, _A, np.random.default_rng(seed).standard_normal(n))
    return 0.5 * x / np.max(np.abs(x))


@pytest.fixture
def stereo() -> np.ndarray:
    return np.stack([_pink(0), _pink(1)])


def _lowpass(x, hz=2000.0):
    return ss.sosfilt(ss.butter(6, hz, fs=SR, output="sos"), x, axis=-1)


def _delay(x, n):
    return np.concatenate([np.zeros((x.shape[0], n)), x[:, :-n]], axis=1)


def _changed(result) -> dict[str, str]:
    return {c["descriptor"]: c["direction"] for c in result["changes"]}


# ---------------------------------------------------------------------------
# No change
# ---------------------------------------------------------------------------


def test_identical_inputs_report_no_significant_change(stereo):
    result = diff_audio(stereo, stereo.copy(), SR)
    assert result["changes"] == []
    assert not any(d["significant"] for d in result["descriptors"].values())
    assert not any(s["significant"] for d in result["descriptors"].values() for s in d["segments"])
    assert not any(b["significant"] for b in result["erb_bands"])
    assert not any(b["significant"] for b in result["named_bands"])
    assert result["residual_rel_db"] is None  # B - A is exactly zero
    assert result["alignment"]["applied"] and result["alignment"]["lag_samples"] == 0
    assert digest_diff({**result}).endswith("no significant change.")


def test_identical_stationary_tone_reports_no_change():
    t = np.arange(N) / SR
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)  # zero repeat floors: only exact equality is "no change"
    assert diff_audio(tone, tone.copy(), SR)["changes"] == []


def test_a_delayed_copy_is_aligned_and_unchanged(stereo):
    result = diff_audio(stereo, _delay(stereo, 300), SR)
    assert result["alignment"]["lag_samples"] == 300
    assert result["changes"] == []
    assert result["residual_rel_db"] is None


def test_b_early_is_aligned_too(stereo):
    early = np.concatenate([stereo[:, 250:], np.zeros((2, 250))], axis=1)
    result = diff_audio(stereo, early, SR)
    assert result["alignment"]["lag_samples"] == -250 and result["changes"] == []


# ---------------------------------------------------------------------------
# Real changes
# ---------------------------------------------------------------------------


def test_lowpass_is_darker_with_the_loss_in_the_top_bands(stereo):
    result = diff_audio(stereo, _lowpass(stereo), SR)
    changed = _changed(result)
    for name in ("spectral_centroid", "spectral_rolloff", "hf_ratio", "sharpness"):
        assert changed.get(name) == "down", name
    bands = {b["band"]: b for b in result["named_bands"]}
    assert bands["air"]["significant"] and bands["air"]["delta_db"] < -40
    assert bands["presence"]["delta_db"] < -10
    assert abs(bands["bass"]["delta_db"]) < 0.5
    # the most significant change comes first
    ratios = [c["ratio_to_floor"] for c in result["changes"] if c["ratio_to_floor"] is not None]
    assert ratios == sorted(ratios, reverse=True)
    assert "spectral_centroid down" in digest_diff(result) and "air" in digest_diff(result)


def test_gain_change_moves_level_not_shape(stereo):
    result = diff_audio(stereo, stereo * 0.5, SR)
    assert _changed(result) == {"rms": "down"}
    assert result["descriptors"]["rms"]["delta"] == pytest.approx(-6.02, abs=0.01)
    assert all(b["delta_db"] == pytest.approx(-6.02, abs=0.01) for b in result["named_bands"])
    assert result["residual_rel_db"] == pytest.approx(-6.02, abs=0.01)


def test_change_in_one_segment_is_located(stereo):
    b = stereo.copy()
    b[:, 2 * SR: 3 * SR] = _lowpass(stereo)[:, 2 * SR: 3 * SR]
    result = diff_audio(stereo, b, SR, segment_sec=1.0)
    seg = result["descriptors"]["spectral_centroid"]["segments"]
    assert len(seg) == 4
    assert seg[2]["significant"] and seg[2]["delta"] < 0
    assert not seg[0]["significant"]


def test_width_change(stereo):
    narrow = np.stack([stereo[0], stereo[0]])
    result = diff_audio(narrow, stereo, SR)
    assert _changed(result).get("stereo_width") == "up"
    assert _changed(result).get("lr_correlation") == "down"


# ---------------------------------------------------------------------------
# Alignment edge cases
# ---------------------------------------------------------------------------


def test_unrelated_material_is_compared_unaligned():
    a = np.stack([_pink(3), _pink(4)])
    b = np.stack([_pink(5), _pink(6)])
    a2, b2, info = align(a, b, SR)
    assert not info["applied"] and info["lag_samples"] == 0 and "correlation" in info["reason"]
    assert a2.shape == b2.shape


def test_an_edge_of_window_lag_is_not_applied(monkeypatch, stereo):
    # Edge detection itself is pinned in test_uat_gates; here: diff must not trust it.
    from audioman.core import diff as diff_mod
    from audioman.core.pdc import LagEstimate

    monkeypatch.setattr(diff_mod, "estimate_lag",
                        lambda *a, **k: LagEstimate(480, 0.95, True, 480, 0))
    _, _, info = align(stereo, _delay(stereo, 4000), SR, max_lag_ms=10.0)
    assert not info["applied"] and info["lag_samples"] == 0 and info["reason"] == "lag on the search edge"


def test_no_align_option_and_unequal_lengths(stereo):
    result = diff_audio(stereo, stereo[:, : SR * 3], SR, do_align=False)
    assert result["alignment"]["reason"] == "disabled (--no-align)"
    assert result["alignment"]["compared_sec"] == 3.0


def test_silent_reference_has_no_residual_and_undefined_deltas(tmp_path):
    silent = np.zeros((2, SR * 2))
    result = diff_audio(silent, silent.copy(), SR)
    assert result["residual_rel_db"] is None
    assert result["descriptors"]["spectral_centroid"]["delta"] is None
    assert result["descriptors"]["spectral_centroid"]["significant"] is False
    from audioman.core.diff_figure import render

    assert render(result, tmp_path / "silent.png").is_file()  # undefined deltas plot as 0


def test_low_sample_rate_omits_bands_above_nyquist():
    sr = 8000
    x = np.stack([_pink(0, sr * 3), _pink(1, sr * 3)])
    result = diff_audio(x, x.copy(), sr)
    assert "air" not in {b["band"] for b in result["named_bands"]}
    assert result["changes"] == []


# ---------------------------------------------------------------------------
# report.v2 and digest
# ---------------------------------------------------------------------------


def test_report_carries_deltas_with_floors(stereo):
    result = {"a": "a.wav", "b": "b.wav", **diff_audio(stereo, _lowpass(stereo), SR)}
    report = build_report("diff", result)
    validate_against(report, _resolve_ref("audioman://schema/report.v2.json"))
    names = {m["name"]: m for m in report["measurements"]}
    assert names["delta.spectral_centroid"]["floor_kind"] == "repeat"
    assert {f"delta.band.{b[0]}" for b in NAMED_BANDS} >= {n for n in names if n.startswith("delta.band.")}
    assert report["verdict"] == "PASS" and report["subject"]["file"] == "b.wav"
    unaligned = build_report("diff", {**result, "alignment": {"applied": False, "reason": "x", "compared_sec": 1.0}})
    assert unaligned["verdict"] == "WARN"
    missing_floor = {**result, "descriptors": {"rms": {"delta": 1.0, "unit": "dBFS", "floor": None}}}
    assert build_report("diff", missing_floor)["measurements"][0]["floor_kind"] == "unknown"


def test_digest_variants():
    base = {"alignment": {"applied": False, "reason": "lag on the search edge"}, "changes": [], "named_bands": []}
    assert "not aligned (lag on the search edge)" in digest_diff(base)
    only_bands = {**base, "named_bands": [{"band": "air", "delta_db": -3.0, "significant": True}]}
    assert digest_diff(only_bands).endswith("Bands: air -3.0 dB.")
    zero_floor = {**base, "changes": [{"descriptor": "rms", "direction": "up", "delta": 1.0, "unit": "dBFS",
                                       "floor": 0.0}]}
    assert "(floor" not in digest_diff(zero_floor)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run(argv):
    """Run in-process; returns (exit code, stdout, stdout + stderr)."""
    from audioman.cli import app

    out, err = io.StringIO(), io.StringIO()
    real = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    plain = os.environ.get("AUDIOMAN_PLAIN")
    try:
        app.main(argv)
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdout, sys.stderr = real
        if plain is None:
            os.environ.pop("AUDIOMAN_PLAIN", None)
    return code, out.getvalue(), out.getvalue() + err.getvalue()


@pytest.fixture
def files(tmp_path, monkeypatch, stereo):
    from audioman.config import settings as settings_module

    monkeypatch.setenv("AUDIOMAN_CACHE_DIR", str(tmp_path / "cache"))
    settings_module.reset_settings()
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    same = tmp_path / "same.wav"
    sf.write(a, stereo.T.astype(np.float32), SR, subtype="FLOAT")
    sf.write(b, _lowpass(stereo).T.astype(np.float32), SR, subtype="FLOAT")
    sf.write(same, stereo.T.astype(np.float32), SR, subtype="FLOAT")
    yield a, b, same
    settings_module.reset_settings()


def test_cli_json_png_and_human(files, tmp_path):
    a, b, same = files
    png = tmp_path / "diff.png"
    code, text, _ = _run(["--json", "diff", str(a), str(b), "--png", str(png), "--detail", "full"])
    assert code == 0
    payload = json.loads(text)
    validate_against(payload, _resolve_ref("audioman://schema/diff.v1.json"))
    assert payload["changes"] and payload["digest"].startswith("B vs A (aligned")
    assert png.is_file() and png.stat().st_size > 10_000 and payload["figure"] == str(png.resolve())

    code, text, _ = _run(["--json", "diff", str(a), str(same)])
    assert code == 0 and json.loads(text)["digest"].endswith("no significant change.")

    code, text, _ = _run(["--json", "diff", str(a), str(b), "--detail", "summary"])
    assert code == 0 and len(text.encode()) <= 4096

    code, _, text = _run(["diff", str(a), str(b), "--no-align"])
    assert code == 0 and "spectral_centroid" in text and "air" in text
    code, _, text = _run(["diff", str(a), str(same)])
    assert code == 0 and "no significant change" in text


def test_cli_errors(files, tmp_path):
    a, _, _ = files
    other = tmp_path / "other.wav"
    sf.write(other, np.zeros((SR, 2), np.float32), 44100)
    assert _run(["diff", str(a), str(other)])[0] == 1
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"nope")
    assert _run(["diff", str(junk), str(a)])[0] == 1
