# Created: 2026-11-04
# Purpose: Compare two renders of (nominally) the same material by texture descriptors:
#          align, describe both, report deltas whole / per segment / per band with floor
#          significance.
# Dependencies: numpy, audioman.core.pdc, audioman.core.texture
#
# "What changed between A and B, and is it real?" is the question an LLM asks of a
# before/after pair. The answer here is a set of deltas (B - A), each judged against
# the repeat floors of both measurements: a delta is significant only when it exceeds
# SIGNIFICANCE_K times the sum of the two floors. Two identical files therefore
# report no significant change, whatever the material.
#
# Alignment: B is shifted onto A by steady-state cross-correlation when the lag can
# be established (|NCC| >= MIN_ALIGN_NCC and not on the search edge). Otherwise the
# files are compared unshifted and the report says so; descriptor deltas remain
# meaningful, the sample-level residual does not.

from __future__ import annotations

import math
from typing import Any

import numpy as np

from audioman.core.pdc import estimate_lag
from audioman.core.texture import describe

SIGNIFICANCE_K = 2.0
MIN_ALIGN_NCC = 0.5
# Smallest delta ever called significant, per unit: below these no listener or
# reader would act on the change even when both floors are zero (stationary tones).
_ABS_EPS = {"Hz": 0.5, "dB": 0.05, "dBFS": 0.05, "ratio": 0.002, "1/s": 0.25, "acum (approx.)": 0.005}
NAMED_BANDS = (
    ("sub", 20.0, 60.0),
    ("bass", 60.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2000.0),
    ("presence", 2000.0, 6000.0),
    ("air", 6000.0, 20000.0),
)


def _as2d(x: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.asarray(x, dtype=np.float64))


def align(a: np.ndarray, b: np.ndarray, sample_rate: int, *, max_lag_ms: float = 500.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Shift B onto A when the lag can be established; trim both to the common length."""
    a2, b2 = _as2d(a), _as2d(b)
    est = estimate_lag(a2, b2, sample_rate, max_lag_ms=max_lag_ms)
    established = abs(est.ncc) >= MIN_ALIGN_NCC and not est.at_edge
    lag = est.lag_samples if established else 0
    if lag > 0:  # B is late: drop its first `lag` samples
        b2 = b2[:, lag:]
    elif lag < 0:  # B is early: drop A's first samples instead
        a2 = a2[:, -lag:]
    n = min(a2.shape[1], b2.shape[1])
    info = {
        "applied": established,
        "lag_samples": lag,
        "measured_lag_samples": est.lag_samples,
        "ncc": round(est.ncc, 6),
        "compared_sec": round(n / sample_rate, 6),
        "reason": "" if established else (
            "lag on the search edge" if est.at_edge else f"correlation {abs(est.ncc):.3f} < {MIN_ALIGN_NCC}"
        ),
    }
    return a2[:, :n], b2[:, :n], info


def _judge(a: float | None, b: float | None, fa: float | None, fb: float | None, unit: str) -> dict[str, Any]:
    if a is None or b is None:
        return {"a": a, "b": b, "delta": None, "floor": None, "significant": False}
    delta = b - a
    floor = (fa or 0.0) + (fb or 0.0)
    threshold = max(SIGNIFICANCE_K * floor, _ABS_EPS.get(unit, 1e-6))
    return {
        "a": a,
        "b": b,
        "delta": round(delta, 4),
        "floor": round(floor, 4),
        "significant": abs(delta) > threshold,
    }


def _residual(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    ref = float(np.sqrt(np.mean(ma * ma))) if len(ma) else 0.0
    res = float(np.sqrt(np.mean((mb - ma) ** 2))) if len(ma) else 0.0
    if ref == 0.0:
        return {"residual_rel_db": None}
    return {"residual_rel_db": round(20.0 * math.log10(res / ref), 2) if res > 0 else None}


def diff_audio(
    a: np.ndarray,
    b: np.ndarray,
    sample_rate: int,
    *,
    segment_sec: float = 1.0,
    do_align: bool = True,
    max_lag_ms: float = 500.0,
) -> dict[str, Any]:
    """Descriptor deltas B - A, whole file, per segment and per band, with significance."""
    if do_align:
        a2, b2, alignment = align(a, b, sample_rate, max_lag_ms=max_lag_ms)
    else:
        a2, b2 = _as2d(a), _as2d(b)
        n = min(a2.shape[1], b2.shape[1])
        a2, b2 = a2[:, :n], b2[:, :n]
        alignment = {"applied": False, "lag_samples": 0, "measured_lag_samples": None, "ncc": None,
                     "compared_sec": round(n / sample_rate, 6), "reason": "disabled (--no-align)"}

    ta = describe(a2, sample_rate, segment_sec=segment_sec)
    tb = describe(b2, sample_rate, segment_sec=segment_sec)

    descriptors: dict[str, Any] = {}
    for name, da in ta["descriptors"].items():
        db = tb["descriptors"][name]
        entry = {"unit": da["unit"], **_judge(da["value"], db["value"], da["floor"], db["floor"], da["unit"])}
        entry["segments"] = [
            _judge(sa, sb, da["segment_floor"], db["segment_floor"], da["unit"])
            for sa, sb in zip(da["segments"], db["segments"])
        ]
        descriptors[name] = entry

    bands = []
    for c, la, lb, fa, fb in zip(ta["erb_profile"]["centers_hz"], ta["erb_profile"]["level_db"],
                                 tb["erb_profile"]["level_db"], ta["erb_profile"]["floor_db"],
                                 tb["erb_profile"]["floor_db"]):
        bands.append({"center_hz": c, **_judge(la, lb, fa, fb, "dB")})

    named = []
    for label, lo, hi in NAMED_BANDS:
        sel = [b_ for b_ in bands if lo <= b_["center_hz"] < hi and b_["delta"] is not None]
        if not sel:
            continue
        delta = float(np.mean([b_["delta"] for b_ in sel]))
        floor = float(np.mean([b_["floor"] for b_ in sel]))
        named.append({"band": label, "low_hz": lo, "high_hz": hi, "n_erb_bands": len(sel),
                      "delta_db": round(delta, 3), "floor_db": round(floor, 3),
                      "significant": abs(delta) > max(SIGNIFICANCE_K * floor, _ABS_EPS["dB"])})

    changes = sorted(
        ({"descriptor": k, "delta": v["delta"], "unit": v["unit"], "direction": "up" if v["delta"] > 0 else "down",
          "floor": v["floor"], "ratio_to_floor": round(abs(v["delta"]) / v["floor"], 2) if v["floor"] else None}
         for k, v in descriptors.items() if v["significant"]),
        key=lambda c: -(c["ratio_to_floor"] if c["ratio_to_floor"] is not None else math.inf),
    )
    result = {
        "sample_rate": sample_rate,
        "segment_sec": segment_sec,
        "significance": f"|delta| > {SIGNIFICANCE_K} x (floor_a + floor_b), and above a per-unit minimum",
        "alignment": alignment,
        **_residual(a2, b2),
        "segments": ta["segments"],
        "changes": changes,
        "descriptors": descriptors,
        "named_bands": named,
        "erb_bands": bands,
    }
    from audioman.core.vocabulary import propose_terms

    result["vocabulary"] = propose_terms(result)
    return result
