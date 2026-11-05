# Created: 2026-11-04
# Purpose: Texture descriptors per segment, each with a repeat floor.
# Dependencies: numpy, audioman.core.texture.spectral
#
# `describe(audio, sr)` reports, for the whole file and for each fixed-length
# segment, a set of texture descriptors (spectral shape, brightness, noisiness,
# change, dynamics, onset density, stereo image) plus an ERB-band level profile.
#
# Repeat floor. A descriptor value only means something to the precision with which
# it can be re-measured. Each descriptor is therefore measured twice: on the signal,
# and on the same signal advanced by half an analysis hop, which moves every frame
# and segment boundary while leaving the content unchanged. The difference between
# the two is the descriptor's floor (`floor` for the whole-file value,
# `segment_floor` as the largest per-segment difference). A change between two
# files smaller than the floor is not evidence of a change. Rolloff, which can only
# land on FFT bins, never gets a floor below one bin.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from audioman.core.texture import pitch as pt
from audioman.core.texture import spectral as sp

FRAME = 2048
HOP = 512
SILENT_FRAME_DB = -70.0  # frames this far below the loudest frame are left out of means

FrameFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

FRAME_DESCRIPTORS: dict[str, tuple[str, FrameFn]] = {
    "spectral_centroid": ("Hz", sp.centroid),
    "spectral_rolloff": ("Hz", sp.rolloff),
    "spectral_bandwidth": ("Hz", sp.bandwidth),
    "spectral_flatness": ("dB", lambda P, f: sp.flatness_db(P)),
    "spectral_contrast": ("dB", sp.contrast_db),
    "spectral_flux": ("ratio", lambda P, f: sp.flux(P)),
    "hf_ratio": ("dB", sp.hf_ratio_db),
    "sharpness": ("acum (approx.)", sp.sharpness),
}
# Descriptors whose value is a bin frequency; their floor is at least one bin.
GRID_QUANTISED = {"spectral_rolloff"}
SIGNAL_DESCRIPTORS = {
    "rms": "dBFS",
    "crest_factor": "dB",
    "dynamic_range": "dB",
    "onset_density": "1/s",
    "stereo_width": "dB",
    "lr_correlation": "ratio",
}
# From the YIN f0 track (core/texture/pitch.py); null on material that is not
# monophonic enough to have one f0.
PITCH_DESCRIPTORS = {
    "pitch": "cents",  # median f0, cents re A4 = 440 Hz
    "pitch_spread": "cents",
    "pitch_deviation": "cents",
    "voiced_fraction": "ratio",
}
# Values are rounded to 4 decimals; no floor is finer than what is reported.
MIN_FLOOR = 1e-4


@dataclass
class _Measured:
    values: dict[str, float | None]  # whole-file value per descriptor
    segments: dict[str, list[float | None]]  # per-segment values
    bounds: list[tuple[float, float]]
    erb_db: np.ndarray


def _finite(x: float) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def _db(x: float) -> float:
    return 20.0 * np.log10(x) if x > 0 else -np.inf


def _segment_bounds(n_samples: int, sample_rate: int, segment_sec: float) -> list[tuple[int, int]]:
    seg = max(1, int(round(segment_sec * sample_rate)))
    n = max(1, n_samples // seg)
    edges = [i * seg for i in range(n)] + [n_samples]
    return list(zip(edges[:-1], edges[1:]))


DYNAMIC_BLOCK_SEC = 0.02
DYNAMIC_FLOOR_DB = -60.0


def _dynamic_range(mono: np.ndarray, sample_rate: int) -> float | None:
    """Spread of short-term level: p95 - p10 of 20 ms block RMS (dB), blocks within 60 dB of the loudest.

    Compression narrows it; crest factor alone can rise under a fast-attack
    compressor (onset peaks pass, the body is turned down), so this is the
    dynamics descriptor to read for compression.
    """
    block = max(1, int(DYNAMIC_BLOCK_SEC * sample_rate))
    if len(mono) < 4 * block:
        return None
    # Overlapping blocks (hop = block / 4) so the statistic barely depends on where
    # the block grid falls; non-overlapping blocks made the repeat floor large.
    hop = max(1, block // 4)
    squares = np.concatenate(([0.0], np.cumsum(mono.astype(np.float64) ** 2)))
    starts = np.arange(0, len(mono) - block + 1, hop)
    levels = np.sqrt(np.maximum(squares[starts + block] - squares[starts], 0.0) / block)
    if levels.max() <= 0:
        return None
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(levels)
    db = db[db >= db.max() + DYNAMIC_FLOOR_DB]
    return float(np.percentile(db, 95) - np.percentile(db, 10))


def _signal_values(x: np.ndarray, sample_rate: int, onsets: np.ndarray, start: int, end: int) -> dict[str, float | None]:
    """Descriptors computed on samples rather than frames, for x[:, start:end] (channels x samples)."""
    seg = x[:, start:end]
    mono = seg.mean(axis=0)
    rms = float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    out: dict[str, float | None] = {
        "rms": _finite(_db(rms)),
        "crest_factor": _finite(_db(peak) - _db(rms)) if rms > 0 else None,
        "dynamic_range": _dynamic_range(mono, sample_rate),
        "onset_density": float(np.count_nonzero((onsets >= start / sample_rate) & (onsets < end / sample_rate)))
        / max((end - start) / sample_rate, 1e-9),
        "stereo_width": None,
        "lr_correlation": None,
    }
    if seg.shape[0] == 2:
        mid, side = (seg[0] + seg[1]) / 2, (seg[0] - seg[1]) / 2
        e_mid, e_side = float(np.mean(mid * mid)), float(np.mean(side * side))
        if e_mid > 0:
            out["stereo_width"] = max(-60.0, 10.0 * np.log10(e_side / e_mid)) if e_side > 0 else -60.0
        sl, sr_ = float(np.std(seg[0])), float(np.std(seg[1]))
        if sl > 0 and sr_ > 0:
            out["lr_correlation"] = float(np.clip(np.corrcoef(seg[0], seg[1])[0, 1], -1.0, 1.0))
    return out


def _frame_mean(values: np.ndarray, keep: np.ndarray) -> float | None:
    v = values[keep & np.isfinite(values)]
    return float(v.mean()) if len(v) else None


def _measure(x: np.ndarray, sample_rate: int, segment_sec: float, erb_centers: np.ndarray) -> _Measured:
    mono = x.mean(axis=0)
    P = sp.power_spectrogram(mono, FRAME, HOP)
    freqs = np.fft.rfftfreq(FRAME, 1.0 / sample_rate)
    energy = P.sum(axis=1)
    loud = energy > energy.max() * 10 ** (SILENT_FRAME_DB / 10) if energy.max() > 0 else np.zeros(len(energy), bool)
    centers = (np.arange(P.shape[0]) * HOP + FRAME / 2) / sample_rate

    bounds = _segment_bounds(x.shape[1], sample_rate, segment_sec)
    onsets = sp.onset_times(P, HOP, sample_rate)
    values: dict[str, float | None] = {}
    segments: dict[str, list[float | None]] = {}
    for name, (_, fn) in FRAME_DESCRIPTORS.items():
        per_frame = fn(P, freqs)
        values[name] = _frame_mean(per_frame, loud)
        segments[name] = [
            _frame_mean(per_frame, loud & (centers >= s / sample_rate) & (centers < e / sample_rate))
            for s, e in bounds
        ]
    whole = _signal_values(x, sample_rate, onsets, 0, x.shape[1])
    per_segment = [_signal_values(x, sample_rate, onsets, s, e) for s, e in bounds]
    for name in SIGNAL_DESCRIPTORS:
        values[name] = whole[name]
        segments[name] = [seg[name] for seg in per_segment]
    track = pt.track(mono, sample_rate)
    whole_pitch = pt.descriptors(track, 0.0, np.inf)
    seg_pitch = [pt.descriptors(track, s / sample_rate, e / sample_rate) for s, e in bounds]
    for name in PITCH_DESCRIPTORS:
        values[name] = whole_pitch[name]
        segments[name] = [seg[name] for seg in seg_pitch]
    erb = sp.erb_profile_db(P[loud] if np.any(loud) else P, freqs, erb_centers)
    return _Measured(values, segments, [(s / sample_rate, e / sample_rate) for s, e in bounds], erb)


def _diff(a: float | None, b: float | None) -> float | None:
    return abs(a - b) if a is not None and b is not None else None


def describe(audio: np.ndarray, sample_rate: int, *, segment_sec: float = 1.0, erb_bands: int = 32) -> dict[str, Any]:
    """Texture descriptors for `audio` ((channels, samples) or (samples,)), whole and per segment.

    Returns a JSON-ready dict:
      descriptors: {name: {unit, value, floor, segment_floor, segments: [...]}}
      segments:    [{start_sec, end_sec}]
      erb_profile: {centers_hz, level_db, floor_db}
    Values that are undefined for the input (stereo descriptors on mono, levels of
    silence) are null.
    """
    x = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    centers = sp.erb_centers(50.0, min(16000.0, 0.45 * sample_rate), erb_bands)
    main = _measure(x, sample_rate, segment_sec, centers)
    repeat = _measure(x[:, HOP // 2:], sample_rate, segment_sec, centers)

    descriptors: dict[str, Any] = {}
    units = {**{k: u for k, (u, _) in FRAME_DESCRIPTORS.items()}, **SIGNAL_DESCRIPTORS, **PITCH_DESCRIPTORS}
    bin_hz = sample_rate / FRAME
    for name, unit in units.items():
        seg_diffs = [_diff(a, b) for a, b in zip(main.segments[name], repeat.segments[name])]
        seg_diffs = [d for d in seg_diffs if d is not None]
        floor = _diff(main.values[name], repeat.values[name])
        segment_floor = max(seg_diffs) if seg_diffs else None
        if name in GRID_QUANTISED:
            # rolloff moves in whole FFT bins: it cannot resolve less than one bin
            floor = max(floor, bin_hz) if floor is not None else None
            segment_floor = max(segment_floor, bin_hz) if segment_floor is not None else None
        floor = max(floor, MIN_FLOOR) if floor is not None else None
        segment_floor = max(segment_floor, MIN_FLOOR) if segment_floor is not None else None
        descriptors[name] = {
            "unit": unit,
            "value": _round(main.values[name]),
            "floor": _round(floor),
            "segment_floor": _round(segment_floor),
            "segments": [_round(v) for v in main.segments[name]],
        }
    return {
        "sample_rate": sample_rate,
        "channels": int(x.shape[0]),
        "duration_sec": round(x.shape[1] / sample_rate, 6),
        "frame_size": FRAME,
        "hop": HOP,
        "segment_sec": segment_sec,
        "floor_method": "re-measured on the signal advanced by half a hop; floor = |difference|, at least 1e-4",
        "pitch_scope": ("pitch descriptors assume one f0 (a voice or solo instrument): they are null when fewer "
                        f"than {pt.MIN_VOICED_FRACTION:.0%} of sounding frames are voiced, and a chord may report "
                        "the common subharmonic of its notes"),
        "segments": [{"start_sec": round(s, 4), "end_sec": round(e, 4)} for s, e in main.bounds],
        "descriptors": descriptors,
        "erb_profile": {
            "centers_hz": [round(float(c), 1) for c in centers],
            "level_db": [_round(float(v)) for v in main.erb_db],
            "floor_db": [_round(max(abs(float(a) - float(b)), MIN_FLOOR)) for a, b in zip(main.erb_db, repeat.erb_db)],
        },
    }


def _round(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 4)


def to_measurements(texture: dict[str, Any]):
    """report.v2 measurements for a describe() result: whole-file values, repeat floors."""
    from audioman.core.report_v2 import Measurement

    out = []
    for name, d in texture["descriptors"].items():
        if d["value"] is None:
            continue
        out.append(Measurement(f"texture.{name}", d["value"], d["unit"], d["floor"],
                               "repeat" if d["floor"] is not None else "unknown", "texture"))
    return out
