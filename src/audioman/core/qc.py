# Created: 2026-04-26
# Purpose: QC (Quality Control) inspection report before mastering delivery.
# Measurement → PASS/WARN/FAIL verdict per target profile → JSON report.
#
# Core measurements:
#   - Loudness: integrated/short-term LUFS, LRA, True Peak
#   - Clipping: count and locations of |sample| >= 1.0
#   - DC offset: per-channel mean
#   - Click/pop: first-difference spike detection (ratio against local RMS)
#   - Phase correlation: stereo L/R correlation coefficient
#   - Channel imbalance: L/R RMS difference
#   - Head/tail silence: verify delivery-standard padding
#   - Format: SR / bit depth / channels

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from audioman.core import loudness as loudness_mod
from audioman.core.audio_file import read_audio
from audioman.core.dsp import measure_dc_offset


# ---------------------------------------------------------------------------
# Target profiles
# ---------------------------------------------------------------------------


@dataclass
class QCTarget:
    name: str
    integrated_lufs: tuple[float, float]      # (min, max) — recommended range
    max_true_peak_dbtp: float                 # absolute ceiling
    min_lra: float = 0.0                      # minimum dynamic range (for broadcast)
    max_lra: float | None = None
    head_silence_ms: tuple[float, float] = (0.0, 1000.0)
    tail_silence_sec: tuple[float, float] = (0.0, 5.0)
    sample_rate: int | None = None            # None → not checked
    min_bit_depth: int | None = None


# Major mastering/streaming standards
TARGETS: dict[str, QCTarget] = {
    "spotify": QCTarget(
        name="Spotify (Loud)",
        integrated_lufs=(-15.0, -13.0),  # -14 ±1
        max_true_peak_dbtp=-1.0,
        head_silence_ms=(100, 700),
        tail_silence_sec=(1.0, 3.0),
    ),
    "apple_music": QCTarget(
        name="Apple Music",
        integrated_lufs=(-17.0, -15.0),  # -16 ±1
        max_true_peak_dbtp=-1.0,
        head_silence_ms=(100, 700),
        tail_silence_sec=(1.0, 3.0),
    ),
    "youtube": QCTarget(
        name="YouTube",
        integrated_lufs=(-15.0, -13.0),
        max_true_peak_dbtp=-1.0,
    ),
    "broadcast_ebu_r128": QCTarget(
        name="EBU R128 (broadcast)",
        integrated_lufs=(-23.5, -22.5),  # -23 ±0.5 LU
        max_true_peak_dbtp=-1.0,
        min_lra=4.0,
    ),
    "cd_master": QCTarget(
        name="CD master (no streaming norm)",
        integrated_lufs=(-12.0, -8.0),
        max_true_peak_dbtp=-0.3,
        sample_rate=44100,
        min_bit_depth=16,
        head_silence_ms=(0, 500),
        tail_silence_sec=(2.0, 5.0),
    ),
}


def list_targets() -> list[str]:
    return list(TARGETS.keys())


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------


def detect_clipping(audio: np.ndarray, threshold: float = 0.999) -> dict:
    """Detect clipping. Count and locations of |sample| >= threshold."""
    abs_audio = np.abs(audio)
    if audio.ndim == 1:
        clipped = np.where(abs_audio >= threshold)[0]
        return {
            "n_samples": int(len(clipped)),
            "first_sample_locations": [int(x) for x in clipped[:10].tolist()],
        }

    # stereo: union across channels
    union = np.zeros(audio.shape[1], dtype=bool)
    per_ch = []
    for ch in range(audio.shape[0]):
        ch_clipped = np.where(abs_audio[ch] >= threshold)[0]
        per_ch.append(int(len(ch_clipped)))
        union[ch_clipped] = True
    union_idx = np.where(union)[0]
    return {
        "n_samples": int(union.sum()),
        "per_channel": per_ch,
        "first_sample_locations": [int(x) for x in union_idx[:10].tolist()],
    }


def detect_clicks(
    audio: np.ndarray,
    sample_rate: int,
    sensitivity: float = 6.0,
    window_ms: float = 50.0,
    min_separation_ms: float = 5.0,
) -> dict:
    """Detect clicks/pops.

    A first difference |x[n+1] - x[n]| exceeding sensitivity times the local RMS
    is judged a click. The local RMS is a sliding average over window_ms.
    Detections chained within min_separation_ms are grouped into one.

    Default sensitivity 6: mastering QC treats a false negative as far more
    dangerous than a false positive, so it errs conservative. Normal musical
    transients are usually 3-5x the RMS, so a ratio above 6 suggests a disk edit
    or digital glitch.
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    n = len(mono)
    if n < 2:
        return {"n_clicks": 0, "locations_sec": []}

    diff = np.abs(np.diff(mono))
    win = max(int(window_ms / 1000.0 * sample_rate), 16)

    if n < win:
        local_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
        ratio = diff / max(local_rms, 1e-6)
        candidates = np.where(ratio > sensitivity)[0]
        if len(candidates) == 0:
            return {"n_clicks": 0, "locations_sec": [], "max_ratio": float(ratio.max())}
        min_sep = max(int(min_separation_ms / 1000.0 * sample_rate), 1)
        grouped = [candidates[0]]
        for c in candidates[1:]:
            if c - grouped[-1] > min_sep:
                grouped.append(c)
        return {
            "n_clicks": len(grouped),
            "locations_sec": [round(int(c) / sample_rate, 4) for c in grouped[:20]],
            "max_ratio": round(float(ratio.max()), 2),
            "sensitivity": sensitivity,
        }

    # local RMS: rolling square mean
    sq = mono.astype(np.float64) ** 2
    cumsum = np.concatenate([[0.0], np.cumsum(sq)])
    rolling_mean_sq = (cumsum[win:] - cumsum[:-win]) / win
    # match the length to diff (pad both ends)
    # `win >= 16` and `n >= win` here, so the rolling mean has n - win + 1 entries and
    # the gap to the n - 1 the ratio needs is exactly win - 2 >= 14: `pad` is never
    # negative, and padding by `pad` on the left and the remainder on the right makes
    # the result exactly n - 1 long, so no truncation or top-up can be needed.
    pad = (n - 1 - len(rolling_mean_sq)) // 2
    rolling_mean_sq = np.pad(
        rolling_mean_sq,
        (pad, n - 1 - len(rolling_mean_sq) - pad),
        mode="edge",
    )

    local_rms = np.sqrt(np.maximum(rolling_mean_sq, 1e-12))
    ratio = diff / local_rms
    candidates = np.where(ratio > sensitivity)[0]

    if len(candidates) == 0:
        return {"n_clicks": 0, "locations_sec": [], "max_ratio": float(ratio.max())}

    # group chains within min_separation into one
    min_sep = max(int(min_separation_ms / 1000.0 * sample_rate), 1)
    grouped = [candidates[0]]
    for c in candidates[1:]:
        if c - grouped[-1] > min_sep:
            grouped.append(c)

    return {
        "n_clicks": len(grouped),
        "locations_sec": [round(int(c) / sample_rate, 4) for c in grouped[:20]],
        "max_ratio": round(float(ratio.max()), 2),
        "sensitivity": sensitivity,
    }


def stereo_phase_correlation(audio: np.ndarray, window_ms: float = 100.0, sample_rate: int = 48000) -> dict:
    """Stereo L/R correlation coefficient. Mono compatibility check.

    -1 (fully out of phase) ~ +1 (fully in phase). A long stretch below 0 means
    cancellation when summed to mono.
    """
    if audio.ndim != 2 or audio.shape[0] != 2:
        return {"applicable": False, "reason": "not stereo"}

    left = audio[0].astype(np.float64)
    right = audio[1].astype(np.float64)

    # global correlation
    if np.std(left) > 0 and np.std(right) > 0:
        global_corr = float(np.corrcoef(left, right)[0, 1])
    else:
        global_corr = 0.0

    # per-window correlation (minimum and negative-region ratio)
    win = max(int(window_ms / 1000.0 * sample_rate), 256)
    n_windows = len(left) // win
    if n_windows < 1:
        return {
            "applicable": True,
            "global_correlation": round(global_corr, 4),
            "min_window_correlation": round(global_corr, 4),
            "negative_correlation_pct": 0.0,
        }

    win_corrs = []
    for i in range(n_windows):
        s, e = i * win, (i + 1) * win
        wl, wr = left[s:e], right[s:e]
        if np.std(wl) < 1e-8 or np.std(wr) < 1e-8:
            continue
        win_corrs.append(float(np.corrcoef(wl, wr)[0, 1]))

    if not win_corrs:
        return {
            "applicable": True,
            "global_correlation": round(global_corr, 4),
            "min_window_correlation": round(global_corr, 4),
            "negative_correlation_pct": 0.0,
        }

    arr = np.array(win_corrs)
    return {
        "applicable": True,
        "global_correlation": round(global_corr, 4),
        "min_window_correlation": round(float(arr.min()), 4),
        "mean_window_correlation": round(float(arr.mean()), 4),
        "negative_correlation_pct": round(100.0 * float((arr < 0).sum()) / len(arr), 2),
        "n_windows": len(win_corrs),
    }


def channel_imbalance_db(audio: np.ndarray) -> dict:
    """L/R RMS difference (dB). 0 means perfectly balanced."""
    if audio.ndim != 2 or audio.shape[0] != 2:
        return {"applicable": False, "reason": "not stereo"}
    rms_l = float(np.sqrt(np.mean(audio[0] ** 2)))
    rms_r = float(np.sqrt(np.mean(audio[1] ** 2)))
    if rms_l <= 0 or rms_r <= 0:
        return {"applicable": True, "imbalance_db": None, "reason": "silent channel"}
    diff = 20.0 * np.log10(rms_l / rms_r)
    return {
        "applicable": True,
        "rms_left_db": round(20.0 * np.log10(rms_l), 2),
        "rms_right_db": round(20.0 * np.log10(rms_r), 2),
        "imbalance_db": round(diff, 2),
    }


def head_tail_silence(audio: np.ndarray, sample_rate: int, threshold_db: float = -60.0) -> dict:
    """Leading/trailing silence length. Verifies mastering delivery padding.

    threshold_db of -60 dB is strict — true silence (actual zeros) or just above
    the noise floor.
    """
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    threshold = 10 ** (threshold_db / 20.0)
    abs_mono = np.abs(mono)
    above = np.where(abs_mono > threshold)[0]
    if len(above) == 0:
        return {"head_ms": None, "tail_sec": None, "all_silence": True}
    head_samples = int(above[0])
    tail_samples = len(mono) - 1 - int(above[-1])
    return {
        "head_ms": round(head_samples / sample_rate * 1000.0, 1),
        "tail_sec": round(tail_samples / sample_rate, 3),
        "threshold_db": threshold_db,
    }


def file_format_info(path: str | Path) -> dict:
    """soundfile metadata (bit depth, subtype, etc)."""
    info = sf.info(str(path))
    bit_depth = None
    sub = info.subtype or ""
    if "PCM_16" in sub:
        bit_depth = 16
    elif "PCM_24" in sub:
        bit_depth = 24
    elif "PCM_32" in sub:
        bit_depth = 32
    elif "FLOAT" in sub:
        bit_depth = 32  # float32
    elif "DOUBLE" in sub:
        bit_depth = 64
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_sec": round(info.duration, 4),
        "frames": info.frames,
        "format": info.format,
        "subtype": info.subtype,
        "bit_depth": bit_depth,
        "file_size_mb": round(Path(path).stat().st_size / (1024 * 1024), 2),
    }


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def _status_for_lufs(lufs: float | None, target_range: tuple[float, float]) -> str:
    if lufs is None:
        return "FAIL"
    lo, hi = target_range
    if lo <= lufs <= hi:
        return "PASS"
    # WARN within 0.5 LU
    if lo - 0.5 <= lufs <= hi + 0.5:
        return "WARN"
    return "FAIL"


def _status_for_tp(tp: float | None, ceiling: float) -> str:
    if tp is None:
        return "PASS"  # silence cannot violate TP
    if tp <= ceiling:
        return "PASS"
    if tp <= ceiling + 0.3:
        return "WARN"
    return "FAIL"


def _status_for_silence(actual: float | None, allowed: tuple[float, float]) -> str:
    if actual is None:
        return "FAIL"
    lo, hi = allowed
    if lo <= actual <= hi:
        return "PASS"
    return "WARN"


def evaluate(
    audio: np.ndarray,
    sample_rate: int,
    file_path: str | Path | None = None,
    target: str | QCTarget = "spotify",
    click_sensitivity: float = 6.0,
) -> dict:
    """All QC measurements + verdict against a target profile. Returns a combined report."""
    if isinstance(target, str):
        if target not in TARGETS:
            raise ValueError(f"Unknown target: {target!r} (supported: {list(TARGETS.keys())})")
        target_obj = TARGETS[target]
        target_name = target
    else:
        target_obj = target
        target_name = target.name

    # 1. measurement
    loud_report = loudness_mod.measure(audio, sample_rate)
    loud_d = loud_report.to_dict()
    clip = detect_clipping(audio)
    dc = measure_dc_offset(audio)
    clicks = detect_clicks(audio, sample_rate, sensitivity=click_sensitivity)
    phase = stereo_phase_correlation(audio, sample_rate=sample_rate)
    imbalance = channel_imbalance_db(audio)
    silences = head_tail_silence(audio, sample_rate)
    fmt = file_format_info(file_path) if file_path else None

    # 2. verdict
    checks: list[dict] = []

    # Loudness
    lufs_status = _status_for_lufs(loud_d["integrated_lufs"], target_obj.integrated_lufs)
    checks.append({
        "category": "loudness",
        "name": "integrated_lufs",
        "value": loud_d["integrated_lufs"],
        "target": list(target_obj.integrated_lufs),
        "status": lufs_status,
    })

    tp_status = _status_for_tp(loud_d["true_peak_dbtp"], target_obj.max_true_peak_dbtp)
    checks.append({
        "category": "loudness",
        "name": "true_peak_dbtp",
        "value": loud_d["true_peak_dbtp"],
        "target": f"<= {target_obj.max_true_peak_dbtp}",
        "status": tp_status,
    })

    if target_obj.min_lra > 0:
        lra_val = loud_d["loudness_range_lu"] or 0.0
        lra_status = "PASS" if lra_val >= target_obj.min_lra else "WARN"
        checks.append({
            "category": "loudness",
            "name": "loudness_range_lu",
            "value": lra_val,
            "target": f">= {target_obj.min_lra}",
            "status": lra_status,
        })

    # Integrity — clipping
    clip_status = "PASS" if clip["n_samples"] == 0 else ("WARN" if clip["n_samples"] < 5 else "FAIL")
    checks.append({
        "category": "integrity",
        "name": "clipping_samples",
        "value": clip["n_samples"],
        "target": "0",
        "status": clip_status,
        "detail": clip,
    })

    # DC offset
    max_dc = max(abs(x) for x in dc) if dc else 0.0
    dc_status = "PASS" if max_dc < 0.001 else ("WARN" if max_dc < 0.01 else "FAIL")
    checks.append({
        "category": "integrity",
        "name": "dc_offset",
        "value": round(max_dc, 6),
        "target": "< 0.001",
        "status": dc_status,
        "per_channel": [round(x, 6) for x in dc],
    })

    # Clicks
    click_status = "PASS" if clicks["n_clicks"] == 0 else ("WARN" if clicks["n_clicks"] < 3 else "FAIL")
    checks.append({
        "category": "integrity",
        "name": "clicks",
        "value": clicks["n_clicks"],
        "target": "0",
        "status": click_status,
        "detail": clicks,
    })

    # Phase correlation (stereo only)
    if phase.get("applicable"):
        neg_pct = phase.get("negative_correlation_pct", 0.0)
        phase_status = "PASS" if neg_pct < 5.0 else ("WARN" if neg_pct < 20.0 else "FAIL")
        checks.append({
            "category": "stereo",
            "name": "phase_correlation",
            "value": phase.get("min_window_correlation"),
            "target": "negative regions < 5%",
            "status": phase_status,
            "detail": phase,
        })

    # Channel imbalance
    if imbalance.get("applicable") and imbalance.get("imbalance_db") is not None:
        imb = abs(imbalance["imbalance_db"])
        imb_status = "PASS" if imb < 0.5 else ("WARN" if imb < 1.5 else "FAIL")
        checks.append({
            "category": "stereo",
            "name": "channel_imbalance_db",
            "value": imbalance["imbalance_db"],
            "target": "< 0.5",
            "status": imb_status,
            "detail": imbalance,
        })

    # Head/tail silence
    head_status = _status_for_silence(silences["head_ms"], target_obj.head_silence_ms)
    checks.append({
        "category": "delivery",
        "name": "head_silence_ms",
        "value": silences["head_ms"],
        "target": list(target_obj.head_silence_ms),
        "status": head_status,
    })
    tail_status = _status_for_silence(silences["tail_sec"], target_obj.tail_silence_sec)
    checks.append({
        "category": "delivery",
        "name": "tail_silence_sec",
        "value": silences["tail_sec"],
        "target": list(target_obj.tail_silence_sec),
        "status": tail_status,
    })

    # Format checks
    if fmt:
        if target_obj.sample_rate is not None:
            sr_status = "PASS" if fmt["sample_rate"] == target_obj.sample_rate else "FAIL"
            checks.append({
                "category": "format",
                "name": "sample_rate",
                "value": fmt["sample_rate"],
                "target": target_obj.sample_rate,
                "status": sr_status,
            })
        if target_obj.min_bit_depth is not None and fmt["bit_depth"] is not None:
            bd_status = "PASS" if fmt["bit_depth"] >= target_obj.min_bit_depth else "FAIL"
            checks.append({
                "category": "format",
                "name": "bit_depth",
                "value": fmt["bit_depth"],
                "target": f">= {target_obj.min_bit_depth}",
                "status": bd_status,
            })

    # 3. overall verdict
    statuses = [c["status"] for c in checks]
    if "FAIL" in statuses:
        verdict = "FAIL"
    elif "WARN" in statuses:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return {
        "target": target_name,
        "target_profile": {
            "name": target_obj.name,
            "integrated_lufs": list(target_obj.integrated_lufs),
            "max_true_peak_dbtp": target_obj.max_true_peak_dbtp,
        },
        "verdict": verdict,
        "checks": checks,
        "loudness": loud_d,
        "format": fmt,
        "summary": {
            "n_pass": statuses.count("PASS"),
            "n_warn": statuses.count("WARN"),
            "n_fail": statuses.count("FAIL"),
        },
    }


def evaluate_file(
    file_path: str | Path,
    target: str | QCTarget = "spotify",
    click_sensitivity: float = 6.0,
) -> dict:
    """evaluate by file path. read_audio + format_info are automatic."""
    audio, sr = read_audio(file_path)
    return evaluate(
        audio, sr,
        file_path=file_path,
        target=target,
        click_sensitivity=click_sensitivity,
    )
