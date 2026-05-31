# Created: 2026-05-11
# Purpose: 오디오 분석 결과 → 통일된 Finding[] 변환기.
# 기존 core/analysis.py 출력(spectrum_diagnostics, detect_silence 등)을 그대로 받아
# Finding 객체로 어댑팅하므로 분석 엔진 재구현이 없다.

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from audioman.core.analysis import SilenceRegion
from audioman.core.findings import (
    Category,
    Code,
    Finding,
    FixHint,
    Severity,
    Where,
)


# --- signal --------------------------------------------------------------

def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def detect_clipping(
    audio: np.ndarray,
    sample_rate: int,
    *,
    file: Optional[str] = None,
    threshold: float = 0.999,
) -> list[Finding]:
    """Sample peak ≥ threshold 가 연속된 구간(런)을 clipping으로 잡는다."""
    mono = _to_mono(audio)
    clipped = np.abs(mono) >= threshold
    if not clipped.any():
        return []

    # 연속 런 찾기
    diffs = np.diff(clipped.astype(np.int8))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if clipped[0]:
        starts = np.concatenate(([0], starts))
    if clipped[-1]:
        ends = np.concatenate((ends, [len(clipped)]))

    findings: list[Finding] = []
    total_clipped = int(clipped.sum())
    n_runs = len(starts)
    peak = float(np.max(np.abs(mono)))
    peak_dbfs = 20.0 * np.log10(peak + 1e-30)

    severity = Severity.CRITICAL if total_clipped > sample_rate * 0.001 else Severity.WARN
    findings.append(Finding(
        code=Code.CLIP_SAMPLE_PEAK_EXCEEDED,
        category=Category.SIGNAL,
        severity=severity,
        where=Where(
            file=file,
            start_sample=int(starts[0]),
            end_sample=int(ends[-1]),
            start_sec=round(float(starts[0]) / sample_rate, 6),
            end_sec=round(float(ends[-1]) / sample_rate, 6),
        ),
        measurement={
            "peak_dbfs": round(peak_dbfs, 3),
            "samples_clipped": total_clipped,
            "run_count": n_runs,
            "threshold": threshold,
        },
        hint=f"{n_runs} clipping run(s), {total_clipped} samples at/above {threshold}. Reduce input gain or apply a limiter.",
        fix_hint=FixHint(
            kind="ffmpeg-plan",
            args=["loudnorm-or-limit", "--ceiling", "-1.0"],
            note="audioman plan loudnorm / limit (Phase B)",
        ),
    ))
    return findings


def detect_dc_offset(
    audio: np.ndarray,
    sample_rate: int,
    *,
    file: Optional[str] = None,
    threshold: float = 0.002,
) -> list[Finding]:
    """채널별 평균이 threshold 초과면 DC offset finding."""
    if audio.ndim == 1:
        means = [float(audio.mean())]
    else:
        means = [float(audio[ch].mean()) for ch in range(audio.shape[0])]

    findings: list[Finding] = []
    for ch, m in enumerate(means):
        if abs(m) > threshold:
            findings.append(Finding(
                code=Code.DC_OFFSET_DETECTED,
                category=Category.SIGNAL,
                severity=Severity.WARN if abs(m) < 0.01 else Severity.CRITICAL,
                where=Where(file=file, channel=ch),
                measurement={"dc_offset": round(m, 6), "threshold": threshold},
                hint=f"Channel {ch} DC offset {m:+.4f} (threshold {threshold}). Apply high-pass at ~20Hz.",
                fix_hint=FixHint(
                    kind="audioman-fx",
                    args=["highpass", "--cutoff", "20"],
                    note="DC removal via 20Hz HPF",
                ),
            ))
    return findings


def detect_channel_imbalance(
    audio: np.ndarray,
    sample_rate: int,
    *,
    file: Optional[str] = None,
    db_threshold: float = 3.0,
) -> list[Finding]:
    """스테레오 채널 RMS 차가 db_threshold 초과면 finding."""
    if audio.ndim != 2 or audio.shape[0] != 2:
        return []
    rms_l = float(np.sqrt(np.mean(audio[0] ** 2)))
    rms_r = float(np.sqrt(np.mean(audio[1] ** 2)))
    if rms_l < 1e-6 or rms_r < 1e-6:
        return []
    diff_db = 20.0 * np.log10(rms_l / rms_r)
    if abs(diff_db) < db_threshold:
        return []
    return [Finding(
        code=Code.CHANNEL_IMBALANCE,
        category=Category.SIGNAL,
        severity=Severity.WARN if abs(diff_db) < 6.0 else Severity.CRITICAL,
        where=Where(file=file),
        measurement={
            "rms_left_db": round(20.0 * np.log10(rms_l + 1e-30), 2),
            "rms_right_db": round(20.0 * np.log10(rms_r + 1e-30), 2),
            "diff_db": round(diff_db, 2),
            "threshold_db": db_threshold,
        },
        hint=f"Left/right RMS differ by {diff_db:+.2f} dB. Investigate panning, mono summing or one-sided signal.",
        fix_hint=FixHint(kind="manual", args=[], note="Inspect track in DAW"),
    )]


def silence_to_findings(
    silence_regions: list[SilenceRegion],
    audio_length_samples: int,
    sample_rate: int,
    *,
    file: Optional[str] = None,
    inner_min_sec: float = 1.0,
) -> list[Finding]:
    """SilenceRegion 리스트를 leading/trailing/inner finding으로 분류."""
    if not silence_regions:
        return []

    findings: list[Finding] = []
    head_thresh = int(0.05 * sample_rate)
    tail_thresh = audio_length_samples - int(0.05 * sample_rate)

    for region in silence_regions:
        if region.start_sample <= head_thresh:
            code = Code.SILENCE_LEADING
            sev = Severity.INFO
            hint = f"Leading silence {region.duration_sec:.3f}s. `audioman plan cut-silence --strategy leading` will trim it."
        elif region.end_sample >= tail_thresh:
            code = Code.SILENCE_TRAILING
            sev = Severity.INFO
            hint = f"Trailing silence {region.duration_sec:.3f}s. `audioman plan cut-silence --strategy trailing` will trim it."
        elif region.duration_sec >= inner_min_sec:
            code = Code.SILENCE_INNER
            sev = Severity.WARN
            hint = f"Inner silence gap {region.duration_sec:.3f}s. Use `audioman plan cut-silence --strategy all` to remove."
        else:
            continue  # 짧은 inner silence는 노이즈로 간주, finding 안 만듦

        findings.append(Finding(
            code=code,
            category=Category.SIGNAL,
            severity=sev,
            where=Where(
                file=file,
                start_sample=region.start_sample,
                end_sample=region.end_sample,
                start_sec=round(region.start_sample / sample_rate, 6),
                end_sec=round(region.end_sample / sample_rate, 6),
            ),
            measurement={"duration_sec": round(region.duration_sec, 6)},
            hint=hint,
            fix_hint=FixHint(
                kind="ffmpeg-plan",
                args=["cut-silence", "--strategy", code.value.split("_")[-1].lower()],
                note="Phase B: audioman plan cut-silence",
            ),
        ))
    return findings


def detect_signal_findings(
    audio: np.ndarray,
    sample_rate: int,
    *,
    file: Optional[str] = None,
) -> list[Finding]:
    """signal 카테고리 detector 일괄."""
    findings: list[Finding] = []
    findings.extend(detect_clipping(audio, sample_rate, file=file))
    findings.extend(detect_dc_offset(audio, sample_rate, file=file))
    findings.extend(detect_channel_imbalance(audio, sample_rate, file=file))
    return findings


# --- spectral ------------------------------------------------------------

def spectrum_to_findings(
    spectrum_dict: dict[str, Any],
    *,
    file: Optional[str] = None,
) -> list[Finding]:
    """core/analysis.py:spectrum_diagnostics() 출력 → spectral Finding[]."""
    findings: list[Finding] = []

    # MAINS_HUM
    for h in spectrum_dict.get("hum_check", []) or []:
        if not h.get("is_hum"):
            continue
        freq = h.get("frequency_hz")
        snr = h.get("snr_db")
        sev = Severity.CRITICAL if (snr is not None and snr > 25.0) else Severity.WARN
        findings.append(Finding(
            code=Code.MAINS_HUM,
            category=Category.SPECTRAL,
            severity=sev,
            where=Where(file=file, frequency_hz=freq),
            measurement={"frequency_hz": freq, "snr_db": snr},
            hint=f"Mains hum at {freq} Hz (SNR {snr:+.1f} dB). Apply a notch filter / RX De-hum.",
            fix_hint=FixHint(
                kind="audioman-process",
                args=["--plugin", "dehum", "--", f"notch_frequency={freq}"],
            ),
        ))

    # HF_NOISE_FLOOR — hf_slope가 양수로 크면 hiss 후보 (단, 발화/음악 자체일 수도 있어 INFO)
    sl = spectrum_dict.get("hf_slope") or {}
    slope = sl.get("slope_db")
    if slope is not None and slope > -3.0 and sl.get("high_db") is not None and sl["high_db"] > -50.0:
        # 고역이 mid 대비 거의 안 빠짐 → hiss/노이즈 floor 가능성
        findings.append(Finding(
            code=Code.HF_NOISE_FLOOR,
            category=Category.SPECTRAL,
            severity=Severity.INFO,
            where=Where(file=file),
            measurement={
                "mid_db": sl.get("mid_db"),
                "high_db": sl.get("high_db"),
                "slope_db": slope,
            },
            hint=f"Flat/elevated HF energy (slope {slope:+.1f} dB). May indicate hiss or wideband noise.",
            fix_hint=FixHint(
                kind="audioman-process",
                args=["--plugin", "denoise"],
                note="Optional — verify against content first",
            ),
        ))

    return findings
