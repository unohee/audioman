# Created: 2026-04-26
# Purpose: ITU-R BS.1770-4 기반 LUFS / True Peak / LRA 측정 + loudness normalization
# Dependencies: pyloudnorm (K-weighting + gating reference 구현), scipy (4x oversample)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyloudnorm
from scipy import signal as scipy_signal


def _to_pyln(audio: np.ndarray) -> np.ndarray:
    """audioman (channels, samples) → pyloudnorm (samples, channels) 또는 1D."""
    if audio.ndim == 1:
        return audio
    return audio.T


def _to_audioman(audio: np.ndarray, original_ndim: int) -> np.ndarray:
    """pyloudnorm 출력 → audioman 형식. 입력이 1D였으면 그대로."""
    if original_ndim == 1:
        return audio
    return audio.T if audio.ndim == 2 else audio


def integrated_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """ITU-R BS.1770-4 integrated loudness (gating 포함).

    무음/짧은 입력은 -inf 반환할 수 있음. NaN/-inf 정규화된 float 반환.
    """
    pyln_audio = _to_pyln(audio)
    meter = pyloudnorm.Meter(sample_rate)
    try:
        lufs = meter.integrated_loudness(pyln_audio)
    except ValueError:
        # 입력이 너무 짧음 (gating block보다 작음, 보통 < 0.4초)
        return float("-inf")
    if not np.isfinite(lufs):
        return float("-inf")
    return float(lufs)


def short_term_lufs(
    audio: np.ndarray,
    sample_rate: int,
    window_sec: float = 3.0,
    hop_sec: float = 0.1,
) -> np.ndarray:
    """Short-term LUFS (3초 sliding window). LRA 계산의 backing 데이터.

    반환: 각 hop의 short-term LUFS 배열. 무음 윈도우는 -inf.
    """
    pyln_audio = _to_pyln(audio)
    n_samples = pyln_audio.shape[0] if pyln_audio.ndim > 1 else len(pyln_audio)
    win = int(window_sec * sample_rate)
    hop = int(hop_sec * sample_rate)
    if n_samples < win:
        return np.array([], dtype=np.float32)

    meter = pyloudnorm.Meter(sample_rate, block_size=window_sec)
    out = []
    for start in range(0, n_samples - win + 1, hop):
        block = pyln_audio[start:start + win]
        try:
            lufs = meter.integrated_loudness(block)
        except ValueError:
            lufs = float("-inf")
        if not np.isfinite(lufs):
            lufs = float("-inf")
        out.append(lufs)
    return np.array(out, dtype=np.float32)


def loudness_range(short_term: np.ndarray) -> float:
    """LRA = (95th percentile − 10th percentile) of short-term LUFS, 무음 제외.

    EBU Tech 3342에 정의된 LRA의 단순화 버전 (실제는 추가 gating이 있지만
    유효 신호 영역에서는 충분히 일치한다).
    """
    finite = short_term[np.isfinite(short_term)]
    if len(finite) < 2:
        return 0.0
    p95 = float(np.percentile(finite, 95))
    p10 = float(np.percentile(finite, 10))
    return round(p95 - p10, 2)


def true_peak_dbtp(
    audio: np.ndarray,
    sample_rate: int,
    oversample: int = 4,
) -> float:
    """True Peak in dBTP (ITU-R BS.1770-4 Annex 2).

    4x oversampling 후 peak 측정. 0.0 dBTP = full scale.
    스테레오는 채널별 max of TP를 반환.
    """
    if audio.ndim == 2:
        ch_peaks = [true_peak_dbtp(audio[ch], sample_rate, oversample)
                    for ch in range(audio.shape[0])]
        return max(ch_peaks)

    # mono
    mono = audio.astype(np.float64)
    if oversample > 1:
        # scipy.signal.resample_poly: polyphase FIR — alias 방지
        upsampled = scipy_signal.resample_poly(mono, oversample, 1)
    else:
        upsampled = mono
    peak = float(np.max(np.abs(upsampled)))
    if peak <= 0:
        return float("-inf")
    return float(20.0 * np.log10(peak))


def sample_peak_dbfs(audio: np.ndarray) -> float:
    """단순 sample peak (oversample 없음). True Peak와 비교용."""
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        return float("-inf")
    return float(20.0 * np.log10(peak))


@dataclass
class LoudnessReport:
    integrated_lufs: float
    short_term_max: float
    short_term_min: float
    lra: float
    true_peak_dbtp: float
    sample_peak_dbfs: float
    duration_sec: float
    sample_rate: int
    channels: int

    def to_dict(self) -> dict:
        def _fmt(v: float) -> float | None:
            if not np.isfinite(v):
                return None
            return round(float(v), 2)
        return {
            "integrated_lufs": _fmt(self.integrated_lufs),
            "short_term_max_lufs": _fmt(self.short_term_max),
            "short_term_min_lufs": _fmt(self.short_term_min),
            "loudness_range_lu": _fmt(self.lra),
            "true_peak_dbtp": _fmt(self.true_peak_dbtp),
            "sample_peak_dbfs": _fmt(self.sample_peak_dbfs),
            "duration_sec": round(self.duration_sec, 4),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


def measure(audio: np.ndarray, sample_rate: int) -> LoudnessReport:
    """LUFS/LRA/TP 통합 측정. QC 리포트의 backing 함수."""
    n_ch = 1 if audio.ndim == 1 else audio.shape[0]
    n_samples = audio.shape[-1]
    duration = n_samples / sample_rate

    integrated = integrated_lufs(audio, sample_rate)
    st = short_term_lufs(audio, sample_rate)
    if len(st) > 0:
        finite = st[np.isfinite(st)]
        st_max = float(finite.max()) if len(finite) > 0 else float("-inf")
        st_min = float(finite.min()) if len(finite) > 0 else float("-inf")
    else:
        st_max = float("-inf")
        st_min = float("-inf")
    lra = loudness_range(st) if len(st) > 0 else 0.0
    tp = true_peak_dbtp(audio, sample_rate)
    sp = sample_peak_dbfs(audio)

    return LoudnessReport(
        integrated_lufs=integrated,
        short_term_max=st_max,
        short_term_min=st_min,
        lra=lra,
        true_peak_dbtp=tp,
        sample_peak_dbfs=sp,
        duration_sec=duration,
        sample_rate=sample_rate,
        channels=n_ch,
    )


# ---------------------------------------------------------------------------
# Loudness normalization (LUFS target with True-Peak ceiling)
# ---------------------------------------------------------------------------


def loudness_normalize(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -14.0,
    max_true_peak_dbtp: float = -1.0,
) -> tuple[np.ndarray, dict]:
    """오디오를 target LUFS로 게인 조정. True Peak 천장 초과 시 추가 감쇠.

    Returns:
        (조정된 오디오, 적용 메타데이터)

    메타데이터:
        - applied_gain_db: 실제로 적용된 게인 (LUFS 기반)
        - tp_limit_attenuation_db: TP 천장 초과 시 추가로 깎은 dB (없으면 0)
        - measured_in / measured_out: LUFS / TP 비교
    """
    measured_in = measure(audio, sample_rate)

    if not np.isfinite(measured_in.integrated_lufs):
        # 무음/측정 불가 — 게인 적용 안 함
        return audio.copy(), {
            "applied_gain_db": 0.0,
            "tp_limit_attenuation_db": 0.0,
            "skipped": "input loudness not measurable (silence or too short)",
            "measured_in": measured_in.to_dict(),
            "measured_out": measured_in.to_dict(),
        }

    # 1차: target LUFS로 맞추는 게인
    gain_db = target_lufs - measured_in.integrated_lufs
    linear = 10 ** (gain_db / 20.0)
    out = audio.astype(np.float32) * np.float32(linear)

    # 2차: True Peak 천장 초과 시 추가 감쇠
    tp_after = true_peak_dbtp(out, sample_rate)
    tp_atten = 0.0
    if np.isfinite(tp_after) and tp_after > max_true_peak_dbtp:
        tp_atten = max_true_peak_dbtp - tp_after  # 음수
        out *= np.float32(10 ** (tp_atten / 20.0))

    measured_out = measure(out, sample_rate)
    return out, {
        "applied_gain_db": round(gain_db, 3),
        "tp_limit_attenuation_db": round(tp_atten, 3),
        "target_lufs": target_lufs,
        "max_true_peak_dbtp": max_true_peak_dbtp,
        "measured_in": measured_in.to_dict(),
        "measured_out": measured_out.to_dict(),
    }


# ---------------------------------------------------------------------------
# Per-utterance leveling — segment-wise LUFS targeting with length preservation
# ---------------------------------------------------------------------------


def _slice_segment(audio: np.ndarray, start: int, end: int) -> np.ndarray:
    """audio (channels, samples) 또는 (samples,)의 [start:end] 슬라이스."""
    if audio.ndim == 1:
        return audio[start:end]
    return audio[:, start:end]


def _apply_gain_with_ramp(
    out: np.ndarray,
    start: int,
    end: int,
    target_gain: float,
    prev_gain: float,
    ramp_samples: int,
) -> None:
    """out의 [start:end] 구간에 target_gain 적용. 시작에서 prev_gain → target_gain 으로 ramp.

    out은 in-place 수정. ramp는 segment 시작 부분에만 적용 (boundary discontinuity 방지).
    """
    n = end - start
    if n <= 0:
        return
    ramp = min(ramp_samples, n)
    gains = np.full(n, target_gain, dtype=np.float32)
    if ramp > 0 and not np.isclose(prev_gain, target_gain):
        # linear interpolation in linear-amplitude domain
        gains[:ramp] = np.linspace(prev_gain, target_gain, ramp, dtype=np.float32)
    if out.ndim == 1:
        out[start:end] *= gains
    else:
        out[:, start:end] *= gains[np.newaxis, :]


def level_utterances(
    audio: np.ndarray,
    sample_rate: int,
    speech_segments: list,  # list[Segment] from vad.py — duck-typed (start, end attrs)
    target_lufs: float = -20.0,
    max_true_peak_dbtp: float = -1.0,
    noise_attenuation_db: float = -12.0,
    ramp_ms: float = 30.0,
    min_segment_ms: float = 200.0,
) -> tuple[np.ndarray, dict]:
    """발화 단위 LUFS 레벨링. 길이 유지.

    Args:
        speech_segments: VAD가 반환한 음성 구간 (start, end 샘플 인덱스 속성 보유)
        target_lufs: 각 발화의 목표 integrated LUFS
        max_true_peak_dbtp: 전체 출력의 TP 천장. 초과 시 모든 게인 비례 축소
        noise_attenuation_db: 음성 외 구간(noise gap)에 적용할 추가 감쇠
        ramp_ms: 게인 변화 시 segment 경계에서의 crossfade 길이
        min_segment_ms: 이보다 짧은 발화는 게인 측정 불가 — skip (이전 게인 유지)

    Returns:
        (조정된 오디오, 메타데이터)
    """
    out = audio.astype(np.float32, copy=True)
    n_total = audio.shape[-1]
    ramp_samples = int(ramp_ms * sample_rate / 1000.0)
    min_samples = int(min_segment_ms * sample_rate / 1000.0)
    noise_gain_lin = float(10 ** (noise_attenuation_db / 20.0))

    per_segment: list[dict] = []
    prev_gain_lin = 1.0
    cursor = 0  # 직전 segment의 end (noise 구간을 채우기 위해)

    # speech segment를 시간순 순회
    speech_sorted = sorted(speech_segments, key=lambda s: s.start)

    for seg in speech_sorted:
        # 1) cursor → seg.start: noise gap → attenuation 적용
        if seg.start > cursor:
            _apply_gain_with_ramp(out, cursor, seg.start, noise_gain_lin, prev_gain_lin, ramp_samples)
            prev_gain_lin = noise_gain_lin

        # 2) seg 자체: LUFS 측정 → 게인 산출
        seg_audio = _slice_segment(audio, seg.start, seg.end)
        seg_lufs = integrated_lufs(seg_audio, sample_rate)

        if seg.end - seg.start < min_samples or not np.isfinite(seg_lufs):
            # 너무 짧거나 측정 불가 — 이전 게인 유지하되 noise보다는 1.0에 가깝게
            target_gain_lin = 1.0
            seg_meta = {
                "start": seg.start,
                "end": seg.end,
                "input_lufs": None if not np.isfinite(seg_lufs) else round(seg_lufs, 2),
                "applied_gain_db": 0.0,
                "skipped": "too_short_or_silent",
            }
        else:
            gain_db = target_lufs - seg_lufs
            target_gain_lin = float(10 ** (gain_db / 20.0))
            seg_meta = {
                "start": seg.start,
                "end": seg.end,
                "input_lufs": round(seg_lufs, 2),
                "applied_gain_db": round(gain_db, 3),
            }

        _apply_gain_with_ramp(out, seg.start, seg.end, target_gain_lin, prev_gain_lin, ramp_samples)
        prev_gain_lin = target_gain_lin
        per_segment.append(seg_meta)
        cursor = seg.end

    # 3) 마지막 segment 이후 trailing noise
    if cursor < n_total:
        _apply_gain_with_ramp(out, cursor, n_total, noise_gain_lin, prev_gain_lin, ramp_samples)

    # 4) 전체 TP 천장 검사
    tp_after = true_peak_dbtp(out, sample_rate)
    tp_atten_db = 0.0
    if np.isfinite(tp_after) and tp_after > max_true_peak_dbtp:
        tp_atten_db = max_true_peak_dbtp - tp_after
        out *= np.float32(10 ** (tp_atten_db / 20.0))

    measured_in = measure(audio, sample_rate)
    measured_out = measure(out, sample_rate)
    return out, {
        "target_lufs": target_lufs,
        "max_true_peak_dbtp": max_true_peak_dbtp,
        "noise_attenuation_db": noise_attenuation_db,
        "tp_limit_attenuation_db": round(tp_atten_db, 3),
        "n_speech_segments": len(speech_sorted),
        "per_segment": per_segment,
        "measured_in": measured_in.to_dict(),
        "measured_out": measured_out.to_dict(),
    }
