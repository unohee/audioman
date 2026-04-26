# Created: 2026-03-21
# Purpose: 오디오 분석 함수 (스펙트럼, silence 감지 등)

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FrameMetrics:
    """프레임 단위 분석 결과"""
    rms: list[float] = field(default_factory=list)
    peak: list[float] = field(default_factory=list)
    spectral_centroid: list[float] = field(default_factory=list)
    spectral_entropy: list[float] = field(default_factory=list)
    zero_crossing_rate: list[float] = field(default_factory=list)


@dataclass
class SilenceRegion:
    start_sample: int
    end_sample: int
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "duration_sec": round(self.duration_sec, 6),
        }


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """(channels, samples) → mono 1D"""
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def compute_frame_metrics(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> FrameMetrics:
    """프레임 단위 오디오 메트릭 계산"""
    mono = _to_mono(audio)
    n_samples = len(mono)
    metrics = FrameMetrics()

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = mono[start : start + frame_size]

        # RMS
        rms = float(np.sqrt(np.mean(frame**2)))
        metrics.rms.append(rms)

        # Peak
        metrics.peak.append(float(np.max(np.abs(frame))))

        # Zero crossing rate
        zcr = float(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_size))
        metrics.zero_crossing_rate.append(zcr)

        # Spectrum
        window = np.hanning(frame_size)
        spectrum = np.abs(np.fft.rfft(frame * window))
        spectrum_sum = np.sum(spectrum)

        # Spectral centroid
        if spectrum_sum > 1e-10:
            centroid = float(np.sum(freqs * spectrum) / spectrum_sum)
        else:
            centroid = 0.0
        metrics.spectral_centroid.append(centroid)

        # Spectral entropy
        if spectrum_sum > 1e-10:
            prob = spectrum / spectrum_sum
            prob = prob[prob > 1e-10]
            entropy = float(-np.sum(prob * np.log2(prob)))
        else:
            entropy = 0.0
        metrics.spectral_entropy.append(entropy)

    return metrics


def compute_summary(metrics: FrameMetrics) -> dict:
    """프레임 메트릭의 요약 통계 (mean/min/max/std)"""
    summary = {}
    for name in ["rms", "peak", "spectral_centroid", "spectral_entropy", "zero_crossing_rate"]:
        values = np.array(getattr(metrics, name))
        if len(values) == 0:
            summary[name] = {"mean": 0, "min": 0, "max": 0, "std": 0}
        else:
            summary[name] = {
                "mean": round(float(np.mean(values)), 6),
                "min": round(float(np.min(values)), 6),
                "max": round(float(np.max(values)), 6),
                "std": round(float(np.std(values)), 6),
            }
    return summary


DEFAULT_OCTAVE_BANDS: list[tuple[float, float, str]] = [
    (20, 60, "sub"),
    (60, 250, "low"),
    (250, 500, "low_mid"),
    (500, 2000, "mid"),
    (2000, 4000, "upper_mid"),
    (4000, 8000, "presence"),
    (8000, 16000, "air"),
    (16000, 24000, "ultra"),
]


def long_term_spectrum(
    audio: np.ndarray,
    sample_rate: int,
    fft_size: int = 16384,
    hop_size: int | None = None,
    min_rms: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, int]:
    """발화/유효 신호 프레임만 평균한 long-term power spectrum 계산.

    min_rms 이하 프레임(무음/잡음)은 제외해 long-term 평균이 무음에 끌려가지 않도록 한다.
    반환: (freqs, power, n_frames_used)
    """
    mono = _to_mono(audio)
    if hop_size is None:
        hop_size = fft_size // 2
    window = np.hanning(fft_size).astype(np.float32)
    n_bins = fft_size // 2 + 1
    power_acc = np.zeros(n_bins, dtype=np.float64)
    n_used = 0

    for start in range(0, len(mono) - fft_size + 1, hop_size):
        frame = mono[start:start + fft_size]
        if np.sqrt(np.mean(frame ** 2)) < min_rms:
            continue
        spectrum = np.abs(np.fft.rfft(frame * window))
        power_acc += spectrum.astype(np.float64) ** 2
        n_used += 1

    if n_used > 0:
        power_acc /= n_used
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    return freqs, power_acc, n_used


def band_energy(
    freqs: np.ndarray,
    power: np.ndarray,
    bands: list[tuple[float, float, str]] | None = None,
) -> list[dict]:
    """대역별 에너지 분포 (% + dB rel total)."""
    if bands is None:
        bands = DEFAULT_OCTAVE_BANDS
    total = float(power.sum())
    out = []
    for lo, hi, name in bands:
        mask = (freqs >= lo) & (freqs < hi)
        e = float(power[mask].sum())
        pct = 100.0 * e / total if total > 0 else 0.0
        db = 10.0 * np.log10(e / total + 1e-30) if total > 0 else -300.0
        out.append({
            "band": name,
            "freq_low": lo,
            "freq_high": hi,
            "percent": round(pct, 4),
            "db_rel_total": round(db, 2),
        })
    return out


def dominant_frequencies(
    freqs: np.ndarray,
    power: np.ndarray,
    n: int = 10,
    min_separation_hz: float = 30.0,
    freq_min: float = 20.0,
    freq_max: float | None = None,
) -> list[dict]:
    """롱텀 스펙트럼에서 피크 주파수 top-N (유사 주파수 dedup)."""
    if freq_max is None:
        freq_max = float(freqs[-1])
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    band_freqs = freqs[mask]
    band_power = power[mask]
    if len(band_power) == 0 or band_power.max() == 0:
        return []

    peak_db = 10.0 * np.log10(band_power.max() + 1e-30)
    sorted_idx = np.argsort(band_power)[::-1]

    selected: list[dict] = []
    seen_freqs: list[float] = []
    for idx in sorted_idx:
        f = float(band_freqs[idx])
        if any(abs(f - s) < min_separation_hz for s in seen_freqs):
            continue
        seen_freqs.append(f)
        db = 10.0 * np.log10(float(band_power[idx]) + 1e-30) - peak_db
        selected.append({
            "frequency_hz": round(f, 1),
            "db_rel_peak": round(db, 2),
        })
        if len(selected) >= n:
            break
    return selected


def detect_hum(
    freqs: np.ndarray,
    power: np.ndarray,
    candidates: tuple[float, ...] = (50, 60, 100, 120, 150, 180),
    snr_threshold_db: float = 10.0,
) -> list[dict]:
    """전원 험(50/60Hz 및 배수) 검출. peak/floor SNR이 threshold 초과 시 hum 의심."""
    out = []
    for hum_f in candidates:
        if hum_f > freqs[-1]:
            continue
        bin_idx = int(np.argmin(np.abs(freqs - hum_f)))
        e = float(power[bin_idx])
        side_lo = power[max(0, bin_idx - 20):max(0, bin_idx - 3)]
        side_hi = power[bin_idx + 3:bin_idx + 20]
        side = np.concatenate([side_lo, side_hi])
        if len(side) == 0:
            continue
        floor = float(np.median(side))
        # 무음/저에너지 입력에서 log10(0) 경고 방지
        if e <= 0 or floor <= 0:
            snr = float("-inf")
            is_hum = False
        else:
            snr = 10.0 * np.log10(e / floor)
            is_hum = bool(snr > snr_threshold_db)
        out.append({
            "frequency_hz": hum_f,
            "snr_db": round(snr, 2) if snr != float("-inf") else None,
            "is_hum": is_hum,
        })
    return out


def hf_slope(
    freqs: np.ndarray,
    power: np.ndarray,
    mid_band: tuple[float, float] = (1000.0, 3000.0),
    high_band: tuple[float, float] = (10000.0, 16000.0),
) -> dict:
    """고역 기울기. high - mid (dB)."""
    mid_mask = (freqs >= mid_band[0]) & (freqs <= mid_band[1])
    hi_mask = (freqs >= high_band[0]) & (freqs <= high_band[1])
    mid_p = power[mid_mask]
    hi_p = power[hi_mask]
    if len(mid_p) == 0 or len(hi_p) == 0 or mid_p.mean() == 0 or hi_p.mean() == 0:
        return {"mid_db": None, "high_db": None, "slope_db": None}
    mid_db = 10.0 * np.log10(float(mid_p.mean()) + 1e-30)
    hi_db = 10.0 * np.log10(float(hi_p.mean()) + 1e-30)
    return {
        "mid_band_hz": list(mid_band),
        "high_band_hz": list(high_band),
        "mid_db": round(mid_db, 2),
        "high_db": round(hi_db, 2),
        "slope_db": round(hi_db - mid_db, 2),
    }


def spectrum_diagnostics(
    audio: np.ndarray,
    sample_rate: int,
    fft_size: int = 16384,
    min_rms: float = 0.01,
    n_dominant: int = 10,
) -> dict:
    """LLM이 읽기 좋은 형태의 통합 스펙트럼 진단.

    band_energy + dominant_frequencies + hum_check + hf_slope 한번에.
    """
    freqs, power, n_used = long_term_spectrum(
        audio, sample_rate, fft_size=fft_size, min_rms=min_rms
    )
    return {
        "fft_size": fft_size,
        "frames_analyzed": n_used,
        "min_rms_threshold": min_rms,
        "band_energy": band_energy(freqs, power),
        "dominant_frequencies": dominant_frequencies(freqs, power, n=n_dominant),
        "hum_check": detect_hum(freqs, power),
        "hf_slope": hf_slope(freqs, power),
    }


def detect_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_duration_sec: float = 0.1,
    frame_size: int = 1024,
    hop_size: int = 512,
) -> list[SilenceRegion]:
    """RMS 임계값 이하 구간을 silence로 감지"""
    mono = _to_mono(audio)
    threshold_linear = 10 ** (threshold_db / 20.0)
    n_samples = len(mono)

    regions = []
    in_silence = False
    silence_start = 0

    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = mono[start : start + frame_size]
        rms = np.sqrt(np.mean(frame**2))

        if rms < threshold_linear:
            if not in_silence:
                silence_start = start
                in_silence = True
        else:
            if in_silence:
                duration = (start - silence_start) / sample_rate
                if duration >= min_duration_sec:
                    regions.append(SilenceRegion(
                        start_sample=silence_start,
                        end_sample=start,
                        duration_sec=duration,
                    ))
                in_silence = False

    # 파일 끝까지 silence인 경우
    if in_silence:
        duration = (n_samples - silence_start) / sample_rate
        if duration >= min_duration_sec:
            regions.append(SilenceRegion(
                start_sample=silence_start,
                end_sample=n_samples,
                duration_sec=duration,
            ))

    return regions
