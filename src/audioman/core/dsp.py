# Created: 2026-03-21
# Purpose: 내장 DSP 함수 (fade, trim, normalize, gate, gain)

import numpy as np


def fade_in(audio: np.ndarray, samples: int) -> np.ndarray:
    """선형 fade in. samples: fade 길이 (샘플 수)"""
    out = audio.copy()
    if audio.ndim == 1:
        n = min(samples, len(out))
        out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    else:
        n = min(samples, out.shape[1])
        curve = np.linspace(0.0, 1.0, n, dtype=np.float32)
        out[:, :n] *= curve
    return out


def fade_out(audio: np.ndarray, samples: int) -> np.ndarray:
    """선형 fade out. samples: fade 길이 (샘플 수)"""
    out = audio.copy()
    if audio.ndim == 1:
        n = min(samples, len(out))
        out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    else:
        n = min(samples, out.shape[1])
        curve = np.linspace(1.0, 0.0, n, dtype=np.float32)
        out[:, -n:] *= curve
    return out


def trim(
    audio: np.ndarray,
    start: int = 0,
    end: int | None = None,
) -> np.ndarray:
    """샘플 단위 트리밍"""
    if audio.ndim == 1:
        return audio[start:end]
    return audio[:, start:end]


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    pad_samples: int = 0,
) -> np.ndarray:
    """앞뒤 silence 제거"""
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    threshold = 10 ** (threshold_db / 20.0)
    above = np.where(np.abs(mono) > threshold)[0]

    if len(above) == 0:
        return audio  # 전체가 silence면 원본 반환

    start = max(0, above[0] - pad_samples)
    end = min(len(mono), above[-1] + 1 + pad_samples)

    if audio.ndim == 1:
        return audio[start:end]
    return audio[:, start:end]


def normalize(
    audio: np.ndarray,
    peak_db: float | None = None,
    target_rms_db: float | None = None,
) -> np.ndarray:
    """피크 또는 RMS 기준 정규화"""
    out = audio.copy().astype(np.float32)

    if peak_db is not None:
        current_peak = np.max(np.abs(out))
        if current_peak < 1e-10:
            return out
        target_peak = 10 ** (peak_db / 20.0)
        out *= target_peak / current_peak

    elif target_rms_db is not None:
        current_rms = np.sqrt(np.mean(out**2))
        if current_rms < 1e-10:
            return out
        target_rms = 10 ** (target_rms_db / 20.0)
        out *= target_rms / current_rms

    return out


def gain(audio: np.ndarray, db: float) -> np.ndarray:
    """dB 단위 게인 적용"""
    return audio * (10 ** (db / 20.0))


def gate(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -50.0,
    attack_sec: float = 0.01,
    release_sec: float = 0.05,
    frame_size: int = 1024,
    hop_size: int = 512,
) -> np.ndarray:
    """RMS 기반 노이즈 게이트. 임계값 이하 구간을 silence로."""
    out = audio.copy()
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    threshold = 10 ** (threshold_db / 20.0)
    n_samples = len(mono)

    # 프레임별 RMS → envelope
    envelope = np.ones(n_samples, dtype=np.float32)
    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame_rms = np.sqrt(np.mean(mono[start : start + frame_size] ** 2))
        if frame_rms < threshold:
            envelope[start : start + hop_size] = 0.0

    # attack/release smoothing
    attack_samples = max(1, int(attack_sec * sample_rate))
    release_samples = max(1, int(release_sec * sample_rate))

    smoothed = np.copy(envelope)
    for i in range(1, n_samples):
        if smoothed[i] > smoothed[i - 1]:
            # attack (opening)
            alpha = 1.0 / attack_samples
            smoothed[i] = smoothed[i - 1] + alpha * (smoothed[i] - smoothed[i - 1])
        else:
            # release (closing)
            alpha = 1.0 / release_samples
            smoothed[i] = smoothed[i - 1] + alpha * (smoothed[i] - smoothed[i - 1])

    if out.ndim == 1:
        out *= smoothed
    else:
        out *= smoothed[np.newaxis, :]

    return out
