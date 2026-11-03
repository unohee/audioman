# Created: 2026-11-04
# Purpose: Frame-wise spectral texture descriptors computed from one STFT.
# Dependencies: numpy
#
# Every function takes the power spectrogram P (frames x bins) and the bin
# frequencies, and returns one value per frame, so callers can aggregate per
# segment or globally. Definitions are the standard ones from the MIR literature;
# sharpness is an approximation of Zwicker's (Bark-band specific loudness with the
# 0.23 power law and the high-band weighting), useful for direction and relative
# comparison, not as a calibrated acum value.

from __future__ import annotations

import numpy as np

_EPS = 1e-20


def power_spectrogram(mono: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Hann-windowed power spectrogram, frames x (frame // 2 + 1), float64."""
    if len(mono) < frame:
        mono = np.pad(mono, (0, frame - len(mono)))
    frames = np.lib.stride_tricks.sliding_window_view(mono, frame)[::hop]
    spec = np.fft.rfft(frames * np.hanning(frame), axis=1)
    return spec.real**2 + spec.imag**2


def centroid(P: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    return (P @ freqs) / (P.sum(axis=1) + _EPS)


def bandwidth(P: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    c = centroid(P, freqs)
    spread = ((freqs[None, :] - c[:, None]) ** 2 * P).sum(axis=1) / (P.sum(axis=1) + _EPS)
    return np.sqrt(spread)


def rolloff(P: np.ndarray, freqs: np.ndarray, fraction: float = 0.85) -> np.ndarray:
    cumulative = np.cumsum(P, axis=1)
    target = fraction * cumulative[:, -1:]
    idx = np.argmax(cumulative >= target, axis=1)
    return freqs[idx]


def flatness_db(P: np.ndarray) -> np.ndarray:
    """Geometric over arithmetic mean of the power spectrum (DC excluded), in dB (0 = white)."""
    p = P[:, 1:] + _EPS
    return 10.0 * (np.mean(np.log10(p), axis=1) - np.log10(np.mean(p, axis=1)))


def contrast_db(P: np.ndarray, freqs: np.ndarray, *, low_hz: float = 200.0, quantile: float = 0.2) -> np.ndarray:
    """Mean over octave bands of (peak - valley) level: mean of the top vs bottom 20% bins."""
    nyquist = freqs[-1]
    edges = [low_hz]
    while edges[-1] * 2 < nyquist:
        edges.append(edges[-1] * 2)
    per_band = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        band = P[:, (freqs >= lo) & (freqs < hi)]
        if band.shape[1] < 5:
            continue
        ordered = np.sort(band, axis=1)
        k = max(1, int(round(quantile * band.shape[1])))
        valley = ordered[:, :k].mean(axis=1) + _EPS
        peak = ordered[:, -k:].mean(axis=1) + _EPS
        per_band.append(10.0 * np.log10(peak / valley))
    if not per_band:
        return np.zeros(P.shape[0])
    return np.mean(per_band, axis=0)


def flux(P: np.ndarray) -> np.ndarray:
    """Positive change of the magnitude spectrum between frames, normalised by its sum (0 = static)."""
    mag = np.sqrt(P)
    rise = np.maximum(np.diff(mag, axis=0, prepend=mag[:1]), 0.0).sum(axis=1)
    return rise / (mag.sum(axis=1) + _EPS)


def hf_ratio_db(P: np.ndarray, freqs: np.ndarray, split_hz: float = 4000.0) -> np.ndarray:
    high = P[:, freqs >= split_hz].sum(axis=1)
    return 10.0 * np.log10((high + _EPS) / (P.sum(axis=1) + _EPS))


def _bark(freqs: np.ndarray) -> np.ndarray:
    return 13.0 * np.arctan(0.00076 * freqs) + 3.5 * np.arctan((freqs / 7500.0) ** 2)


def sharpness(P: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Approximate sharpness: weighted first moment of Bark-band specific loudness."""
    z_of_bin = _bark(freqs)
    bands = np.arange(0.5, 24.5, 1.0)
    specific = np.zeros((P.shape[0], len(bands)))
    for i, z in enumerate(bands):
        sel = (z_of_bin >= z - 0.5) & (z_of_bin < z + 0.5)
        if np.any(sel):
            specific[:, i] = P[:, sel].sum(axis=1) ** 0.23
    weight = np.where(bands <= 15.8, 1.0, 0.066 * np.exp(0.171 * bands))
    return 0.11 * (specific @ (weight * bands)) / (specific.sum(axis=1) + _EPS)


def erb_centers(low_hz: float, high_hz: float, n: int) -> np.ndarray:
    """n centre frequencies equally spaced on the ERB-rate scale (Glasberg & Moore)."""
    def rate(f):
        return 21.4 * np.log10(1.0 + 0.00437 * f)

    def inverse(r):
        return (10.0 ** (r / 21.4) - 1.0) / 0.00437

    return inverse(np.linspace(rate(low_hz), rate(high_hz), n))


def erb_profile_db(P: np.ndarray, freqs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Mean level (dB) per ERB band over all frames; bands are one ERB wide."""
    mean_power = P.mean(axis=0)
    widths = 24.7 * (0.00437 * centers + 1.0)
    out = np.empty(len(centers))
    for i, (c, w) in enumerate(zip(centers, widths, strict=True)):
        sel = (freqs >= c - w / 2) & (freqs < c + w / 2)
        if not np.any(sel):  # narrower than one bin: take the nearest bin
            sel = np.zeros_like(freqs, dtype=bool)
            sel[int(np.argmin(np.abs(freqs - c)))] = True
        out[i] = 10.0 * np.log10(mean_power[sel].mean() + _EPS)
    return out


def onset_times(P: np.ndarray, hop: int, sample_rate: int, *, min_gap_sec: float = 0.05,
                k: float = 4.0) -> np.ndarray:
    """Onset times (s): peaks of the spectral-flux novelty above median + k * MAD."""
    mag = np.sqrt(P)
    novelty = np.maximum(np.diff(mag, axis=0, prepend=mag[:1]), 0.0).sum(axis=1)
    if len(novelty) < 3:
        return np.zeros(0)
    median = np.median(novelty)
    mad = np.median(np.abs(novelty - median)) + _EPS
    threshold = median + k * 1.4826 * mad
    peaks = np.flatnonzero((novelty[1:-1] > threshold) & (novelty[1:-1] >= novelty[:-2])
                           & (novelty[1:-1] > novelty[2:])) + 1
    times, last = [], -np.inf
    for p in peaks:
        t = p * hop / sample_rate
        if t - last >= min_gap_sec:
            times.append(t)
            last = t
    return np.asarray(times)
