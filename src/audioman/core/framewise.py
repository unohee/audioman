# Created: 2026-11-03
# Purpose: Batched per-frame features (RMS, band power, spectral flatness, ZCR).
# Dependencies: numpy
#
# The aesthetic screeners used to call one rfft per frame and per band, so a
# five-minute file cost ~130k FFTs plus as many `np.hanning` constructions. This
# module computes the same per-frame quantities with one strided frame matrix and
# one batched rfft per window dtype, processed in bounded chunks so memory stays
# flat on long files.
#
# Parity contract: every value equals the single-frame helpers in
# `audioman.core.aesthetic` (`_band_power`, `_spectral_flatness`,
# `_zero_crossing_rate`) to floating-point rounding. The dtypes deliberately mirror
# them: band power windows with a float32 Hann (float32 FFT), flatness with a
# float64 Hann (float64 FFT).

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Frames per batched FFT. 1024 frames of 100 ms at 48 kHz is ~40 MB of float64,
# which keeps peak memory bounded independently of the file length.
_CHUNK_FRAMES = 1024


@dataclass
class FrameFeatures:
    """Per-frame features, one entry per (start, end) bound."""

    rms: np.ndarray
    band_power: list[np.ndarray] = field(default_factory=list)
    flatness: np.ndarray | None = None
    zcr: np.ndarray | None = None


def _band_masks(frame_len: int, sample_rate: int, bands: tuple[tuple[float, float], ...]) -> list[np.ndarray]:
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sample_rate)
    return [(freqs >= lo) & (freqs < min(hi, sample_rate / 2.0)) for lo, hi in bands]


def frame_features(
    mono: np.ndarray,
    bounds: list[tuple[int, int]],
    sample_rate: int,
    *,
    bands: tuple[tuple[float, float], ...] = (),
    flatness: bool = False,
    zcr: bool = False,
) -> FrameFeatures:
    """Compute per-frame features for equally sized frames.

    Args:
        mono: 1-D float32 signal.
        bounds: (start, end) pairs from `aesthetic._frame_bounds`; all frames share
            one length, which that function guarantees.
        sample_rate: sample rate in Hz (band edges and Nyquist clamp).
        bands: (low, high) Hz ranges; mean Hann-windowed power per band is returned
            in the same order.
        flatness: also compute spectral flatness (DC bin excluded).
        zcr: also compute the zero-crossing rate.
    """
    n_frames = len(bounds)
    out = FrameFeatures(
        rms=np.zeros(n_frames, dtype=np.float64),
        band_power=[np.zeros(n_frames, dtype=np.float64) for _ in bands],
        flatness=np.zeros(n_frames, dtype=np.float64) if flatness else None,
        zcr=np.zeros(n_frames, dtype=np.float64) if zcr else None,
    )
    if n_frames == 0:
        return out

    frame_len = bounds[0][1] - bounds[0][0]
    starts = np.fromiter((b[0] for b in bounds), dtype=np.int64, count=n_frames)
    view = np.lib.stride_tricks.sliding_window_view(mono, frame_len)

    spectral = frame_len >= 2  # the single-frame helpers return 0.0 below this
    if spectral:
        hann32 = np.hanning(frame_len).astype(np.float32)
        hann64 = np.hanning(frame_len)
        masks = _band_masks(frame_len, sample_rate, bands)

    for lo in range(0, n_frames, _CHUNK_FRAMES):
        hi = min(lo + _CHUNK_FRAMES, n_frames)
        frames = view[starts[lo:hi]]
        out.rms[lo:hi] = np.sqrt(np.mean(frames**2, axis=1))
        if not spectral:
            continue
        if bands:
            power32 = np.abs(np.fft.rfft(frames * hann32, axis=1)) ** 2
            for dest, mask in zip(out.band_power, masks, strict=True):
                if np.any(mask):
                    dest[lo:hi] = np.mean(power32[:, mask], axis=1)
        if flatness:
            power64 = np.abs(np.fft.rfft(frames * hann64, axis=1)) ** 2
            power64 = power64[:, 1:] + 1e-18
            out.flatness[lo:hi] = np.exp(np.mean(np.log(power64), axis=1)) / np.mean(power64, axis=1)
        if zcr:
            out.zcr[lo:hi] = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    return out
