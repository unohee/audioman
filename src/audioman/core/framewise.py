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
# `_zero_crossing_rate`) to floating-point rounding. The dtypes mirror them: band
# power uses a float32 Hann and FFT, flatness a float64 one. When both are asked for,
# band power is read from the float64 spectrum, which differs from the float32 one
# only by float32 rounding. RMS and ZCR come from whole-signal cumulative sums.

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
    hops = np.diff(starts)
    # `_frame_bounds` yields a constant hop from 0, so the frames are a strided view
    # of the signal and need no gather copy; other bounds fall back to fancy indexing.
    hop = int(hops[0]) if len(hops) and np.all(hops == hops[0]) and starts[0] == 0 else None

    # RMS from one float64 cumulative sum of squares: O(samples), not O(frames x frame_len).
    squares = np.concatenate(([0.0], np.cumsum(mono.astype(np.float64) ** 2)))
    out.rms[:] = np.sqrt(np.maximum(squares[starts + frame_len] - squares[starts], 0.0) / frame_len)

    if frame_len < 2 or not (bands or flatness or zcr):
        return out  # the single-frame helpers return 0.0 below two samples

    if zcr:
        # Sign changes between neighbouring samples, counted once for the whole
        # signal; a frame's rate is its count over frame_len - 1 pairs. Integer
        # counts, so this equals the per-frame mean exactly.
        changes = np.concatenate(([0], np.cumsum(np.signbit(mono[1:]) != np.signbit(mono[:-1]), dtype=np.int32)))
        out.zcr[:] = (changes[starts + frame_len - 1] - changes[starts]) / (frame_len - 1)

    hann32 = np.hanning(frame_len).astype(np.float32)  # band-power window (float32 FFT)
    hann64 = np.hanning(frame_len)
    # Band masks are contiguous bin ranges; slicing keeps the means copy-free.
    band_slices = [_mask_to_slice(m) for m in _band_masks(frame_len, sample_rate, bands)]

    # Work buffers are allocated once per call and reused for every chunk: on some
    # macOS Python builds a fresh multi-MB temporary costs 2-4x the arithmetic done
    # in it, so per-chunk allocation, not the FFT, would dominate.
    rows = min(_CHUNK_FRAMES, n_frames)
    n_bins = frame_len // 2 + 1
    # When flatness needs the float64 spectrum anyway, band power is read from it
    # too (one FFT instead of two). It then differs from the float32-FFT single-frame
    # helper only by float32 rounding, well inside the parity tolerance.
    bands32 = bool(bands) and not flatness
    if bands32:
        win32 = np.empty((rows, frame_len), dtype=np.float32)
        spec32 = np.empty((rows, n_bins), dtype=np.complex64)
        pow32 = np.empty((rows, n_bins), dtype=np.float32)
    if flatness:
        win64 = np.empty((rows, frame_len), dtype=np.float64)
        spec64 = np.empty((rows, n_bins), dtype=np.complex128)
        pow64 = np.empty((rows, n_bins), dtype=np.float64)
        log64 = np.empty((rows, n_bins - 1), dtype=np.float64)

    for lo in range(0, n_frames, _CHUNK_FRAMES):
        hi = min(lo + _CHUNK_FRAMES, n_frames)
        m = hi - lo
        frames = view[lo * hop : (hi - 1) * hop + 1 : hop] if hop else view[starts[lo:hi]]
        if bands32:
            np.multiply(frames, hann32, out=win32[:m])
            np.fft.rfft(win32[:m], axis=1, out=spec32[:m])
            np.abs(spec32[:m], out=pow32[:m])
            np.square(pow32[:m], out=pow32[:m])
            for dest, band in zip(out.band_power, band_slices, strict=True):
                if band is not None:
                    dest[lo:hi] = np.mean(pow32[:m, band], axis=1)
        if flatness:
            np.multiply(frames, hann64, out=win64[:m])
            np.fft.rfft(win64[:m], axis=1, out=spec64[:m])
            np.abs(spec64[:m], out=pow64[:m])
            np.square(pow64[:m], out=pow64[:m])
            for dest, band in zip(out.band_power, band_slices, strict=True):
                if band is not None:
                    dest[lo:hi] = np.mean(pow64[:m, band], axis=1)
            ac = pow64[:m, 1:]  # DC excluded
            np.add(ac, 1e-18, out=ac)
            np.log(ac, out=log64[:m])
            out.flatness[lo:hi] = np.exp(np.mean(log64[:m], axis=1)) / np.mean(ac, axis=1)
    return out


def _mask_to_slice(mask: np.ndarray) -> slice | None:
    """A contiguous boolean mask as a slice; None when it selects nothing."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    return slice(int(idx[0]), int(idx[-1]) + 1)
