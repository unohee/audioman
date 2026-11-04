# Created: 2026-11-04
# Purpose: Monophonic f0 track (YIN) and the pitch descriptors derived from it.
# Dependencies: numpy, scipy.signal
#
# YIN (de Cheveigne & Kawahara, JASA 2002): for every frame, the difference function
# d(tau) = sum_j (x_j - x_{j+tau})^2 over an integration window, normalised by its
# cumulative mean (d'), and the period is the first tau where d' dips below an
# absolute threshold (then refined to that dip's local minimum and by parabolic
# interpolation). d' at the chosen tau is the frame's aperiodicity: near 0 for a
# periodic frame, near 1 for noise. A frame is voiced when it is periodic enough, its
# f0 is in range and it is not silent.
#
# The track runs on a 16 kHz downmix: f0 up to 1 kHz needs nothing above that, and
# it keeps the cost of a minute of audio well under a second.
#
# Scope: monophonic material (a voice, a solo instrument). A chord or a mix has no
# single f0; such material is mostly unvoiced here, and the descriptors are null when
# fewer than MIN_VOICED_FRACTION of the non-silent frames are voiced.

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy import signal as scipy_signal

TRACK_RATE = 16000
WINDOW = 512  # integration window, 32 ms
HOP = 160  # 10 ms
F0_MIN_HZ = 50.0
F0_MAX_HZ = 1000.0
THRESHOLD = 0.15  # absolute threshold on d'; also the voicing threshold
SILENT_DB = -70.0  # frames this far below the loudest are silent (as in texture)
MIN_VOICED_FRACTION = 0.2
_CHUNK = 512  # frames per FFT batch, bounds memory on long files

TAU_MIN = int(TRACK_RATE // F0_MAX_HZ)
TAU_MAX = int(np.ceil(TRACK_RATE / F0_MIN_HZ))
_FRAME = WINDOW + TAU_MAX + 1


@dataclass
class PitchTrack:
    times_sec: np.ndarray  # frame centres, in seconds of the input
    f0_hz: np.ndarray  # nan where unvoiced
    aperiodicity: np.ndarray
    voiced: np.ndarray  # bool
    silent: np.ndarray  # bool


def _downmix_16k(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TRACK_RATE:
        return mono.astype(np.float64)
    g = gcd(TRACK_RATE, int(sample_rate))
    return scipy_signal.resample_poly(mono.astype(np.float64), TRACK_RATE // g, int(sample_rate) // g)


def _cmnd(frames: np.ndarray) -> np.ndarray:
    """Cumulative-mean-normalised difference d'(tau), tau = 0..TAU_MAX, for each frame."""
    n_fft = 1 << int(np.ceil(np.log2(_FRAME + WINDOW)))
    head = np.zeros_like(frames)
    head[:, :WINDOW] = frames[:, :WINDOW]
    # r(tau) = sum_{j<W} x_j x_{j+tau}
    acf = np.fft.irfft(np.conj(np.fft.rfft(head, n_fft)) * np.fft.rfft(frames, n_fft), n_fft)[:, : TAU_MAX + 1]
    sq = np.concatenate((np.zeros((len(frames), 1)), np.cumsum(frames * frames, axis=1)), axis=1)
    taus = np.arange(TAU_MAX + 1)
    energy_shifted = sq[:, taus + WINDOW] - sq[:, taus]  # sum_{j<W} x_{j+tau}^2
    d = np.maximum(sq[:, [WINDOW]] + energy_shifted - 2.0 * acf, 0.0)
    out = np.ones_like(d)
    running = np.cumsum(d[:, 1:], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, 1:] = np.where(running > 0, d[:, 1:] * taus[1:] / running, 1.0)
    return out


def _pick(dp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Period (fractional samples) and aperiodicity per frame from d'."""
    rows = np.arange(len(dp))
    search = dp[:, TAU_MIN: TAU_MAX]
    below = search < THRESHOLD
    first = np.where(below.any(axis=1), below.argmax(axis=1), search.argmin(axis=1))
    # walk down to the bottom of the dip that crossed the threshold
    tau = first + TAU_MIN
    while True:
        nxt = np.minimum(tau + 1, TAU_MAX - 1)
        step = (nxt > tau) & (dp[rows, nxt] < dp[rows, tau])
        if not step.any():
            break
        tau = np.where(step, nxt, tau)
    left, mid, right = dp[rows, tau - 1], dp[rows, tau], dp[rows, np.minimum(tau + 1, TAU_MAX)]
    curve = left - 2.0 * mid + right
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(curve > 0, 0.5 * (left - right) / curve, 0.0)
    return tau + np.clip(shift, -1.0, 1.0), mid


def track(mono: np.ndarray, sample_rate: int) -> PitchTrack:
    """YIN f0 track of a mono signal; empty when it is shorter than one frame."""
    x = _downmix_16k(np.asarray(mono, dtype=np.float64), sample_rate)
    if len(x) < _FRAME:
        empty = np.zeros(0)
        return PitchTrack(empty, empty, empty, empty.astype(bool), empty.astype(bool))
    starts = np.arange(0, len(x) - _FRAME + 1, HOP)
    view = np.lib.stride_tricks.sliding_window_view(x, _FRAME)[::HOP]
    periods, aper = [], []
    for i in range(0, len(starts), _CHUNK):
        p, a = _pick(_cmnd(np.ascontiguousarray(view[i: i + _CHUNK])))
        periods.append(p)
        aper.append(a)
    period, aperiodicity = np.concatenate(periods), np.concatenate(aper)
    energy = np.einsum("ij,ij->i", view[:, :WINDOW], view[:, :WINDOW])
    silent = energy <= energy.max() * 10 ** (SILENT_DB / 10) if energy.max() > 0 else np.ones(len(starts), bool)
    f0 = TRACK_RATE / period
    voiced = (aperiodicity < THRESHOLD) & (f0 >= F0_MIN_HZ) & (f0 <= F0_MAX_HZ) & ~silent
    return PitchTrack(
        times_sec=(starts + WINDOW / 2) / TRACK_RATE,
        f0_hz=np.where(voiced, f0, np.nan),
        aperiodicity=aperiodicity,
        voiced=voiced,
        silent=silent,
    )


def descriptors(t: PitchTrack, start_sec: float, end_sec: float) -> dict[str, float | None]:
    """Pitch descriptors over frames centred in [start_sec, end_sec).

    pitch            median f0, cents re A4 (440 Hz)
    pitch_spread     interquartile range of f0, cents (vibrato, drift)
    pitch_deviation  mean distance to the nearest 12-TET semitone, cents (0..50)
    voiced_fraction  voiced share of the non-silent frames
    """
    sel = (t.times_sec >= start_sec) & (t.times_sec < end_sec)
    sounding = int(np.count_nonzero(sel & ~t.silent))
    voiced = sel & t.voiced
    n_voiced = int(np.count_nonzero(voiced))
    out: dict[str, float | None] = {"pitch": None, "pitch_spread": None, "pitch_deviation": None,
                                    "voiced_fraction": n_voiced / sounding if sounding else None}
    if not sounding or n_voiced < 3 or n_voiced / sounding < MIN_VOICED_FRACTION:
        return out
    cents = 1200.0 * np.log2(t.f0_hz[voiced] / 440.0)
    q25, q50, q75 = np.percentile(cents, [25, 50, 75])
    out["pitch"] = float(q50)
    out["pitch_spread"] = float(q75 - q25)
    out["pitch_deviation"] = float(np.mean(np.abs(cents - 100.0 * np.round(cents / 100.0))))
    return out
