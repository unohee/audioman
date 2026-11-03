# Created: 2026-11-04
# Purpose: Steady-state latency (plugin delay compensation) measurement by cross-correlation.
# Dependencies: numpy, scipy
#
# A plugin's reported latency is a promise; this module measures what the audio
# actually does. The first onset of output is a poor latency estimate (look-ahead,
# warm-up and fade-in all move it), so the estimate is the lag that maximises the
# cross-correlation between the dry reference and the processed signal *after*
# a settling period has been cut from both.
#
# Probe with broadband material: a periodic signal correlates almost as well at
# every period (and, with polarity allowed, every half period), so its lag is
# ambiguous by construction.
#
# The search is two-sided: a processed signal that arrives early (negative lag)
# is a real failure mode and must be measurable. A best lag sitting on the edge of
# the search window means the true lag may lie outside it, so the estimate is
# flagged instead of trusted.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

DEFAULT_TOLERANCE_MS = 2.0
DEFAULT_MIN_NCC = 0.70


@dataclass
class LagEstimate:
    lag_samples: int  # processed is late by this many samples (negative: early)
    ncc: float  # normalised correlation at that lag, sign included
    at_edge: bool  # best lag on the search boundary: not trustworthy
    max_lag_samples: int
    settle_samples: int

    def to_dict(self) -> dict:
        return {
            "lag_samples": self.lag_samples,
            "ncc": round(self.ncc, 6),
            "at_edge": self.at_edge,
            "max_lag_samples": self.max_lag_samples,
            "settle_samples": self.settle_samples,
        }


def _mono64(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    x = x.mean(axis=0) if x.ndim == 2 else x
    # Non-finite samples would turn the FFT correlation into NaN everywhere; they are
    # judged by the finiteness gates, so here they only must not poison the lag.
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _ncc_at(reference: np.ndarray, processed: np.ndarray, lag: int) -> float:
    """Pearson-free normalised correlation of processed[n] against reference[n - lag]."""
    if lag >= 0:
        r, p = reference[: len(reference) - lag], processed[lag:]
    else:
        r, p = reference[-lag:], processed[: len(processed) + lag]
    n = min(len(r), len(p))
    r, p = r[:n], p[:n]
    denom = float(np.sqrt(np.dot(r, r) * np.dot(p, p)))
    return float(np.dot(r, p) / denom) if denom > 0 else 0.0


def estimate_lag(
    reference: np.ndarray,
    processed: np.ndarray,
    sample_rate: int,
    *,
    max_lag_ms: float = 500.0,
    settle_sec: float = 0.25,
) -> LagEstimate:
    """Lag of `processed` relative to `reference` from steady-state cross-correlation.

    Args:
        reference: dry input, (channels, samples) or (samples,).
        processed: plugin output, same layout.
        max_lag_ms: search window, both directions.
        settle_sec: cut from the start of both signals before correlating, so
            warm-up and onset transients do not decide the estimate. Reduced
            automatically for short signals.
    """
    ref = _mono64(reference)
    proc = _mono64(processed)
    n = min(len(ref), len(proc))
    max_lag = max(1, int(round(max_lag_ms * sample_rate / 1000.0)))
    max_lag = min(max_lag, max(1, n // 4))
    settle = int(settle_sec * sample_rate)
    settle = min(settle, max(0, (n - 2 * max_lag) // 2))
    ref, proc = ref[settle:n], proc[settle:n]

    corr = scipy_signal.correlate(proc, ref, mode="full", method="fft")
    lags = scipy_signal.correlation_lags(len(proc), len(ref), mode="full")
    window = np.abs(lags) <= max_lag
    corr, lags = corr[window], lags[window]
    best = int(lags[int(np.argmax(np.abs(corr)))])
    return LagEstimate(
        lag_samples=best,
        ncc=_ncc_at(ref, proc, best),
        at_edge=abs(best) >= max_lag,
        max_lag_samples=max_lag,
        settle_samples=settle,
    )


@dataclass
class PdcVerdict:
    status: str  # PASS | FAIL
    measured_samples: int
    reported_samples: int
    residual_samples: int
    residual_ms: float
    tolerance_ms: float
    ncc: float
    reason: str


def judge_pdc(
    estimate: LagEstimate,
    reported_samples: int,
    sample_rate: int,
    *,
    tolerance_ms: float = DEFAULT_TOLERANCE_MS,
    min_ncc: float = DEFAULT_MIN_NCC,
) -> PdcVerdict:
    """PASS when the measured lag matches the reported latency within tolerance.

    A weak correlation or an edge-of-window lag fails: the measurement could not
    establish the latency, and an unestablished latency is not a pass.
    """
    residual = estimate.lag_samples - int(reported_samples)
    residual_ms = residual * 1000.0 / sample_rate
    reasons = []
    if abs(estimate.ncc) < min_ncc:
        reasons.append(f"correlation {abs(estimate.ncc):.3f} < {min_ncc} (latency not established)")
    if estimate.at_edge:
        reasons.append(f"best lag on the search edge (+/-{estimate.max_lag_samples} samples)")
    if abs(residual_ms) > tolerance_ms:
        reasons.append(f"measured {estimate.lag_samples} vs reported {reported_samples} samples "
                       f"({residual_ms:+.3f} ms > +/-{tolerance_ms} ms)")
    return PdcVerdict(
        status="FAIL" if reasons else "PASS",
        measured_samples=estimate.lag_samples,
        reported_samples=int(reported_samples),
        residual_samples=residual,
        residual_ms=round(residual_ms, 4),
        tolerance_ms=tolerance_ms,
        ncc=round(estimate.ncc, 6),
        reason="; ".join(reasons),
    )
