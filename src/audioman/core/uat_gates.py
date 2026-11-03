# Created: 2026-11-04
# Purpose: Host/plugin acceptance gates: real-time take vs offline render, smoke checks,
#          dropout blocks. Results are report.v2 gates and measurements.
# Dependencies: numpy, audioman.core.pdc, audioman.core.report_v2
#
# `compare_takes` judges four renders of the same source through the same plugin:
#
#   source     the dry input
#   offline    the host's offline (bounce) render
#   realtime   the host's real-time recording of the plugin output
#   bypassed   (optional) a real-time recording with the plugin bypassed
#
# Blocks are 256 samples so that any lost host buffer of 512 samples or more fully
# covers at least one analysis block, wherever it falls.
#
# A correct plugin in a correct host produces a real-time take identical to the
# offline render, sample-aligned with the source, with no dropped or unprocessed
# blocks, and a bypass that returns the source untouched and undelayed. Each of
# those statements is one gate. Gates that need an optional input are only
# declared when that input is given; a declared gate that cannot be evaluated
# fails (report_v2.require).

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from audioman.core.pdc import estimate_lag
from audioman.core.report_v2 import Gate, Measurement, Report, require

SILENT_DBFS = -60.0  # a block below this is silent
ACTIVE_DBFS = -40.0  # a reference block above this should not be silent
NULL_MATCH_DBFS = -90.0  # residual peak at or below this counts as identical
PROCESSED_MIN_DB = -40.0  # offline vs source residual must exceed this to count as processed
UNPROCESSED_TOL = 1e-4  # a block within this peak of the source was passed through


def _mono64(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    return x.mean(axis=0) if x.ndim == 2 else x


def _db(value: float) -> float:
    return 20.0 * math.log10(value) if value > 0 else -math.inf


def _round_db(value: float) -> float | None:
    return round(value, 2) if math.isfinite(value) else None


def _blocks(x: np.ndarray, block: int) -> np.ndarray:
    n = (len(x) // block) * block
    return x[:n].reshape(-1, block)


def block_rms_dbfs(audio: np.ndarray, block: int) -> np.ndarray:
    b = _blocks(_mono64(audio), block)
    rms = np.sqrt(np.mean(b * b, axis=1)) if len(b) else np.zeros(0)
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(rms)


def dropout_blocks(
    output: np.ndarray,
    reference: np.ndarray,
    block: int = 256,
    *,
    silent_dbfs: float = SILENT_DBFS,
    active_dbfs: float = ACTIVE_DBFS,
) -> list[int]:
    """Indices of blocks where `output` is silent while `reference` is clearly active.

    `reference` is what the output should resemble (the offline render, or the
    input for a level-preserving plugin). Blocks that are quiet in the reference
    are never counted, so a plugin that gates silence is not a dropout.
    """
    out_db = block_rms_dbfs(output, block)
    ref_db = block_rms_dbfs(reference, block)
    n = min(len(out_db), len(ref_db))
    hits = (out_db[:n] < silent_dbfs) & (ref_db[:n] >= active_dbfs)
    return [int(i) for i in np.flatnonzero(hits)]


def residual_peak_dbfs(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _mono64(a), _mono64(b)
    n = min(len(x), len(y))
    if n == 0:
        return -math.inf
    diff = np.abs(x[:n] - y[:n])
    if not np.all(np.isfinite(diff)):
        return math.inf  # NaN/Inf can never count as a match
    return _db(float(np.max(diff)))


@dataclass
class TakeSet:
    source: np.ndarray
    offline: np.ndarray
    realtime: np.ndarray
    sample_rate: int
    bypassed: np.ndarray | None = None
    render_seconds: float | None = None


def _signal_gate(name: str, audio: np.ndarray, ms: list[Measurement]) -> Gate:
    x = _mono64(audio)
    finite = bool(np.all(np.isfinite(x)))
    rms_db = _db(float(np.sqrt(np.mean(x * x)))) if finite and len(x) else -math.inf
    ms.append(Measurement(f"{name}.rms_dbfs", _round_db(rms_db), "dBFS", 0.005, "rounding", "uat"))
    if not finite:
        return Gate(f"{name}_present", "FAIL", reason="non-finite samples")
    if rms_db <= SILENT_DBFS:
        return Gate(f"{name}_present", "FAIL", measurement=f"{name}.rms_dbfs",
                    threshold=f"> {SILENT_DBFS} dBFS", reason="output is silent")
    return Gate(f"{name}_present", "PASS", measurement=f"{name}.rms_dbfs", threshold=f"> {SILENT_DBFS} dBFS")


def compare_takes(
    takes: TakeSet,
    *,
    block: int = 256,
    match_dbfs: float = NULL_MATCH_DBFS,
    expect_processing: bool = True,
) -> Report:
    sr = takes.sample_rate
    one_sample = 1.0 / sr
    ms: list[Measurement] = []
    gates: list[Gate] = []
    required = ["realtime_present", "offline_present", "rt_matches_offline", "no_dropout_blocks",
                "no_unprocessed_blocks", "delay_compensated"]

    gates.append(_signal_gate("realtime", takes.realtime, ms))
    gates.append(_signal_gate("offline", takes.offline, ms))

    # real-time take vs offline render: same samples, same position
    rt_lag = estimate_lag(takes.offline, takes.realtime, sr, settle_sec=0.0)
    rt_null = residual_peak_dbfs(takes.realtime, takes.offline)
    ms += [
        Measurement("rt_vs_offline.lag_samples", rt_lag.lag_samples, "samples", 0.0, "exact", "uat.compare"),
        Measurement("rt_vs_offline.residual_peak_dbfs", _round_db(rt_null), "dBFS", 0.005, "rounding", "uat.compare"),
    ]
    reasons = []
    if rt_lag.lag_samples != 0 and abs(rt_lag.ncc) >= 0.7:
        reasons.append(f"real-time take is offset by {rt_lag.lag_samples} samples")
    if rt_null > match_dbfs:
        reasons.append(f"residual peak {_round_db(rt_null)} dBFS > {match_dbfs} dBFS")
    gates.append(Gate("rt_matches_offline", "FAIL" if reasons else "PASS",
                      measurement="rt_vs_offline.residual_peak_dbfs", threshold=f"<= {match_dbfs} dBFS at lag 0",
                      reason="; ".join(reasons)))

    # blocks the real-time take lost (silent where the offline render is active)
    drops = dropout_blocks(takes.realtime, takes.offline, block)
    ms.append(Measurement("dropout_blocks", len(drops), "count", 0.0, "exact", "uat.compare"))
    gates.append(Gate("no_dropout_blocks", "FAIL" if drops else "PASS", measurement="dropout_blocks",
                      threshold=f"0 blocks of {block} silent (< {SILENT_DBFS} dBFS) where offline >= {ACTIVE_DBFS} dBFS",
                      reason=f"first at {drops[0] * block / sr:.4f} s" if drops else ""))

    # blocks the real-time take passed through unprocessed (equal to the source
    # where the offline render differs from it)
    rt, src, off = (_blocks(_mono64(x), block) for x in (takes.realtime, takes.source, takes.offline))
    n = min(len(rt), len(src), len(off))
    as_source = np.max(np.abs(rt[:n] - src[:n]), axis=1) <= UNPROCESSED_TOL
    processed_here = np.max(np.abs(off[:n] - src[:n]), axis=1) > 10 * UNPROCESSED_TOL
    unprocessed = np.flatnonzero(as_source & processed_here)
    ms.append(Measurement("unprocessed_blocks", int(len(unprocessed)), "count", 0.0, "exact", "uat.compare"))
    gates.append(Gate("no_unprocessed_blocks", "FAIL" if len(unprocessed) else "PASS",
                      measurement="unprocessed_blocks", threshold="0",
                      reason=f"first at {unprocessed[0] * block / sr:.4f} s" if len(unprocessed) else ""))

    # offline render vs source: aligned (PDC compensated) and actually processed
    pdc = estimate_lag(takes.source, takes.offline, sr)
    ms += [
        Measurement("offline_vs_source.lag_samples", pdc.lag_samples, "samples", 0.0, "exact", "uat.compare"),
        Measurement("offline_vs_source.ncc", round(pdc.ncc, 6), "ratio", 0.5e-6, "rounding", "uat.compare"),
    ]
    if abs(pdc.ncc) < 0.7:
        status, reason = "FAIL", f"correlation {abs(pdc.ncc):.3f} < 0.7: alignment not measurable"
    elif pdc.lag_samples != 0:
        status, reason = "FAIL", f"offline render is {pdc.lag_samples:+d} samples from the source"
    else:
        status, reason = "PASS", ""
    gates.append(Gate("delay_compensated", status, measurement="offline_vs_source.lag_samples",
                      threshold="0 samples", reason=reason))

    if expect_processing:
        required.append("processed")
        rel = residual_peak_dbfs(takes.offline, takes.source) - _db(float(np.max(np.abs(_mono64(takes.source))) or 1e-30))
        ms.append(Measurement("offline_vs_source.residual_rel_db", _round_db(rel), "dB", 0.005, "rounding", "uat.compare"))
        gates.append(Gate("processed", "PASS" if rel > PROCESSED_MIN_DB else "FAIL",
                          measurement="offline_vs_source.residual_rel_db", threshold=f"> {PROCESSED_MIN_DB} dB",
                          reason="" if rel > PROCESSED_MIN_DB else "offline render is the source: nothing was processed"))

    if takes.bypassed is not None:
        required.append("bypass_equals_source")
        lag = estimate_lag(takes.source, takes.bypassed, sr, settle_sec=0.0)
        null = residual_peak_dbfs(takes.bypassed, takes.source)
        ms += [
            Measurement("bypass_vs_source.lag_samples", lag.lag_samples, "samples", 0.0, "exact", "uat.compare"),
            Measurement("bypass_vs_source.residual_peak_dbfs", _round_db(null), "dBFS", 0.005, "rounding", "uat.compare"),
        ]
        reasons = []
        if lag.lag_samples != 0:
            reasons.append(f"bypass delays the source by {lag.lag_samples} samples")
        if null > match_dbfs:
            reasons.append(f"bypass residual {_round_db(null)} dBFS > {match_dbfs} dBFS")
        gates.append(Gate("bypass_equals_source", "FAIL" if reasons else "PASS",
                          measurement="bypass_vs_source.residual_peak_dbfs", threshold=f"<= {match_dbfs} dBFS at lag 0",
                          reason="; ".join(reasons)))

    duration = len(_mono64(takes.source)) / sr
    if takes.render_seconds is not None:
        required.append("faster_than_real_time")
        ratio = takes.render_seconds / duration if duration else math.inf
        ms.append(Measurement("render_realtime_ratio", round(ratio, 4), "ratio", 0.5e-4, "rounding", "uat.compare"))
        gates.append(Gate("faster_than_real_time", "PASS" if ratio < 1.0 else "FAIL",
                          measurement="render_realtime_ratio", threshold="< 1.0"))

    subject = {"file": None, "sample_rate": sr, "channels": None, "duration_sec": round(duration, 6)}
    ms.insert(0, Measurement("duration_sec", subject["duration_sec"], "s", one_sample, "resolution", "subject"))
    return Report("uat compare", subject, ms, [], require(gates, required))


def smoke_check(
    source: np.ndarray,
    output: np.ndarray,
    sample_rate: int,
    *,
    n_parameters: int | None = None,
    label: str = "",
) -> list[Gate]:
    """Minimal 'does it work' gates for one render: shape, finite, audible after warm-up."""
    prefix = f"{label}." if label else ""
    src = np.asarray(source)
    out = np.asarray(output)
    gates = [Gate(f"{prefix}shape_matches", "PASS" if out.shape == src.shape else "FAIL",
                  threshold=list(src.shape), reason="" if out.shape == src.shape else f"got {list(out.shape)}")]
    finite = bool(np.all(np.isfinite(out)))
    gates.append(Gate(f"{prefix}finite", "PASS" if finite else "FAIL", reason="" if finite else "NaN/Inf in output"))
    half = _mono64(out)[len(_mono64(out)) // 2:]
    tail_db = _db(float(np.sqrt(np.mean(half * half)))) if finite and len(half) else -math.inf
    audible = tail_db > SILENT_DBFS
    gates.append(Gate(f"{prefix}second_half_audible", "PASS" if audible else "FAIL",
                      threshold=f"> {SILENT_DBFS} dBFS",
                      reason="" if audible else f"second half at {_round_db(tail_db)} dBFS"))
    if n_parameters is not None:
        gates.append(Gate(f"{prefix}parameters_exposed", "PASS" if n_parameters >= 1 else "FAIL",
                          threshold=">= 1", reason="" if n_parameters >= 1 else "no parameters exposed"))
    return gates
