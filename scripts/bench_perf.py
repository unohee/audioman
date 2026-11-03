#!/usr/bin/env python
"""Speed of the vectorised analysis paths against their per-window references.

Each case times the shipped code path and the same path with the previous
per-window implementations patched back in, on one deterministic synthetic
file, in one process. The assertion is the *ratio* between the two, so the
result does not depend on how fast the machine is; only on the algorithms.

    python scripts/bench_perf.py            # print the report
    python scripts/bench_perf.py --assert   # exit 1 if a ratio is below its floor
    python scripts/bench_perf.py --json     # machine-readable report

Numeric parity of the same paths is asserted by tests/unit/test_perf_parity.py;
this script only measures speed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import pyloudnorm
import soundfile as sf

from audioman.core import aesthetic, loudness, qc
from audioman.core.framewise import FrameFeatures

SR = 48000
DURATION_SEC = 300.0
FLOORS = {"master_qc": 5.0, "screen": 3.0}


# ---------------------------------------------------------------------------
# The synthetic programme: speech-like bursts, hiss, a mains hum, sparse clicks.
# ---------------------------------------------------------------------------


def make_programme(seconds: float = DURATION_SEC, sample_rate: int = SR) -> np.ndarray:
    rng = np.random.default_rng(20261103)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    voiced = 0.2 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0)
    hiss = 0.03 * rng.standard_normal(len(t)) * (np.sin(2 * np.pi * 1.3 * t) > 0.6)
    hum = 0.005 * np.sin(2 * np.pi * 60 * t)
    mono = voiced + hiss + hum
    mono[:: sample_rate * 7] += 0.8  # isolated clicks
    return np.stack([mono, 0.9 * mono + 0.01 * rng.standard_normal(len(t))]).astype(np.float32)


# ---------------------------------------------------------------------------
# References: the implementations the fast paths replaced.
# ---------------------------------------------------------------------------


def _reference_short_term(audio, sample_rate, window_sec=3.0, hop_sec=0.1):
    data = audio.T if audio.ndim == 2 else audio
    win, hop = int(window_sec * sample_rate), int(hop_sec * sample_rate)
    if data.shape[0] < win:
        return np.array([], dtype=np.float32)
    meter = pyloudnorm.Meter(sample_rate, block_size=window_sec)
    out = []
    for start in range(0, data.shape[0] - win + 1, hop):
        try:
            value = meter.integrated_loudness(data[start:start + win])
        except ValueError:
            value = float("-inf")
        out.append(value if np.isfinite(value) else float("-inf"))
    return np.array(out, dtype=np.float32)


def _reference_true_peak(audio, sample_rate, oversample=4):
    return loudness.true_peak_dbtp(np.asarray(audio, dtype=np.float64), sample_rate, oversample)


def _reference_measure(audio, sample_rate):
    """measure() as it was: K-weighting once for integrated, once per short-term window."""
    n_ch = 1 if audio.ndim == 1 else audio.shape[0]
    integrated = loudness.integrated_lufs(audio, sample_rate)
    st = _reference_short_term(audio, sample_rate)
    finite = st[np.isfinite(st)]
    return loudness.LoudnessReport(
        integrated_lufs=integrated,
        short_term_max=float(finite.max()) if len(finite) else float("-inf"),
        short_term_min=float(finite.min()) if len(finite) else float("-inf"),
        lra=loudness.loudness_range(st) if len(st) else 0.0,
        true_peak_dbtp=_reference_true_peak(audio, sample_rate),
        sample_peak_dbfs=loudness.sample_peak_dbfs(audio),
        duration_sec=audio.shape[-1] / sample_rate,
        sample_rate=sample_rate,
        channels=n_ch,
    )


def _reference_phase_correlation(audio, window_ms=100.0, sample_rate=48000):
    if audio.ndim != 2 or audio.shape[0] != 2:
        return {"applicable": False, "reason": "not stereo"}
    left, right = audio[0].astype(np.float64), audio[1].astype(np.float64)
    global_corr = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 0 and np.std(right) > 0 else 0.0
    win = max(int(window_ms / 1000.0 * sample_rate), 256)
    corrs = []
    for i in range(len(left) // win):
        wl, wr = left[i * win:(i + 1) * win], right[i * win:(i + 1) * win]
        if np.std(wl) < 1e-8 or np.std(wr) < 1e-8:
            continue
        corrs.append(float(np.corrcoef(wl, wr)[0, 1]))
    arr = np.array(corrs) if corrs else np.array([global_corr])
    return {
        "applicable": True,
        "global_correlation": round(global_corr, 4),
        "min_window_correlation": round(float(arr.min()), 4),
        "mean_window_correlation": round(float(arr.mean()), 4),
        "negative_correlation_pct": round(100.0 * float((arr < 0).sum()) / len(arr), 2),
        "n_windows": len(corrs),
    }


def _reference_frame_features(mono, bounds, sample_rate, *, bands=(), flatness=False, zcr=False):
    frames = [mono[s:e] for s, e in bounds]
    return FrameFeatures(
        rms=np.array([float(np.sqrt(np.mean(f**2))) for f in frames]),
        band_power=[np.array([aesthetic._band_power(f, sample_rate, b) for f in frames]) for b in bands],
        flatness=np.array([aesthetic._spectral_flatness(f) for f in frames]) if flatness else None,
        zcr=np.array([aesthetic._zero_crossing_rate(f) for f in frames]) if zcr else None,
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _best_of(fn, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def bench(repeats: int = 2) -> dict:
    audio = make_programme()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "programme.wav"
        sf.write(path, audio.T, SR, subtype="FLOAT")

        def master_qc():
            qc.evaluate_file(path)

        def master_qc_reference():
            with ExitStack() as stack:
                # qc calls `loudness_mod.measure`, i.e. the attribute on the loudness module.
                stack.enter_context(mock.patch.object(loudness, "measure", _reference_measure))
                stack.enter_context(mock.patch.object(qc, "stereo_phase_correlation", _reference_phase_correlation))
                qc.evaluate_file(path)

        def screen():
            aesthetic.screen_audio(audio, SR)

        def screen_reference():
            with mock.patch.object(aesthetic, "frame_features", _reference_frame_features):
                aesthetic.screen_audio(audio, SR)

        cases = {
            "master_qc": (master_qc, master_qc_reference),
            "screen": (screen, screen_reference),
        }
        report = {"duration_sec": DURATION_SEC, "sample_rate": SR, "channels": 2, "cases": {}}
        for name, (fast, reference) in cases.items():
            fast()  # warm caches and lazy imports outside the timed region
            t_fast = _best_of(fast, repeats)
            t_ref = _best_of(reference, 1)
            report["cases"][name] = {
                "fast_sec": round(t_fast, 3),
                "reference_sec": round(t_ref, 3),
                "speedup": round(t_ref / t_fast, 2),
                "floor": FLOORS[name],
            }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assert", dest="check", action="store_true", help="exit 1 if a speedup is below its floor")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--repeats", type=int, default=2, help="best-of repeats for the fast path")
    args = parser.parse_args(argv)

    report = bench(args.repeats)
    failures = [n for n, c in report["cases"].items() if c["speedup"] < c["floor"]]
    report["passed"] = not failures
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for name, case in report["cases"].items():
            mark = "ok  " if name not in failures else "FAIL"
            print(f"{mark} {name:10s} fast {case['fast_sec']:7.3f}s  reference {case['reference_sec']:7.3f}s"
                  f"  speedup {case['speedup']:6.2f}x  (floor {case['floor']}x)")
    return 1 if (args.check and failures) else 0


if __name__ == "__main__":
    sys.exit(main())
