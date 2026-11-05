#!/usr/bin/env python
"""Texture-QA mini bench: does audioman tell an agent what changed, within a token budget?

Each case is a pair (A, B) where B is A after a known process (lowpass, added noise,
widening, gain, compression, ...). The pair goes through the same path an agent
uses, the MCP `diff` tool, and the case is scored on what comes back:

  direction  every expected term (e.g. "darker") is proposed and its opposite
             ("brighter") is not; control pairs (identical, delayed copy) must
             report no significant change at all
  budget     digest + structured summary fit in TOKEN_BUDGET tokens
             (estimated as UTF-8 bytes / 4)

    python scripts/texture_qa.py            # print the table
    python scripts/texture_qa.py --assert   # exit 1 below 90% direction accuracy or over budget
    python scripts/texture_qa.py --json     # machine-readable report

The material is synthetic and the terms are audioman's experimental vocabulary, so
this measures whether the tool's evidence points the right way, not whether a
listener would use the same word.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from scipy import signal as ss

SR = 48000
SECONDS = 4.0
TOKEN_BUDGET = 1500
MIN_ACCURACY = 0.90


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


def _pink(seed: int, n: int) -> np.ndarray:
    b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
    x = ss.lfilter(b, a, np.random.default_rng(seed).standard_normal(n))
    return x / np.max(np.abs(x))


def material(name: str) -> np.ndarray:
    n = int(SR * SECONDS)
    t = np.arange(n) / SR
    rng = np.random.default_rng(11)
    if name == "pink":
        return 0.4 * np.stack([_pink(0, n), _pink(1, n)])
    if name == "speechlike":
        voiced = sum(np.sin(2 * np.pi * f0 * k * t) / k for f0 in (140.0,) for k in range(1, 12))
        envelope = (np.sin(2 * np.pi * 3.0 * t) > 0) * 0.5 * (1 + np.sin(2 * np.pi * 0.5 * t))
        hiss = 0.04 * rng.standard_normal(n) * (np.sin(2 * np.pi * 1.7 * t) > 0.7)
        mono = 0.12 * voiced * envelope + hiss
        return np.stack([mono, 0.95 * mono])
    if name in ("sung", "sung_corrected"):
        # one sung note per second, A3..E4; the raw take is sharp by 25 cents with
        # a 5.5 Hz +/-40-cent vibrato, the corrected take sits on the semitone
        notes = np.array([-12.0, -10.0, -8.0, -5.0])  # semitones re A4
        semis = notes[np.minimum((t / (SECONDS / len(notes))).astype(int), len(notes) - 1)]
        cents = 100.0 * semis
        if name == "sung":
            cents = cents + 25.0 + 40.0 * np.sin(2 * np.pi * 5.5 * t)
        phase = 2 * np.pi * np.cumsum(440.0 * 2 ** (cents / 1200.0)) / SR
        voiced = sum(np.sin(k * phase) / k for k in range(1, 9))
        envelope = np.minimum(1.0, 20 * (t % 1.0)) * np.minimum(1.0, 20 * (1.0 - t % 1.0))
        mono = 0.15 * voiced * envelope
        return np.stack([mono, mono])
    if name == "chords":
        tones = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6, 440.0))
        decay = np.exp(-((t % 0.5) / 0.25))
        mono = 0.12 * tones * decay
        side = 0.03 * np.sin(2 * np.pi * 660.0 * t) * decay
        return np.stack([mono + side, mono - side])
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Processing with a known effect
# ---------------------------------------------------------------------------


def lowpass(x, cutoff_hz=2500.0):
    return ss.sosfilt(ss.butter(6, cutoff_hz, fs=SR, output="sos"), x, axis=-1)


def lowpass_300(x):
    return lowpass(x, 300.0)


def high_shelf_300(x):
    high = ss.sosfilt(ss.butter(2, 300, btype="high", fs=SR, output="sos"), x, axis=-1)
    return x + 1.8 * high


def pitch_correct(x):
    # the "plug-in" here is resynthesis on the semitone grid; the bench only needs a
    # pair whose difference is known
    return material("sung_corrected")


def pitch_up_semitone(x):
    # resample 17 -> 18: every frequency x 18/17 (+99 cents); trim/pad to the length
    y = ss.resample_poly(x, 17, 18, axis=-1)
    out = np.zeros_like(x)
    out[:, : min(y.shape[1], x.shape[1])] = y[:, : x.shape[1]]
    return out


def high_shelf(x):
    high = ss.sosfilt(ss.butter(2, 4000, btype="high", fs=SR, output="sos"), x, axis=-1)
    return x + 1.8 * high


def add_noise(x):
    level = np.sqrt(np.mean(x * x))
    return x + 0.3 * level * np.random.default_rng(5).standard_normal(x.shape)


def saturate(x):
    drive = 4.0 / np.max(np.abs(x))
    return np.tanh(drive * x) / np.tanh(drive) * np.max(np.abs(x))


def gain_down(x):
    return x * 10 ** (-6 / 20)


def gain_up(x):
    return x * 10 ** (4 / 20)


def widen(x):
    mid, side = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2
    extra = 0.6 * np.sqrt(np.mean(mid * mid)) * _pink(9, x.shape[1])
    return np.stack([mid + side + extra, mid - side - extra])


def to_mono(x):
    mid = x.mean(axis=0)
    return np.stack([mid, mid])


def compress(x):
    from pedalboard import Compressor, Pedalboard

    return Pedalboard([Compressor(threshold_db=-30, ratio=10, attack_ms=1, release_ms=50)])(
        x.astype(np.float32), SR).astype(np.float64)


def add_transients(x):
    clicks = np.zeros(x.shape[1])
    clicks[:: SR // 10] = 1.0
    burst = np.exp(-np.arange(300) / 30.0) * np.random.default_rng(3).standard_normal(300)
    hits = np.convolve(clicks, burst)[: x.shape[1]] * 0.4 * np.max(np.abs(x))
    return x + hits


def identity(x):
    return x.copy()


def delayed(x):
    return np.concatenate([np.zeros((x.shape[0], 333)), x[:, :-333]], axis=1)


@dataclass
class Case:
    material: str
    process: str
    fn: Callable[[np.ndarray], np.ndarray]
    expect: tuple[str, ...]  # terms that must be proposed
    forbid: tuple[str, ...] = ()  # terms that must not be
    control: bool = False  # expects no significant change at all
    result: dict = field(default_factory=dict)


def cases() -> list[Case]:
    """Expectations follow from what each process physically does to that material.

    - A lowpass only darkens material that has energy above its cutoff: the
      chords (sines <= 440 Hz) are filtered at 300 Hz, broadband material at 2.5 kHz.
    - Compression changes dynamics only where there are dynamics: stationary pink
      noise is not compressed in any perceptual sense, so it is not a case.
    - Saturation always lowers the crest of what it clips ("more compressed"); it
      only reads as "harmonically richer" on material that was nearly pure tones.
    - Likewise a high shelf only brightens what has energy above its corner: the
      chords get theirs at 300 Hz, broadband material at 4 kHz.
    - Decorrelated widening cannot widen material whose channels are already
      uncorrelated (the pink pair: width ~0 dB, L/R correlation ~0), so it is a
      case only for speechlike and chords.
    - Pitch terms apply to material with one f0 (the sung notes); every other case
      forbids them, so a spurious "pitched up" on a level change is a miss.
    """
    pitch_terms = ("pitch-corrected", "looser pitch", "pitched up", "pitched down")
    out = []
    for mat in ("pink", "speechlike", "chords"):
        shelf = high_shelf_300 if mat == "chords" else high_shelf
        out += [
            Case(mat, "high shelf +" + (" 300 Hz" if mat == "chords" else ""), shelf, ("brighter",), ("darker",)),
            Case(mat, "gain -6 dB", gain_down, ("quieter",), ("louder", "darker", "brighter")),
            Case(mat, "gain +4 dB", gain_up, ("louder",), ("quieter", "darker", "brighter")),
            Case(mat, "fold to mono", to_mono, ("narrower",), ("wider",)),
            *([Case(mat, "decorrelated widening", widen, ("wider",), ("narrower",))] if mat != "pink" else []),
            Case(mat, "identical copy", identity, (), control=True),
            Case(mat, "333-sample delay", delayed, (), control=True),
        ]
    out += [
        Case("pink", "lowpass 2.5 kHz", lowpass, ("darker",), ("brighter",)),
        Case("speechlike", "lowpass 2.5 kHz", lowpass, ("darker",), ("brighter",)),
        Case("chords", "lowpass 300 Hz", lowpass_300, ("darker",), ("brighter",)),
    ]
    for mat in ("speechlike", "chords"):  # material with dynamics and tonal structure
        out += [
            Case(mat, "compression", compress, ("more compressed",), ("more dynamic",)),
            Case(mat, "added noise", add_noise, ("noisier",), ("cleaner",)),
            Case(mat, "added transients", add_transients, ("more transient",), ("smoother",)),
        ]
    out += [
        Case("speechlike", "saturation", saturate, ("more compressed",), ("more dynamic",)),
        Case("chords", "saturation", saturate, ("more compressed", "harmonically richer"), ("more dynamic", "purer")),
    ]
    for case in out:
        case.forbid = tuple(case.forbid) + pitch_terms
    out += [
        Case("sung", "pitch correction", pitch_correct, ("pitch-corrected",), ("looser pitch",)),
        Case("sung", "pitch up a semitone", pitch_up_semitone, ("pitched up",), ("pitched down",)),
        Case("sung", "gain -6 dB", gain_down, ("quieter",), pitch_terms),
    ]
    return out


# ---------------------------------------------------------------------------
# Scoring through the MCP tool
# ---------------------------------------------------------------------------


def _write(path: Path, audio: np.ndarray) -> str:
    peak = np.max(np.abs(audio))
    if peak > 0.999:
        raise ValueError(f"{path.name} clips ({peak:.3f}); lower the material level")
    sf.write(path, audio.T.astype(np.float32), SR, subtype="FLOAT")
    return str(path)


def score(case: Case, workdir: Path) -> None:
    from audioman.cli.mcp import McpServer

    a = material(case.material)
    b = case.fn(a)
    stem = f"{case.material}-{case.process}".replace(" ", "_").replace("/", "_")
    pa, pb = _write(workdir / f"{stem}-a.wav", a), _write(workdir / f"{stem}-b.wav", b)
    server = McpServer()
    server.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "diff", "arguments": {"a": pa, "b": pb}}})
    result = response["result"]
    digest = result["content"][0]["text"]
    visible = digest + json.dumps(result.get("structuredContent", {}), ensure_ascii=False)
    tokens = len(visible.encode("utf-8")) // 4
    full = json.loads(Path(result["structuredContent"]["result_ref"]).read_text())
    terms = {t["term"] for t in full["vocabulary"]["terms"]}
    no_change = digest.endswith("no significant change.")
    if case.control:
        ok = no_change
        why = "" if ok else f"reported a change: {sorted(terms)}"
    else:
        missing = [t for t in case.expect if t not in terms]
        wrong = [t for t in case.forbid if t in terms]
        in_digest = all(t in digest for t in case.expect)
        ok = not missing and not wrong and in_digest
        why = "; ".join(filter(None, [
            f"missing {missing}" if missing else "",
            f"contradicting {wrong}" if wrong else "",
            "" if in_digest else "expected term not in the digest",
        ]))
    case.result = {"material": case.material, "process": case.process, "expect": list(case.expect),
                   "terms": sorted(terms), "correct": ok, "tokens": tokens,
                   "within_budget": tokens <= TOKEN_BUDGET, "why": why}


def run() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ.setdefault("AUDIOMAN_CACHE_DIR", str(Path(tmp) / "cache"))
        all_cases = cases()
        for case in all_cases:
            score(case, Path(tmp))
    rows = [c.result for c in all_cases]
    accuracy = sum(r["correct"] for r in rows) / len(rows)
    return {
        "cases": rows,
        "n_cases": len(rows),
        "direction_accuracy": round(accuracy, 4),
        "min_accuracy": MIN_ACCURACY,
        "token_budget": TOKEN_BUDGET,
        "max_tokens": max(r["tokens"] for r in rows),
        "over_budget": [f"{r['material']}/{r['process']}" for r in rows if not r["within_budget"]],
        "passed": accuracy >= MIN_ACCURACY and all(r["within_budget"] for r in rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assert", dest="check", action="store_true",
                        help=f"exit 1 below {MIN_ACCURACY:.0%} accuracy or over the token budget")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    report = run()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for r in report["cases"]:
            mark = "ok  " if r["correct"] else "MISS"
            print(f"{mark} {r['material']:10s} {r['process']:24s} {r['tokens']:5d} tok  "
                  f"terms={','.join(r['terms']) or '-'}" + (f"  ({r['why']})" if r["why"] else ""))
        print(f"\ndirection accuracy {report['direction_accuracy']:.1%} over {report['n_cases']} cases "
              f"(min {MIN_ACCURACY:.0%}); max {report['max_tokens']} tokens (budget {TOKEN_BUDGET})")
    return 1 if args.check and not report["passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
