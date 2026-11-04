# Created: 2026-11-04
# Purpose: One-paragraph plain-English digests of analysis payloads, for LLM readers.
# Dependencies: stdlib only
#
# A digest is computed from the *full* payload before any detail shaping, so it
# always describes everything that was measured, even when the arrays behind it
# were spilled to files. It states measurements and counts; it does not judge
# quality. Every digest stays under DIGEST_MAX_CHARS.

from __future__ import annotations

import math
from typing import Any, Callable

DIGEST_MAX_CHARS = 600


def _db(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v <= 0 or not math.isfinite(v):
        return "-inf dBFS"
    return f"{20.0 * math.log10(v):.1f} dBFS"


def _duration(seconds: Any) -> str:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "unknown length"
    minutes, rest = divmod(s, 60.0)
    return f"{int(minutes)}:{rest:04.1f}" if minutes else f"{rest:.1f} s"


def _format(payload: dict) -> str:
    channels = payload.get("channels")
    layout = {1: "mono", 2: "stereo"}.get(channels, f"{channels} ch" if channels else "")
    sr = payload.get("sample_rate")
    rate = f"{sr / 1000:g} kHz" if isinstance(sr, (int, float)) and sr else ""
    dur = payload.get("duration_sec", payload.get("duration"))
    return ", ".join(p for p in (_duration(dur), layout, rate) if p)


def _counts(items: list[dict], field: str) -> str:
    counts: dict[str, int] = {}
    for item in items:
        counts[str(item.get(field, "?"))] = counts.get(str(item.get(field, "?")), 0) + 1
    return ", ".join(f"{n} {k}" for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _findings_sentence(findings: list[dict]) -> str:
    if not findings:
        return "No findings."
    codes = sorted({str(f.get("code", "?")) for f in findings})
    shown = ", ".join(codes[:6]) + (", ..." if len(codes) > 6 else "")
    return f"{len(findings)} findings ({_counts(findings, 'severity')}): {shown}."


def _clip(text: str) -> str:
    return text if len(text) <= DIGEST_MAX_CHARS else text[: DIGEST_MAX_CHARS - 3] + "..."


def digest_analyze(p: dict) -> str:
    parts = [f"{_format(p)}. RMS {_db(p.get('rms'))}, peak {_db(p.get('peak'))}."]
    centroid = (p.get("summary") or {}).get("spectral_centroid", {}).get("mean")
    if isinstance(centroid, (int, float)):
        parts.append(f"Mean frame spectral centroid {centroid:.0f} Hz (magnitude-weighted; texture descriptors use power).")
    regions = p.get("silence_regions") or []
    if regions:
        parts.append(f"{len(regions)} silent regions, {p.get('silence_total_sec', 0)} s total.")
    spectrum = p.get("spectrum") or {}
    hums = [h for h in spectrum.get("hum_check", []) if isinstance(h, dict) and h.get("is_hum")]
    if hums:
        parts.append("Mains hum at " + ", ".join(f"{h.get('frequency_hz')} Hz" for h in hums[:3]) + ".")
    parts.append(_findings_sentence(p.get("findings") or []))
    return _clip(" ".join(parts))


def digest_observe(p: dict) -> str:
    return _clip(f"{_format(p)}. {_findings_sentence(p.get('findings') or [])}")


def digest_screen(p: dict) -> str:
    if "files" in p:  # batch
        reports = p.get("files") or []
        n_events = sum(len(r.get("events") or []) for r in reports)
        return _clip(f"{len(reports)} files screened, {n_events} events in total.")
    events = p.get("events") or []
    head = f"{_format(p)}. Screened for {', '.join(p.get('issues') or [])}."
    if not events:
        return _clip(f"{head} No events.")
    return _clip(f"{head} {len(events)} events ({_counts(events, 'type')}); severities: {_counts(events, 'severity')}.")


def digest_master_qc(p: dict) -> str:
    checks = p.get("checks") or []
    failing = [c for c in checks if c.get("status") in ("FAIL", "WARN")]
    head = f"QC against {p.get('target', '?')}: {p.get('verdict', '?')}."
    loud = p.get("loudness") or {}
    lufs = loud.get("integrated_lufs")
    tp = loud.get("true_peak_dbtp")
    level = ""
    if lufs is not None or tp is not None:
        level = f" Integrated {lufs} LUFS, true peak {tp} dBTP."
    if not failing:
        return _clip(f"{head}{level} All {len(checks)} checks pass.")
    named = ", ".join(f"{c.get('name')} {c.get('status')} ({c.get('value')})" for c in failing[:5])
    return _clip(f"{head}{level} {len(failing)} of {len(checks)} checks not passing: {named}.")


def digest_report(report: dict) -> str:
    """Digest of a report.v2 block: verdict, then the gates that did not pass and why."""
    gates = report.get("gates") or []
    bad = [g for g in gates if g.get("status") != "PASS"]
    head = f"{report.get('command', '?')}: {report.get('verdict', '?')}."
    if not bad:
        return _clip(f"{head} All {len(gates)} gates pass.")
    named = "; ".join(f"{g.get('name')} {g.get('status')}" + (f" ({g['reason']})" if g.get("reason") else "")
                      for g in bad[:5])
    return _clip(f"{head} {len(bad)} of {len(gates)} gates not passing: {named}.")


def digest_diff(p: dict) -> str:
    align = p.get("alignment") or {}
    if align.get("applied"):
        how = f"aligned by {align.get('lag_samples', 0):+d} samples" + (
            " from the envelopes" if align.get("method") == "envelope" else "")
    else:
        how = f"not aligned ({align.get('reason', 'n/a')})"
    changes = p.get("changes") or []
    bands = [b for b in p.get("named_bands") or [] if b.get("significant")]
    if not changes and not bands:
        return _clip(f"B vs A ({how}): no significant change.")
    terms = (p.get("vocabulary") or {}).get("terms") or []
    vocab = ""
    if terms:
        vocab = " Terms (experimental): " + ", ".join(f"{t['term']} ({t['confidence']:.2f})" for t in terms[:8]) + "."
    parts = [f"{c['descriptor']} {c['direction']} {abs(c['delta'])} {c['unit']}"
             + (f" (floor {c['floor']})" if c.get("floor") is not None else "") for c in changes[:6]]
    text = f"B vs A ({how}): {len(changes)} significant descriptor change(s): {', '.join(parts)}."
    if bands:
        text += " Bands: " + ", ".join(f"{b['band']} {b['delta_db']:+.1f} dB" for b in bands) + "."
    # the vocabulary goes first so it survives the length clip
    return _clip(f"B vs A ({how}):{vocab} " + text.split("): ", 1)[1]) if vocab else _clip(text)


_TEXTURE_LEAD = ("spectral_centroid", "spectral_rolloff", "spectral_flatness", "hf_ratio", "sharpness",
                 "crest_factor", "onset_density", "stereo_width", "lr_correlation", "pitch", "pitch_deviation")


def digest_texture(p: dict) -> str:
    """Key texture descriptors with their repeat floors, for describe()."""
    d = p.get("descriptors") or {}
    parts = []
    for name in _TEXTURE_LEAD:
        entry = d.get(name) or {}
        if entry.get("value") is None:
            continue
        floor = entry.get("floor")
        parts.append(f"{name} {entry['value']:g} {entry.get('unit', '')}".rstrip()
                     + (f" (+/-{floor:g})" if floor is not None else ""))
    head = f"{_format(p)}, {len(p.get('segments') or [])} segments."
    return _clip(f"{head} {'; '.join(parts)}." if parts else f"{head} No defined descriptors (silent input).")


DIGESTS: dict[str, Callable[[dict], str]] = {
    "analyze": digest_analyze,
    "observe": digest_observe,
    "screen": digest_screen,
    "master qc": digest_master_qc,
    "diff": digest_diff,
}


def build_digest(command: str, payload: dict) -> str:
    """Digest for `command`'s payload; payloads that report an error say so."""
    if "error" in payload:
        return _clip(f"{command} failed: {payload['error']}")
    return DIGESTS[command](payload)
