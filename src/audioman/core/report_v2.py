# Created: 2026-11-04
# Purpose: report.v2 — one contract (subject, measurements with floors, findings, gates,
#          verdict) shared by analyze, observe, screen, master qc and doctor.
# Dependencies: stdlib only
#
# Each command keeps its own v1 payload; `build_report(command, payload)` derives a
# report.v2 block from it, which the CLI attaches as the top-level `report` key.
#
# Measurements carry a *floor*: the smallest difference in the value that means
# anything. An LLM comparing two reports must not read a change below the floor as
# a change. Floor kinds:
#   exact         counts and configuration values (floor 0)
#   rounding      the producer rounded the value; floor is half the last digit kept
#   resolution    bounded by a grid (sample period, FFT bin width)
#   quantization  bounded by the file's PCM word length
#   repeat        difference between two measurements of the same content
#                 (texture descriptors: analysis grid shifted by half a hop)
#   unknown       no defensible floor; `floor` is null
#
# Gates turn measurements into PASS/WARN/FAIL. Each command declares the gates it
# must produce; a declared gate the payload does not provide is added as FAIL with
# reason "missing" — a check that did not run is never a pass.

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

REPORT_VERSION = "2"
REPORT_SCHEMA = "audioman://schema/report.v2.json"
GATE_STATUSES = ("PASS", "WARN", "FAIL")
FLOOR_KINDS = ("exact", "rounding", "resolution", "quantization", "repeat", "unknown")
_PCM_BITS = {"PCM_U8": 8, "PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32}


@dataclass
class Measurement:
    name: str
    value: float | int | str | bool | None
    unit: str
    floor: float | None
    floor_kind: str
    source: str


@dataclass
class Gate:
    name: str
    status: str
    measurement: str | None = None
    threshold: Any = None
    required: bool = True
    reason: str = ""


@dataclass
class Report:
    command: str
    subject: dict[str, Any]
    measurements: list[Measurement] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        statuses = {g.status for g in self.gates}
        verdict = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
        return {
            "schema": REPORT_SCHEMA,
            "version": REPORT_VERSION,
            "command": self.command,
            "subject": self.subject,
            "measurements": [asdict(m) for m in self.measurements],
            "findings": self.findings,
            "gates": [_gate_dict(g) for g in self.gates],
            "missing_gates": [g.name for g in self.gates if g.reason == "missing"],
            "verdict": verdict,
        }


def _gate_dict(gate: Gate) -> dict[str, Any]:
    """Gate as JSON, omitting fields at their default (no measurement/threshold,
    no reason, required) — a report is read through a context window."""
    out: dict[str, Any] = {"name": gate.name, "status": gate.status}
    if gate.measurement is not None:
        out["measurement"] = gate.measurement
    if gate.threshold is not None:
        out["threshold"] = gate.threshold
    if not gate.required:
        out["required"] = False
    if gate.reason:
        out["reason"] = gate.reason
    return out


# ---------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------


def rounding_floor(value: Any) -> tuple[float | None, str]:
    """Floor implied by how many decimals a producer kept (half a unit in the last place).

    Integers are counts or settings (exact). Floats with more than six decimals
    were not rounded by the producer, so their precision says nothing: unknown.
    """
    if isinstance(value, bool) or isinstance(value, str) or value is None:
        return 0.0, "exact"
    if isinstance(value, int):
        return 0.0, "exact"
    if not math.isfinite(value):
        return None, "unknown"
    text = repr(float(value))
    if "e" in text or "E" in text:
        return None, "unknown"
    decimals = len(text.split(".")[1].rstrip("0")) if "." in text else 0
    if decimals > 6:
        return None, "unknown"
    return 0.5 * 10.0 ** (-decimals), "rounding"


def _measure(name: str, value: Any, unit: str, source: str) -> Measurement:
    floor, kind = rounding_floor(value)
    return Measurement(name, value, unit, floor, kind, source)


def _level_measure(name: str, value: Any, bits: int | None, source: str) -> Measurement:
    """Linear sample level: a PCM word length bounds it at one LSB, if coarser than rounding."""
    m = _measure(name, value, "linear", source)
    if bits:
        lsb = 2.0 ** -(bits - 1)
        if m.floor is None or lsb > m.floor:
            m.floor, m.floor_kind = lsb, "quantization"
    return m


def _subject(payload: dict, *, file_key: str = "file") -> tuple[dict[str, Any], list[Measurement]]:
    sr = payload.get("sample_rate")
    duration = payload.get("duration_sec", payload.get("duration"))
    subject = {
        "file": payload.get(file_key),
        "sample_rate": sr if isinstance(sr, int) else None,
        "channels": payload.get("channels") if isinstance(payload.get("channels"), int) else None,
        "duration_sec": duration if isinstance(duration, (int, float)) else None,
    }
    out = []
    if subject["duration_sec"] is not None:
        resolution = 1.0 / subject["sample_rate"] if subject["sample_rate"] else None
        out.append(Measurement(
            "duration_sec", subject["duration_sec"], "s",
            resolution, "resolution" if resolution else "unknown", "subject",
        ))
    if subject["sample_rate"] is not None:
        out.append(Measurement("sample_rate", subject["sample_rate"], "Hz", 0.0, "exact", "subject"))
    if subject["channels"] is not None:
        out.append(Measurement("channels", subject["channels"], "count", 0.0, "exact", "subject"))
    return subject, out


def _pcm_bits(subtype: str | None) -> int | None:
    return _PCM_BITS.get(subtype or "")


def _file_subtype(path: Any) -> str | None:
    if not path:
        return None
    try:
        import soundfile as sf

        return sf.info(str(path)).subtype
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _severity_gate(name: str, items: list[dict], *, fail: tuple[str, ...], warn: tuple[str, ...], what: str) -> Gate:
    severities = [str(i.get("severity", "")).lower() for i in items]
    n_fail = sum(s in fail for s in severities)
    n_warn = sum(s in warn for s in severities)
    if n_fail:
        return Gate(name, "FAIL", threshold=f"no {what} rated {'/'.join(fail)}", reason=f"{n_fail} {what} rated {'/'.join(fail)}")
    if n_warn:
        return Gate(name, "WARN", threshold=f"no {what} rated {'/'.join(fail)}", reason=f"{n_warn} {what} rated {'/'.join(warn)}")
    return Gate(name, "PASS", threshold=f"no {what} rated {'/'.join(fail)}")


def require(gates: list[Gate], required: list[str]) -> list[Gate]:
    """Append a FAIL gate for every required name the producer did not emit."""
    present = {g.name for g in gates}
    return gates + [Gate(name, "FAIL", reason="missing") for name in required if name not in present]


# ---------------------------------------------------------------------------
# Adapters (v1 payload -> report.v2)
# ---------------------------------------------------------------------------


def _report_analyze(p: dict) -> Report:
    subject, ms = _subject(p)
    bits = _pcm_bits(_file_subtype(subject["file"]))
    for key in ("rms", "peak"):
        if isinstance(p.get(key), (int, float)):
            ms.append(_level_measure(key, p[key], bits, "analyze"))
    units = {"rms": "linear", "peak": "linear", "spectral_centroid": "Hz", "spectral_entropy": "bits",
             "zero_crossing_rate": "ratio"}
    for metric, stats in (p.get("summary") or {}).items():
        for stat, value in (stats or {}).items():
            if isinstance(value, (int, float)):
                ms.append(_measure(f"{metric}.{stat}", value, units.get(metric, ""), "analyze.summary"))
    if isinstance(p.get("silence_total_sec"), (int, float)):
        ms.append(_measure("silence_total_sec", p["silence_total_sec"], "s", "analyze"))
    if isinstance(p.get("texture"), dict):
        from audioman.core.texture import to_measurements

        ms.extend(to_measurements(p["texture"]))
    findings = list(p.get("findings") or [])
    gates = [_severity_gate("no_critical_findings", findings, fail=("critical",), warn=("warn",), what="findings")]
    return Report("analyze", subject, ms, findings, require(gates, ["no_critical_findings"]))


def _report_observe(p: dict) -> Report:
    subject, ms = _subject(p)
    findings = list(p.get("findings") or [])
    counts = (p.get("summary") or {}).get("by_severity") or {}
    for severity, n in counts.items():
        ms.append(Measurement(f"findings.{severity}", n, "count", 0.0, "exact", "observe.summary"))
    gates = [_severity_gate("no_critical_findings", findings, fail=("critical",), warn=("warn",), what="findings")]
    return Report("observe", subject, ms, findings, require(gates, ["no_critical_findings"]))


def _report_screen(p: dict) -> Report:
    if "files" in p:  # batch: one report over every file's events
        subject = {"file": p.get("input"), "sample_rate": None, "channels": None, "duration_sec": None}
        ms, events = [], []
        for report in p.get("files") or []:
            file_events = list(report.get("events") or [])
            events.extend(file_events)
            ms.append(Measurement(f"events@{report.get('file')}", len(file_events), "count", 0.0, "exact", "screen.files"))
        gates = [_severity_gate("no_fail_events", events, fail=("fail",), warn=("warn",), what="events")]
        return Report("screen", subject, ms, [], require(gates, ["no_fail_events"]))
    subject, ms = _subject(p)
    events = list(p.get("events") or [])
    for issue, n in (p.get("summary") or {}).items():
        if isinstance(n, int):
            ms.append(Measurement(f"events.{issue}", n, "count", 0.0, "exact", "screen.summary"))
    for event in events:
        confidence = event.get("confidence")
        if isinstance(confidence, (int, float)):
            ms.append(_measure(f"event.{event.get('type')}@{event.get('start_sec')}.confidence",
                               confidence, "ratio", "screen.events"))
    gates = [_severity_gate("no_fail_events", events, fail=("fail",), warn=("warn",), what="events")]
    return Report("screen", subject, ms, [], require(gates, ["no_fail_events"]))


# Checks qc.evaluate always produces; stereo adds the two inter-channel checks.
QC_REQUIRED = ["integrated_lufs", "true_peak_dbtp", "clipping_samples", "dc_offset", "clicks",
               "head_silence_ms", "tail_silence_sec"]
QC_REQUIRED_STEREO = ["phase_correlation", "channel_imbalance_db"]
_QC_UNITS = {"integrated_lufs": "LUFS", "true_peak_dbtp": "dBTP", "loudness_range_lu": "LU",
             "clipping_samples": "count", "dc_offset": "linear", "clicks": "count",
             "phase_correlation": "correlation", "channel_imbalance_db": "dB",
             "head_silence_ms": "ms", "tail_silence_sec": "s", "sample_rate": "Hz", "bit_depth": "bits"}


def _report_master_qc(p: dict) -> Report:
    fmt = p.get("format") or {}
    subject, ms = _subject({**fmt, "file": p.get("input"), "duration_sec": fmt.get("duration_sec")})
    for name, value in (p.get("loudness") or {}).items():
        if isinstance(value, (int, float)) and name not in ("sample_rate", "channels", "duration_sec"):
            ms.append(_measure(f"loudness.{name}", value, _QC_UNITS.get(name, ""), "master_qc.loudness"))
    gates = []
    for check in p.get("checks") or []:
        name = check.get("name")
        value = check.get("value")
        if isinstance(value, (int, float)):
            ms.append(_measure(f"check.{name}", value, _QC_UNITS.get(name, ""), "master_qc.checks"))
        status = str(check.get("status", "FAIL")).upper()
        gates.append(Gate(
            name, status if status in GATE_STATUSES else "FAIL",
            measurement=f"check.{name}" if isinstance(value, (int, float)) else None,
            threshold=check.get("target"),
        ))
    required = QC_REQUIRED + (QC_REQUIRED_STEREO if subject["channels"] == 2 else [])
    return Report("master qc", subject, ms, [], require(gates, required))


_DOCTOR_RESULT_KEYS = {"attack-release": "attack_release"}
_DOCTOR_ALL = ["linear", "thd", "imd", "sweep", "dynamics", "attack-release", "waveshaper", "performance"]


def _report_doctor(p: dict) -> Report:
    subject = {"file": p.get("plugin"), "sample_rate": None, "channels": None, "duration_sec": None}
    requested = _DOCTOR_ALL if p.get("mode") == "all" else [p.get("mode")]
    ms: list[Measurement] = []
    gates: list[Gate] = []
    for mode in [m for m in requested if m]:
        key = _DOCTOR_RESULT_KEYS.get(mode, mode)
        block = p.get(key)
        if not isinstance(block, dict):
            continue  # require() below records it as missing
        if "error" in block:
            gates.append(Gate(f"completed.{mode}", "FAIL", reason=str(block["error"])))
            continue
        gates.append(Gate(f"completed.{mode}", "PASS"))
        for name, value in block.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                ms.append(_measure(f"{key}.{name}", value, "", f"doctor.{key}"))
    required = [f"completed.{m}" for m in requested if m]
    return Report("doctor", subject, ms, [], require(gates, required))


def _report_diff(p: dict) -> Report:
    align = p.get("alignment") or {}
    sr = p.get("sample_rate")
    subject = {"file": p.get("b"), "sample_rate": sr if isinstance(sr, int) else None, "channels": None,
               "duration_sec": align.get("compared_sec")}
    ms: list[Measurement] = []
    for name, d in (p.get("descriptors") or {}).items():
        if d.get("delta") is not None:
            ms.append(Measurement(f"delta.{name}", d["delta"], d.get("unit", ""), d.get("floor"),
                                  "repeat" if d.get("floor") is not None else "unknown", "diff"))
    for band in p.get("named_bands") or []:
        ms.append(Measurement(f"delta.band.{band['band']}", band["delta_db"], "dB", band["floor_db"], "repeat", "diff"))
    if p.get("residual_rel_db") is not None:
        ms.append(_measure("residual_rel_db", p["residual_rel_db"], "dB", "diff"))
    if align.get("applied"):
        gate = Gate("alignment", "PASS", threshold="lag established by cross-correlation")
    else:
        gate = Gate("alignment", "WARN", threshold="lag established by cross-correlation",
                    reason=f"compared unaligned: {align.get('reason') or 'unknown'}")
    return Report("diff", subject, ms, [], require([gate], ["alignment"]))


ADAPTERS: dict[str, Callable[[dict], Report]] = {
    "analyze": _report_analyze,
    "observe": _report_observe,
    "screen": _report_screen,
    "master qc": _report_master_qc,
    "doctor": _report_doctor,
    "diff": _report_diff,
}


def build_report(command: str, payload: dict) -> dict[str, Any]:
    """report.v2 dict for a command's v1 payload. Error payloads yield a FAIL report."""
    if "error" in payload:
        subject = {"file": payload.get("file"), "sample_rate": None, "channels": None, "duration_sec": None}
        return Report(command, subject, gates=[Gate("completed", "FAIL", reason=str(payload["error"]))]).to_dict()
    return ADAPTERS[command](payload).to_dict()
