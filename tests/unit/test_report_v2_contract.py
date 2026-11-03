"""report.v2: every analysis command emits it, it validates, floors are present, missing gates fail."""

from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audioman.core import report_v2
from audioman.core.report_v2 import QC_REQUIRED, QC_REQUIRED_STEREO, build_report, rounding_floor
from test_json_contract import _resolve_ref, validate_against

SR = 48000
REPORT_SCHEMA = _resolve_ref("audioman://schema/report.v2.json")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@dataclass
class _Result:
    code: int
    stdout: str
    stderr: str


def _run(argv: list[str]) -> _Result:
    from audioman.cli import app

    plain_before = os.environ.get("AUDIOMAN_PLAIN")
    out, err = io.StringIO(), io.StringIO()
    real = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        app.main(argv)
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdout, sys.stderr = real
        if plain_before is None:
            os.environ.pop("AUDIOMAN_PLAIN", None)
        else:
            os.environ["AUDIOMAN_PLAIN"] = plain_before
    return _Result(code, out.getvalue(), err.getvalue())


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    from audioman.config import settings as settings_module

    monkeypatch.setenv("AUDIOMAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_module.reset_settings()
    yield
    settings_module.reset_settings()


def assert_valid_report(report: dict, command: str) -> None:
    validate_against(report, REPORT_SCHEMA, where="report")
    assert report["command"] == command
    statuses = {g["status"] for g in report["gates"]}
    expected = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
    assert report["verdict"] == expected
    assert report["gates"], "every report has at least one gate"
    for m in report["measurements"]:
        # a floor is null only when it is declared unknown
        assert (m["floor"] is None) == (m["floor_kind"] == "unknown"), m
        if m["floor"] is not None:
            assert m["floor"] >= 0
    assert report["missing_gates"] == [g["name"] for g in report["gates"] if g["reason"] == "missing"]


@pytest.fixture
def pcm16_stereo(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(SR * 4) / SR
    x = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.01 * rng.standard_normal(len(t))
    x[SR: SR + 4800] = 0.0
    x[SR * 3] = 0.99
    path = tmp_path / "tone16.wav"
    sf.write(path, np.stack([x, 0.8 * x]).T, SR, subtype="PCM_16")
    return path


@pytest.fixture
def float_mono(tmp_path):
    t = np.arange(SR * 3) / SR
    path = tmp_path / "mono.wav"
    sf.write(path, (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR, subtype="FLOAT")
    return path


# ---------------------------------------------------------------------------
# Every command, every detail level
# ---------------------------------------------------------------------------

COMMANDS = [
    (["analyze", "{f}", "--spectrum", "--frames"], "analyze"),
    (["observe", "{f}"], "observe"),
    (["screen", "{f}"], "screen"),
    (["master", "qc", "{f}"], "master qc"),
]


@pytest.mark.parametrize("level", ["summary", "standard", "full"])
@pytest.mark.parametrize("argv,command", COMMANDS)
def test_cli_emits_valid_report(pcm16_stereo, argv, command, level):
    argv = [a.replace("{f}", str(pcm16_stereo)) for a in argv]
    result = _run(["--json", *argv, "--detail", level])
    assert result.code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert_valid_report(payload["report"], command)
    assert payload["report"]["subject"]["file"].endswith("tone16.wav")
    if level == "full":
        # the whole v1 payload, report included, still validates against its v1 schema
        validate_against(payload, _resolve_ref(payload["$schema"]))


def test_master_qc_report_mirrors_checks_and_requires_stereo_gates(pcm16_stereo):
    payload = json.loads(_run(["--json", "master", "qc", str(pcm16_stereo), "--detail", "full"]).stdout)
    report = payload["report"]
    gate_names = {g["name"] for g in report["gates"]}
    assert set(QC_REQUIRED + QC_REQUIRED_STEREO) <= gate_names
    assert report["missing_gates"] == []
    by_name = {g["name"]: g["status"] for g in report["gates"]}
    for check in payload["checks"]:
        assert by_name[check["name"]] == check["status"]
    assert report["verdict"] == payload["verdict"]


def test_level_floors_follow_the_file_word_length(pcm16_stereo, float_mono):
    pcm = json.loads(_run(["--json", "analyze", str(pcm16_stereo), "--detail", "full"]).stdout)["report"]
    peak = next(m for m in pcm["measurements"] if m["name"] == "peak")
    assert peak["floor_kind"] == "quantization" and peak["floor"] == pytest.approx(2.0 ** -15)

    flt = json.loads(_run(["--json", "analyze", str(float_mono), "--detail", "full"]).stdout)["report"]
    peak = next(m for m in flt["measurements"] if m["name"] == "peak")
    assert peak["floor_kind"] == "rounding"  # float file: the producer's rounding is the floor
    duration = next(m for m in flt["measurements"] if m["name"] == "duration_sec")
    assert duration["floor_kind"] == "resolution" and duration["floor"] == pytest.approx(1 / SR)


def test_batch_outputs_carry_reports(pcm16_stereo, tmp_path):
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.wav").write_bytes(pcm16_stereo.read_bytes())
    (batch / "bad.wav").write_bytes(b"nope")

    lines = [json.loads(line) for line in _run(["--json", "analyze", str(batch)]).stdout.splitlines() if line.strip()]
    reports = {Path(r.get("file", "")).name: r["report"] for r in lines}
    assert_valid_report(reports["a.wav"], "analyze")
    assert reports["bad.wav"]["verdict"] == "FAIL"
    assert_valid_report(reports["bad.wav"], "analyze")

    screen = json.loads(_run(["--json", "screen", str(batch / "a.wav")]).stdout)
    assert_valid_report(screen["report"], "screen")
    only = tmp_path / "only"
    only.mkdir()
    (only / "a.wav").write_bytes(pcm16_stereo.read_bytes())
    batch_screen = json.loads(_run(["--json", "screen", str(only)]).stdout)
    assert_valid_report(batch_screen["report"], "screen")
    assert batch_screen["report"]["subject"]["file"] == str(only)


def test_doctor_emits_report_and_records_failures(monkeypatch, tmp_path):
    import audioman.core.plugin_analysis as pa

    @dataclass
    class _THD:  # doctor serialises the result with dataclasses.asdict
        fundamental_freq: float = 1000.0
        fundamental_db: float = -6.02
        thd_percent: float = 0.01
        thd_plus_n_percent: float = 0.02
        harmonics: list = None

        def __post_init__(self):
            self.harmonics = []

    bundle = tmp_path / "Fake.vst3"
    bundle.mkdir()
    thd = _THD()
    monkeypatch.setattr(pa, "measure_thd", lambda *a, **k: thd)
    out_file = tmp_path / "doctor.json"
    result = _run(["--json", "doctor", "-p", str(bundle), "--mode", "thd", "--output", str(out_file)])
    assert result.code == 0, result.stderr
    report = json.loads(result.stdout)["report"]
    assert_valid_report(report, "doctor")
    assert [g["name"] for g in report["gates"]] == ["completed.thd"] and report["verdict"] == "PASS"
    assert any(m["name"] == "thd.thd_percent" for m in report["measurements"])
    assert json.loads(out_file.read_text())["report"] == report

    def boom(*a, **k):
        raise RuntimeError("plugin crashed")

    monkeypatch.setattr(pa, "measure_thd", boom)
    failed = json.loads(_run(["--json", "doctor", "-p", str(bundle), "--mode", "thd"]).stdout)["report"]
    assert failed["verdict"] == "FAIL"
    assert failed["gates"][0]["reason"] == "plugin crashed"


# ---------------------------------------------------------------------------
# Adapters directly: missing gates, error payloads, floors
# ---------------------------------------------------------------------------


def test_missing_qc_gate_fails_the_report():
    payload = {
        "input": "x.wav", "verdict": "PASS",
        "format": {"sample_rate": SR, "channels": 2, "duration_sec": 1.0},
        "loudness": {"integrated_lufs": -14.0, "sample_rate": SR},
        "checks": [{"name": name, "value": 0, "target": "t", "status": "PASS"} for name in QC_REQUIRED],
    }
    report = build_report("master qc", payload)
    assert_valid_report(report, "master qc")
    assert report["missing_gates"] == QC_REQUIRED_STEREO  # stereo but no inter-channel checks
    assert report["verdict"] == "FAIL"

    mono = build_report("master qc", {**payload, "format": {**payload["format"], "channels": 1}})
    assert mono["missing_gates"] == [] and mono["verdict"] == "PASS"

    odd = build_report("master qc", {**payload, "checks": payload["checks"] + [{"name": "x", "status": "maybe"}]})
    assert next(g for g in odd["gates"] if g["name"] == "x")["status"] == "FAIL"


def test_doctor_all_mode_with_absent_results_reports_missing():
    report = build_report("doctor", {"plugin": "p.vst3", "mode": "all", "thd": {"thd_percent": 0.1},
                                     "attack_release": {"attack_ms": 5.0, "flag": True}})
    assert_valid_report(report, "doctor")
    assert "completed.attack-release" not in report["missing_gates"]
    assert set(report["missing_gates"]) == {
        "completed.linear", "completed.imd", "completed.sweep", "completed.dynamics",
        "completed.waveshaper", "completed.performance",
    }
    assert not any(m["name"] == "attack_release.flag" for m in report["measurements"])


@pytest.mark.parametrize("command", list(report_v2.ADAPTERS))
def test_error_payloads_yield_fail_reports(command):
    report = build_report(command, {"file": "x.wav", "error": "cannot decode"})
    assert_valid_report(report, command)
    assert report["verdict"] == "FAIL" and report["gates"][0]["reason"] == "cannot decode"


def test_finding_and_event_severities_drive_gates():
    warn = build_report("observe", {"findings": [{"id": "a", "code": "DC_OFFSET", "category": "signal",
                                                   "severity": "warn", "where": {}}],
                                    "summary": {"by_severity": {"warn": 1}}})
    assert warn["verdict"] == "WARN"
    assert any(m["name"] == "findings.warn" and m["floor_kind"] == "exact" for m in warn["measurements"])
    screen = build_report("screen", {"events": [{"type": "hum", "severity": "fail", "start_sec": 0.0,
                                                  "confidence": 0.75}], "summary": {"hum": 1}})
    assert screen["verdict"] == "FAIL"
    assert_valid_report(screen, "screen")


@pytest.mark.parametrize(
    "value,floor,kind",
    [
        (3, 0.0, "exact"), (True, 0.0, "exact"), ("x", 0.0, "exact"), (None, 0.0, "exact"),
        (-14.25, 0.005, "rounding"), (12.0, 0.5, "rounding"), (0.1234567891, None, "unknown"),
        (float("inf"), None, "unknown"), (1e-9, None, "unknown"),
    ],
)
def test_rounding_floor(value, floor, kind):
    got_floor, got_kind = rounding_floor(value)
    assert got_kind == kind
    assert got_floor == (pytest.approx(floor) if floor is not None else None)


def test_subject_without_sample_rate_has_unknown_duration_floor():
    report = build_report("analyze", {"file": None, "duration": 2.0, "rms": 0.1, "peak": 0.2, "findings": []})
    duration = next(m for m in report["measurements"] if m["name"] == "duration_sec")
    assert duration["floor"] is None and duration["floor_kind"] == "unknown"
    assert_valid_report(report, "analyze")


def test_quantization_floor_only_raises_a_finer_rounding_floor():
    coarse = report_v2._level_measure("peak", 0.5, 16, "t")  # rounding floor 0.05 > one LSB
    assert coarse.floor_kind == "rounding"
    fine = report_v2._level_measure("peak", 0.123456789012, 16, "t")  # unknown -> LSB
    assert fine.floor_kind == "quantization"
    assert report_v2._file_subtype("/does/not/exist.wav") is None
