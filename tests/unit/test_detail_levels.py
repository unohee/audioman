"""`--detail` levels, digests, spill files and byte ceilings for analysis commands."""

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

from audioman.core import detail as detail_mod
from audioman.core.detail import POLICIES, apply_detail, printed_size
from audioman.core.digest import DIGEST_MAX_CHARS, build_digest

SR = 48000


@dataclass
class _Result:
    code: int
    stdout: str
    stderr: str


def run_command(argv: list[str]) -> _Result:
    """Run `audioman <argv>` in-process with swapped stdout/stderr (rich-safe)."""
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
    """Spill files go to <cache dir>/spill by default: keep them in tmp_path."""
    from audioman.config import settings as settings_module

    monkeypatch.setenv("AUDIOMAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_module.reset_settings()
    yield
    settings_module.reset_settings()


def _load_ref(ref: dict | str):
    path = ref["$ref"] if isinstance(ref, dict) else ref
    return json.loads(Path(path).read_text(encoding="utf-8"))["value"]


# ---------------------------------------------------------------------------
# apply_detail
# ---------------------------------------------------------------------------


def _payload(n_values=1000, n_findings=120):
    severities = ["info", "warn", "critical"]
    return {
        "$schema": "audioman://schema/analyze.v1.json",
        "audioman_version": "0",
        "command": "analyze",
        "digest": "d",
        "rms": 0.1,
        "frame_metrics": {"n_frames": n_values, "rms": [i / 7 for i in range(n_values)], "short": [1, 2, 3]},
        "findings": [
            {"code": f"C{i}", "severity": severities[i % 3], "where": {"sec_start": float(n_findings - i)}}
            for i in range(n_findings)
        ],
    }


def test_full_level_is_unchanged(tmp_path):
    payload = _payload()
    out = apply_detail(payload, "full", spill_dir=tmp_path, label="t")
    assert {k: v for k, v in out.items() if k != "detail"} == payload
    assert out["detail"] == {"level": "full", "byte_ceiling": None, "within_ceiling": True, "spilled": [], "truncated": {}}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("level", ["summary", "standard"])
def test_levels_spill_long_arrays_and_rank_findings(tmp_path, level):
    payload = _payload()
    out = apply_detail(payload, level, spill_dir=tmp_path, label="t")
    policy = POLICIES[level]
    assert printed_size(out) <= policy.byte_ceiling
    assert out["detail"]["within_ceiling"] is True
    # envelope metadata is never moved
    assert [k for k in out][:4] == ["$schema", "audioman_version", "command", "digest"]

    ref = out["frame_metrics"]["rms"]
    assert ref["length"] == 1000
    assert ref["min"] == 0.0 and ref["max"] == pytest.approx(999 / 7, abs=1e-6)
    assert _load_ref(ref) == payload["frame_metrics"]["rms"]  # nothing is lost
    assert out["frame_metrics"]["short"] == [1, 2, 3]  # short arrays stay inline

    kept = out["findings"]
    assert len(kept) == policy.top_k
    rank = {"critical": 0, "warn": 1, "info": 2}
    assert [rank[f["severity"]] for f in kept] == sorted(rank[f["severity"]] for f in kept)  # most severe first
    assert kept[0]["severity"] == "critical"
    note = out["detail"]["truncated"]["findings"]
    assert note["total"] == 120
    assert _load_ref(note["$ref"]) == payload["findings"]
    assert {s["key"] for s in out["detail"]["spilled"]} >= {"frame_metrics.rms", "findings"}


def test_ranking_uses_status_for_checks_and_start_time_as_tiebreak(tmp_path):
    checks = [{"name": f"c{i}", "status": s} for i, s in enumerate(["PASS"] * 8 + ["WARN", "FAIL"])]
    events = [{"type": "x", "severity": "warn", "start_sec": t, "confidence": 0.5} for t in (5.0, 1.0, "bad")]
    out = apply_detail({"checks": checks, "events": events * 3}, "summary", spill_dir=tmp_path, label="t")
    assert [c["status"] for c in out["checks"][:2]] == ["FAIL", "WARN"]
    assert out["events"][0]["start_sec"] in (0.0, "bad", 1.0)


def test_long_strings_and_nested_lists_are_shaped(tmp_path):
    payload = {"ascii_waveform": "#" * 9000, "nested": [[0.5] * 100, {"deep": list(range(70))}]}
    out = apply_detail(payload, "standard", spill_dir=tmp_path, label="w")
    assert out["ascii_waveform"]["length"] == 9000
    assert _load_ref(out["ascii_waveform"]) == "#" * 9000
    assert out["nested"][0]["length"] == 100
    assert out["nested"][1]["deep"]["length"] == 70


def test_non_finite_values_have_null_stats(tmp_path):
    out = apply_detail({"x": [float("nan")] * 100}, "standard", spill_dir=tmp_path, label="n")
    assert out["x"]["min"] is None and out["x"]["mean"] is None


def test_ceiling_tightens_until_it_fits(tmp_path):
    # Sections of many short arrays: each array is under the inline limit, and a
    # reference per array would be larger than the array, so only moving whole
    # sections gets under 4 KiB.
    payload = {
        "bands": {f"b{i}": [float(j) for j in range(8)] for i in range(60)},
        "per_channel": {f"c{i}": [float(j) for j in range(8)] for i in range(40)},
        "records": [{"v": i} for i in range(9)],
        "rms": 0.5,
    }
    out = apply_detail(payload, "summary", spill_dir=tmp_path, label="m")
    assert printed_size(out) <= 4096
    assert out["detail"]["within_ceiling"] is True
    sections = [k for k, v in out.items() if isinstance(v, dict) and v.get("section")]
    assert sections  # whole sections moved out once per-array references stopped helping
    for key in sections:
        assert _load_ref(out[key]) == payload[key]


def test_unspillable_metadata_reports_ceiling_miss(tmp_path):
    payload = {f"key_{i}": f"value {i}" for i in range(400)}  # short scalars only
    out = apply_detail(payload, "summary", spill_dir=tmp_path, label="s")
    assert out["detail"]["within_ceiling"] is False


def test_unknown_level_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown detail level"):
        apply_detail({}, "verbose", spill_dir=tmp_path, label="x")


def test_spill_names_are_safe_and_content_addressed(tmp_path):
    a = apply_detail({"v": list(range(100))}, "standard", spill_dir=tmp_path, label="my file/../x")
    b = apply_detail({"v": list(range(100))}, "standard", spill_dir=tmp_path, label="my file/../x")
    assert a["v"]["$ref"] == b["v"]["$ref"]
    assert Path(a["v"]["$ref"]).parent == tmp_path.resolve()
    assert "/" not in Path(a["v"]["$ref"]).name.replace("my_file_.._x", "")
    empty = apply_detail({"v": list(range(100))}, "standard", spill_dir=tmp_path, label="")
    assert Path(empty["v"]["$ref"]).name.startswith("payload-")


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def test_analyze_digest_mentions_level_silence_hum_and_findings():
    payload = {
        "duration_sec": 125.0, "channels": 2, "sample_rate": 48000, "rms": 0.1, "peak": 1.0,
        "summary": {"spectral_centroid": {"mean": 1234.4}},
        "silence_regions": [{}, {}], "silence_total_sec": 3.5,
        "spectrum": {"hum_check": [{"frequency_hz": 60.0, "is_hum": True}, {"frequency_hz": 50.0, "is_hum": False}]},
        "findings": [{"code": "CLIPPING", "severity": "warn"}, {"code": "DC_OFFSET", "severity": "info"}],
    }
    text = build_digest("analyze", payload)
    assert text.startswith("2:05.0, stereo, 48 kHz. RMS -20.0 dBFS, peak 0.0 dBFS.")
    for fragment in ("1234 Hz", "2 silent regions", "Mains hum at 60.0 Hz", "2 findings", "CLIPPING"):
        assert fragment in text


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ({"channels": 1, "sample_rate": 44100, "duration_sec": 2.0, "rms": 0, "peak": "x"}, "mono, 44.1 kHz"),
        ({"channels": 6, "duration": "?", "rms": float("nan"), "peak": 0.5}, "unknown length, 6 ch"),
        ({"rms": 0.5, "peak": 0.5, "findings": [{"code": c, "severity": "info"} for c in "ABCDEFGH"]}, "..."),
    ],
)
def test_analyze_digest_formats_edge_values(payload, fragment):
    assert fragment in build_digest("analyze", payload)


def test_observe_screen_and_qc_digests():
    assert build_digest("observe", {"duration_sec": 1.0, "findings": []}).endswith("No findings.")
    screen_none = build_digest("screen", {"duration": 1.0, "issues": ["click"], "events": []})
    assert screen_none.endswith("No events.")
    screen = build_digest("screen", {"issues": ["hum"], "events": [{"type": "hum", "severity": "warn"}] * 2})
    assert "2 events (2 hum)" in screen
    batch = build_digest("screen", {"files": [{"events": [{}]}, {"events": []}]})
    assert batch == "2 files screened, 1 events in total."
    ok = build_digest("master qc", {"target": "spotify", "verdict": "PASS", "checks": [{"status": "PASS"}]})
    assert ok == "QC against spotify: PASS. All 1 checks pass."
    bad = build_digest("master qc", {
        "target": "spotify", "verdict": "FAIL",
        "loudness": {"integrated_lufs": -9.0, "true_peak_dbtp": 0.2},
        "checks": [{"name": "integrated_lufs", "status": "FAIL", "value": -9.0}, {"status": "PASS"}],
    })
    assert "Integrated -9.0 LUFS" in bad and "1 of 2 checks not passing: integrated_lufs FAIL (-9.0)" in bad


def test_digest_reports_errors_and_is_bounded():
    assert build_digest("observe", {"error": "boom"}) == "observe failed: boom"
    long = build_digest("observe", {"findings": [{"code": "X" * 200 + str(i), "severity": "info"} for i in range(9)]})
    assert len(long) == DIGEST_MAX_CHARS and long.endswith("...")


# ---------------------------------------------------------------------------
# CLI: every analysis command, default level bounded, levels selectable
# ---------------------------------------------------------------------------


@pytest.fixture
def long_file(tmp_path):
    """60 s stereo with clicks, a silent stretch and hiss: enough to overflow 32 KiB at --detail full."""
    rng = np.random.default_rng(0)
    t = np.arange(SR * 60) / SR
    x = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))
    x[SR * 10: SR * 12] = 0.0
    x[:: SR // 2] += 0.9
    path = tmp_path / "long.wav"
    sf.write(path, np.stack([x, 0.9 * x]).T.astype(np.float32), SR)
    return path


def _run_json(argv):
    result = run_command(["--json", *argv])
    assert result.code == 0, result.stderr
    return result.stdout, json.loads(result.stdout)


COMMANDS = [
    (["analyze", "{f}", "--frames", "--spectrum"], "analyze"),
    (["observe", "{f}"], "observe"),
    (["screen", "{f}"], "screen"),
    (["master", "qc", "{f}"], "master qc"),
]


@pytest.mark.parametrize("argv,command", COMMANDS)
def test_default_output_has_digest_and_stays_under_ceiling(long_file, argv, command):
    argv = [a.replace("{f}", str(long_file)) for a in argv]
    text, payload = _run_json(argv)
    assert len(text.encode("utf-8")) <= POLICIES["standard"].byte_ceiling
    assert payload["detail"]["level"] == "standard"
    assert payload["detail"]["within_ceiling"] is True
    assert isinstance(payload["digest"], str) and 0 < len(payload["digest"]) <= DIGEST_MAX_CHARS
    assert list(payload)[:4] == ["$schema", "audioman_version", "command", "digest"]

    summary_text, summary = _run_json([*argv, "--detail", "summary"])
    assert len(summary_text.encode("utf-8")) <= POLICIES["summary"].byte_ceiling
    assert summary["digest"] == payload["digest"]


def test_analyze_frames_round_trip_through_spill_files(long_file, tmp_path):
    spill = tmp_path / "spill"
    full_text, full = _run_json(["analyze", str(long_file), "--frames", "--detail", "full"])
    assert len(full_text.encode("utf-8")) > POLICIES["standard"].byte_ceiling  # the problem being solved
    _, std = _run_json(["analyze", str(long_file), "--frames", "--spill-dir", str(spill)])
    for key in ("rms", "peak", "spectral_centroid", "spectral_entropy", "zero_crossing_rate"):
        ref = std["frame_metrics"][key]
        assert Path(ref["$ref"]).parent == spill.resolve()
        assert _load_ref(ref) == full["frame_metrics"][key]
        assert ref["length"] == full["frame_metrics"]["n_frames"]
    assert std["digest"] == full["digest"]


def test_analyze_waveform_text_spills_at_summary(long_file):
    _, out = _run_json(["analyze", str(long_file), "--waveform", "--detail", "summary"])
    assert "$ref" in out["ascii_waveform"]


def test_batch_modes_are_shaped_per_record(long_file, tmp_path):
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.wav").write_bytes(long_file.read_bytes())
    (batch / "broken.wav").write_bytes(b"not audio")

    analyze = run_command(["--json", "analyze", str(batch), "--frames"])
    lines = [json.loads(line) for line in analyze.stdout.splitlines() if line.strip()]
    assert analyze.code == 1  # the broken file still fails the batch
    ok = [r for r in lines if "error" not in r][0]
    assert "$ref" in ok["frame_metrics"]["rms"] and ok["digest"]
    err = [r for r in lines if "error" in r][0]
    assert err["digest"].startswith("analyze failed:")

    observe = run_command(["--json", "observe", str(batch)])
    assert observe.code == 1
    assert observe.stdout.count('"digest"') == 2

    screen_text, screen = _run_json(["screen", str(batch / "a.wav")])
    assert screen["detail"]["level"] == "standard"
    screen_batch = run_command(["--json", "screen", str(tmp_path / "only"), "--detail", "summary"])
    assert screen_batch.code in (0, 1)


def test_screen_batch_digest(long_file, tmp_path):
    batch = tmp_path / "ok"
    batch.mkdir()
    (batch / "a.wav").write_bytes(long_file.read_bytes())
    _, payload = _run_json(["screen", str(batch)])
    assert payload["digest"].startswith("1 files screened")


def test_default_spill_dir_is_under_the_cache_dir(monkeypatch, tmp_path):
    from audioman.cli import _detail
    from audioman.config import settings as settings_module

    monkeypatch.setenv("AUDIOMAN_CACHE_DIR", str(tmp_path / "cache"))
    settings_module.reset_settings()
    try:
        assert _detail.default_spill_dir() == tmp_path / "cache" / "spill"
    finally:
        settings_module.reset_settings()


def test_detail_policy_table_is_ordered():
    s, d = detail_mod.POLICIES["summary"], detail_mod.POLICIES["standard"]
    assert s.byte_ceiling < d.byte_ceiling and s.inline_values < d.inline_values and s.top_k < d.top_k
