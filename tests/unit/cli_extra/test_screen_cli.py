# tests/unit/cli_extra/test_screen_cli.py
# Purpose: cover `audioman screen` end to end (AUD-1851) — single-file and
#          batch modes, the JSON envelope, the event-note detail branches and
#          the failure paths.
#
# This host has no `essentia` (an optional extra), so the `auto`/`essentia`
# backends are exercised for their import-error handling and the detectors run
# through their documented fallback; `--backend fallback` forces that path
# deterministically.

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from .conftest import (
    write_click_wav,
    write_corrupt_wav,
    write_hum_wav,
    write_tone,
)


def write_sibilant_wav(path, *, sample_rate: int = 44100, duration: float = 1.0):
    """A high-band burst over a low-mid body -> heuristic sibilance event."""
    n = int(sample_rate * duration)
    half = n // 2
    t = np.arange(half, dtype=np.float32) / sample_rate
    hf = (0.3 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    lf = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.concatenate([hf, lf[: n - half]]), sample_rate, subtype="PCM_16")
    return path


def write_rf_wav(path, *, sample_rate: int = 44100, duration: float = 1.0):
    """Sustained narrowband HF tones -> heuristic rf_noise event with `tones`."""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = (0.05 * np.sin(2 * np.pi * 8000 * t)
            + 0.05 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mono, sample_rate, subtype="PCM_16")
    return path


class TestSingleFileJson:
    def test_envelope_and_metadata_for_a_clean_tone(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav", sample_rate=44100, channels=2)
        result = run_cli(["--json", "screen", str(src)])
        assert result.code == 0, result.stderr

        payload = result.payload
        assert payload["$schema"] == "audioman://schema/screen.v1.json"
        assert payload["command"] == "screen"
        assert payload["file"] == str(src)
        assert payload["sample_rate"] == 44100
        assert payload["channels"] == 2
        assert payload["duration"] == 1.0

    def test_default_issues_cover_every_published_detector(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav")
        payload = run_cli(["--json", "screen", str(src)]).payload

        assert payload["issues"] == [
            "click", "hum", "mouth_click", "sibilance", "breath",
            "background_noise", "rf_noise",
        ]
        assert payload["unsupported_issues"] == []
        assert set(payload["summary"]) == set(payload["issues"])

    def test_summary_counts_agree_with_the_event_list(self, run_cli, tmp_path):
        src = write_click_wav(tmp_path / "clicks.wav")
        payload = run_cli([
            "--json", "screen", str(src), "--issues", "click", "--backend", "fallback",
        ]).payload

        assert payload["events"]
        counted = {}
        for event in payload["events"]:
            counted[event["type"]] = counted.get(event["type"], 0) + 1
        for issue, n in counted.items():
            assert payload["summary"][issue] == n

    def test_events_are_sorted_by_start_time(self, run_cli, tmp_path):
        src = write_click_wav(tmp_path / "clicks.wav")
        payload = run_cli([
            "--json", "screen", str(src), "--issues", "click", "--backend", "fallback",
        ]).payload

        starts = [e["start_sec"] for e in payload["events"]]
        assert starts == sorted(starts)


class TestDetectorEvidence:
    def test_click_events_come_from_the_fallback_detector(self, run_cli, tmp_path):
        src = write_click_wav(tmp_path / "clicks.wav")
        payload = run_cli([
            "--json", "screen", str(src), "--issues", "click", "--backend", "fallback",
        ]).payload

        assert payload["backends"] == {"click": "fallback"}
        # The fallback result is reported as unavailable, not as "no essentia".
        assert payload["essentia_available"] is None
        event = payload["events"][0]
        assert event["type"] == "click"
        assert event["backend"] == "fallback"
        assert event["detector"] == "qc.detect_clicks"

    def test_hum_event_carries_frequency_and_snr(self, run_cli, tmp_path):
        src = write_hum_wav(tmp_path / "mains.wav", frequency=60.0)
        payload = run_cli([
            "--json", "screen", str(src), "--issues", "hum", "--backend", "fallback",
        ]).payload

        hums = [e for e in payload["events"] if e["type"] == "hum"]
        assert hums
        assert all("frequency_hz" in e and "snr_db" in e for e in hums)

    def test_sibilance_event_carries_rms_and_ratio(self, run_cli, tmp_path):
        src = write_sibilant_wav(tmp_path / "sib.wav")
        payload = run_cli([
            "--json", "screen", str(src), "--issues", "sibilance", "--backend", "fallback",
        ]).payload

        assert payload["backends"] == {"sibilance": "heuristic"}
        event = payload["events"][0]
        assert event["type"] == "sibilance"
        assert "rms_db" in event and "ratio_db" in event

    def test_rf_noise_event_lists_the_detected_tones(self, run_cli, tmp_path):
        src = write_rf_wav(tmp_path / "rf.wav")
        payload = run_cli([
            "--json", "screen", str(src), "--issues", "rf_noise", "--backend", "fallback",
        ]).payload

        assert payload["backends"] == {"rf_noise": "heuristic"}
        event = next(e for e in payload["events"] if e["type"] == "rf_noise")
        assert event["tones"]
        assert all("frequency_hz" in tone for tone in event["tones"])

    def test_unknown_issue_is_reported_not_crashed(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav")
        payload = run_cli([
            "--json", "screen", str(src), "--issues", "click,teleport",
        ]).payload

        assert payload["unsupported_issues"] == ["teleport"]
        assert "teleport" not in payload["backends"]

    def test_issue_aliases_are_normalised(self, run_cli, tmp_path):
        """`de-ess` is an alias for sibilance and must not be 'unsupported'."""
        src = write_sibilant_wav(tmp_path / "sib.wav")
        payload = run_cli([
            "--json", "screen", str(src), "--issues", "de-ess", "--backend", "fallback",
        ]).payload

        assert payload["unsupported_issues"] == []
        assert payload["issues"] == ["sibilance"]

    def test_auto_backend_falls_back_when_essentia_is_absent(self, run_cli, tmp_path):
        """`auto` tries essentia then silently uses the fallback on ImportError."""
        src = write_click_wav(tmp_path / "clicks.wav")
        result = run_cli([
            "--json", "screen", str(src), "--issues", "click", "--backend", "auto",
        ])
        assert result.code == 0, result.stderr

        payload = result.payload
        assert payload["backends"] == {"click": "fallback"}
        # `auto` probes for essentia, so the field is a real boolean here.
        assert payload["essentia_available"] is False


class TestHumanOutput:
    def test_single_file_header_and_backend_line(self, run_cli, tmp_path):
        src = write_click_wav(tmp_path / "clicks.wav")
        result = run_cli([
            "screen", str(src), "--issues", "click", "--backend", "fallback",
        ])
        assert result.code == 0, result.stderr

        assert str(src) in result.stdout
        assert "Duration: 1.0s | SR: 44100Hz | CH: 1" in result.stdout
        assert "Backend: click=fallback" in result.stdout

    def test_event_table_lists_type_times_and_severity(self, run_cli, tmp_path):
        src = write_click_wav(tmp_path / "clicks.wav")
        result = run_cli([
            "screen", str(src), "--issues", "click", "--backend", "fallback",
        ])

        assert "Type\tStart\tEnd\tSeverity\tBackend\tDetail" in result.stdout
        rows = [line for line in result.stdout.splitlines() if line.startswith("click\t")]
        assert len(rows) == 3
        assert rows[0].split("\t")[:5] == ["click", "0.100", "0.105", "fail", "fallback"]

    def test_detail_column_renders_frequency_and_snr(self, run_cli, tmp_path):
        src = write_hum_wav(tmp_path / "mains.wav", frequency=60.0)
        result = run_cli([
            "screen", str(src), "--issues", "hum", "--backend", "fallback",
        ])

        row = next(line for line in result.stdout.splitlines() if line.startswith("hum\t"))
        assert "Hz" in row
        assert "SNR" in row

    def test_detail_column_renders_rms_and_ratio(self, run_cli, tmp_path):
        src = write_sibilant_wav(tmp_path / "sib.wav")
        result = run_cli([
            "screen", str(src), "--issues", "sibilance", "--backend", "fallback",
        ])

        row = next(line for line in result.stdout.splitlines()
                   if line.startswith("sibilance\t"))
        assert "ratio" in row
        assert "RMS" in row

    def test_detail_column_renders_tones(self, run_cli, tmp_path):
        src = write_rf_wav(tmp_path / "rf.wav")
        result = run_cli([
            "screen", str(src), "--issues", "rf_noise", "--backend", "fallback",
        ])

        row = next(line for line in result.stdout.splitlines()
                   if line.startswith("rf_noise\t"))
        assert "tones" in row
        assert "Hz" in row

    def test_no_events_path_prints_the_empty_message(self, run_cli, tmp_path):
        src = write_hum_wav(tmp_path / "mains.wav")
        result = run_cli([
            "screen", str(src), "--issues", "click", "--backend", "fallback",
        ])

        assert result.code == 0, result.stderr
        assert "No events detected" in result.stdout
        assert "Type\tStart" not in result.stdout

    def test_unsupported_issues_are_warned_about(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav")
        result = run_cli([
            "screen", str(src), "--issues", "click,teleport", "--backend", "fallback",
        ])

        assert result.code == 0, result.stderr
        assert "unsupported issues skipped: teleport" in result.stderr

    def test_backend_line_reads_none_when_no_issue_was_detected(self, run_cli, tmp_path):
        """Every requested issue is unknown -> no backend entry is recorded."""
        src = write_tone(tmp_path / "tone.wav")
        result = run_cli([
            "screen", str(src), "--issues", "teleport", "--backend", "fallback",
        ])

        assert result.code == 0, result.stderr
        assert "Backend: none" in result.stdout


class TestBatch:
    def test_directory_mode_emits_one_envelope_holding_every_report(self, run_cli, tmp_path):
        src_dir = tmp_path / "audio"
        src_dir.mkdir()
        write_click_wav(src_dir / "a.wav")
        write_click_wav(src_dir / "b.wav")

        result = run_cli([
            "--json", "screen", str(src_dir), "--issues", "click", "--backend", "fallback",
        ])
        assert result.code == 0, result.stderr

        payload = result.payload
        assert payload["$schema"] == "audioman://schema/screen.v1.json"
        assert payload["command"] == "screen"
        assert payload["input"] == str(src_dir)
        assert {r["file"] for r in payload["files"]} == {
            str(src_dir / "a.wav"), str(src_dir / "b.wav"),
        }

    def test_recursive_flag_reaches_nested_directories(self, run_cli, tmp_path):
        src_dir = tmp_path / "audio"
        nested = src_dir / "nested"
        nested.mkdir(parents=True)
        write_click_wav(src_dir / "top.wav")
        write_click_wav(nested / "deep.wav")

        flat = run_cli([
            "--json", "screen", str(src_dir), "--issues", "click", "--backend", "fallback",
        ]).payload
        assert [r["file"] for r in flat["files"]] == [str(src_dir / "top.wav")]

        deep = run_cli([
            "--json", "screen", str(src_dir), "--issues", "click",
            "--backend", "fallback", "--recursive",
        ]).payload
        assert len(deep["files"]) == 2

    def test_human_batch_prints_a_summary_row_per_file(self, run_cli, tmp_path):
        src_dir = tmp_path / "audio"
        src_dir.mkdir()
        write_click_wav(src_dir / "a.wav")
        write_hum_wav(src_dir / "b.wav")

        result = run_cli([
            "screen", str(src_dir), "--issues", "click,hum", "--backend", "fallback",
        ])
        assert result.code == 0, result.stderr

        assert "File\tEvents\tSummary" in result.stdout
        rows = result.stdout.splitlines()[2:]
        assert len(rows) == 2
        assert rows[0].startswith(str(src_dir / "a.wav"))
        assert "click:" in rows[0] and "hum:" in rows[0]

    def test_empty_directory_prints_an_empty_batch_table(self, run_cli, tmp_path):
        src_dir = tmp_path / "empty"
        src_dir.mkdir()
        result = run_cli(["screen", str(src_dir)])

        assert result.code == 0, result.stderr
        assert "File\tEvents\tSummary" in result.stdout
        assert len(result.stdout.splitlines()) == 2


class TestFailures:
    def test_missing_input_file_exits_nonzero(self, run_cli, tmp_path):
        result = run_cli(["screen", str(tmp_path / "ghost.wav")])

        assert result.code == 1
        assert "파일 없음" in result.stderr
        assert "Traceback" not in result.stderr

    def test_undecodable_input_file_exits_nonzero(self, run_cli, tmp_path):
        bad = write_corrupt_wav(tmp_path / "corrupt.wav")
        result = run_cli(["screen", str(bad)])

        assert result.code == 1
        assert "Format not recognised" in result.stderr
        assert "Traceback" not in result.stderr

    def test_explicit_essentia_backend_reports_the_missing_extra(self, run_cli, tmp_path):
        """`--backend essentia` must name the extra instead of a bare ImportError."""
        src = write_click_wav(tmp_path / "clicks.wav")
        result = run_cli([
            "screen", str(src), "--issues", "click", "--backend", "essentia",
        ])

        assert result.code == 1
        assert "Install essentia or use --backend fallback." in result.stderr
        assert "Traceback" not in result.stderr

    def test_empty_issue_list_is_rejected(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav")
        result = run_cli(["screen", str(src), "--issues", "", "--backend", "fallback"])

        assert result.code == 1
        assert "--issues must include at least one issue" in result.stderr

    def test_whitespace_only_issue_list_is_rejected(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav")
        result = run_cli(["screen", str(src), "--issues", "  , ", "--backend", "fallback"])

        assert result.code == 1
        assert "--issues must include at least one issue" in result.stderr


class TestArgumentValidation:
    def test_invalid_backend_is_rejected_by_argparse(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav")
        result = run_cli(["screen", str(src), "--backend", "magic"])
        assert result.code == 2
        assert "invalid choice" in result.stderr

    def test_input_positional_is_required(self, run_cli):
        result = run_cli(["screen"])
        assert result.code == 2
        assert "input" in result.stderr

    @pytest.mark.parametrize("backend", ["auto", "essentia", "fallback"])
    def test_backend_choices_are_accepted_on_a_clean_tone(self, run_cli, tmp_path, backend):
        """Only the fallback path can run here; essentia is asserted above."""
        if backend == "essentia":
            pytest.skip("essentia is an optional extra and is not installed on this host")
        src = write_tone(tmp_path / "tone.wav")
        result = run_cli(["--json", "screen", str(src), "--backend", backend])
        assert result.code == 0, result.stderr
