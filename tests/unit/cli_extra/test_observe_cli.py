# tests/unit/cli_extra/test_observe_cli.py
# Purpose: cover `audioman observe` end to end (AUD-1851) — the filters, the
#          findings payload, the human renderer's location branches, the batch
#          path and the failure handling added for AUD-1859.
#
# The envelope comes from core.findings.json_envelope, so the tests assert the
# published $schema/command/summary contract rather than the raw print shape.

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from .conftest import SR, parse_json_stream, write_corrupt_wav, write_tone


def write_clipped_wav(path, *, sample_rate: int = SR, duration: float = 1.0):
    """Hard-clipped tone: CLIP_SAMPLE_PEAK_EXCEEDED at critical severity (mono)."""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = np.clip(2.0 * np.sin(2 * np.pi * 1000 * t), -1.0, 1.0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mono, sample_rate, subtype="PCM_16")
    return path


def write_mains_wav(path, *, sample_rate: int = SR, duration: float = 2.0,
                    frequency: float = 60.0):
    """60 Hz mains tone -> MAINS_HUM finding (spectral, warn)."""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = (0.05 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mono, sample_rate, subtype="PCM_16")
    return path


def write_stereo_imbalance_wav(path, *, sample_rate: int = SR, duration: float = 1.0):
    """Left/right RMS differ by far more than 3 dB -> CHANNEL_IMBALANCE."""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    left = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack([left, right], axis=1), sample_rate, subtype="PCM_16")
    return path


class TestEnvelope:
    def test_json_payload_is_the_published_envelope(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav")
        result = run_cli(["--json", "observe", str(src)])
        assert result.code == 0, result.stderr

        payload = result.payload
        assert payload["$schema"] == "audioman://schema/observe.v1.json"
        assert payload["command"] == "observe"
        assert "audioman_version" in payload

    def test_metadata_matches_the_input_file(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav", sample_rate=SR, duration=1.0, channels=2)
        result = run_cli(["--json", "observe", str(src)])
        payload = result.payload

        assert payload["file"] == str(src)
        assert payload["sample_rate"] == SR
        assert payload["channels"] == 2
        assert payload["total_samples"] == SR
        assert payload["duration_sec"] == 1.0

    def test_filter_block_defaults_to_every_category_at_info(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav")
        payload = run_cli(["--json", "observe", str(src)]).payload

        assert payload["filter"] == {
            "categories": ["container", "plugin", "signal", "spectral"],
            "min_severity": "info",
        }

    def test_summary_counts_match_the_findings_list(self, run_cli, tmp_path):
        src = write_clipped_wav(tmp_path / "clip.wav")
        payload = run_cli(["--json", "observe", str(src)]).payload

        summary = payload["summary"]
        assert summary["total"] == len(payload["findings"])
        assert sum(summary["by_severity"].values()) == summary["total"]
        assert set(summary["by_category"]) == {"signal", "spectral", "plugin", "container"}


class TestFindings:
    def test_clipping_is_reported_with_its_measurement_and_hint(self, run_cli, tmp_path):
        src = write_clipped_wav(tmp_path / "clip.wav")
        payload = run_cli(["--json", "observe", str(src)]).payload

        clip = next(f for f in payload["findings"]
                    if f["code"] == "CLIP_SAMPLE_PEAK_EXCEEDED")
        assert clip["category"] == "signal"
        assert clip["severity"] == "critical"
        assert clip["where"]["file"] == str(src)
        assert clip["where"]["end_sec"] >= clip["where"]["start_sec"]
        assert clip["hint"]
        assert clip["id"]

    def test_mains_hum_lands_in_the_spectral_category(self, run_cli, tmp_path):
        src = write_mains_wav(tmp_path / "mains.wav")
        payload = run_cli(["--json", "observe", str(src)]).payload

        hums = [f for f in payload["findings"] if f["code"] == "MAINS_HUM"]
        assert hums
        assert all(f["category"] == "spectral" for f in hums)
        # The spectral findings carry a frequency instead of a time range.
        assert all("frequency_hz" in f["where"] for f in hums)

    def test_channel_imbalance_is_detected_on_stereo(self, run_cli, tmp_path):
        src = write_stereo_imbalance_wav(tmp_path / "imbalanced.wav")
        payload = run_cli(["--json", "observe", str(src)]).payload

        codes = {f["code"] for f in payload["findings"]}
        assert "CHANNEL_IMBALANCE" in codes

    def test_plugin_and_container_categories_yield_no_findings(self, run_cli, tmp_path):
        """Those detectors are not implemented yet; the schema stays consistent."""
        src = write_clipped_wav(tmp_path / "clip.wav")
        payload = run_cli([
            "--json", "observe", str(src), "--category", "plugin,container",
        ]).payload

        assert payload["findings"] == []
        assert payload["summary"]["total"] == 0
        assert payload["filter"]["categories"] == ["container", "plugin"]


class TestFilters:
    def test_category_filter_restricts_the_scan(self, run_cli, tmp_path):
        src = write_clipped_wav(tmp_path / "clip.wav")
        payload = run_cli([
            "--json", "observe", str(src), "--category", "spectral",
        ]).payload

        assert all(f["category"] == "spectral" for f in payload["findings"])
        assert payload["filter"]["categories"] == ["spectral"]

    def test_multiple_categories_are_accepted(self, run_cli, tmp_path):
        src = write_mains_wav(tmp_path / "mains.wav")
        payload = run_cli([
            "--json", "observe", str(src), "--category", "signal, spectral",
        ]).payload
        assert payload["filter"]["categories"] == ["signal", "spectral"]

    def test_severity_filter_drops_lower_severities(self, run_cli, tmp_path):
        src = write_mains_wav(tmp_path / "mains.wav")
        payload = run_cli([
            "--json", "observe", str(src), "--severity", "warn",
        ]).payload

        ranks = {"info": 0, "warn": 1, "critical": 2}
        assert payload["filter"]["min_severity"] == "warn"
        assert all(ranks[f["severity"]] >= 1 for f in payload["findings"])

    def test_critical_filter_can_leave_nothing(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav")
        payload = run_cli([
            "--json", "observe", str(src), "--severity", "critical",
        ]).payload
        assert payload["summary"]["by_severity"]["info"] == 0
        assert payload["summary"]["by_severity"]["warn"] == 0

    def test_unknown_category_exits_nonzero(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav")
        result = run_cli(["observe", str(src), "--category", "bogus"])
        assert result.code == 1
        assert "unknown category" in result.stderr
        assert "bogus" in result.stderr

    def test_silence_threshold_is_honoured(self, run_cli, tmp_path):
        """A -10 dB threshold flags the quiet head that -40 dB treats as signal."""
        src = tmp_path / "pattern.wav"
        sr = 8000
        # The gap is a low-level tone (~-34 dBFS), not digital zero: it sits
        # above the default -40 dB threshold and below the -10 dB probe.
        gap = (0.02 * np.sin(2 * np.pi * 300 * np.arange(int(sr * 0.3)) / sr)).astype(np.float32)
        tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(sr * 0.7)) / sr)).astype(np.float32)
        sf.write(str(src), np.concatenate([gap, tone]), sr, subtype="PCM_16")

        permissive = run_cli([
            "--json", "observe", str(src), "--silence-threshold", "-40",
        ]).payload
        strict = run_cli([
            "--json", "observe", str(src), "--silence-threshold", "-10",
        ]).payload

        assert permissive["findings"] == []
        codes = {f["code"] for f in strict["findings"]}
        assert "SILENCE_LEADING" in codes

    def test_spectrum_fft_size_changes_the_harmonic_resolution(self, run_cli, tmp_path):
        """A smaller FFT has coarser bins, so more harmonics clear the hum test."""
        src = write_mains_wav(tmp_path / "mains.wav", frequency=60.0)

        coarse = run_cli([
            "--json", "observe", str(src), "--category", "spectral",
            "--spectrum-fft", "4096",
        ]).payload
        fine = run_cli([
            "--json", "observe", str(src), "--category", "spectral",
            "--spectrum-fft", "16384",
        ]).payload

        coarse_hz = {f["where"]["frequency_hz"] for f in coarse["findings"]}
        fine_hz = {f["where"]["frequency_hz"] for f in fine["findings"]}
        assert 60.0 in fine_hz
        assert len(coarse_hz) > len(fine_hz)

    def test_spectrum_min_rms_gates_the_spectral_detectors(self, run_cli, tmp_path):
        """Above the signal's own RMS every frame is skipped and nothing reports."""
        src = write_mains_wav(tmp_path / "mains.wav", frequency=60.0)

        detected = run_cli([
            "--json", "observe", str(src), "--category", "spectral",
            "--spectrum-min-rms", "0.001",
        ]).payload
        gated = run_cli([
            "--json", "observe", str(src), "--category", "spectral",
            "--spectrum-min-rms", "0.2",
        ]).payload

        assert any(f["code"] == "MAINS_HUM" for f in detected["findings"])
        assert gated["findings"] == []


class TestHumanOutput:
    def test_header_and_summary_lines_are_printed(self, run_cli, tmp_path):
        src = write_clipped_wav(tmp_path / "clip.wav")
        result = run_cli(["observe", str(src)])
        assert result.code == 0, result.stderr

        out = result.stdout
        assert str(src) in out
        assert f"1.0s @ {SR}Hz, 1 ch, {SR} samples" in out
        assert "Findings:" in out

    def test_findings_table_renders_a_row_per_finding(self, run_cli, tmp_path):
        src = write_clipped_wav(tmp_path / "clip.wav")
        result = run_cli(["observe", str(src)])

        assert "Sev\tCategory\tCode\tWhere\tHint" in result.stdout
        assert "CRITICAL\tsignal\tCLIP_SAMPLE_PEAK_EXCEEDED" in result.stdout

    def test_time_range_location_is_rendered(self, run_cli, tmp_path):
        """`where.start_sec` + distinct `end_sec` -> "a.bbbs-c.ddds"."""
        src = write_clipped_wav(tmp_path / "clip.wav")
        result = run_cli(["observe", str(src)])

        where = next(line.split("\t")[3] for line in result.stdout.splitlines()
                     if "CLIP_SAMPLE_PEAK_EXCEEDED" in line)
        assert where.endswith("s")
        assert "-" in where

    def test_frequency_location_is_rendered_for_spectral_findings(self, run_cli, tmp_path):
        src = write_mains_wav(tmp_path / "mains.wav")
        result = run_cli(["observe", str(src), "--severity", "warn"])

        where = next(line.split("\t")[3] for line in result.stdout.splitlines()
                     if "MAINS_HUM" in line)
        assert where.endswith("Hz")
        assert where[:-2].isdigit() or "." in where[:-2]

    def test_no_findings_path_prints_the_green_message(self, run_cli, tmp_path):
        src = write_clipped_wav(tmp_path / "clip.wav")
        result = run_cli([
            "observe", str(src), "--category", "plugin",
        ])

        assert result.code == 0, result.stderr
        assert "Findings: 0 (critical=0, warn=0, info=0)" in result.stdout
        assert "No findings at requested severity." in result.stdout
        assert "Sev\tCategory" not in result.stdout


class TestBatch:
    def test_directory_mode_emits_one_envelope_per_file(self, run_cli, tmp_path):
        src_dir = tmp_path / "audio"
        src_dir.mkdir()
        write_tone(src_dir / "a.wav")
        write_tone(src_dir / "b.wav", frequency=660.0)

        result = run_cli(["--json", "observe", str(src_dir)])
        assert result.code == 0, result.stderr

        # One pretty-printed envelope per file, concatenated on stdout.
        payloads = parse_json_stream(result.stdout)
        assert len(payloads) == 2
        assert all(p["command"] == "observe" for p in payloads)
        assert {p["file"] for p in payloads} == {
            str(src_dir / "a.wav"), str(src_dir / "b.wav"),
        }

    def test_recursive_flag_reaches_nested_directories(self, run_cli, tmp_path):
        src_dir = tmp_path / "audio"
        nested = src_dir / "nested"
        nested.mkdir(parents=True)
        write_tone(src_dir / "top.wav")
        write_tone(nested / "deep.wav")

        flat = run_cli(["--json", "observe", str(src_dir)]).payload
        assert flat["file"] == str(src_dir / "top.wav")

        deep_result = run_cli(["--json", "observe", str(src_dir), "--recursive"])
        payloads = parse_json_stream(deep_result.stdout)
        assert len(payloads) == 2

    def test_human_batch_prints_a_block_per_file(self, run_cli, tmp_path):
        src_dir = tmp_path / "audio"
        src_dir.mkdir()
        write_clipped_wav(src_dir / "a.wav")
        write_clipped_wav(src_dir / "b.wav")

        result = run_cli(["observe", str(src_dir)])
        assert result.code == 0, result.stderr
        assert result.stdout.count("Findings:") == 2
        assert "Batch complete: 2 succeeded, 0 failed / 2 total" in result.stderr

    def test_empty_directory_exits_nonzero(self, run_cli, tmp_path):
        src_dir = tmp_path / "audio"
        src_dir.mkdir()
        result = run_cli(["observe", str(src_dir)])
        assert result.code == 1
        assert "No audio files in" in result.stderr


class TestFailures:
    """AUD-1859: a missing or undecodable file must report, not traceback."""

    def test_missing_input_file_exits_nonzero(self, run_cli, tmp_path):
        ghost = tmp_path / "ghost.wav"
        result = run_cli(["observe", str(ghost)])

        assert result.code == 1
        assert "File not found" in result.stderr
        assert "Traceback" not in result.stderr

    def test_undecodable_input_file_exits_nonzero(self, run_cli, tmp_path):
        bad = write_corrupt_wav(tmp_path / "corrupt.wav")
        result = run_cli(["observe", str(bad)])

        assert result.code == 1
        assert "Format not recognised" in result.stderr
        assert "Traceback" not in result.stderr

    def test_undecodable_input_in_json_mode_still_reports(self, run_cli, tmp_path):
        bad = write_corrupt_wav(tmp_path / "corrupt.wav")
        result = run_cli(["--json", "observe", str(bad)])

        assert result.code == 1
        assert result.stdout == ""
        assert "Format not recognised" in result.stderr

    def test_batch_continues_past_a_broken_file(self, run_cli, tmp_path):
        src_dir = tmp_path / "audio"
        src_dir.mkdir()
        write_clipped_wav(src_dir / "good_a.wav")
        write_corrupt_wav(src_dir / "broken.wav")
        write_clipped_wav(src_dir / "good_b.wav")

        result = run_cli(["--json", "observe", str(src_dir)])

        assert result.code == 1
        payloads = parse_json_stream(result.stdout)
        assert len(payloads) == 3
        errors = [p for p in payloads if "error" in p]
        assert len(errors) == 1
        assert errors[0]["file"] == str(src_dir / "broken.wav")
        assert "Format not recognised" in errors[0]["error"]
        # The healthy neighbours still produced real findings.
        assert sum(1 for p in payloads if p.get("findings")) == 2

    def test_batch_human_mode_warns_and_reports_the_counts(self, run_cli, tmp_path):
        src_dir = tmp_path / "audio"
        src_dir.mkdir()
        write_clipped_wav(src_dir / "good.wav")
        write_corrupt_wav(src_dir / "broken.wav")

        result = run_cli(["observe", str(src_dir)])

        assert result.code == 1
        assert "warning:" in result.stderr
        assert "broken.wav" in result.stderr
        assert "Batch complete: 1 succeeded, 1 failed / 2 total" in result.stderr

    def test_batch_exit_is_zero_when_every_file_succeeds(self, run_cli, tmp_path):
        src_dir = tmp_path / "audio"
        src_dir.mkdir()
        write_clipped_wav(src_dir / "a.wav")

        result = run_cli(["observe", str(src_dir)])
        assert result.code == 0
        assert "Batch complete: 1 succeeded, 0 failed / 1 total" in result.stderr


class TestArgumentValidation:
    @pytest.mark.parametrize("severity", ["info", "warn", "critical"])
    def test_every_severity_choice_is_accepted(self, run_cli, tmp_path, severity):
        src = write_tone(tmp_path / "tone.wav")
        result = run_cli(["--json", "observe", str(src), "--severity", severity])
        assert result.code == 0, result.stderr
        assert result.payload["filter"]["min_severity"] == severity

    def test_invalid_severity_is_rejected_by_argparse(self, run_cli, tmp_path):
        src = write_tone(tmp_path / "tone.wav")
        result = run_cli(["observe", str(src), "--severity", "fatal"])
        assert result.code == 2
        assert "invalid choice" in result.stderr

    def test_input_positional_is_required(self, run_cli):
        result = run_cli(["observe"])
        assert result.code == 2
        assert "input" in result.stderr
