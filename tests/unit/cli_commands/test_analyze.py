# Created: 2026-09-28
# Purpose: coverage for cli/analyze.py — single/batch analysis, --frames, --waveform,
#          --spectrum (AUD-1851).
#
# analyze is a pure analysis command that uses no plugins at all. These tests exercise the
# real numbers in the JSON payload (sample rate/channels/silence regions/hum) and every
# branch of the human-facing output (no silence, more than 10 silence regions,
# spectrum hum present/absent, hf_slope available/not).

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from harness import run_command, write_wav


def _json(result) -> dict:
    return json.loads(result.out)


def _write_hummed(path, *, sample_rate=44100, duration=0.5, freq=60.0, amp=0.5):
    """A single 60Hz hum component. Reliably exercises hum detection in spectrum_diagnostics."""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack([mono, mono], axis=1), sample_rate, subtype="PCM_16")
    return path


def _write_many_silences(path, *, sample_rate=44100, regions=12, seg_sec=0.2):
    """A file alternating tone and silence, producing `regions` silence regions."""
    n = int(sample_rate * seg_sec)
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(n, dtype=np.float32) / sample_rate)
    parts = []
    for _ in range(regions):
        parts.append(np.zeros(n, dtype=np.float32))
        parts.append(tone.astype(np.float32))
    mono = np.concatenate(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack([mono, mono], axis=1), sample_rate, subtype="PCM_16")
    return path


class TestSingleJson:
    def test_payload_carries_core_measurements(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav", sample_rate=44100, duration=1.0, amp=0.5)
        result = run_command(["--json", "analyze", str(src)])
        assert result.code == 0
        payload = _json(result)
        assert payload["command"] == "analyze"
        assert payload["file"] == str(src)
        assert payload["sample_rate"] == 44100
        assert payload["channels"] == 2
        assert payload["duration"] == pytest.approx(1.0, abs=1e-3)
        assert payload["duration_sec"] == pytest.approx(1.0, abs=1e-3)
        assert payload["frames"] == 44100
        assert payload["total_samples"] == 44100
        assert payload["rms"] == pytest.approx(0.5 / np.sqrt(2), abs=1e-2)
        assert payload["peak"] == pytest.approx(0.5, abs=1e-2)
        assert "summary" in payload
        assert payload["silence_regions"] == []
        assert payload["silence_total_sec"] == 0.0
        assert isinstance(payload["findings"], list)

    def test_summary_has_all_expected_metrics(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav")
        payload = _json(run_command(["--json", "analyze", str(src)]))
        assert set(payload["summary"]) >= {
            "rms", "peak", "spectral_centroid", "spectral_entropy", "zero_crossing_rate",
        }
        rms_stats = payload["summary"]["rms"]
        assert set(rms_stats) == {"mean", "min", "max", "std"}

    def test_frames_flag_adds_per_frame_arrays(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav", duration=1.0)
        result = run_command([
            "--json", "analyze", str(src), "--frames", "--frame-size", "1024", "--hop", "256",
        ])
        assert result.code == 0
        block = _json(result)["frame_metrics"]
        assert block["frame_size"] == 1024
        assert block["hop_size"] == 256
        assert block["n_frames"] == len(block["rms"]) == len(block["peak"])
        assert block["n_frames"] == len(block["spectral_centroid"])
        assert block["n_frames"] == len(block["spectral_entropy"])
        assert block["n_frames"] == len(block["zero_crossing_rate"])
        assert block["n_frames"] > 0

    def test_custom_frame_size_and_hop_change_frame_count(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav")
        coarse = _json(run_command([
            "--json", "analyze", str(src), "--frames", "--frame-size", "4096", "--hop", "4096",
        ]))["frame_metrics"]["n_frames"]
        fine = _json(run_command([
            "--json", "analyze", str(src), "--frames", "--frame-size", "512", "--hop", "512",
        ]))["frame_metrics"]["n_frames"]
        assert fine > coarse

    def test_silence_threshold_reports_regions(self, tmp_path):
        src = _write_many_silences(tmp_path / "pattern.wav", regions=3)
        result = run_command([
            "--json", "analyze", str(src), "--silence-threshold", "-40",
        ])
        assert result.code == 0
        payload = _json(result)
        assert len(payload["silence_regions"]) == 3
        for region in payload["silence_regions"]:
            assert set(region) >= {"start_sample", "end_sample", "duration_sec"}
        assert payload["silence_total_sec"] == pytest.approx(
            sum(r["duration_sec"] for r in payload["silence_regions"]), abs=1e-4)

    def test_threshold_below_noise_floor_finds_no_silence(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav", amp=0.5)
        payload = _json(run_command([
            "--json", "analyze", str(src), "--silence-threshold", "-120",
        ]))
        assert payload["silence_regions"] == []


class TestSpectrum:
    def test_spectrum_block_present_with_flags(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav", freq=440.0)
        result = run_command([
            "--json", "analyze", str(src), "--spectrum", "--spectrum-fft", "4096",
            "--spectrum-min-rms", "0.005",
        ])
        assert result.code == 0
        spec = _json(result)["spectrum"]
        assert spec["fft_size"] == 4096
        assert spec["min_rms_threshold"] == 0.005
        assert spec["frames_analyzed"] > 0
        assert spec["band_energy"]
        assert spec["dominant_frequencies"]
        # The dominant frequency of a 440Hz tone must really be near 440.
        top = max(spec["dominant_frequencies"], key=lambda d: d["db_rel_peak"])
        assert top["frequency_hz"] == pytest.approx(440.0, abs=30.0)
        assert "hum_check" in spec and "hf_slope" in spec

    def test_hum_is_detected_for_60hz_tone(self, tmp_path):
        src = _write_hummed(tmp_path / "hum.wav", freq=60.0)
        spec = _json(run_command(["--json", "analyze", str(src), "--spectrum"]))["spectrum"]
        hum_bins = [h for h in spec["hum_check"] if h["is_hum"]]
        assert any(abs(h["frequency_hz"] - 60.0) < 1e-6 for h in hum_bins)
        for entry in hum_bins:
            assert entry["snr_db"] > 10.0

    def test_clean_tone_reports_no_hum_at_440(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav", freq=440.0)
        spec = _json(run_command(["--json", "analyze", str(src), "--spectrum"]))["spectrum"]
        assert [h for h in spec["hum_check"] if h["is_hum"]] == []

    def test_low_sample_rate_makes_hf_slope_unavailable(self, tmp_path):
        # At 8kHz there is no 10-16kHz band, so the slope cannot be computed.
        src = write_wav(tmp_path / "low.wav", sample_rate=8000, duration=0.5)
        spec = _json(run_command([
            "--json", "analyze", str(src), "--spectrum", "--spectrum-fft", "2048",
        ]))["spectrum"]
        assert spec["hf_slope"]["slope_db"] is None

    def test_spectrum_adds_spectral_findings(self, tmp_path):
        src = _write_hummed(tmp_path / "hum.wav", freq=60.0)
        plain = _json(run_command(["--json", "analyze", str(src)]))["findings"]
        with_spectrum = _json(run_command([
            "--json", "analyze", str(src), "--spectrum",
        ]))["findings"]
        assert len(with_spectrum) > len(plain)
        assert any(f["category"] == "spectral" for f in with_spectrum)


class TestWaveform:
    def test_waveform_flags_add_ascii_blocks(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav")
        result = run_command([
            "--json", "analyze", str(src), "--waveform", "--waveform-width", "40",
            "--waveform-height", "8",
        ])
        assert result.code == 0
        payload = _json(result)
        assert isinstance(payload["ascii_waveform"], str)
        assert payload["ascii_waveform"].strip()
        assert payload["ascii_envelope"].strip()
        assert payload["ascii_spectral"].strip()
        assert len(payload["ascii_waveform"].splitlines()) > 1

    def test_waveform_mode_rms_is_accepted(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav")
        payload = _json(run_command([
            "--json", "analyze", str(src), "-w", "--waveform-mode", "rms",
        ]))
        assert payload["ascii_waveform"].strip()

    def test_json_without_waveform_omits_ascii_keys(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav")
        payload = _json(run_command(["--json", "analyze", str(src)]))
        assert "ascii_waveform" not in payload
        assert "ascii_envelope" not in payload
        assert "ascii_spectral" not in payload


class TestHumanOutput:
    def test_plain_output_prints_header_and_summary_table(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav", amp=0.5)
        result = run_command(["--plain", "analyze", str(src)])
        assert result.code == 0
        assert "Duration:" in result.out
        assert "SR: 44100Hz" in result.out
        assert "CH: 2" in result.out
        assert "RMS:" in result.out
        assert "Metric\tMean\tMin\tMax\tStd" in result.out
        assert "Silence regions: none" in result.out

    def test_plain_output_reports_silence_regions(self, tmp_path):
        src = _write_many_silences(tmp_path / "pattern.wav", regions=2)
        result = run_command(["--plain", "analyze", str(src)])
        assert result.code == 0
        assert "Silence regions: 2" in result.out

    def test_more_than_ten_silence_regions_are_truncated(self, tmp_path):
        src = _write_many_silences(tmp_path / "many.wav", regions=12)
        result = run_command(["--plain", "analyze", str(src)])
        assert result.code == 0
        assert "Silence regions: 12" in result.out
        assert "... +2 more" in result.out

    def test_plain_waveform_sections_are_printed(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav")
        result = run_command([
            "--plain", "analyze", str(src), "--waveform", "--waveform-width", "30",
        ])
        assert result.code == 0
        assert "Waveform" in result.out
        assert "RMS Envelope" in result.out
        assert "Spectral" in result.out

    def test_spectrum_section_renders_bands_and_dominant_frequencies(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav", freq=440.0)
        result = run_command([
            "--plain", "analyze", str(src), "--spectrum", "--spectrum-fft", "4096",
        ])
        assert result.code == 0
        assert "Spectrum diagnostics" in result.out
        assert "Band energy" in result.out
        assert "Dominant frequencies:" in result.out
        assert "Hz" in result.out
        assert "HF slope:" in result.out
        assert "Mains hum: not detected" in result.out

    def test_spectrum_section_flags_detected_hum(self, tmp_path):
        src = _write_hummed(tmp_path / "hum.wav", freq=60.0)
        result = run_command(["--plain", "analyze", str(src), "--spectrum"])
        assert result.code == 0
        assert "HUM detected" in result.out
        assert "60" in result.out

    def test_rich_output_path_runs(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav")
        result = run_command(["analyze", str(src)])
        assert result.code == 0
        assert "Duration:" in result.out
        assert "Summary" in result.out

    def test_missing_input_file_exits_1(self, tmp_path):
        result = run_command(["--json", "analyze", str(tmp_path / "ghost.wav")])
        assert result.code == 1
        assert "File not found" in result.err


class TestArgValidation:
    @pytest.mark.parametrize("flag", ["--frame-size", "--hop", "--waveform-width",
                                      "--waveform-height", "--spectrum-fft"])
    def test_non_positive_int_arguments_are_rejected(self, tmp_path, flag):
        src = write_wav(tmp_path / "tone.wav")
        result = run_command(["--json", "analyze", str(src), flag, "0"])
        assert result.code == 2
        assert "must be a positive integer" in result.err

    def test_bad_waveform_mode_is_rejected(self, tmp_path):
        src = write_wav(tmp_path / "tone.wav")
        result = run_command(["--json", "analyze", str(src), "--waveform-mode", "nope"])
        assert result.code == 2
        assert "invalid choice" in result.err


class TestBatch:
    def test_plain_directory_analysis_summarises_each_file(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        write_wav(in_dir / "b.wav", freq=880.0)

        result = run_command(["--plain", "analyze", str(in_dir)])
        assert result.code == 0
        assert "[1/2]" in result.out and "[2/2]" in result.out
        assert "a.wav:" in result.out and "b.wav:" in result.out
        assert "RMS=" in result.out and "Centroid=" in result.out
        assert "Analysis complete: 2 files" in result.err

    def test_json_directory_analysis_emits_jsonl(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        write_wav(in_dir / "b.wav")

        result = run_command(["--json", "analyze", str(in_dir)])
        assert result.code == 0
        payloads = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert len(payloads) == 2
        assert all(p["command"] == "analyze" for p in payloads)
        assert {p["file"] for p in payloads} == {str(in_dir / "a.wav"), str(in_dir / "b.wav")}

    def test_recursive_flag_descends_into_subdirectories(self, tmp_path):
        in_dir = tmp_path / "in"
        (in_dir / "nested").mkdir(parents=True)
        write_wav(in_dir / "top.wav")
        write_wav(in_dir / "nested" / "deep.wav")

        result = run_command(["--json", "analyze", str(in_dir), "--recursive"])
        payloads = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert {p["file"] for p in payloads} == {
            str(in_dir / "top.wav"), str(in_dir / "nested" / "deep.wav"),
        }

    def test_empty_directory_exits_1(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = run_command(["--plain", "analyze", str(empty)])
        assert result.code == 1
        assert "No audio files found" in result.err

    def test_per_file_error_is_reported_and_exits_1(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "good.wav")
        # Audio extension but corrupt contents -> read_audio fails.
        (in_dir / "broken.wav").write_bytes(b"not really audio")

        result = run_command(["--plain", "analyze", str(in_dir)])
        assert result.code == 1
        assert "broken.wav: ERROR" in result.out
        assert "good.wav:" in result.out

    def test_per_file_error_json_payload(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        (in_dir / "broken.wav").write_bytes(b"not really audio")

        result = run_command(["--json", "analyze", str(in_dir)])
        assert result.code == 1
        payloads = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert len(payloads) == 1
        assert payloads[0]["file"] == str(in_dir / "broken.wav")
        assert payloads[0]["error"]
        assert "broken.wav" in payloads[0]["error"]
