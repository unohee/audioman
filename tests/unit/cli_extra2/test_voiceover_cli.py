# tests/unit/cli_extra2/test_voiceover_cli.py
# Purpose: cover `audioman vo analyze` and `audioman vo process` — both output
#          modes, the denoise-plugin switch, the segment listing and the error
#          paths.
#
# VAD is stubbed in most tests (its silero decisions are irrelevant to the CLI
# wiring under test — segment bookkeeping, denoise dispatch, reporting). One
# test runs the real silero VAD end to end to prove the command works with the
# shipping backend, which is available offline from the installed wheel.

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from audioman.core.vad import Segment

from .conftest import install_registry, install_wrapper, make_meta, write_wav

SR = 16000


@pytest.fixture
def speech_wav(tmp_path):
    """Speech-like fixture: 300 Hz tone inside a syllabic envelope."""
    duration = 2.0
    n = int(duration * SR)
    t = np.arange(n, dtype=np.float32) / SR
    envelope = (np.sin(2 * np.pi * 1.5 * t) > 0).astype(np.float32)
    mono = (0.3 * np.sin(2 * np.pi * 300 * t) * envelope).astype(np.float32)
    data = np.stack([mono, mono], axis=1)
    path = tmp_path / "vo.wav"
    sf.write(str(path), data, SR, subtype="FLOAT")
    return path


@pytest.fixture
def stubbed_vad(monkeypatch):
    """Deterministic VAD: 0.0-0.8s speech, then noise."""
    import audioman.core.voiceover as core_voiceover

    segments = [Segment(start=0, end=int(0.8 * SR), kind="speech")]
    monkeypatch.setattr(core_voiceover, "detect_speech", lambda audio, sr, **kw: list(segments))
    return segments


class TestAnalyze:
    def test_missing_file_is_a_usage_error(self, run_cli, tmp_path):
        result = run_cli(["vo", "analyze", str(tmp_path / "absent.wav")])
        assert result.code == 1
        assert "File not found" in result.stderr

    def test_missing_file_is_reported_twice_and_never_crashes(self, run_cli, tmp_path, silent_error):
        """The existence check has no bare `else` around the analysis call.

        With `print_error`'s exit neutralised the analyzer still runs on the
        missing path and raises `FileNotFoundError`, which the surrounding
        `except Exception` converts into a second, more specific report. The
        contract asserted here is that this stays a *report* — the user sees
        "File not found" and, if the first report were ever bypassed, still gets a
        clean "Analysis failed: ..." instead of a traceback.
        """
        seen = silent_error("audioman.cli.voiceover")
        result = run_cli(["vo", "analyze", str(tmp_path / "absent.wav")])

        assert result.code == 0, result.stderr
        assert seen and "File not found" in seen[0]
        assert "Analysis failed" in seen[-1]
        assert "Traceback" not in result.stdout

    def test_analysis_failure_is_reported(self, run_cli, speech_wav, monkeypatch):
        import audioman.core.voiceover as core_voiceover

        def _boom(*args, **kwargs):
            raise RuntimeError("vad exploded")

        monkeypatch.setattr(core_voiceover, "detect_speech", _boom)

        result = run_cli(["vo", "analyze", str(speech_wav)])

        assert result.code == 1
        assert "Analysis failed" in result.stderr
        assert "vad exploded" in result.stderr

    def test_json_summary_omits_segment_lists(self, run_cli, speech_wav, stubbed_vad):
        result = run_cli(["vo", "analyze", str(speech_wav)], json_mode=True)

        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "vo analyze"
        assert payload["input"] == str(speech_wav)
        assert payload["sample_rate"] == SR
        assert payload["n_speech_segments"] == 1
        assert payload["speech_total_sec"] == pytest.approx(0.8, abs=0.01)
        # Default (no --segments): the long lists are dropped from the payload.
        assert "speech_segments" not in payload
        assert "noise_segments" not in payload

    def test_json_segments_flag_keeps_the_lists(self, run_cli, speech_wav, stubbed_vad):
        result = run_cli(["vo", "analyze", str(speech_wav), "--segments"], json_mode=True)

        assert result.code == 0, result.stderr
        payload = result.payload
        assert [s["start"] for s in payload["speech_segments"]] == [0]
        assert payload["speech_segments"][0]["end"] == int(0.8 * SR)
        assert sum(s["duration_sec"] for s in payload["noise_segments"]) == pytest.approx(
            1.2, abs=0.01
        )

    def test_human_report_prints_stats_and_loudness(self, run_cli, speech_wav, stubbed_vad):
        result = run_cli(["vo", "analyze", str(speech_wav)])

        assert result.code == 0, result.stderr
        assert "Voiceover analysis" in result.stdout
        assert f"Duration: 2.0s @ {SR}Hz" in result.stdout
        assert "Speech segments: 1" in result.stdout
        assert "Speech: 0.8s (40.0%) | Noise: 1.2s" in result.stdout
        assert "Integrated LUFS:" in result.stdout
        assert "True Peak:" in result.stdout
        assert "LRA:" in result.stdout
        # The per-segment listing is opt-in: without --segments only the count
        # above is printed, and no start/end pair is rendered. In --plain mode
        # the rich markup around the listing header is stripped, so the header
        # is visible as its bare text and the count line is the only
        # "Speech segments" occurrence.
        assert result.stdout.count("Speech segments") == 1
        assert "s -" not in result.stdout  # no "<start>s - <end>s" listing rows

    def test_human_segment_listing_is_printed_on_request(self, run_cli, speech_wav, stubbed_vad):
        result = run_cli(["vo", "analyze", str(speech_wav), "--segments"])

        assert result.code == 0, result.stderr
        # The listing header appears a second time (the count line is the first
        # occurrence). The header is authored with rich markup, and --plain must
        # strip it, so the user-visible text carries no "[dim]" token.
        assert result.stdout.count("Speech segments") == 2
        assert "0.00s -    0.80s" in result.stdout
        assert "(0.80s)" in result.stdout
        assert "[dim]" not in result.stdout
        assert "[bold]" not in result.stdout
        assert "[/dim]" not in result.stdout
        # The "+N more" overflow line only appears past 30 segments.
        assert "more" not in result.stdout

    def test_more_than_thirty_segments_truncates_with_a_counter(self, run_cli, speech_wav, monkeypatch):
        import audioman.core.voiceover as core_voiceover

        segments = [Segment(start=i * 10, end=i * 10 + 5, kind="speech") for i in range(35)]
        monkeypatch.setattr(core_voiceover, "detect_speech", lambda audio, sr, **kw: list(segments))

        result = run_cli(["vo", "analyze", str(speech_wav), "--segments"])

        assert result.code == 0, result.stderr
        assert "+5 more" in result.stdout

    def test_real_silero_vad_backend_runs_offline(self, run_cli, speech_wav):
        """No stub: the shipping silero VAD must analyse the file without network."""
        result = run_cli(["vo", "analyze", str(speech_wav)], json_mode=True)

        assert result.code == 0, result.stderr
        assert result.payload["n_speech_segments"] >= 0
        assert result.payload["duration_sec"] == pytest.approx(2.0, abs=0.01)


class TestProcess:
    def test_missing_file_is_a_usage_error(self, run_cli, tmp_path):
        result = run_cli(
            ["vo", "process", str(tmp_path / "absent.wav"), "-o", str(tmp_path / "out.wav")],
        )
        assert result.code == 1
        assert "File not found" in result.stderr

    def test_processing_failure_is_reported(self, run_cli, speech_wav, monkeypatch, tmp_path):
        import audioman.core.voiceover as core_voiceover

        def _boom(*args, **kwargs):
            raise RuntimeError("leveling exploded")

        monkeypatch.setattr(core_voiceover, "detect_speech", _boom)

        result = run_cli(
            ["vo", "process", str(speech_wav), "-o", str(tmp_path / "o.wav")],
        )

        assert result.code == 1
        assert "Processing failed" in result.stderr
        assert "leveling exploded" in result.stderr
        assert "Voiceover complete" not in result.stderr
        assert not (tmp_path / "o.wav").exists()

    def test_processing_failure_returns_before_reporting(
        self, run_cli, speech_wav, monkeypatch, tmp_path, silent_error,
    ):
        """The `return` after the error must stop the command, not just print.

        Without it the code would call `result.to_dict()` on an unbound local
        and die on a NameError after having printed the error.
        """
        import audioman.core.voiceover as core_voiceover

        def _boom(*args, **kwargs):
            raise RuntimeError("leveling exploded")

        monkeypatch.setattr(core_voiceover, "detect_speech", _boom)
        seen = silent_error("audioman.cli.voiceover")

        result = run_cli(["vo", "process", str(speech_wav), "-o", str(tmp_path / "o.wav")])

        assert result.code == 0, result.stderr
        assert seen and "Processing failed" in seen[0]
        assert "Voiceover complete" not in result.stderr

    def test_human_success_report_is_reachable(self, run_cli, speech_wav, stubbed_vad, tmp_path):
        """Regression guard for the `print_success` NameError fixed in 0378bd2.

        The command must print its success line instead of raising after the
        output file has already been written.
        """
        out = tmp_path / "out.wav"
        result = run_cli(
            ["vo", "process", str(speech_wav), "-o", str(out), "--denoise-plugin", "none"],
        )

        assert result.code == 0, result.stderr
        assert "Voiceover complete" in result.stderr
        assert f"Input:  {speech_wav}" in result.stdout
        assert f"Output: {out}" in result.stdout
        assert "Speech: 1 segments" in result.stdout
        assert "Denoise:" not in result.stdout  # skipped with "none"
        assert "Target: -20.0 LUFS" in result.stdout
        assert "Loudness  in:" in result.stdout
        assert "Loudness  out:" in result.stdout
        assert out.exists()
        assert not result.stdout.startswith("Traceback")

    def test_json_payload_summarises_per_segment_leveling(self, run_cli, speech_wav, stubbed_vad, tmp_path):
        result = run_cli(
            ["vo", "process", str(speech_wav), "-o", str(tmp_path / "out.wav"),
             "--denoise-plugin", "none"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "vo process"
        leveling = payload["leveling"]
        # The verbose per-segment list is replaced by a count in JSON mode.
        assert "per_segment" not in leveling
        assert leveling["per_segment_count"] == 1
        assert payload["denoise_plugin"] is None
        assert payload["measured_in"]["sample_rate"] == SR
        assert payload["measured_out"]["sample_rate"] == SR

    def test_denoise_none_skips_the_plugin_entirely(self, run_cli, speech_wav, stubbed_vad, tmp_path, monkeypatch):
        import audioman.core.voiceover as core_voiceover

        def _fail(*args, **kwargs):
            raise AssertionError("denoise must not run when 'none' is given")

        monkeypatch.setattr(core_voiceover, "_apply_denoise", _fail)

        out = tmp_path / "clean.wav"
        result = run_cli(
            ["vo", "process", str(speech_wav), "-o", str(out), "--denoise-plugin", "none"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        assert result.payload["denoise_plugin"] is None
        assert out.exists()

    def test_denoise_plugin_name_and_params_reach_the_wrapper(
        self, run_cli, speech_wav, stubbed_vad, tmp_path, monkeypatch,
    ):
        meta = make_meta(short_name="voice-de-noise", name="RX 10 Voice De-noise")
        stub = install_registry(monkeypatch, ("audioman.core.voiceover",), [meta])
        wrappers = install_wrapper(monkeypatch, ("audioman.core.voiceover",), gain=0.5)

        out = tmp_path / "denoised.wav"
        result = run_cli(
            ["vo", "process", str(speech_wav), "-o", str(out),
             "--denoise-param", "reduction=12.0"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        assert stub.get_calls == ["voice-de-noise"]
        assert len(wrappers) == 1
        assert wrappers[0].loaded is True
        assert wrappers[0].applied == [{"reduction": 12.0}]
        # The report carries the plugin's full display name, not the short name.
        assert result.payload["denoise_plugin"] == "RX 10 Voice De-noise"

    def test_human_report_shows_the_denoise_plugin_line(
        self, run_cli, speech_wav, stubbed_vad, tmp_path, monkeypatch,
    ):
        meta = make_meta(short_name="voice-de-noise", name="RX 10 Voice De-noise")
        install_registry(monkeypatch, ("audioman.core.voiceover",), [meta])
        install_wrapper(monkeypatch, ("audioman.core.voiceover",))

        result = run_cli(["vo", "process", str(speech_wav), "-o", str(tmp_path / "d.wav")])

        assert result.code == 0, result.stderr
        assert "Denoise: RX 10 Voice De-noise" in result.stdout

    def test_unknown_denoise_plugin_is_reported(self, run_cli, speech_wav, stubbed_vad, tmp_path):
        result = run_cli(
            ["vo", "process", str(speech_wav), "-o", str(tmp_path / "o.wav"),
             "--denoise-plugin", "definitely-not-installed"],
        )
        assert result.code == 1
        assert "Processing failed" in result.stderr
        assert "definitely-not-installed" in result.stderr

    def test_leveling_parameters_are_forwarded(self, run_cli, speech_wav, stubbed_vad, tmp_path, monkeypatch):
        import audioman.core.voiceover as core_voiceover

        captured = {}
        original = core_voiceover.process

        def _spy(*, input_path, output_path, **kwargs):
            captured.update(kwargs)
            return original(input_path=input_path, output_path=output_path, **kwargs)

        monkeypatch.setattr(core_voiceover, "process", _spy)

        result = run_cli(
            ["vo", "process", str(speech_wav), "-o", str(tmp_path / "o.wav"),
             "--denoise-plugin", "none",
             "--target-lufs", "-16", "--max-true-peak", "-2",
             "--noise-attenuation", "-20",
             "--vad-threshold", "0.7", "--min-speech-ms", "100",
             "--min-silence-ms", "150", "--speech-pad-ms", "40"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        assert captured == {
            "target_lufs": -16.0,
            "max_true_peak_dbtp": -2.0,
            "noise_attenuation_db": -20.0,
            "denoise_plugin": None,
            "denoise_params": None,
            "vad_threshold": 0.7,
            "min_speech_ms": 100,
            "min_silence_ms": 150,
            "speech_pad_ms": 40,
        }
        assert result.payload["leveling"]["target_lufs"] == -16.0

    def test_output_directory_is_created(self, run_cli, speech_wav, stubbed_vad, tmp_path):
        out = tmp_path / "nested" / "deep" / "vo.wav"
        result = run_cli(
            ["vo", "process", str(speech_wav), "-o", str(out), "--denoise-plugin", "none"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        assert out.exists()
        info = sf.info(str(out))
        assert info.samplerate == SR
