# tests/unit/cli_extra2/test_undecodable_input.py
# Purpose: an undecodable file must be a reported error, never a traceback.
#
# `soundfile.LibsndfileError` derives from RuntimeError, not OSError. Catching OSError
# therefore misses it, which is how `analyze`/`fx`/`visualize`/`stream` all printed a
# raw traceback for a corrupt input. These tests pin the reported-error behaviour for
# each command and cover the guard branches that fix introduced.

from __future__ import annotations

import pytest

from .conftest import write_wav

CORRUPT = b"not an audio file"


@pytest.fixture
def corrupt_file(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(CORRUPT)
    return path


@pytest.fixture
def good_file(tmp_path):
    # Long enough for the default spectrogram frame size (2048 samples), so the
    # success-path assertions below exercise the real command rather than its
    # too-short-input guard.
    return write_wav(tmp_path / "good.wav", sample_rate=8000, duration=1.0)


class TestUndecodableInputIsReported:
    def test_analyze_reports_the_format_error(self, run_cli, corrupt_file):
        result = run_cli(["analyze", str(corrupt_file)])

        assert result.code == 1
        assert "Traceback" not in result.stderr
        assert "Format not recognised" in result.stderr

    def test_analyze_with_waveform_reports_it_too(self, run_cli, corrupt_file):
        result = run_cli(["analyze", str(corrupt_file), "--waveform"])

        assert result.code == 1
        assert "Traceback" not in result.stderr

    def test_fx_reports_the_format_error(self, run_cli, corrupt_file, tmp_path):
        out = tmp_path / "out.wav"

        result = run_cli(["fx", str(corrupt_file), "gain", "--db", "1", "-o", str(out)])

        assert result.code == 1
        assert "Traceback" not in result.stderr
        assert not out.exists()

    def test_fx_reports_an_unreadable_clip(self, run_cli, good_file, corrupt_file, tmp_path):
        result = run_cli([
            "fx", str(good_file), "splice", "--clip", str(corrupt_file),
            "--position", "0", "-o", str(tmp_path / "o.wav"),
        ])

        assert result.code == 1
        assert "Traceback" not in result.stderr
        assert "cannot read clip" in result.stderr

    def test_visualize_reports_the_format_error(self, run_cli, corrupt_file, tmp_path):
        result = run_cli(["visualize", str(corrupt_file), "-o", str(tmp_path / "v.svl")])

        assert result.code == 1
        assert "Traceback" not in result.stderr

    def test_stream_reports_the_format_error(self, run_cli, corrupt_file):
        result = run_cli(["stream", "triage", str(corrupt_file), "-p", "builtin:reverb"])

        assert result.code == 1
        assert "Traceback" not in result.stderr
        assert "cannot read" in result.stderr

    def test_stream_reports_a_missing_file_without_falling_through(self, run_cli, tmp_path):
        """The missing-input path used to print an error and then keep going."""
        result = run_cli(["stream", "triage", str(tmp_path / "ghost.wav"), "-p", "builtin:reverb"])

        assert result.code == 1
        assert "Traceback" not in result.stderr
        assert "input not found" in result.stderr


class TestValidInputStillWorks:
    """The guards must not reject input the commands previously accepted."""

    def test_analyze_still_analyzes(self, run_cli, good_file):
        result = run_cli(["analyze", str(good_file)])
        assert result.code == 0

    def test_fx_still_applies(self, run_cli, good_file, tmp_path):
        out = tmp_path / "g.wav"
        result = run_cli(["fx", str(good_file), "gain", "--db", "1", "-o", str(out)])
        assert result.code == 0
        assert out.exists()

    def test_visualize_still_writes_svl(self, run_cli, good_file, tmp_path):
        out = tmp_path / "v.svl"
        result = run_cli(["visualize", str(good_file), "-o", str(out)])
        assert result.code == 0
        assert out.exists()

    def test_stream_still_triages(self, run_cli, good_file):
        result = run_cli(["stream", "triage", str(good_file), "-p", "builtin:reverb"])
        assert result.code == 0


class TestGuardsStopTheCommandWhenTheExitIsNeutralised:
    """`print_error` exits, so a missing guard is invisible in production.

    The repo's CLI tests neutralise that exit to make the guards observable; the same
    technique is used here, because each of these lines is load-bearing precisely when
    the exit is bypassed (which is how the original tracebacks were found).
    """

    def test_analyze_returns_after_reporting(self, run_cli, silent_error, corrupt_file):
        seen = silent_error("audioman.cli.analyze")

        run_cli(["analyze", str(corrupt_file)])

        assert seen, "the format error must be reported"
        assert "Format not recognised" in seen[0]

    def test_visualize_returns_after_reporting_on_the_builtin_path(
        self, run_cli, silent_error, corrupt_file, tmp_path,
    ):
        seen = silent_error("audioman.cli.visualize")

        run_cli(["visualize", str(corrupt_file), "-o", str(tmp_path / "v.svl")])

        assert seen, "the format error must be reported"
        assert not (tmp_path / "v.svl").exists()

    def test_visualize_returns_after_reporting_on_the_vamp_path(
        self, run_cli, silent_error, corrupt_file, tmp_path,
    ):
        """The Vamp branch reads the file before the plugin runs, so it needs its own guard."""
        seen = silent_error("audioman.cli.visualize")

        run_cli([
            "visualize", str(corrupt_file), "--plugin", "some-vamp-plugin",
            "-o", str(tmp_path / "v.svl"),
        ])

        assert seen, "the format error must be reported"
        # Reaching the plugin would mean the reader guard did not fire.
        assert not any("Running Vamp plugin" in m for m in seen)

    def test_stream_exits_after_reporting_a_missing_file(self, run_cli, silent_error, tmp_path):
        seen = silent_error("audioman.cli.stream")

        result = run_cli([
            "stream", "triage", str(tmp_path / "ghost.wav"), "-p", "builtin:reverb",
        ])

        assert seen and "input not found" in seen[0]
        assert result.code != 0

    def test_stream_exits_after_reporting_an_unreadable_file(self, run_cli, silent_error, corrupt_file):
        seen = silent_error("audioman.cli.stream")

        result = run_cli([
            "stream", "triage", str(corrupt_file), "-p", "builtin:reverb",
        ])

        assert seen and "cannot read" in seen[0]
        assert result.code != 0
