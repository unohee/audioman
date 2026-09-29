# tests/unit/cli_extra2/test_cli_error_guards.py
# Purpose: prove the `return` guards added to cli/fx.py and cli/preset.py are what
#          stop the command, not the implicit exit inside print_error (AUD-1859).
#
# `print_error` calls sys.exit(1), so in production a missing guard is invisible: the
# command ends anyway. The bug surfaces only when the exit is bypassed (which the CLI
# tests do, and which is how the original tracebacks were found). Neutralising the exit
# here makes each guard observable, so removing one fails a test rather than passing
# silently because of the exit.

from __future__ import annotations

from .conftest import write_wav


class TestFxErrorGuards:
    def test_returns_after_a_rejected_effect(self, run_cli, silent_error, tmp_path):
        seen = silent_error("audioman.cli.fx")
        src = write_wav(tmp_path / "base.wav")
        out = tmp_path / "out.wav"

        # With no --start/--end this cut would remove every sample and is rejected.
        run_cli(["fx", str(src), "cut-region", "-o", str(out)])

        assert seen, "the rejection must be reported"
        assert "remove the entire file" in seen[0]
        assert not out.exists(), "the guard must stop the command before it writes"

    def test_returns_after_a_missing_input(self, run_cli, silent_error, tmp_path):
        seen = silent_error("audioman.cli.fx")
        out = tmp_path / "out.wav"

        run_cli(["fx", str(tmp_path / "ghost.wav"), "gain", "--db", "1", "-o", str(out)])

        assert seen, "a missing input must be reported"
        assert not out.exists(), "the guard must stop the command before the effect runs"


class TestPresetSaveErrorGuards:
    def test_returns_after_a_rejected_name(self, run_cli, silent_error):
        seen = silent_error("audioman.cli.preset")

        result = run_cli(["preset", "save", "../escape", "-p", "plug"])

        assert seen, "the rejection must be reported"
        assert "single path component" in seen[0]
        # Falling through the guard would reach the success branch.
        assert "Preset saved" not in result.stderr

    def test_returns_after_a_bad_param(self, run_cli, silent_error, tmp_path):
        seen = silent_error("audioman.cli.preset")

        run_cli(["preset", "save", "bad", "-p", "plug", "--param", "not-a-pair"])

        assert seen, "the malformed param must be reported"
        assert "key=value" in seen[0]
        assert not (tmp_path / "presets" / "plug" / "bad.json").exists()
