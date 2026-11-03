# tests/unit/cli_commands/test_plain_tag_leaks.py
# Purpose: command-level guard that rich tag text never leaks into `--plain` output (AUD-1853).
#
# The `--plain` console uses `markup=False`, so passing a tagged string such as
# `[dim]...[/dim]` straight to `console.print` prints the tags verbatim. This file
# runs representative commands through the real parser and checks that no tag text
# appears in stdout+stderr, while bracket text that is not a tag stays
# (`[dry-run]`, `[0, 1]`, `[1/2]`).

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audioman.cli import chain, output, process
from audioman.core.engine import ProcessResult
from audioman.core.pipeline import PipelineStep
from harness import (
    FakePlugin,
    assert_no_leaked_tag_text,
    capture_streams,
    find_leaked_tag_text,
    run_command,
    wrapper_factory,
    write_wav,
)


@pytest.fixture
def stub_engine(monkeypatch):
    """Stand-in for process.process_file so batch/single paths run without VST3."""
    calls = []

    def _process_file(input_path, output_path, plugin_name, params=None, passes=1):
        calls.append(plugin_name)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"RIFF")
        return ProcessResult(
            input_path=str(input_path),
            output_path=str(output_path),
            plugin_name=plugin_name,
            params_applied=params or {},
            input_stats={"rms": 0.3, "peak": 0.5, "duration": 1.0, "frames": 44100},
            output_stats={"rms": 0.15, "peak": 0.25, "duration": 1.0, "frames": 44100},
            duration_seconds=0.25,
        )

    monkeypatch.setattr(process, "process_file", _process_file)
    return calls


@pytest.fixture
def fixed_steps(monkeypatch):
    """Pin chain parsing so the plan path can be observed without resolving plugins."""
    steps = [
        PipelineStep(plugin_name="dehum", params={"freq": 60.0}),
        PipelineStep(plugin_name="declick", params={}),
    ]
    monkeypatch.setattr(chain, "parse_chain_string", lambda raw: list(steps))
    return steps


class TestPlainCommandsEmitNoTagText:
    """Representative commands leave no tag text in plain output."""

    def test_process_dry_run(self, tmp_path, stub_engine):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--plain", "process", str(src), "-p", "denoise", "-o", str(tmp_path / "out.wav"),
            "--dry-run", "--param", "x=1",
        ])
        assert result.code == 0
        assert_no_leaked_tag_text(result)
        assert result.out.count("[dry-run]") == 1
        assert "[denoise]" in result.out

    def test_process_batch_dry_run(self, tmp_path, stub_engine):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        result = run_command([
            "--plain", "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"),
            "--dry-run",
        ])
        assert result.code == 0
        assert_no_leaked_tag_text(result)
        assert f"[dry-run] batch: 1 files → [denoise] → {tmp_path / 'out'}" in result.out

    def test_process_batch_failure_is_reported_with_a_clean_message(self, tmp_path, monkeypatch):
        """Failure-path messages must not carry tag text either."""
        def _boom(*args, **kwargs):
            raise ValueError("Plugin not found: 'ghost'")

        monkeypatch.setattr(process, "process_file", _boom)
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        result = run_command([
            "--plain", "process", str(in_dir), "-p", "ghost", "-o", str(tmp_path / "out"),
        ])
        assert result.code == 1
        assert_no_leaked_tag_text(result)
        assert "Plugin not found: 'ghost'" in result.err

    def test_chain_plan_lines(self, tmp_path, fixed_steps):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--plain", "chain", str(src), "-s", "dehum:freq=60,declick",
            "-o", str(tmp_path / "out.wav"), "--dry-run",
        ])
        assert result.code == 0
        assert_no_leaked_tag_text(result)
        assert f"[dry-run] {src}" in result.out
        assert "  → [dehum ({'freq': 60.0})]" in result.out
        assert "  → [declick]" in result.out

    def test_chain_batch_plan_line(self, tmp_path, fixed_steps):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        result = run_command([
            "--plain", "chain", str(in_dir), "-s", "dehum,declick", "-o", str(tmp_path / "out"),
            "--dry-run",
        ])
        assert result.code == 0
        assert_no_leaked_tag_text(result)
        assert "[dehum → declick]" in result.out

    def test_chain_json_plan_is_unaffected(self, tmp_path, fixed_steps):
        """JSON mode keeps stdout pure JSON, regardless of tags/markup."""
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--plain", "--json", "chain", str(src), "-s", "dehum,declick",
            "-o", str(tmp_path / "out.wav"), "--dry-run",
        ])
        assert result.code == 0
        assert json.loads(result.out)["dry_run"] is True


class _RecordingPlugin(FakePlugin):
    """Fake plugin that actually holds parameter attributes (for the dump path)."""

    def __init__(self):
        super().__init__(threshold=-20.0, mode="fast", bypass=False, label=object())
        self.parameters = {
            "threshold": object(),
            "mode": object(),
            "bypass": object(),
            "label": object(),
        }


class _BoomPlugin(_RecordingPlugin):
    def __init__(self):
        raise RuntimeError("cannot load plugin")


class TestPlainProgressMessagesKeepTheirEvidence:
    """The `[N/M]` progress marker in failure messages must survive plain mode.

    The old `_strip_markup` removed whole bracket groups, which also erased the
    evidence in `[1/2]`. Now only real rich tags are stripped.
    """

    def test_batch_failure_warning_keeps_the_index_and_message(
        self, fake_registry, tmp_path, monkeypatch
    ):
        from audioman.cli import dump

        monkeypatch.setattr(dump, "VST3PluginWrapper", wrapper_factory(_BoomPlugin))
        result = run_command([
            "--plain", "dump", "--all", "--output-file", str(tmp_path / "failed.jsonl"),
        ])
        assert result.code == 0
        assert_no_leaked_tag_text(result)
        assert "  [1/2] fake-denoise: cannot load plugin" in result.err
        assert "  [2/2] fake-comp: cannot load plugin" in result.err

    def test_rich_batch_progress_row_is_visible(self, fake_registry, tmp_path, monkeypatch):
        from audioman.cli import dump

        monkeypatch.setattr(dump, "VST3PluginWrapper", wrapper_factory(_RecordingPlugin))
        target = tmp_path / "dumped.jsonl"
        result = run_command(["dump", "--all", "--output-file", str(target)])
        assert result.code == 0
        assert_no_leaked_tag_text(result)
        assert "[1/2] fake-denoise (4 params)" in result.out


class TestTagLeakHelper:
    """The guard itself catches real leaks (it is not a no-op that always passes)."""

    def test_detects_a_leaked_tag(self):
        assert find_leaked_tag_text("  [dim]Built-in analysis[/dim]") == ["[dim]", "[/dim]"]

    def test_ignores_untagged_bracket_text(self):
        assert find_leaked_tag_text("[dry-run] in.wav → [denoise]") == []

    def test_detects_the_visualize_leak_shape(self):
        """The line the old `--plain visualize` actually emitted."""
        line = "[dim]Built-in analysis: spectrogram (frame=999999, hop=512)[/dim]"
        assert find_leaked_tag_text(line) == ["[dim]", "[/dim]"]

    def test_plain_message_printer_is_clean(self):
        """Tags passed to print_info never survive on the plain path."""
        output.set_plain(True)
        with capture_streams() as (_out, err):
            output.print_info("[dim]Built-in analysis: rms[/dim]")
        assert find_leaked_tag_text(err.getvalue()) == []
        assert "Built-in analysis: rms" in err.getvalue()
