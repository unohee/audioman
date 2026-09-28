# tests/unit/cli_commands/test_plain_tag_leaks.py
# Purpose: `--plain` 출력에 rich 태그 텍스트가 새지 않는지 명령 단위로 막는다 (AUD-1853).
#
# `--plain` 콘솔은 `markup=False`이므로, `[dim]...[/dim]`처럼 태그가 붙은 문자열을
# `console.print`에 그대로 넘기면 태그가 글자 그대로 찍힌다. 이 파일은 대표 명령을
# 실제 파서로 실행해 stdout+stderr 어디에도 태그 텍스트가 없음을 확인하고,
# 태그가 아닌 대괄호 텍스트(`[dry-run]`, `[0, 1]`, `[1/2]`)는 남아 있음을 확인한다.

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
    """process.process_file 대체: VST3 없이 배치/단일 경로가 돌게 한다."""
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
    """체인 파싱을 고정해 플러그인 해석 없이 계획 경로만 관찰한다."""
    steps = [
        PipelineStep(plugin_name="dehum", params={"freq": 60.0}),
        PipelineStep(plugin_name="declick", params={}),
    ]
    monkeypatch.setattr(chain, "parse_chain_string", lambda raw: list(steps))
    return steps


class TestPlainCommandsEmitNoTagText:
    """대표 명령의 plain 출력에 태그 텍스트가 남지 않는다."""

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
        assert f"[dry-run] 배치: 1개 파일 → [denoise] → {tmp_path / 'out'}" in result.out

    def test_process_batch_failure_is_reported_with_a_clean_message(self, tmp_path, monkeypatch):
        """실패 경로 메시지에도 태그 텍스트가 없어야 한다."""
        def _boom(*args, **kwargs):
            raise ValueError("플러그인을 찾을 수 없습니다: 'ghost'")

        monkeypatch.setattr(process, "process_file", _boom)
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        result = run_command([
            "--plain", "process", str(in_dir), "-p", "ghost", "-o", str(tmp_path / "out"),
        ])
        assert result.code == 1
        assert_no_leaked_tag_text(result)
        assert "플러그인을 찾을 수 없습니다: 'ghost'" in result.err

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
        """JSON 모드는 stdout이 순수 JSON이어야 한다 (태그/마크업 무관)."""
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--plain", "--json", "chain", str(src), "-s", "dehum,declick",
            "-o", str(tmp_path / "out.wav"), "--dry-run",
        ])
        assert result.code == 0
        assert json.loads(result.out)["dry_run"] is True


class _RecordingPlugin(FakePlugin):
    """파라미터 attribute를 실제로 들고 있는 페이크 플러그인 (dump 경로용)."""

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
    """실패 메시지의 `[N/M]` 진행 표시는 plain 모드에서도 남아야 한다.

    예전 `_strip_markup`은 대괄호 그룹을 통째로 지워 `[1/2]` 같은 근거까지
    사라졌다. 이제는 실제 rich 태그만 지운다.
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
    """가드 자체가 실제 누수를 잡는지 (통과만 하는 상태가 아님)."""

    def test_detects_a_leaked_tag(self):
        assert find_leaked_tag_text("  [dim]내장 분석[/dim]") == ["[dim]", "[/dim]"]

    def test_ignores_untagged_bracket_text(self):
        assert find_leaked_tag_text("[dry-run] in.wav → [denoise]") == []

    def test_detects_the_visualize_leak_shape(self):
        """수정 전 `--plain visualize`가 실제로 내보내던 줄."""
        line = "[dim]내장 분석: spectrogram (frame=999999, hop=512)[/dim]"
        assert find_leaked_tag_text(line) == ["[dim]", "[/dim]"]

    def test_plain_message_printer_is_clean(self):
        """print_info에 태그를 넘겨도 plain에서는 태그가 남지 않는다."""
        output.set_plain(True)
        with capture_streams() as (_out, err):
            output.print_info("[dim]내장 분석: rms[/dim]")
        assert find_leaked_tag_text(err.getvalue()) == []
        assert "내장 분석: rms" in err.getvalue()
