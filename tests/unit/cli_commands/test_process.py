# Created: 2026-09-28
# Purpose: cli/process.py 커버리지 — 단일/배치, dry-run, workers, 종료코드 (AUD-1851).
#
# process_file은 실제 VST3를 요구하므로(core/engine) 이 호스트에서는 실행할 수
# 없다. 여기서는 core.engine.process_file을 대체해 CLI의 배치 루프/JSONL/종료코드
# 계약만 관찰하고, 배치 워커(_process_one)는 실제 process_file을 태우되 registry가
# 플러그인을 못 찾는 경로(이 환경의 실제 동작)를 검증한다.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audioman.cli import process
from audioman.core.engine import ProcessResult
from harness import FakePool, run_command, write_wav


def _result(input_path, output_path, plugin="fake-plugin", params=None, passes=1):
    return ProcessResult(
        input_path=str(input_path),
        output_path=str(output_path),
        plugin_name=plugin,
        params_applied=params or {},
        input_stats={"rms": 0.3, "peak": 0.5, "duration": 1.0, "frames": 44100},
        output_stats={"rms": 0.15, "peak": 0.25, "duration": 1.0, "frames": 44100},
        duration_seconds=0.25,
    )


@pytest.fixture
def stub_engine(monkeypatch):
    """process.process_file을 기록형 페이크로 교체."""
    calls = []

    def _process_file(input_path, output_path, plugin_name, params=None, passes=1):
        calls.append({"input": str(input_path), "output": str(output_path),
                      "plugin": plugin_name, "params": params, "passes": passes})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"RIFF")
        return _result(input_path, output_path, plugin=plugin_name, params=params)

    monkeypatch.setattr(process, "process_file", _process_file)
    return calls


class TestSingleDryRun:
    def test_json_plan_contains_inputs_and_params(self, tmp_path, stub_engine):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--json", "process", str(src), "-p", "denoise", "-o", str(tmp_path / "out.wav"),
            "--dry-run", "--param", "threshold=-20",
        ])
        assert result.code == 0
        plan = json.loads(result.out)
        assert plan["command"] == "process"
        assert plan["dry_run"] is True
        assert plan["input"] == str(src)
        assert plan["output"] == str(tmp_path / "out.wav")
        assert plan["plugin"] == "denoise"
        assert plan["params"] == {"threshold": -20.0}
        assert stub_engine == []                       # dry-run은 엔진을 부르지 않는다
        assert not (tmp_path / "out.wav").exists()

    def test_plain_dry_run_prints_plan(self, tmp_path, stub_engine):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--plain", "process", str(src), "-p", "denoise", "-o", str(tmp_path / "out.wav"),
            "--dry-run", "--param", "x=1",
        ])
        assert result.code == 0
        assert str(src) in result.out
        assert str(tmp_path / "out.wav") in result.out
        assert "params: {'x': 1.0}" in result.out

    def test_dry_run_brackets_are_eaten_by_rich_markup(self, tmp_path, stub_engine):
        """현재 동작 고정: `[dry-run]`/`[denoise]` 토큰은 출력에서 사라진다.

        process.py는 `output_console`을 import 시점에 값으로 바인딩하고, 그
        문장을 rich에 그대로 넘긴다. rich는 대괄호를 markup으로 해석하므로
        태그가 통째로 지워져 `[dry-run] /in.wav → [denoise] → /out.wav`가
        ` /in.wav →  → /out.wav`가 된다 (AUD-1853 계열: 조용한 정보 손실).
        수정되면 이 테스트가 깨지도록 남겨둔다.
        """
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "process", str(src), "-p", "denoise", "-o", str(tmp_path / "out.wav"), "--dry-run",
        ])
        assert result.code == 0
        assert "[dry-run]" not in result.out
        assert "[denoise]" not in result.out

    def test_rich_dry_run_reaches_the_same_facts(self, tmp_path, stub_engine):
        """rich 경로에서도 입력/출력/플러그인 정보가 나온다.

        `[dry-run]` / `[plugin]` 같은 대괄호 토큰은 rich가 markup으로 해석해
        사라진다(별도 이슈 AUD-1853 소관). 여기서는 사라지지 않는 부분만 단정한다.
        """
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "process", str(src), "-p", "denoise", "-o", str(tmp_path / "out.wav"), "--dry-run",
        ])
        assert result.code == 0
        assert str(src) in result.out
        assert str(tmp_path / "out.wav") in result.out


class TestSingleRun:
    def test_json_payload_reports_engine_result(self, tmp_path, stub_engine):
        src = write_wav(tmp_path / "in.wav")
        out = tmp_path / "out.wav"
        result = run_command([
            "--json", "process", str(src), "-p", "denoise", "-o", str(out), "--passes", "2",
        ])
        assert result.code == 0
        payload = json.loads(result.out)
        assert payload["command"] == "process"
        assert payload["plugin_name"] == "denoise"
        assert payload["input_path"] == str(src)
        assert payload["output_path"] == str(out)
        assert payload["input_stats"]["rms"] == 0.3
        assert payload["output_stats"]["rms"] == 0.15
        assert payload["duration_seconds"] == 0.25
        assert stub_engine[0]["passes"] == 2
        assert stub_engine[0]["input"] == str(src)

    def test_plain_output_prints_stats_and_completion(self, tmp_path, stub_engine):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--plain", "process", str(src), "-p", "denoise", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 0
        assert "Processing complete" in result.err
        assert "완료" in result.err
        assert "RMS:    0.3000 → 0.1500" in result.out
        assert "Peak:   0.5000 → 0.2500" in result.out

    def test_rich_output_path_runs(self, tmp_path, stub_engine):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "process", str(src), "-p", "denoise", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 0
        assert "Processing complete" in result.err
        assert "Plugin: denoise" in result.out

    def test_value_error_from_engine_exits_1_with_message(self, tmp_path, monkeypatch):
        src = write_wav(tmp_path / "in.wav")

        def _raise(input_path, output_path, plugin_name, params=None, passes=1):
            raise ValueError(f"플러그인을 찾을 수 없습니다: '{plugin_name}'")

        monkeypatch.setattr(process, "process_file", _raise)
        result = run_command([
            "--json", "process", str(src), "-p", "ghost", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "플러그인을 찾을 수 없습니다" in result.err
        assert result.out == ""

    def test_file_not_found_from_engine_exits_1(self, tmp_path, monkeypatch):
        src = write_wav(tmp_path / "in.wav")

        def _raise(input_path, output_path, plugin_name, params=None, passes=1):
            raise FileNotFoundError(f"파일 없음: {input_path}")

        monkeypatch.setattr(process, "process_file", _raise)
        result = run_command([
            "--json", "process", str(src), "-p", "p", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "파일 없음" in result.err

    def test_unexpected_engine_error_is_wrapped_with_prefix(self, tmp_path, monkeypatch):
        src = write_wav(tmp_path / "in.wav")

        def _raise(input_path, output_path, plugin_name, params=None, passes=1):
            raise RuntimeError("plugin exploded")

        monkeypatch.setattr(process, "process_file", _raise)
        result = run_command([
            "--json", "process", str(src), "-p", "p", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "처리 실패: plugin exploded" in result.err

    def test_missing_required_arguments_exit_2(self, tmp_path):
        src = write_wav(tmp_path / "in.wav")
        result = run_command(["--json", "process", str(src), "-o", str(tmp_path / "o.wav")])
        assert result.code == 2
        assert "--plugin" in result.err


class TestBatchDryRun:
    def test_json_batch_plan_lists_files(self, tmp_path, stub_engine):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        write_wav(in_dir / "b.wav")

        result = run_command([
            "--json", "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"),
            "--dry-run",
        ])
        assert result.code == 0
        plan = json.loads(result.out)
        assert plan["batch"] is True
        assert plan["file_count"] == 2
        assert plan["input_dir"] == str(in_dir)
        assert plan["output_dir"] == str(tmp_path / "out")
        assert plan["files"] == [str(in_dir / "a.wav"), str(in_dir / "b.wav")]

    def test_plain_batch_dry_run_prints_count(self, tmp_path, stub_engine):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        result = run_command([
            "--plain", "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"),
            "--dry-run",
        ])
        assert result.code == 0
        assert "배치: 1개 파일" in result.out
        assert str(tmp_path / "out") in result.out

    def test_rich_batch_dry_run_still_reports_the_count(self, tmp_path, stub_engine):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        result = run_command([
            "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"), "--dry-run",
        ])
        assert result.code == 0
        assert "배치: 1개 파일" in result.out


class TestBatchRun:
    def test_sequential_json_emits_one_payload_per_file(self, tmp_path, stub_engine):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        write_wav(in_dir / "b.wav")
        out_dir = tmp_path / "out"

        result = run_command([
            "--json", "process", str(in_dir), "-p", "denoise", "-o", str(out_dir), "--suffix", "_p",
        ])
        assert result.code == 0
        payloads = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert len(payloads) == 2
        assert all(p["command"] == "process" for p in payloads)
        assert {Path(p["output_path"]).name for p in payloads} == {"a_p.wav", "b_p.wav"}

    def test_sequential_plain_reports_success_count(self, tmp_path, stub_engine):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        write_wav(in_dir / "b.wav")

        result = run_command([
            "--plain", "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"),
        ])
        assert result.code == 0
        assert "배치 완료: 2 성공, 0 실패 / 2 전체" in result.err

    def test_parallel_json_uses_pool_and_emits_payloads(self, tmp_path, stub_engine, monkeypatch):
        import multiprocessing

        monkeypatch.setattr(multiprocessing, "Pool", FakePool)
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        write_wav(in_dir / "b.wav")

        result = run_command([
            "--json", "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"),
            "--workers", "2",
        ])
        assert result.code == 0
        payloads = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert len(payloads) == 2
        assert all(p["command"] == "process" for p in payloads)

    def test_parallel_plain_reports_worker_count(self, tmp_path, stub_engine, monkeypatch):
        import multiprocessing

        monkeypatch.setattr(multiprocessing, "Pool", FakePool)
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        result = run_command([
            "--plain", "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"),
            "--workers", "3",
        ])
        assert result.code == 0
        assert "(3 workers)" in result.err

    def test_sequential_failure_exits_1(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        def _raise(input_path, output_path, plugin_name, params=None, passes=1):
            raise RuntimeError("plugin exploded")

        monkeypatch.setattr(process, "process_file", _raise)
        result = run_command([
            "--json", "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"),
        ])
        assert result.code == 1
        payload = json.loads(result.out.strip())
        assert payload["error"] == "plugin exploded"
        assert payload["input"] == str(in_dir / "a.wav")

    def test_sequential_plain_reports_warning_per_file(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        def _raise(input_path, output_path, plugin_name, params=None, passes=1):
            raise RuntimeError("bad plugin")

        monkeypatch.setattr(process, "process_file", _raise)
        result = run_command([
            "--plain", "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"),
        ])
        assert result.code == 1
        assert "warning:" in result.err and "bad plugin" in result.err
        assert "배치 완료: 0 성공, 1 실패 / 1 전체" in result.err

    def test_parallel_failure_exits_1(self, tmp_path, monkeypatch):
        import multiprocessing

        monkeypatch.setattr(multiprocessing, "Pool", FakePool)
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        def _raise(input_path, output_path, plugin_name, params=None, passes=1):
            raise RuntimeError("parallel boom")

        monkeypatch.setattr(process, "process_file", _raise)
        result = run_command([
            "--json", "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"),
            "--workers", "2",
        ])
        assert result.code == 1
        payload = json.loads(result.out.strip())
        assert payload["error"] == "parallel boom"

    def test_empty_directory_exits_1(self, tmp_path, stub_engine):
        in_dir = tmp_path / "empty"
        in_dir.mkdir()
        result = run_command([
            "--plain", "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"),
        ])
        assert result.code == 1
        assert "오디오 파일이 없습니다" in result.err

    def test_recursive_flag_includes_subdirectories(self, tmp_path, stub_engine):
        in_dir = tmp_path / "in"
        (in_dir / "nested").mkdir(parents=True)
        write_wav(in_dir / "top.wav")
        write_wav(in_dir / "nested" / "deep.wav")

        result = run_command([
            "--json", "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"),
            "--recursive",
        ])
        assert result.code == 0
        payloads = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert len(payloads) == 2


class TestProcessWorker:
    """_process_one은 예외를 per-file 실패 결과로 바꾼다 (풀 밖으로 새면 안 된다)."""

    def test_success_result_shape(self, tmp_path, stub_engine):
        src = write_wav(tmp_path / "in.wav")
        out = tmp_path / "out.wav"
        r = process._process_one((src, out, "denoise", {"x": 1.0}, 1))
        assert r["ok"] is True
        assert r["input"] == str(src)
        assert r["output"] == str(out)
        assert r["result"]["plugin_name"] == "denoise"

    def test_failure_becomes_ok_false(self, tmp_path):
        src = write_wav(tmp_path / "in.wav")
        r = process._process_one((src, tmp_path / "o.wav", "unresolvable-plugin-xyz", {}, 1))
        assert r["ok"] is False
        assert r["input"] == str(src)
        assert "unresolvable-plugin-xyz" in r["error"]


class TestArgValidation:
    @pytest.mark.parametrize("flag", ["--passes", "--workers"])
    def test_non_positive_int_rejected(self, tmp_path, flag):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--json", "process", str(src), "-p", "p", "-o", str(tmp_path / "o.wav"), flag, "0",
        ])
        assert result.code == 2
        assert "must be a positive integer" in result.err
