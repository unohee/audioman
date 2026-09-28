# Created: 2026-09-28
# Purpose: cli/chain.py 커버리지 — 단일/배치, dry-run, 순차/병렬, 종료코드 (AUD-1851).
#
# run_pipeline은 실제 VST3를 요구하므로 여기서는 core.pipeline.run_pipeline을
# 대체하고 CLI의 계획/루프/JSONL/종료코드만 관찰한다. 워커(_chain_one)는 실제
# run_pipeline을 태우되 이 호스트에서 플러그인이 해석되지 않는 경로를 검증한다.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audioman.cli import chain
from audioman.core.pipeline import PipelineResult, PipelineStep
from harness import FakePool, run_command, write_wav


def _pipeline_result(input_path, output_path, steps):
    return PipelineResult(
        input_path=str(input_path),
        output_path=str(output_path),
        steps=[s.to_dict() for s in steps],
        input_stats={"rms": 0.3, "peak": 0.5},
        output_stats={"rms": 0.2, "peak": 0.4},
        duration_seconds=0.5,
    )


@pytest.fixture
def stub_pipeline(monkeypatch):
    """chain.run_pipeline을 기록형 페이크로 교체."""
    calls: list[dict] = []

    def _run_pipeline(input_path, output_path, steps):
        calls.append({"input": str(input_path), "output": str(output_path),
                      "steps": [(s.plugin_name, dict(s.params)) for s in steps]})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"RIFF")
        return _pipeline_result(input_path, output_path, steps)

    monkeypatch.setattr(chain, "run_pipeline", _run_pipeline)
    return calls


class TestChainStringParsing:
    def test_empty_chain_exits_1(self, tmp_path, stub_pipeline):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--json", "chain", str(src), "-s", ", ,", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "처리 단계가 비어있습니다" in result.err
        assert stub_pipeline == []

    def test_parses_plugins_and_params(self, tmp_path, stub_pipeline):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--json", "chain", str(src),
            "-s", "dehum:freq=60;q=0.7,declick,denoise:threshold=-20",
            "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 0
        assert stub_pipeline[0]["steps"] == [
            ("dehum", {"freq": 60.0, "q": 0.7}),
            ("declick", {}),
            ("denoise", {"threshold": -20.0}),
        ]


class TestSingleDryRun:
    def test_json_plan_lists_steps(self, tmp_path, stub_pipeline):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--json", "chain", str(src), "-s", "dehum:freq=60,declick",
            "-o", str(tmp_path / "out.wav"), "--dry-run",
        ])
        assert result.code == 0
        plan = json.loads(result.out)
        assert plan["command"] == "chain"
        assert plan["dry_run"] is True
        assert plan["steps"] == [
            {"plugin": "dehum", "params": {"freq": 60.0}},
            {"plugin": "declick", "params": {}},
        ]
        assert stub_pipeline == []
        assert not (tmp_path / "out.wav").exists()

    def test_plain_plan_prints_each_step_with_params(self, tmp_path, stub_pipeline):
        """step당 한 줄 + 출력 줄. 플러그인/파라미터 텍스트는 아직 살아 있다.

        rich 마크업에 먹히는 부분은 별도 테스트에서 고정한다.
        """
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--plain", "chain", str(src), "-s", "dehum:freq=60,declick",
            "-o", str(tmp_path / "out.wav"), "--dry-run",
        ])
        assert result.code == 0
        assert str(src) in result.out
        assert str(tmp_path / "out.wav") in result.out
        assert result.out.count("→") == 3          # two steps + output


class TestSingleRun:
    def test_json_payload_describes_pipeline_result(self, tmp_path, stub_pipeline):
        src = write_wav(tmp_path / "in.wav")
        out = tmp_path / "out.wav"
        result = run_command([
            "--json", "chain", str(src), "-s", "dehum,declick", "-o", str(out),
        ])
        assert result.code == 0
        payload = json.loads(result.out)
        assert payload["command"] == "chain"
        assert payload["input_path"] == str(src)
        assert payload["output_path"] == str(out)
        assert [s["plugin"] for s in payload["steps"]] == ["dehum", "declick"]
        assert payload["duration_seconds"] == 0.5

    def test_plain_output_lists_steps_and_files(self, tmp_path, stub_pipeline):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--plain", "chain", str(src), "-s", "dehum,declick", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 0
        assert "Chain complete" in result.err
        assert "완료" in result.err
        assert "Steps: 2" in result.out
        assert "1. dehum" in result.out
        assert "2. declick" in result.out

    def test_rich_output_path_runs(self, tmp_path, stub_pipeline):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "chain", str(src), "-s", "dehum", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 0
        assert "Chain complete" in result.err
        assert "Output:" in result.out

    def test_value_error_from_pipeline_exits_1(self, tmp_path, monkeypatch):
        src = write_wav(tmp_path / "in.wav")

        def _raise(input_path, output_path, steps):
            raise ValueError("플러그인을 찾을 수 없습니다: 'ghost' (step 1)")

        monkeypatch.setattr(chain, "run_pipeline", _raise)
        result = run_command([
            "--json", "chain", str(src), "-s", "ghost", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "플러그인을 찾을 수 없습니다: 'ghost' (step 1)" in result.err

    def test_file_not_found_from_pipeline_exits_1(self, tmp_path, monkeypatch):
        src = write_wav(tmp_path / "in.wav")

        def _raise(input_path, output_path, steps):
            raise FileNotFoundError("파일 없음: missing.wav")

        monkeypatch.setattr(chain, "run_pipeline", _raise)
        result = run_command([
            "--json", "chain", str(src), "-s", "dehum", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "파일 없음: missing.wav" in result.err

    def test_unexpected_error_is_wrapped(self, tmp_path, monkeypatch):
        src = write_wav(tmp_path / "in.wav")

        def _raise(input_path, output_path, steps):
            raise RuntimeError("chain exploded")

        monkeypatch.setattr(chain, "run_pipeline", _raise)
        result = run_command([
            "--json", "chain", str(src), "-s", "dehum", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "체인 처리 실패: chain exploded" in result.err


class TestBatchDryRun:
    def test_json_batch_plan_lists_files_and_output_dir(self, tmp_path, stub_pipeline):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        write_wav(in_dir / "b.wav")

        result = run_command([
            "--json", "chain", str(in_dir), "-s", "dehum,declick",
            "-o", str(tmp_path / "out"), "--dry-run",
        ])
        assert result.code == 0
        plan = json.loads(result.out)
        assert plan["batch"] is True
        assert plan["file_count"] == 2
        assert plan["input_dir"] == str(in_dir)
        assert plan["output_dir"] == str(tmp_path / "out")
        assert plan["files"] == [str(in_dir / "a.wav"), str(in_dir / "b.wav")]

    def test_plain_batch_plan_prints_file_count_and_output_dir(self, tmp_path, stub_pipeline):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        result = run_command([
            "--plain", "chain", str(in_dir), "-s", "dehum,declick",
            "-o", str(tmp_path / "out"), "--dry-run",
        ])
        assert result.code == 0
        assert "배치: 1개 파일" in result.out
        assert str(tmp_path / "out") in result.out

    def test_step_names_are_dropped_by_rich_markup_in_plan_lines(self, tmp_path, stub_pipeline):
        """현재 동작 고정: `[dehum]` / `[dehum (…)]` 대괄호 토큰이 사라진다.

        chain.py는 `output_console`을 import하고 rich가 대괄호를 markup으로
        해석한다. 그 결과 dry-run 계획에서 플러그인 이름이 통째로 지워지고
        화살표만 남는다 (`  → `, `  → `). AUD-1853 계열의 조용한 정보 손실.
        수정되면 이 테스트가 깨지도록 남겨둔다.
        """
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "chain", str(src), "-s", "dehum:freq=60,declick",
            "-o", str(tmp_path / "out.wav"), "--dry-run",
        ])
        assert result.code == 0
        assert "[dehum" not in result.out
        assert "dehum" not in result.out
        assert "{'freq': 60.0}" not in result.out


class TestBatchRun:
    def test_sequential_json_emits_one_payload_per_file(self, tmp_path, stub_pipeline):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        write_wav(in_dir / "b.wav")
        out_dir = tmp_path / "out"

        result = run_command([
            "--json", "chain", str(in_dir), "-s", "dehum", "-o", str(out_dir), "--suffix", "_c",
        ])
        assert result.code == 0
        payloads = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert len(payloads) == 2
        assert all(p["command"] == "chain" for p in payloads)
        assert {Path(p["output_path"]).name for p in payloads} == {"a_c.wav", "b_c.wav"}

    def test_sequential_plain_reports_counts(self, tmp_path, stub_pipeline):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        write_wav(in_dir / "b.wav")

        result = run_command([
            "--plain", "chain", str(in_dir), "-s", "dehum", "-o", str(tmp_path / "out"),
        ])
        assert result.code == 0
        assert "배치 완료: 2 성공, 0 실패 / 2 전체" in result.err

    def test_parallel_json_uses_pool(self, tmp_path, stub_pipeline, monkeypatch):
        import multiprocessing

        monkeypatch.setattr(multiprocessing, "Pool", FakePool)
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        write_wav(in_dir / "b.wav")

        result = run_command([
            "--json", "chain", str(in_dir), "-s", "dehum", "-o", str(tmp_path / "out"),
            "--workers", "2",
        ])
        assert result.code == 0
        payloads = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert len(payloads) == 2
        assert all(p["command"] == "chain" for p in payloads)

    def test_parallel_plain_reports_worker_count(self, tmp_path, stub_pipeline, monkeypatch):
        import multiprocessing

        monkeypatch.setattr(multiprocessing, "Pool", FakePool)
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        result = run_command([
            "--plain", "chain", str(in_dir), "-s", "dehum", "-o", str(tmp_path / "out"),
            "--workers", "4",
        ])
        assert result.code == 0
        assert "(4 workers)" in result.err

    def test_sequential_failure_exits_1_with_error_payload(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        def _raise(input_path, output_path, steps):
            raise RuntimeError("chain exploded")

        monkeypatch.setattr(chain, "run_pipeline", _raise)
        result = run_command([
            "--json", "chain", str(in_dir), "-s", "dehum", "-o", str(tmp_path / "out"),
        ])
        assert result.code == 1
        payload = json.loads(result.out.strip())
        assert payload["error"] == "chain exploded"
        assert payload["input"] == str(in_dir / "a.wav")

    def test_sequential_plain_warns_per_failing_file(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        def _raise(input_path, output_path, steps):
            raise RuntimeError("bad chain")

        monkeypatch.setattr(chain, "run_pipeline", _raise)
        result = run_command([
            "--plain", "chain", str(in_dir), "-s", "dehum", "-o", str(tmp_path / "out"),
        ])
        assert result.code == 1
        assert "warning:" in result.err and "bad chain" in result.err
        assert "배치 완료: 0 성공, 1 실패 / 1 전체" in result.err

    def test_parallel_failure_exits_1(self, tmp_path, monkeypatch):
        import multiprocessing

        monkeypatch.setattr(multiprocessing, "Pool", FakePool)
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        def _raise(input_path, output_path, steps):
            raise RuntimeError("parallel chain boom")

        monkeypatch.setattr(chain, "run_pipeline", _raise)
        result = run_command([
            "--json", "chain", str(in_dir), "-s", "dehum", "-o", str(tmp_path / "out"),
            "--workers", "2",
        ])
        assert result.code == 1
        assert json.loads(result.out.strip())["error"] == "parallel chain boom"

    def test_empty_directory_exits_1(self, tmp_path, stub_pipeline):
        in_dir = tmp_path / "empty"
        in_dir.mkdir()
        result = run_command([
            "--plain", "chain", str(in_dir), "-s", "dehum", "-o", str(tmp_path / "out"),
        ])
        assert result.code == 1
        assert "오디오 파일이 없습니다" in result.err

    def test_recursive_flag_includes_nested_files(self, tmp_path, stub_pipeline):
        in_dir = tmp_path / "in"
        (in_dir / "nested").mkdir(parents=True)
        write_wav(in_dir / "top.wav")
        write_wav(in_dir / "nested" / "deep.wav")

        result = run_command([
            "--json", "chain", str(in_dir), "-s", "dehum", "-o", str(tmp_path / "out"),
            "--recursive",
        ])
        assert result.code == 0
        payloads = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert len(payloads) == 2


class TestWorkerAndStepRebuild:
    """워커는 예외를 per-file 실패 결과로 바꾸고, step dict는 라운드트립한다."""

    def test_steps_from_dicts_uses_plugin_key(self):
        steps = chain._steps_from_dicts([
            {"plugin": "dehum", "params": {"freq": 60.0}},
            {"plugin": "declick", "params": {}},
        ])
        assert steps == [
            PipelineStep(plugin_name="dehum", params={"freq": 60.0}),
            PipelineStep(plugin_name="declick", params={}),
        ]

    def test_steps_from_dicts_accepts_plugin_name_key_and_missing_params(self):
        steps = chain._steps_from_dicts([
            {"plugin_name": "legacy", "params": None},
            {"plugin": "plain"},
        ])
        assert steps[0] == PipelineStep(plugin_name="legacy", params={})
        assert steps[1] == PipelineStep(plugin_name="plain", params={})

    def test_worker_success_shape(self, tmp_path, stub_pipeline):
        src = write_wav(tmp_path / "in.wav")
        out = tmp_path / "out.wav"
        r = chain._chain_one((str(src), str(out), [{"plugin": "dehum", "params": {"freq": 60.0}}]))
        assert r["ok"] is True
        assert r["input"] == str(src)
        assert r["result"]["steps"] == [{"plugin": "dehum", "params": {"freq": 60.0}}]

    def test_worker_failure_from_pipeline(self, tmp_path, monkeypatch):
        src = write_wav(tmp_path / "in.wav")

        def _raise(input_path, output_path, steps):
            raise RuntimeError("worker boom")

        monkeypatch.setattr(chain, "run_pipeline", _raise)
        r = chain._chain_one((str(src), str(tmp_path / "o.wav"), [{"plugin": "x", "params": {}}]))
        assert r["ok"] is False
        assert r["error"] == "worker boom"
        assert r["input"] == str(src)

    def test_worker_reports_step_rebuild_errors_as_file_failures(self, tmp_path):
        """dict가 아닌 steps는 워커 밖으로 예외를 새게 하지 않는다."""
        r = chain._chain_one((str(tmp_path / "in.wav"), str(tmp_path / "o.wav"), ["not-a-dict"]))
        assert r["ok"] is False
        assert r["input"] == str(tmp_path / "in.wav")


class TestArgValidation:
    @pytest.mark.parametrize("flag", ["--workers"])
    def test_non_positive_int_rejected(self, tmp_path, flag):
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--json", "chain", str(src), "-s", "dehum", "-o", str(tmp_path / "o.wav"), flag, "0",
        ])
        assert result.code == 2
        assert "must be a positive integer" in result.err

    def test_steps_and_output_are_required(self, tmp_path):
        src = write_wav(tmp_path / "in.wav")
        result = run_command(["--json", "chain", str(src)])
        assert result.code == 2
        assert "--steps" in result.err
        assert "--output" in result.err
