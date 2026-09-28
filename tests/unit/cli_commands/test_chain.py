# Created: 2026-09-28
# Purpose: coverage for cli/chain.py - single/batch, dry-run, sequential/parallel, exit codes (AUD-1851).
#
# run_pipeline needs real VST3 plugins, so these tests replace
# core.pipeline.run_pipeline and observe only the CLI's plan/loop/JSONL/exit-code
# contract. The worker (_chain_one) runs the real run_pipeline and exercises the
# path where plugin resolution fails on this host.

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
    """Replace chain.run_pipeline with a recording fake."""
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
        assert "Processing chain is empty" in result.err
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
        """One line per step plus the output line, with plugin/parameter text intact.

        What rich markup would swallow is pinned by a separate test.
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
        assert "Done" in result.err
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
            raise ValueError("Plugin not found: 'ghost' (step 1)")

        monkeypatch.setattr(chain, "run_pipeline", _raise)
        result = run_command([
            "--json", "chain", str(src), "-s", "ghost", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "Plugin not found: 'ghost' (step 1)" in result.err

    def test_file_not_found_from_pipeline_exits_1(self, tmp_path, monkeypatch):
        src = write_wav(tmp_path / "in.wav")

        def _raise(input_path, output_path, steps):
            raise FileNotFoundError("File not found: missing.wav")

        monkeypatch.setattr(chain, "run_pipeline", _raise)
        result = run_command([
            "--json", "chain", str(src), "-s", "dehum", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "File not found: missing.wav" in result.err

    def test_unexpected_error_is_wrapped(self, tmp_path, monkeypatch):
        src = write_wav(tmp_path / "in.wav")

        def _raise(input_path, output_path, steps):
            raise RuntimeError("chain exploded")

        monkeypatch.setattr(chain, "run_pipeline", _raise)
        result = run_command([
            "--json", "chain", str(src), "-s", "dehum", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "Chain processing failed: chain exploded" in result.err


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
        assert "batch: 1 files" in result.out
        assert str(tmp_path / "out") in result.out

    def test_step_names_survive_in_the_plan_lines(self, tmp_path, stub_pipeline):
        """The `[dehum (…)]` bracket tokens must survive in the plan lines.

        Previously the sentence went straight to `output_console.print`, rich
        parsed the brackets as markup, and the plugin names vanished, leaving
        only the arrows (`  → `, `  → `). Silent information loss, AUD-1853.
        """
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "chain", str(src), "-s", "dehum:freq=60,declick",
            "-o", str(tmp_path / "out.wav"), "--dry-run",
        ])
        assert result.code == 0
        assert f"[dry-run] {src}" in result.out
        assert "  → [dehum ({'freq': 60.0})]" in result.out
        assert "  → [declick]" in result.out

    def test_plain_plan_lines_keep_the_bracket_tokens(self, tmp_path, stub_pipeline):
        """Plain mode emits the same plan lines."""
        src = write_wav(tmp_path / "in.wav")
        result = run_command([
            "--plain", "chain", str(src), "-s", "dehum:freq=60,declick",
            "-o", str(tmp_path / "out.wav"), "--dry-run",
        ])
        assert result.code == 0
        assert f"[dry-run] {src}" in result.out
        assert "  → [dehum ({'freq': 60.0})]" in result.out
        assert "  → [declick]" in result.out
        assert "[" not in result.err

    def test_batch_plan_line_keeps_the_step_names(self, tmp_path, stub_pipeline):
        """The `[step → step]` token in the batch plan must survive too."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        out_dir = tmp_path / "out"

        result = run_command([
            "chain", str(in_dir), "-s", "dehum,declick", "-o", str(out_dir), "--dry-run",
        ])
        assert result.code == 0
        assert f"[dry-run] batch: 1 files → [dehum → declick] → {out_dir}" in result.out


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
        assert "Batch complete: 2 succeeded, 0 failed / 2 total" in result.err

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
        assert "Batch complete: 0 succeeded, 1 failed / 1 total" in result.err

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
        assert "No audio files found" in result.err

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
    """The worker turns exceptions into per-file failures and round-trips step dicts."""

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
        """Non-dict steps must not leak an exception past the worker."""
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
