# Created: 2026-09-28
# Purpose: coverage for cli/process.py - single/batch, dry-run, workers, exit codes (AUD-1851).
#
# process_file needs real VST3 plugins (core/engine) and cannot run on this host.
# These tests replace core.engine.process_file to observe only the CLI's batch
# loop / JSONL / exit-code contract, while the batch worker (_process_one) runs the
# real process_file and exercises the path where the registry cannot find the
# plugin (what actually happens on this machine).

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
    """Replace process.process_file with a recording fake."""
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
        assert stub_engine == []                       # dry-run never calls the engine
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

    def test_dry_run_tokens_survive_in_rich_mode(self, tmp_path, stub_engine):
        """The `[dry-run]`/`[denoise]` tokens must survive in rich mode too.

        Previously the sentence went straight to `output_console.print`, rich
        parsed the brackets as markup, and the tokens were deleted entirely:
        `[dry-run] /in.wav → [denoise] → /out.wav` became
        ` /in.wav →  → /out.wav` (AUD-1853: silent information loss).
        """
        src = write_wav(tmp_path / "in.wav")
        out = tmp_path / "out.wav"
        result = run_command([
            "process", str(src), "-p", "denoise", "-o", str(out), "--dry-run",
        ])
        assert result.code == 0
        assert f"[dry-run] {src} → [denoise] → {out}" in result.out

    def test_plain_dry_run_keeps_the_tokens_and_the_plan(self, tmp_path, stub_engine):
        """Plain mode reports the same information (only tags are stripped)."""
        src = write_wav(tmp_path / "in.wav")
        out = tmp_path / "out.wav"
        result = run_command([
            "--plain", "process", str(src), "-p", "denoise", "-o", str(out), "--dry-run",
        ])
        assert result.code == 0
        assert f"[dry-run] {src} → [denoise] → {out}" in result.out
        assert "[" not in result.err

    def test_rich_dry_run_reaches_the_same_facts(self, tmp_path, stub_engine):
        """The rich path also reports input/output/plugin information."""
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
        assert "Done" in result.err
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
            raise ValueError(f"Plugin not found: '{plugin_name}'")

        monkeypatch.setattr(process, "process_file", _raise)
        result = run_command([
            "--json", "process", str(src), "-p", "ghost", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "Plugin not found" in result.err
        assert result.out == ""

    def test_file_not_found_from_engine_exits_1(self, tmp_path, monkeypatch):
        src = write_wav(tmp_path / "in.wav")

        def _raise(input_path, output_path, plugin_name, params=None, passes=1):
            raise FileNotFoundError(f"File not found: {input_path}")

        monkeypatch.setattr(process, "process_file", _raise)
        result = run_command([
            "--json", "process", str(src), "-p", "p", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "File not found" in result.err

    def test_unexpected_engine_error_is_wrapped_with_prefix(self, tmp_path, monkeypatch):
        src = write_wav(tmp_path / "in.wav")

        def _raise(input_path, output_path, plugin_name, params=None, passes=1):
            raise RuntimeError("plugin exploded")

        monkeypatch.setattr(process, "process_file", _raise)
        result = run_command([
            "--json", "process", str(src), "-p", "p", "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "Processing failed: plugin exploded" in result.err

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
        assert "batch: 1 files" in result.out
        assert str(tmp_path / "out") in result.out

    def test_rich_batch_dry_run_still_reports_the_count(self, tmp_path, stub_engine):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        result = run_command([
            "process", str(in_dir), "-p", "denoise", "-o", str(tmp_path / "out"), "--dry-run",
        ])
        assert result.code == 0
        assert "batch: 1 files" in result.out


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
        assert "Batch complete: 2 succeeded, 0 failed / 2 total" in result.err

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
        assert "Batch complete: 0 succeeded, 1 failed / 1 total" in result.err

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
        assert "No audio files found" in result.err

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
    """`_process_one` turns exceptions into per-file failure results (they must not
    escape the pool)."""

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
