# Created: 2026-09-28
# Purpose: 배치 CLI 회귀 — chain 워커 ImportError, 병렬 JSONL 오염, 배치 실패 종료코드.
#
# VST3 플러그인 없이 돌아야 하므로 "해석 불가능한 플러그인 이름"을 쓴다:
# 이 환경에서 plugin lookup은 실패할 수밖에 없고(레지스트리 0개), 그래서
# 워커/배치 경로가 예외를 per-file 실패로 바꾸는지, 그리고 실패가
# 종료코드에 반영되는지를 관찰한다.

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audioman.cli.chain import _chain_one, _steps_from_dicts
from audioman.core.pipeline import PipelineStep

REPO_ROOT = Path(__file__).resolve().parents[2]
UNRESOLVABLE_PLUGIN = "audioman-test-nonexistent-plugin-xyz"


def _write_wav(path: Path, sample_rate: int = 44100, duration: float = 0.25) -> Path:
    """440Hz 사인파 스테레오 WAV (tests/conftest.py 스타일)."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), np.stack([mono, mono]).T, sample_rate, subtype="PCM_16")
    return path


def _run_cli(args: list[str], cache_dir: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # 사용자 홈에 캐시를 쓰지 않도록 격리 (레지스트리 스캔은 읽기 전용).
    env["AUDIOMAN_CACHE_DIR"] = str(cache_dir)
    return subprocess.run(
        [sys.executable, "-m", "audioman", *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def _json_lines(stdout: str) -> list[dict]:
    """stdout을 JSONL로 파싱. JSON이 아닌 줄이 하나라도 있으면 실패한다."""
    payloads = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        payloads.append(json.loads(line))
    return payloads


class TestChainWorkerResult:
    """_chain_one은 워커 프로세스에서 절대 예외를 새로 밖으로 던지지 않는다."""

    def test_unresolvable_plugin_becomes_per_file_failure(self, tmp_path, monkeypatch):
        """수정 전: 워커가 import 단계에서 ImportError로 즉사했다."""
        import audioman.core.pipeline as pipeline_mod

        class _EmptyRegistry:
            def get(self, name):
                return None

        monkeypatch.setattr(pipeline_mod, "get_registry", lambda: _EmptyRegistry())

        src = _write_wav(tmp_path / "in.wav")
        out = tmp_path / "out.wav"
        steps_dicts = PipelineStep(plugin_name=UNRESOLVABLE_PLUGIN, params={}).to_dict()

        result = _chain_one((str(src), str(out), [steps_dicts]))

        assert result["ok"] is False
        assert result["input"] == str(src)
        assert UNRESOLVABLE_PLUGIN in result["error"]
        assert not out.exists()

    def test_steps_from_dicts_round_trips_to_dict(self):
        """to_dict() -> 복원 -> 원본 동일. (`plugin` 키 vs 생성자 인자 불일치 회귀)"""
        steps = [
            PipelineStep(plugin_name="dehum", params={"freq": 60.0}),
            PipelineStep(plugin_name="declick", params={}),
        ]

        restored = _steps_from_dicts([s.to_dict() for s in steps])

        assert restored == steps
        assert [s.plugin_name for s in restored] == ["dehum", "declick"]


class TestChainJsonStdoutPurity:
    """--json 배치에서 stdout은 순수 JSONL이어야 한다 (진행바 미혼입)."""

    def _assert_pure_jsonl_and_no_crash(self, result: subprocess.CompletedProcess, expected_lines: int):
        # 워커 예외가 풀로 새면 traceback이 찍히고 JSONL이 깨진다.
        assert "Traceback" not in result.stderr, result.stderr
        assert "ImportError" not in result.stderr, result.stderr
        assert "ChainStep" not in result.stderr, result.stderr

        payloads = _json_lines(result.stdout)
        assert len(payloads) == expected_lines
        assert all(p["command"] == "chain" for p in payloads)
        return payloads

    def test_workers_2_completes_with_pure_jsonl(self, tmp_path):
        """AUD-1848 핵심 회귀: --workers 2 가 ImportError로 즉사하지 않는다."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_wav(in_dir / "a.wav")
        _write_wav(in_dir / "b.wav")

        result = _run_cli(
            ["--json", "chain", str(in_dir), "--steps", UNRESOLVABLE_PLUGIN,
             "-o", str(tmp_path / "out"), "--suffix", "_x", "--workers", "2"],
            tmp_path / "cache",
        )

        payloads = self._assert_pure_jsonl_and_no_crash(result, expected_lines=2)
        assert {Path(p["input"]).name for p in payloads} == {"a.wav", "b.wav"}
        assert all("error" in p for p in payloads)
        # 두 파일 모두 실패 -> 실패를 숨기지 않고 non-zero.
        assert result.returncode != 0

    def test_workers_1_completes_with_pure_jsonl(self, tmp_path):
        """순차 경로도 동일 계약을 지킨다."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_wav(in_dir / "a.wav")
        _write_wav(in_dir / "b.wav")

        result = _run_cli(
            ["--json", "chain", str(in_dir), "--steps", UNRESOLVABLE_PLUGIN,
             "-o", str(tmp_path / "out"), "--suffix", "_x", "--workers", "1"],
            tmp_path / "cache",
        )

        payloads = self._assert_pure_jsonl_and_no_crash(result, expected_lines=2)
        assert all("error" in p for p in payloads)
        assert result.returncode != 0


class TestChainBatchExitSemantics:
    """순차 배치의 종료코드 계약 (in-process).

    이 환경에서는 어떤 플러그인도 해석되지 않으므로(vst3 레지스트리 0개)
    실제 CLI로 exit 0 인 chain 배치를 만들 수 없다. 대신 배치 루프가
    카운터 -> 종료코드로 가는 경로를 직접 단정한다.
    """

    def _args(self, json_mode: bool):
        import argparse
        return argparse.Namespace(json=json_mode, workers=1)

    def test_all_success_does_not_exit(self, tmp_path, monkeypatch):
        import audioman.cli.chain as chain_cli
        import audioman.core.pipeline as pipeline_mod

        result = pipeline_mod.PipelineResult(
            input_path="in.wav", output_path="out.wav",
            steps=[{"plugin": "x", "params": {}}],
            input_stats={}, output_stats={}, duration_seconds=0.0,
        )
        monkeypatch.setattr(
            chain_cli, "run_pipeline",
            lambda input_path, output_path, steps: result,
        )

        src = _write_wav(tmp_path / "a.wav")
        jobs = [(str(src), str(tmp_path / "out.wav"), [{"plugin": "x", "params": {}}])]

        # SystemExit가 나면 테스트 실패 (성공 배치는 종료코드 0)
        chain_cli._run_chain_sequential(self._args(json_mode=False), jobs, [], 1)

    def test_any_failure_exits_1(self, tmp_path, monkeypatch):
        import audioman.cli.chain as chain_cli

        def _boom(input_path, output_path, steps):
            raise RuntimeError("plugin exploded")

        monkeypatch.setattr(chain_cli, "run_pipeline", _boom)

        src = _write_wav(tmp_path / "a.wav")
        jobs = [(str(src), str(tmp_path / "out.wav"), [{"plugin": "x", "params": {}}])]

        with pytest.raises(SystemExit) as excinfo:
            chain_cli._run_chain_sequential(self._args(json_mode=True), jobs, [], 1)

        assert excinfo.value.code == 1

    def test_json_mode_keeps_stdout_pure_jsonl(self, tmp_path, monkeypatch, capsys):
        import audioman.cli.chain as chain_cli

        def _boom(input_path, output_path, steps):
            raise RuntimeError("plugin exploded")

        monkeypatch.setattr(chain_cli, "run_pipeline", _boom)

        src = _write_wav(tmp_path / "a.wav")
        jobs = [(str(src), str(tmp_path / "out.wav"), [{"plugin": "x", "params": {}}])]

        with pytest.raises(SystemExit):
            chain_cli._run_chain_sequential(self._args(json_mode=True), jobs, [], 1)

        captured = capsys.readouterr()
        payloads = _json_lines(captured.out)
        assert len(payloads) == 1
        assert payloads[0]["error"] == "plugin exploded"


class TestBatchExitCodes:
    """실패한 파일이 하나라도 있으면 프로세스 종료코드는 non-zero."""

    def test_chain_batch_with_corrupt_file_exits_nonzero(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_wav(in_dir / "good.wav")
        (in_dir / "bad.wav").write_bytes(b"this is not a wav file")

        result = _run_cli(
            ["--json", "chain", str(in_dir), "--steps", UNRESOLVABLE_PLUGIN,
             "-o", str(tmp_path / "out"), "--workers", "2"],
            tmp_path / "cache",
        )

        payloads = _json_lines(result.stdout)
        assert len(payloads) == 2
        bad = [p for p in payloads if Path(p["input"]).name == "bad.wav"]
        assert bad and "error" in bad[0]
        assert result.returncode != 0

    def test_chain_dry_run_exits_zero(self, tmp_path):
        """실패 파일이 없으면 dry-run은 exit 0 (플러그인 해석 전에 끝난다)."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_wav(in_dir / "a.wav")
        _write_wav(in_dir / "b.wav")

        result = _run_cli(
            ["--json", "chain", str(in_dir), "--steps", UNRESOLVABLE_PLUGIN,
             "-o", str(tmp_path / "out"), "--dry-run"],
            tmp_path / "cache",
        )

        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert plan["dry_run"] is True
        assert plan["file_count"] == 2

    def test_analyze_batch_all_success_exits_zero(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_wav(in_dir / "a.wav")
        _write_wav(in_dir / "b.wav")

        result = _run_cli(
            ["--json", "analyze", str(in_dir)],
            tmp_path / "cache",
        )

        assert result.returncode == 0, result.stderr
        assert len(_json_lines(result.stdout)) == 2

    def test_analyze_batch_with_corrupt_file_exits_nonzero(self, tmp_path):
        """수정 전: 깨진 파일을 포함해도 exit 0 이었다."""
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        _write_wav(in_dir / "good.wav")
        (in_dir / "bad.wav").write_bytes(b"this is not a wav file")

        result = _run_cli(
            ["--json", "analyze", str(in_dir)],
            tmp_path / "cache",
        )

        payloads = _json_lines(result.stdout)
        errors = [p for p in payloads if "error" in p]
        assert len(errors) == 1
        assert Path(errors[0]["file"]).name == "bad.wav"
        assert result.returncode != 0
