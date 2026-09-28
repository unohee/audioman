# Created: 2026-09-28
# Purpose: cli/dump.py 커버리지 — 단일 플러그인 JSON, --all JSONL, --preset,
#          --save-preset, --output-file (AUD-1851).
#
# 이 호스트에는 실제 VST3가 없으므로(AUD-1857) 래퍼를 페이크 플러그인으로
# 대체한다. 검증 대상은 CLI의 상태 추출/기록 로직과 파일 출력이다.

from __future__ import annotations

import json

import pytest

from audioman.cli import dump
from harness import FakePlugin, run_command, wrapper_factory


class RecordingPlugin(FakePlugin):
    """파라미터 attribute를 실제로 들고 있는 페이크 플러그인."""

    def __init__(self):
        super().__init__(
            threshold=-20.0, mode="fast", bypass=False, label=object(),
        )
        self.parameters = {
            "threshold": object(),
            "mode": object(),
            "bypass": object(),
            "label": object(),
        }


class ExplodingAttribute(RecordingPlugin):
    def __getattr__(self, item):
        if item == "mode":
            raise RuntimeError("attribute read failed")
        return super().__getattr__(item)


@pytest.fixture
def stub_wrapper(monkeypatch):
    """dump.VST3PluginWrapper를 페이크로 교체."""
    monkeypatch.setattr(dump, "VST3PluginWrapper", wrapper_factory(RecordingPlugin))


class TestSingleDump:
    def test_json_payload_serialises_parameter_values(self, fake_registry, stub_wrapper):
        result = run_command(["--json", "dump", "fake-denoise"])
        assert result.code == 0
        payload = json.loads(result.out)
        assert payload["command"] == "dump"
        assert payload["plugin"] == "Fake De-noise"
        assert payload["short_name"] == "fake-denoise"
        assert payload["path"] == "/nonexistent/Fake De-noise.vst3"
        assert payload["format"] == "vst3"
        assert payload["identifier"] == "com.fake.plugin"
        assert payload["version"] == "9.9"
        assert payload["parameter_count"] == 4
        assert payload["parameters"]["threshold"] == -20.0
        assert payload["parameters"]["mode"] == "fast"
        assert payload["parameters"]["bypass"] is False
        assert isinstance(payload["parameters"]["label"], str)

    def test_missing_attribute_becomes_none(self, fake_registry, monkeypatch):
        monkeypatch.setattr(dump, "VST3PluginWrapper", wrapper_factory(ExplodingAttribute))
        result = run_command(["--json", "dump", "fake-denoise"])
        assert result.code == 0
        assert json.loads(result.out)["parameters"]["mode"] is None

    def test_alias_resolves(self, fake_registry, stub_wrapper):
        result = run_command(["--json", "dump", "denoise"])
        assert result.code == 0
        assert json.loads(result.out)["short_name"] == "fake-denoise"

    def test_unknown_plugin_exits_1(self, fake_registry, stub_wrapper):
        result = run_command(["--json", "dump", "nope"])
        assert result.code == 1
        assert "플러그인을 찾을 수 없습니다" in result.err
        assert result.out == ""

    def test_neither_plugin_nor_all_exits_1(self, fake_registry):
        result = run_command(["--json", "dump"])
        assert result.code == 1
        assert "플러그인 이름 또는 --all" in result.err

    def test_cli_params_are_applied_to_the_wrapper(self, fake_registry, stub_wrapper):
        result = run_command([
            "--json", "dump", "fake-denoise", "--param", "threshold=-30", "--param", "mode=slow",
        ])
        assert result.code == 0
        assert FakeWrapperLast().applied == [{"threshold": -30.0, "mode": "slow"}]


def FakeWrapperLast():
    """dump가 마지막으로 만든 페이크 래퍼 (파라미터 적용 확인용)."""
    from harness import FakeWrapper

    assert FakeWrapper.instances, "no wrapper was constructed"
    return FakeWrapper.instances[-1]


class TestPresetIntegration:
    def test_preset_is_loaded_and_applied(self, fake_registry, stub_wrapper, tmp_path):
        # 실제 프리셋을 먼저 저장한 뒤 dump가 그것을 읽어 적용하는지 본다.
        preset_dir = tmp_path / "presets" / "fake-denoise"
        preset_dir.mkdir(parents=True)
        (preset_dir / "vocal.json").write_text(json.dumps({
            "name": "vocal", "plugin": "fake-denoise",
            "parameters": {"threshold": -12.0}, "description": "", "created": "",
        }))

        result = run_command(["--json", "dump", "fake-denoise", "--preset", "vocal"])
        assert result.code == 0
        assert FakeWrapperLast().applied == [{"threshold": -12.0}]

    def test_missing_preset_exits_1(self, fake_registry, stub_wrapper):
        result = run_command(["--json", "dump", "fake-denoise", "--preset", "ghost"])
        assert result.code == 1
        assert "프리셋을 찾을 수 없습니다" in result.err

    def test_preset_and_param_are_applied_in_order(self, fake_registry, stub_wrapper, tmp_path):
        preset_dir = tmp_path / "presets" / "fake-denoise"
        preset_dir.mkdir(parents=True)
        (preset_dir / "base.json").write_text(json.dumps({
            "name": "base", "plugin": "fake-denoise",
            "parameters": {"threshold": -12.0}, "description": "", "created": "",
        }))

        result = run_command([
            "--json", "dump", "fake-denoise", "--preset", "base", "--param", "mode=slow",
        ])
        assert result.code == 0
        assert FakeWrapperLast().applied == [{"threshold": -12.0}, {"mode": "slow"}]

    def test_save_preset_writes_file_and_reports_name(self, fake_registry, stub_wrapper, tmp_path):
        out_preset = tmp_path / "presets" / "fake-denoise" / "snapshot.json"
        result = run_command([
            "--json", "dump", "fake-denoise", "--save-preset", "snapshot",
        ])
        assert result.code == 0
        payload = json.loads(result.out)
        assert payload["saved_as_preset"] == "snapshot"

        assert out_preset.exists()
        saved = json.loads(out_preset.read_text())
        assert saved["name"] == "snapshot"
        assert saved["plugin"] == "fake-denoise"
        assert saved["parameters"]["threshold"] == -20.0
        assert saved["description"] == "dump from Fake De-noise"


class TestBatchDump:
    def test_all_writes_jsonl_to_stdout(self, fake_registry, stub_wrapper):
        result = run_command(["--json", "dump", "--all"])
        assert result.code == 0
        lines = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert len(lines) == 2
        assert all(line["batch"] is True for line in lines)
        assert {line["short_name"] for line in lines} == {"fake-denoise", "fake-comp"}
        assert all(line["command"] == "dump" for line in lines)

    def test_format_filter_selects_one_plugin(self, fake_registry, stub_wrapper):
        result = run_command(["--json", "dump", "--all", "--format-filter", "au"])
        lines = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert [line["short_name"] for line in lines] == ["fake-comp"]

    def test_keyword_filter_matches_name(self, fake_registry, stub_wrapper):
        result = run_command(["--json", "dump", "--all", "--filter", "DENOISE"])
        lines = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert [line["short_name"] for line in lines] == ["fake-denoise"]

    def test_no_plugin_matches_filter_exits_1(self, fake_registry, stub_wrapper):
        result = run_command(["--json", "dump", "--all", "--filter", "zzz"])
        assert result.code == 1
        assert "조건에 맞는 플러그인이 없습니다" in result.err

    def test_output_file_receives_jsonl_and_stdout_reports_progress(
        self, fake_registry, stub_wrapper, tmp_path
    ):
        target = tmp_path / "dumped.jsonl"
        result = run_command(["--json", "dump", "--all", "--output-file", str(target)])
        assert result.code == 0
        assert target.exists()

        lines = [json.loads(line) for line in target.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        assert all(line["batch"] is True for line in lines)
        # 진행 상황은 stdout 콘솔(=result.out)로, 성공 요약은 stderr로 나간다.
        assert "[1/2] fake-denoise (4 params)" in result.out
        assert "Dump complete: 2 ok, 0 failed / 2 total" in result.err
        # rich는 긴 경로를 줄바꿈하므로 경로 자체의 등장만 확인한다.
        assert str(target) in result.out
        assert "출력:" in result.out

    def test_wrapper_failure_is_recorded_as_error_record(self, fake_registry, monkeypatch, tmp_path):
        class Boom(RecordingPlugin):
            def __init__(self):
                raise RuntimeError("cannot load plugin")

        monkeypatch.setattr(dump, "VST3PluginWrapper", wrapper_factory(Boom))
        target = tmp_path / "failed.jsonl"
        result = run_command(["--json", "dump", "--all", "--output-file", str(target)])
        assert result.code == 0  # 실패해도 종료코드는 올리지 않는다 (계약)

        lines = [json.loads(line) for line in target.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        assert all(line["error"] == "cannot load plugin" for line in lines)
        assert all("parameters" not in line for line in lines)
        assert "warning:" in result.err
        assert "Dump complete: 0 ok, 2 failed / 2 total" in result.err

    def test_empty_registry_exits_1(self, empty_registry, stub_wrapper):
        result = run_command(["--json", "dump", "--all"])
        assert result.code == 1
        assert "조건에 맞는 플러그인이 없습니다" in result.err
