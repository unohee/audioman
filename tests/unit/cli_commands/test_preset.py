# Created: 2026-09-28
# Purpose: cli/preset.py 커버리지 — save/load/list/delete + not-found 경로 (AUD-1851).
#
# 프리셋 저장은 실제 파일을 만든다(AUDIOMAN_PRESET_DIR은 conftest가 tmp_path로
# 돌려놓는다). not-found 경로는 PresetManager를 실제로 태워 예외를 유도한다.

from __future__ import annotations

import json

import pytest

from harness import run_command


class TestPresetSave:
    def test_creates_preset_file_and_emits_json(self, tmp_path):
        result = run_command([
            "--json", "preset", "save", "clean-vocal", "-p", "fake-denoise",
            "--param", "threshold=-20", "--param", "mode=fast", "-d", "vocal chain",
        ])
        assert result.code == 0
        payload = json.loads(result.out)
        assert payload["command"] == "preset save"
        assert payload["name"] == "clean-vocal"
        assert payload["$schema"].startswith("audioman://schema/")

        written = tmp_path / "presets" / "fake-denoise" / "clean-vocal.json"
        assert str(written) == payload["path"]
        on_disk = json.loads(written.read_text())
        assert on_disk["name"] == "clean-vocal"
        assert on_disk["plugin"] == "fake-denoise"
        assert on_disk["parameters"] == {"threshold": -20.0, "mode": "fast"}
        assert on_disk["description"] == "vocal chain"

    def test_no_params_writes_empty_parameters(self, tmp_path):
        result = run_command(["--json", "preset", "save", "empty", "-p", "plug"])
        assert result.code == 0
        written = tmp_path / "presets" / "plug" / "empty.json"
        assert json.loads(written.read_text())["parameters"] == {}

    def test_plain_mode_reports_saved_path(self, tmp_path):
        result = run_command(["--plain", "preset", "save", "p1", "-p", "plug"])
        assert result.code == 0
        assert "프리셋 저장:" in result.err
        assert str(tmp_path / "presets" / "plug" / "p1.json") in result.err
        assert result.out == ""

    def test_rejects_path_traversal_name(self):
        # 잘못된 이름은 PresetManager가 ValueError로 거부한다. CLI는 이 예외를
        # 잡지 않는다(= 실제 실행에서는 traceback과 함께 exit 1). 크래시를
        # 'exit code'로만 뭉개지 않고 어떤 예외인지까지 고정한다.
        with pytest.raises(ValueError, match="single path component"):
            run_command(["--json", "preset", "save", "../escape", "-p", "plug"])

    def test_invalid_param_string_raises_before_writing(self, tmp_path):
        with pytest.raises(ValueError, match="key=value"):
            run_command([
                "--json", "preset", "save", "bad", "-p", "plug", "--param", "not-a-pair",
            ])
        assert not (tmp_path / "presets" / "plug" / "bad.json").exists()


class TestPresetLoad:
    def test_json_loads_saved_preset(self, tmp_path):
        run_command([
            "--json", "preset", "save", "round-trip", "-p", "plug",
            "--param", "gain=3", "-d", "desc",
        ])
        result = run_command(["--json", "preset", "load", "round-trip", "-p", "plug"])
        assert result.code == 0
        payload = json.loads(result.out)
        assert payload["command"] == "preset load"
        assert payload["name"] == "round-trip"
        assert payload["plugin"] == "plug"
        assert payload["parameters"] == {"gain": 3.0}
        assert payload["description"] == "desc"
        assert payload["created"]  # timestamp filled in on save

    def test_load_without_plugin_searches_all_dirs(self):
        run_command(["--json", "preset", "save", "anywhere", "-p", "some-plug"])
        result = run_command(["--json", "preset", "load", "anywhere"])
        assert result.code == 0
        assert json.loads(result.out)["plugin"] == "some-plug"

    def test_plain_mode_prints_details(self):
        run_command([
            "--json", "preset", "save", "pretty", "-p", "plug",
            "--param", "gain=3", "-d", "a description",
        ])
        result = run_command(["--plain", "preset", "load", "pretty", "-p", "plug"])
        assert result.code == 0
        assert "pretty (plug)" in result.out
        assert "a description" in result.out
        assert "Created:" in result.out
        assert "gain: 3.0" in result.out

    def test_plain_mode_without_description_omits_it(self):
        run_command(["--json", "preset", "save", "nodesc", "-p", "plug"])
        result = run_command(["--plain", "preset", "load", "nodesc", "-p", "plug"])
        assert result.code == 0
        assert "Created:" in result.out

    def test_missing_preset_exits_1_with_message(self):
        result = run_command(["--json", "preset", "load", "ghost", "-p", "plug"])
        assert result.code == 1
        assert "프리셋을 찾을 수 없습니다" in result.err
        assert "ghost" in result.err
        assert result.out == ""

    def test_missing_preset_without_plugin_exits_1(self):
        result = run_command(["--plain", "preset", "load", "ghost"])
        assert result.code == 1
        assert "error: 프리셋을 찾을 수 없습니다" in result.err


class TestPresetList:
    def test_json_lists_saved_presets_with_count(self):
        run_command(["--json", "preset", "save", "a", "-p", "plug", "--param", "x=1"])
        run_command(["--json", "preset", "save", "b", "-p", "plug", "--param", "x=2"])
        result = run_command(["--json", "preset", "list", "-p", "plug"])
        assert result.code == 0
        payload = json.loads(result.out)
        assert payload["command"] == "preset list"
        assert payload["count"] == 2
        assert sorted(p["name"] for p in payload["presets"]) == ["a", "b"]

    def test_json_empty_list_returns_zero_count(self):
        result = run_command(["--json", "preset", "list"])
        assert result.code == 0
        payload = json.loads(result.out)
        assert payload["count"] == 0
        assert payload["presets"] == []

    def test_plain_mode_prints_empty_notice(self):
        result = run_command(["--plain", "preset", "list"])
        assert result.code == 0
        assert "저장된 프리셋 없음" in result.out

    def test_plain_mode_table_has_name_plugin_params_description(self):
        run_command([
            "--json", "preset", "save", "listed", "-p", "plug",
            "--param", "x=1", "-d", "described",
        ])
        result = run_command(["--plain", "preset", "list", "-p", "plug"])
        assert result.code == 0
        assert "프리셋 목록" in result.out
        assert "Name\tPlugin\tParams\tDescription" in result.out
        assert "listed\tplug\t1\tdescribed" in result.out

    def test_plain_mode_table_dash_for_missing_description(self):
        run_command(["--json", "preset", "save", "nodesc-list", "-p", "plug"])
        result = run_command(["--plain", "preset", "list", "-p", "plug"])
        assert result.code == 0
        assert "nodesc-list\tplug\t0\t-" in result.out

    def test_rich_mode_table_path_runs(self):
        run_command(["--json", "preset", "save", "rich", "-p", "plug"])
        result = run_command(["preset", "list", "-p", "plug"])
        assert result.code == 0
        assert "rich" in result.out


class TestPresetDelete:
    def test_json_delete_removes_file(self, tmp_path):
        run_command(["--json", "preset", "save", "doomed", "-p", "plug"])
        target = tmp_path / "presets" / "plug" / "doomed.json"
        assert target.exists()

        result = run_command(["--json", "preset", "delete", "doomed", "-p", "plug"])
        assert result.code == 0
        payload = json.loads(result.out)
        assert payload["command"] == "preset delete"
        assert payload["name"] == "doomed"
        assert not target.exists()

    def test_delete_without_plugin_searches_all_dirs(self, tmp_path):
        run_command(["--json", "preset", "save", "free", "-p", "some-plug"])
        result = run_command(["--json", "preset", "delete", "free"])
        assert result.code == 0
        assert not (tmp_path / "presets" / "some-plug" / "free.json").exists()

    def test_plain_mode_reports_deletion(self):
        run_command(["--json", "preset", "save", "gone", "-p", "plug"])
        result = run_command(["--plain", "preset", "delete", "gone", "-p", "plug"])
        assert result.code == 0
        assert "프리셋 삭제: gone" in result.err

    def test_missing_preset_exits_1_without_deleting(self):
        result = run_command(["--json", "preset", "delete", "ghost", "-p", "plug"])
        assert result.code == 1
        assert "프리셋을 찾을 수 없습니다" in result.err
        assert result.out == ""


class TestPresetBareInvocation:
    def test_bare_preset_prints_help(self):
        result = run_command(["preset"])
        assert result.code == 0
        assert "save" in result.out
        assert "load" in result.out
        assert "delete" in result.out
