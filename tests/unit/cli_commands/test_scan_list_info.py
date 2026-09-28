# Created: 2026-09-28
# Purpose: cli/scan.py, cli/list_cmd.py, cli/info.py 커버리지 (AUD-1851).
#
# 이 세 커맨드는 전부 registry를 조회한다. 이 호스트에는 등록 가능한 VST3가
# 하나도 없으므로(AUD-1857), registry를 스텁으로 바꿔 실제 필터/출력 로직만
# 관찰한다. info.py는 추가로 플러그인 래퍼를 로드하므로 래퍼도 대체한다.

from __future__ import annotations

import json

import pytest

from audioman.cli import info, list_cmd, scan
from audioman.core import findings
from harness import make_meta, run_command


def _json(result) -> dict:
    return json.loads(result.out)


class TestScan:
    def test_json_payload_lists_plugins_with_envelope(self, fake_registry):
        result = run_command(["--json", "scan"])
        assert result.code == 0
        payload = _json(result)
        assert payload["command"] == "scan"
        assert payload["$schema"] == findings.schema_uri("scan")
        assert payload["count"] == 2
        assert [p["short_name"] for p in payload["plugins"]] == ["fake-denoise", "fake-comp"]
        assert payload["plugins"][0]["aliases"] == ["denoise", "deno"]

    def test_json_forwards_paths_and_refresh_to_registry(self, fake_registry):
        result = run_command(["--json", "scan", "--paths", "/tmp/a", "/tmp/b", "--refresh"])
        assert result.code == 0
        assert fake_registry.scan_calls == [{"extra_paths": ["/tmp/a", "/tmp/b"], "refresh": True}]

    def test_json_defaults_are_no_paths_and_no_refresh(self, fake_registry):
        run_command(["--json", "scan"])
        assert fake_registry.scan_calls == [{"extra_paths": None, "refresh": False}]

    def test_plain_table_shows_names_aliases_and_success_line(self, fake_registry):
        result = run_command(["--plain", "scan"])
        assert result.code == 0
        assert "발견된 플러그인 (2개)" in result.out
        assert "Short Name\tFull Name\tFormat\tAliases" in result.out
        assert "fake-denoise\tFake De-noise\tvst3\tdenoise, deno" in result.out
        assert "fake-comp\tFake Comp\tau\t-" in result.out  # no aliases -> dash
        assert "2개 플러그인 스캔 완료" in result.err

    def test_empty_registry_reports_zero_and_creates_app_dirs(self, empty_registry, tmp_path,
                                                            monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        result = run_command(["--plain", "scan"])
        assert result.code == 0
        assert "발견된 플러그인 (0개)" in result.out
        assert "0개 플러그인 스캔 완료" in result.err
        assert (tmp_path / ".audioman" / "cache").is_dir()

    def test_rich_table_path_runs_without_plain(self, fake_registry):
        """rich 경로(print_table의 Console 출력)도 실제로 태운다."""
        result = run_command(["scan"])
        assert result.code == 0
        assert "fake-denoise" in result.out


class TestList:
    def test_json_payload_envelope_and_fields(self, fake_registry):
        result = run_command(["--json", "list"])
        assert result.code == 0
        payload = _json(result)
        assert payload["command"] == "list"
        assert payload["$schema"] == findings.schema_uri("list")
        assert payload["count"] == 2
        assert payload["plugins"][1]["param_count"] == 0

    def test_format_and_vendor_filters_are_forwarded_and_applied(self, fake_registry):
        result = run_command(["--json", "list", "--format", "au", "--vendor", "other"])
        assert result.code == 0
        payload = _json(result)
        assert fake_registry.list_calls == [{"fmt": "au", "vendor": "other"}]
        assert [p["short_name"] for p in payload["plugins"]] == ["fake-comp"]

    def test_format_filter_excludes_non_matching(self, fake_registry):
        payload = _json(run_command(["--json", "list", "--format", "vst3"]))
        assert [p["short_name"] for p in payload["plugins"]] == ["fake-denoise"]

    def test_plain_table_lists_params_and_aliases_columns(self, fake_registry):
        fake_registry.plugins = [
            make_meta(short_name="p1", name="P One", aliases=["one"], param_count=7)
        ]
        result = run_command(["--plain", "list"])
        assert result.code == 0
        assert "플러그인 목록 (1개)" in result.out
        assert "Short Name\tFull Name\tFormat\tParams\tAliases" in result.out
        assert "p1\tP One\tvst3\t7\tone" in result.out

    def test_plain_table_dash_for_missing_aliases_and_empty_list(self, empty_registry):
        result = run_command(["--plain", "list"])
        assert result.code == 0
        assert "플러그인 목록 (0개)" in result.out


class TestInfo:
    @pytest.fixture(autouse=True)
    def stub_wrapper(self, monkeypatch):
        from harness import stub_plugin_params

        class _Wrapper:
            def __init__(self, path):
                self.path = path

            def get_parameters(self):
                return stub_plugin_params()

        monkeypatch.setattr(info, "VST3PluginWrapper", _Wrapper)

    def test_json_payload_has_params_and_updates_param_count(self, fake_registry):
        result = run_command(["--json", "info", "fake-denoise"])
        assert result.code == 0
        payload = _json(result)
        assert payload["command"] == "info"
        assert payload["plugin"]["short_name"] == "fake-denoise"
        assert payload["plugin"]["param_count"] == 4  # set from the loaded params
        assert [p["name"] for p in payload["parameters"]] == [
            "threshold", "mode", "bypass", "unbounded",
        ]

    def test_alias_resolves_to_plugin(self, fake_registry):
        result = run_command(["--json", "info", "denoise"])
        assert result.code == 0
        assert _json(result)["plugin"]["short_name"] == "fake-denoise"

    def test_unknown_plugin_exits_1_with_message(self, fake_registry):
        result = run_command(["--json", "info", "nope"])
        assert result.code == 1
        assert "플러그인을 찾을 수 없습니다" in result.err
        assert "nope" in result.err
        assert result.out == ""  # no payload emitted on the error path

    def test_plain_output_renders_each_param_type_range(self, fake_registry):
        result = run_command(["--plain", "info", "fake-denoise"])
        assert result.code == 0
        assert "Fake De-noise" in result.out
        assert "Short name: fake-denoise" in result.out
        assert "Aliases: denoise, deno" in result.out
        assert "파라미터 (4개)" in result.out
        assert "[-60.0, 0.0]" in result.out       # float with bounds
        assert "(enum)" in result.out             # enum param
        assert "(bool)" in result.out             # bool param
        assert "threshold\tfloat\t-20.0\t[-60.0, 0.0]" in result.out
        assert "unbounded\tfloat\t-\t-" in result.out  # None current and range

    def test_plain_output_without_aliases_skips_alias_line(self, fake_registry):
        fake_registry.plugins = [make_meta(short_name="solo", name="Solo", aliases=[])]
        result = run_command(["--plain", "info", "solo"])
        assert result.code == 0
        assert "Aliases:" not in result.out

    def test_rich_output_path_runs_without_plain(self, fake_registry):
        result = run_command(["info", "fake-denoise"])
        assert result.code == 0
        assert "Fake De-noise" in result.out
        assert "Path: /nonexistent/Fake De-noise.vst3" in result.out
