# Created: 2026-09-28
# Purpose: cli/app.py 커버리지 — 최상위 파서/디스패치 (AUD-1851).
#
# app.main은 모든 커맨드의 진입점이므로, 여기서는 (a) 커맨드 없는 호출이 도움말을
# 내고 exit 0 하는 경로, (b) --verbose가 로깅을 켜는 경로, (c) --plain/환경변수가
# plain 콘솔을 재구성하는 경로, (d) 실제 디스패치를 확인한다.

from __future__ import annotations

import logging

import pytest

from audioman import __version__
from audioman.cli import app
from audioman.cli import output as output_module
from harness import run_cli


@pytest.fixture(autouse=True)
def restore_plain():
    before = output_module.is_plain()
    yield
    output_module.set_plain(before)


class TestParser:
    def test_build_parser_registers_every_command(self):
        parser = app.build_parser()
        # subparsers action이 노출하는 choices로 등록 여부를 본다.
        subparsers = next(
            action for action in parser._actions if getattr(action, "choices", None)
            and "scan" in action.choices
        )
        for name in ("scan", "list", "info", "process", "chain", "preset", "dump",
                     "analyze", "fx", "visualize", "doctor", "schemas", "stream"):
            assert name in subparsers.choices

    def test_json_and_plain_flags_default_to_false(self):
        args = app.build_parser().parse_args(["scan"])
        assert args.json is False
        assert args.plain is False
        assert args.verbose is False
        assert args.command == "scan"

    def test_verbose_flag_is_recorded(self):
        args = app.build_parser().parse_args(["--verbose", "scan"])
        assert args.verbose is True

    def test_version_flag_prints_version(self):
        result = run_cli(["--version"])
        assert result.code == 0
        assert __version__ in result.out


class TestDispatch:
    def test_no_command_prints_help_and_exits_zero(self):
        result = run_cli([])
        assert result.code == 0
        assert "Available commands" in result.out
        assert "scan" in result.out

    def test_unknown_command_exits_2(self):
        result = run_cli(["definitely-not-a-command"])
        assert result.code == 2
        assert "invalid choice" in result.err

    def test_command_is_dispatched_to_its_run_function(self, fake_registry):
        result = run_cli(["--plain", "scan"])
        assert result.code == 0
        assert fake_registry.scan_calls, "scan command did not reach the registry"
        assert "발견된 플러그인" in result.out


class TestVerboseLogging:
    def test_verbose_enables_debug_logging(self, fake_registry, monkeypatch):
        calls = []
        monkeypatch.setattr(logging, "basicConfig", lambda **kw: calls.append(kw))

        result = run_cli(["--verbose", "--plain", "scan"])
        assert result.code == 0
        assert calls == [{"level": logging.DEBUG, "format": "%(name)s: %(message)s"}]

    def test_without_verbose_logging_is_not_configured(self, fake_registry, monkeypatch):
        calls = []
        monkeypatch.setattr(logging, "basicConfig", lambda **kw: calls.append(kw))

        result = run_cli(["--plain", "scan"])
        assert result.code == 0
        assert calls == []


class TestPlainModeWiring:
    def test_plain_flag_enables_plain_output(self, fake_registry):
        result = run_cli(["--plain", "scan"])
        assert result.code == 0
        assert output_module.is_plain() is True

    def test_env_var_enables_plain_output(self, fake_registry, monkeypatch):
        monkeypatch.setenv("AUDIOMAN_PLAIN", "1")
        result = run_cli(["scan"])
        assert result.code == 0
        assert output_module.is_plain() is True

    def test_plain_flag_sets_env_for_i18n_detection(self, fake_registry, monkeypatch):
        """`--plain`은 import 시점 i18n 감지를 위해 env도 세팅해야 한다."""
        monkeypatch.delenv("AUDIOMAN_PLAIN", raising=False)
        run_cli(["--plain", "scan"])
        # run_cli가 원래 값을 복원하므로, 세팅 여부는 _early_plain_detect로 확인한다.
        assert app._early_plain_detect(["--plain"]) is True

    def test_early_plain_detect_reads_argv_then_env(self, monkeypatch):
        monkeypatch.delenv("AUDIOMAN_PLAIN", raising=False)
        assert app._early_plain_detect(["--plain", "scan"]) is True
        # 위 호출은 env까지 세팅하는 부작용이 있으므로 지우고 다시 본다.
        monkeypatch.delenv("AUDIOMAN_PLAIN", raising=False)
        assert app._early_plain_detect(["scan"]) is False
        monkeypatch.setenv("AUDIOMAN_PLAIN", "yes")
        assert app._early_plain_detect(["scan"]) is True

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_early_plain_detect_accepts_env_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("AUDIOMAN_PLAIN", value)
        assert app._early_plain_detect(["scan"]) is True

    @pytest.mark.parametrize("value", ["", "0", "off", "no"])
    def test_early_plain_detect_rejects_env_falsey_values(self, monkeypatch, value):
        monkeypatch.setenv("AUDIOMAN_PLAIN", value)
        assert app._early_plain_detect(["scan"]) is False

    def test_early_plain_detect_uses_process_argv_when_none(self, monkeypatch):
        monkeypatch.setenv("AUDIOMAN_PLAIN", "1")
        assert app._early_plain_detect(None) is True

    def test_non_plain_run_leaves_plain_disabled(self, fake_registry, monkeypatch):
        monkeypatch.delenv("AUDIOMAN_PLAIN", raising=False)
        result = run_cli(["scan"])
        assert result.code == 0
        assert output_module.is_plain() is False
