# Created: 2026-09-28
# Purpose: coverage for cli/app.py — top-level parser/dispatch (AUD-1851).
#
# app.main is the entry point for every command, so these tests cover (a) a call with no
# command printing help and exiting 0, (b) --verbose turning logging on, (c) --plain/env
# vars rebuilding the plain console, and (d) the actual dispatch.

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
        # Check registration through the choices the subparsers action exposes.
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
        assert "Plugins found" in result.out


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
        """`--plain` must also set the env var for the import-time i18n detection."""
        monkeypatch.delenv("AUDIOMAN_PLAIN", raising=False)
        run_cli(["--plain", "scan"])
        # run_cli restores the original value, so check via _early_plain_detect.
        assert app._early_plain_detect(["--plain"]) is True

    def test_early_plain_detect_reads_argv_then_env(self, monkeypatch):
        monkeypatch.delenv("AUDIOMAN_PLAIN", raising=False)
        assert app._early_plain_detect(["--plain", "scan"]) is True
        # The call above also sets the env var, so clear it and check again.
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
