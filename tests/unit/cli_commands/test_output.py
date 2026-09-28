# Created: 2026-09-28
# Purpose: cli/output.py 계약 — JSON/table/plain 포매터 (AUD-1851).
#
# 출력 경로는 두 갈래다: rich 경로(TTY용)와 --plain 경로(LLM용, markup 제거 + TSV).
# 여기서는 둘 다 실제로 호출하고 stdout/stderr에 무엇이 나가는지 검증한다.
#
# 캡처는 harness.capture_streams를 쓴다: pytest 캡처가 켜져 있으면
# monkeypatch로 sys.stdout을 바꿔도 pytest가 자기 스트림으로 되돌려 놓는다.

from __future__ import annotations

import json

import pytest

from audioman.cli import output
from harness import capture_streams


@pytest.fixture(autouse=True)
def restore_plain_mode():
    """각 테스트 후 plain 모드를 원상복구 (전역 console 재구성 포함)."""
    before = output.is_plain()
    yield
    output.set_plain(before)


class TestPlainFlag:
    def test_set_plain_toggles_is_plain(self):
        output.set_plain(True)
        assert output.is_plain() is True
        output.set_plain(False)
        assert output.is_plain() is False

    def test_set_plain_rebuilds_consoles_with_plain_settings(self):
        """plain 콘솔은 markup/highlight/emoji를 끄고 stderr 라우팅은 유지한다.

        `no_color`는 TTY가 아닌 환경에서 기본 콘솔도 True이므로 구분자가 못 된다.
        rich가 이를 공개 API로 노출하지 않아 콘솔 인스턴스의 플래그를 직접 본다.
        """
        output.set_plain(True)
        for console in (output.console, output.output_console):
            assert console._markup is False
            assert console._highlight is False
            assert console._emoji is False
        assert output.console.stderr is True
        assert output.output_console.stderr is False

    def test_set_plain_false_restores_rich_consoles(self):
        output.set_plain(True)
        output.set_plain(False)
        for console in (output.console, output.output_console):
            assert console._markup is True
            assert console._highlight is True
        assert output.console.stderr is True
        assert output.output_console.stderr is False

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on", " True "])
    def test_env_var_variants_enable_plain(self, monkeypatch, value):
        monkeypatch.setenv("AUDIOMAN_PLAIN", value)
        assert output._is_plain_env() is True

    @pytest.mark.parametrize("value", ["", "0", "off", "no", "maybe"])
    def test_env_var_non_truthy_values_disable_plain(self, monkeypatch, value):
        monkeypatch.setenv("AUDIOMAN_PLAIN", value)
        assert output._is_plain_env() is False


class TestMarkupStripping:
    @pytest.mark.parametrize("raw,expected", [
        ("[bold]Title[/bold]", "Title"),
        ("[red]error:[/red] boom", "error: boom"),
        ("plain text", "plain text"),
        ("[dim]a[/dim][green]b[/green]", "ab"),
    ])
    def test_strip_markup_removes_rich_tokens(self, raw, expected):
        assert output._strip_markup(raw) == expected

    def test_strip_markup_also_swallows_numeric_bracket_groups(self):
        # 알려진 한계를 고정한다: 정규식이 태그가 아닌 "[0, 1]" 형태도 함께 지운다.
        # info/analyze의 파라미터 range 컬럼이 plain 모드에서 대괄호를 잃는다.
        assert output._strip_markup("range [0, 1]") == "range "


class TestPrintJson:
    def test_prints_indented_utf8_json(self):
        with capture_streams() as (out, _err):
            output.print_json({"name": "값", "n": 1})
        assert json.loads(out.getvalue()) == {"name": "값", "n": 1}
        assert "값" in out.getvalue()  # ensure_ascii=False
        assert out.getvalue().startswith("{\n  ")  # indent=2

    def test_non_serializable_values_fall_back_to_str(self):
        with capture_streams() as (out, _err):
            output.print_json({"cls": output.Console})
        assert json.loads(out.getvalue())["cls"].startswith("<class")

    def test_json_ignores_plain_mode(self):
        """print_json은 plain 여부와 무관하게 같은 JSON을 낸다 (계약 고정)."""
        output.set_plain(True)
        with capture_streams() as (out, _err):
            output.print_json({"a": 1})
        assert json.loads(out.getvalue()) == {"a": 1}


class TestPrintTable:
    def test_plain_mode_emits_tsv_with_title_comment(self):
        output.set_plain(True)
        with capture_streams() as (out, _err):
            output.print_table("플러그인", ["A", "B"], [["1", 2], ["3", "4"]])
        lines = out.getvalue().splitlines()
        assert lines[0] == "# 플러그인"
        assert lines[1] == "A\tB"
        assert lines[2] == "1\t2"  # non-str 셀도 str() 변환
        assert lines[3] == "3\t4"

    def test_plain_mode_without_title_skips_comment(self):
        output.set_plain(True)
        with capture_streams() as (out, _err):
            output.print_table("", ["A"], [["1"]])
        assert out.getvalue().splitlines()[0] == "A"

    def test_rich_mode_prints_table_to_stdout(self):
        output.set_plain(False)
        with capture_streams() as (out, err):
            output.print_table("T", ["A", "B"], [["1", "2"]])
        assert "T" in out.getvalue()
        assert "A" in out.getvalue() and "2" in out.getvalue()
        assert err.getvalue() == ""


class TestMessagePrinters:
    def test_rich_mode_prints_messages_to_stderr(self):
        output.set_plain(False)
        with capture_streams() as (out, err):
            output.print_info("info-msg")
            output.print_success("success-msg")
            output.print_warning("warn-msg")
        assert "info-msg" in err.getvalue()
        assert "success-msg" in err.getvalue()
        assert "warn-msg" in err.getvalue()
        assert out.getvalue() == ""

    def test_plain_mode_strips_markup_and_writes_stderr(self):
        output.set_plain(True)
        with capture_streams() as (out, err):
            output.print_info("[dim]info-msg[/dim]")
            output.print_success("[green]done[/green]")
            output.print_warning("[yellow]careful[/yellow]")
        assert "info-msg" in err.getvalue()
        assert "done" in err.getvalue()
        assert "warning: careful" in err.getvalue()
        assert "[" not in err.getvalue()
        assert out.getvalue() == ""

    def test_print_error_raises_systemexit_1_in_rich_mode(self):
        output.set_plain(False)
        with capture_streams() as (_out, err):
            with pytest.raises(SystemExit) as excinfo:
                output.print_error("boom")
        assert excinfo.value.code == 1
        assert "boom" in err.getvalue()

    def test_print_error_plain_mode_prefixes_and_exits(self):
        output.set_plain(True)
        with capture_streams() as (_out, err):
            with pytest.raises(SystemExit) as excinfo:
                output.print_error("[red]bad thing[/red]")
        assert excinfo.value.code == 1
        assert err.getvalue().strip() == "error: bad thing"
