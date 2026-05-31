# Created: 2026-03-21
# Purpose: CLI 출력 포매터 (human-readable / JSON / plain)
#
# Plain 모드 (--plain 또는 AUDIOMAN_PLAIN=1) 사용 시:
#   - Rich Console이 색상/markup/highlight를 모두 끄도록 재설정
#   - print_table은 TSV로 fallback
#   - rich markup 토큰 ([red], [bold] 등)이 제거된 채 출력

import json
import os
import re
import sys
from typing import Any

from rich.console import Console
from rich.table import Table

_PLAIN: bool = False


def _is_plain_env() -> bool:
    val = os.environ.get("AUDIOMAN_PLAIN", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _make_console(*, stderr: bool) -> Console:
    if _PLAIN:
        return Console(
            stderr=stderr,
            no_color=True,
            force_terminal=False,
            markup=False,
            highlight=False,
            emoji=False,
        )
    return Console(stderr=stderr)


def set_plain(enabled: bool) -> None:
    """app.py에서 --plain 결정 후 호출. console 인스턴스를 재구성한다."""
    global _PLAIN, console, output_console
    _PLAIN = bool(enabled)
    console = _make_console(stderr=True)
    output_console = _make_console(stderr=False)


def is_plain() -> bool:
    return _PLAIN


_PLAIN = _is_plain_env()
console = _make_console(stderr=True)
output_console = _make_console(stderr=False)


_RICH_MARKUP_RE = re.compile(r"\[/?[a-zA-Z0-9 _#=,\.\-]+\]")


def _strip_markup(text: str) -> str:
    """Plain 모드용: [bold]X[/bold] 같은 rich markup 토큰을 제거."""
    return _RICH_MARKUP_RE.sub("", text)


def print_json(data: Any) -> None:
    """JSON 모드 출력 (stdout)"""
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def print_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    """테이블 출력. Plain 모드면 TSV로 stdout에 출력."""
    if _PLAIN:
        if title:
            print(f"# {title}")
        print("\t".join(columns))
        for row in rows:
            print("\t".join(str(c) for c in row))
        return

    table = Table(title=title)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    output_console.print(table)


def print_info(message: str) -> None:
    if _PLAIN:
        print(_strip_markup(message), file=sys.stderr)
        return
    console.print(f"[dim]{message}[/dim]", highlight=False)


def print_success(message: str) -> None:
    if _PLAIN:
        print(_strip_markup(message), file=sys.stderr)
        return
    console.print(f"[green]{message}[/green]", highlight=False)


def print_error(message: str) -> None:
    if _PLAIN:
        print(f"error: {_strip_markup(message)}", file=sys.stderr)
        sys.exit(1)
    console.print(f"[red]error:[/red] {message}", highlight=False)
    sys.exit(1)


def print_warning(message: str) -> None:
    if _PLAIN:
        print(f"warning: {_strip_markup(message)}", file=sys.stderr)
        return
    console.print(f"[yellow]warning:[/yellow] {message}", highlight=False)
