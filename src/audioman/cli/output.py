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
from rich.errors import StyleSyntaxError
from rich.style import Style
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


_BRACKET_GROUP_RE = re.compile(r"\[([^\[\]]*)\]")

# rich의 RE_TAGS와 같은 진입 조건: 여는 대괄호 바로 뒤 글자.
_RICH_TAG_START = frozenset("abcdefghijklmnopqrstuvwxyz#/@")
_RICH_OPEN_TAG_START = frozenset("abcdefghijklmnopqrstuvwxyz#@")


def _is_rich_style(text: str) -> bool:
    """rich가 스타일 정의로 해석할 수 있는 텍스트인지."""
    try:
        Style.parse(text)
    except StyleSyntaxError:
        return False
    return True


def _is_rich_open_tag(body: str) -> bool:
    """rich가 여는 태그로 인정하는 본문인지 (스타일/`link=`/`@handler`)."""
    if not body or body[0] not in _RICH_OPEN_TAG_START:
        return False  # `[BOLD]`는 rich의 RE_TAGS에 걸리지 않는다
    name, equals, _parameters = body.partition("=")
    name = name.strip().lower()
    if name.startswith("@"):
        return True
    if name == "link":
        return True
    return not equals and _is_rich_style(body)


def _is_rich_tag(inner: str) -> bool:
    """대괄호 안 텍스트가 rich가 태그로 해석하는 형태인지.

    rich는 `[a-z#/@]`로 시작하는 대괄호만 태그 후보로 보고, 그중에서도
    스타일(`[bold]`)·`[link=...]`·`[@handler]`로 해석되는 것만 스타일로
    적용한다. 스타일로 해석되지 않는 그룹(`[0, 1]`, `[dry-run]`)은 rich
    모드에서도 텍스트로 남으므로 plain 모드에서도 지우면 안 된다.
    """
    if not inner or inner[0] not in _RICH_TAG_START:
        return False  # `[0, 1]`, `[-60.0, 0.0]`, `[1/2]` 등
    if inner.startswith("/"):
        body = inner[1:].strip()
        if not body:
            return True  # `[/]`는 rich의 암묵적 닫기 태그
        return _is_rich_open_tag(body)
    return _is_rich_open_tag(inner)


def _strip_markup(text: str) -> str:
    """Plain 모드용: [bold]X[/bold] 같은 rich markup 토큰만 제거.

    태그가 아닌 대괄호 텍스트(`[0, 1]`, `[dry-run]`, `[1/2]`)는 보존한다.
    """
    return _BRACKET_GROUP_RE.sub(
        lambda match: "" if _is_rich_tag(match.group(1)) else match.group(0), text
    )


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


def print_literal(message: str) -> None:
    """stdout에 대괄호 텍스트를 그대로 출력한다.

    `[dry-run]` / `[plugin]` 같은 토큰은 rich 모드에서 마크업으로 해석돼
    통째로 사라진다. 계획(plan) 텍스트는 입력값을 그대로 보여줘야 하므로
    두 모드 모두 `markup=False` 경로로 보낸다.
    """
    output_console.print(message, markup=False, soft_wrap=True, highlight=False)


def print_markup(message: str) -> None:
    """stdout에 마크업이 섞인 텍스트를 출력한다.

    rich 모드에서는 마크업을 적용하고(기존 `output_console.print` 경로와
    동일), plain 모드에서는 태그를 걷어낸 텍스트를 낸다. `--plain` 콘솔은
    `markup=False`라 태그가 그대로 새어 나가므로 이 경로를 쓴다.
    """
    if _PLAIN:
        print(_strip_markup(message), file=sys.stdout)
        return
    output_console.print(message)


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
