# Created: 2026-03-21
# Purpose: CLI 출력 포매터 (human-readable / JSON)

import json
import sys
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console(stderr=True)
output_console = Console()


def print_json(data: Any) -> None:
    """JSON 모드 출력 (stdout)"""
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def print_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    """Rich 테이블 출력"""
    table = Table(title=title)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    output_console.print(table)


def print_info(message: str) -> None:
    console.print(f"[dim]{message}[/dim]", highlight=False)


def print_success(message: str) -> None:
    console.print(f"[green]{message}[/green]", highlight=False)


def print_error(message: str) -> None:
    console.print(f"[red]error:[/red] {message}", highlight=False)
    sys.exit(1)


def print_warning(message: str) -> None:
    console.print(f"[yellow]warning:[/yellow] {message}", highlight=False)
