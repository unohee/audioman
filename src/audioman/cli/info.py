# Created: 2026-03-21
# Purpose: audioman info 서브커맨드

import argparse

from audioman.cli.output import print_error, print_json, print_table, output_console
from audioman.core.registry import get_registry
from audioman.plugins.vst3 import VST3PluginWrapper


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("info", help="플러그인 상세 정보 + 파라미터 목록")
    parser.add_argument("plugin", help="플러그인 이름 (short_name 또는 별칭)")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    registry = get_registry()
    meta = registry.get(args.plugin)

    if not meta:
        print_error(f"플러그인을 찾을 수 없습니다: '{args.plugin}'")

    # 플러그인 로드하여 파라미터 추출
    wrapper = VST3PluginWrapper(meta.path)
    params = wrapper.get_parameters()
    meta.param_count = len(params)

    if args.json:
        print_json({
            "command": "info",
            "plugin": meta.to_dict(),
            "parameters": [p.to_dict() for p in params],
        })
        return

    # 기본 정보
    output_console.print(f"\n[bold]{meta.name}[/bold]")
    output_console.print(f"  Short name: {meta.short_name}")
    output_console.print(f"  Path: {meta.path}")
    output_console.print(f"  Format: {meta.format}")
    if meta.aliases:
        output_console.print(f"  Aliases: {', '.join(meta.aliases)}")
    output_console.print()

    # 파라미터 테이블
    rows = []
    for p in params:
        if p.type == "float":
            range_str = f"[{p.min_value}, {p.max_value}]" if p.min_value is not None else "-"
        elif p.type == "enum":
            range_str = f"(enum)"
        else:
            range_str = f"(bool)"

        current = str(p.current_value) if p.current_value is not None else "-"
        rows.append([p.name, p.type, current, range_str])

    print_table(
        f"파라미터 ({len(params)}개)",
        ["Name", "Type", "Current", "Range"],
        rows,
    )
