# Created: 2026-03-21
# Purpose: audioman list 서브커맨드

import argparse

from audioman.cli.output import print_json, print_table
from audioman.core.registry import get_registry


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("list", help="등록된 플러그인 목록")
    parser.add_argument("--format", choices=["vst3", "au"], help="포맷 필터")
    parser.add_argument("--vendor", help="벤더 필터")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    registry = get_registry()
    plugins = registry.list(fmt=args.format, vendor=args.vendor)

    if args.json:
        print_json({
            "command": "list",
            "count": len(plugins),
            "plugins": [p.to_dict() for p in plugins],
        })
        return

    rows = []
    for p in plugins:
        aliases = ", ".join(p.aliases) if p.aliases else "-"
        rows.append([p.short_name, p.name, p.format, str(p.param_count), aliases])

    print_table(
        f"플러그인 목록 ({len(plugins)}개)",
        ["Short Name", "Full Name", "Format", "Params", "Aliases"],
        rows,
    )
