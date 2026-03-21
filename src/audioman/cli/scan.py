# Created: 2026-03-21
# Purpose: audioman scan 서브커맨드

import argparse

from audioman.cli.output import print_json, print_success, print_table
from audioman.config.paths import ensure_app_dirs
from audioman.core.registry import get_registry


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("scan", help="시스템에서 VST3/AU 플러그인 검색")
    parser.add_argument("--paths", nargs="*", help="추가 검색 경로")
    parser.add_argument("--refresh", action="store_true", help="캐시 무시하고 재스캔")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    ensure_app_dirs()
    registry = get_registry()
    plugins = registry.scan(extra_paths=args.paths, refresh=args.refresh)

    if args.json:
        print_json({
            "command": "scan",
            "count": len(plugins),
            "plugins": [p.to_dict() for p in plugins],
        })
        return

    rows = []
    for p in plugins:
        aliases = ", ".join(p.aliases) if p.aliases else "-"
        rows.append([p.short_name, p.name, p.format, aliases])

    print_table(
        f"발견된 플러그인 ({len(plugins)}개)",
        ["Short Name", "Full Name", "Format", "Aliases"],
        rows,
    )
    print_success(f"{len(plugins)}개 플러그인 스캔 완료 (캐시 저장됨)")
