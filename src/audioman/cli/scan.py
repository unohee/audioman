# Created: 2026-03-21
# Purpose: audioman scan subcommand

import argparse

from audioman.cli.output import print_json, print_success, print_table
from audioman.config.paths import ensure_app_dirs
from audioman.core.findings import json_envelope, schema_uri
from audioman.core.registry import get_registry


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("scan", help="Scan system for VST3/AU plugins")
    parser.add_argument("--paths", nargs="*", help="Additional search paths")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and rescan")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    ensure_app_dirs()
    registry = get_registry()
    plugins = registry.scan(extra_paths=args.paths, refresh=args.refresh)

    if args.json:
        print_json(json_envelope(
            "scan",
            {
                "count": len(plugins),
                "plugins": [p.to_dict() for p in plugins],
            },
            schema=schema_uri("scan"),
        ))
        return

    rows = []
    for p in plugins:
        aliases = ", ".join(p.aliases) if p.aliases else "-"
        rows.append([p.short_name, p.name, p.format, aliases])

    print_table(
        f"Plugins found ({len(plugins)})",
        ["Short Name", "Full Name", "Format", "Aliases"],
        rows,
    )
    print_success(f"Scanned {len(plugins)} plugins (cache saved)")
