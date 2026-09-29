# Created: 2026-03-21
# Purpose: audioman info subcommand

import argparse

from audioman.cli.output import print_error, print_json, print_table, output_console
from audioman.cli.output import print_markup
from audioman.core.findings import json_envelope, schema_uri
from audioman.core.registry import get_registry
from audioman.plugins.vst3 import VST3PluginWrapper


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("info", help="Plugin details + parameter list")
    parser.add_argument("plugin", help="Plugin name (short_name or alias)")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    registry = get_registry()
    meta = registry.get(args.plugin)

    if not meta:
        print_error(f"Plugin not found: '{args.plugin}'")

    # Load the plugin to extract its parameters
    wrapper = VST3PluginWrapper(meta.path)
    params = wrapper.get_parameters()
    meta.param_count = len(params)

    if args.json:
        print_json(json_envelope(
            "info",
            {
                "plugin": meta.to_dict(),
                "parameters": [p.to_dict() for p in params],
            },
            schema=schema_uri("info"),
        ))
        return

    # Basic info
    print_markup(f"\n[bold]{meta.name}[/bold]")
    output_console.print(f"  Short name: {meta.short_name}")
    output_console.print(f"  Path: {meta.path}")
    output_console.print(f"  Format: {meta.format}")
    if meta.aliases:
        output_console.print(f"  Aliases: {', '.join(meta.aliases)}")
    output_console.print()

    # Parameter table
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
        f"Parameters ({len(params)})",
        ["Name", "Type", "Current", "Range"],
        rows,
    )
