# Created: 2026-03-21
# Purpose: audioman dump — dump plugin parameter state as JSON/JSONL

import argparse
import json
import sys

from audioman.cli.output import print_error, print_json, print_success, print_warning, output_console
from audioman.core.findings import json_envelope, schema_uri
from audioman.core.registry import get_registry
from audioman.core.engine import parse_params
from audioman.plugins.vst3 import VST3PluginWrapper


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "dump",
        help="Dump plugin parameter state to JSON/JSONL (always machine-readable; --json is implied)",
        description=(
            "Dump plugin parameter state as JSON (single plugin) or JSONL (--all). "
            "This command is always machine-readable, so --json is implied and not "
            "needed; the flag is accepted only for CLI uniformity."
        ),
    )
    # Single mode: name a plugin
    parser.add_argument("plugin", nargs="?", default=None, help="Plugin name (omit for --all)")
    parser.add_argument("--param", action="append", default=[], help="Set parameter before dump (key=value)")
    parser.add_argument("--preset", help="Preset name (apply before dump)")
    parser.add_argument("--save-preset", metavar="NAME", help="Save dump as preset")
    # Batch mode
    parser.add_argument("--all", action="store_true", help="Dump all plugins as JSONL")
    parser.add_argument("--filter", metavar="KEYWORD", help="Plugin name filter (with --all)")
    parser.add_argument("--format-filter", choices=["vst3", "au"], help="Format filter (with --all)")
    parser.add_argument("--output-file", "-o", metavar="PATH", help="JSONL output file (default: stdout)")
    parser.set_defaults(func=run)


def _dump_plugin_state(wrapper: VST3PluginWrapper, meta) -> dict:
    """Extract the full parameter state of a plugin as a dict"""
    plugin = wrapper._plugin
    state = {}

    for attr_name, param in plugin.parameters.items():
        try:
            val = getattr(plugin, attr_name)
            if isinstance(val, bool):
                state[attr_name] = val
            elif isinstance(val, (int, float)):
                state[attr_name] = float(val)
            elif isinstance(val, str):
                state[attr_name] = val
            else:
                state[attr_name] = str(val)
        except Exception:
            state[attr_name] = None

    return {
        "plugin": meta.name,
        "short_name": meta.short_name,
        "path": meta.path,
        "format": meta.format,
        "identifier": getattr(plugin, "identifier", ""),
        "version": getattr(plugin, "version", ""),
        "parameter_count": len(state),
        "parameters": state,
    }


def run(args: argparse.Namespace) -> None:
    if args.all:
        _run_batch(args)
    elif args.plugin:
        _run_single(args)
    else:
        print_error("A plugin name or the --all flag is required")


def _run_single(args: argparse.Namespace) -> None:
    registry = get_registry()
    meta = registry.get(args.plugin)
    if not meta:
        print_error(f"Plugin not found: '{args.plugin}'")

    wrapper = VST3PluginWrapper(meta.path)
    wrapper.load()

    # Apply preset
    if args.preset:
        from audioman.core.preset_manager import PresetManager
        manager = PresetManager()
        try:
            preset = manager.load(args.preset, plugin=meta.short_name)
            wrapper.set_parameters(preset.parameters)
        except FileNotFoundError as e:
            print_error(str(e))

    # Apply CLI parameters
    if args.param:
        params = parse_params(args.param)
        wrapper.set_parameters(params)

    state = _dump_plugin_state(wrapper, meta)

    # Save as preset
    if args.save_preset:
        from audioman.core.preset_manager import PresetManager
        from audioman.config.paths import ensure_app_dirs
        ensure_app_dirs()
        manager = PresetManager()
        manager.save(
            name=args.save_preset,
            plugin=meta.short_name,
            params=state["parameters"],
            description=f"dump from {meta.name}",
        )
        state["saved_as_preset"] = args.save_preset

    print_json(json_envelope("dump", state, schema=schema_uri("dump")))


def _run_batch(args: argparse.Namespace) -> None:
    """Dump the default parameter state of every plugin as JSONL"""
    registry = get_registry()
    plugins = registry.list(fmt=args.format_filter)

    # Keyword filter
    if args.filter:
        keyword = args.filter.lower()
        plugins = [p for p in plugins if keyword in p.name.lower() or keyword in p.short_name]

    if not plugins:
        print_error("No plugins match the filter")

    # Output target
    if args.output_file:
        out_file = open(args.output_file, "w")
    else:
        out_file = sys.stdout

    ok, fail = 0, 0
    try:
        for i, meta in enumerate(plugins):
            try:
                wrapper = VST3PluginWrapper(meta.path)
                wrapper.load()
                state = _dump_plugin_state(wrapper, meta)
                line = json.dumps(
                    json_envelope("dump", {**state, "batch": True}, schema=schema_uri("dump")),
                    ensure_ascii=False, default=str,
                )
                out_file.write(line + "\n")
                ok += 1

                if out_file is not sys.stdout:
                    output_console.print(f"  [{i+1}/{len(plugins)}] {meta.short_name} ({state['parameter_count']} params)")

            except Exception as e:
                fail += 1
                # Record an error record in the JSONL even on failure
                err_line = json.dumps(json_envelope("dump", {
                    "plugin": meta.name,
                    "short_name": meta.short_name,
                    "path": meta.path,
                    "error": str(e),
                    "batch": True,
                }, schema=schema_uri("dump")), ensure_ascii=False)
                out_file.write(err_line + "\n")

                if out_file is not sys.stdout:
                    print_warning(f"  [{i+1}/{len(plugins)}] {meta.short_name}: {e}")

    finally:
        if out_file is not sys.stdout:
            out_file.close()

    if out_file is not sys.stdout or args.output_file:
        print_success(f"Dump complete: {ok} ok, {fail} failed / {len(plugins)} total")
        if args.output_file:
            output_console.print(f"Output: {args.output_file}")
