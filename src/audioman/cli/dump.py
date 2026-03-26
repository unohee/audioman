# Created: 2026-03-21
# Purpose: audioman dump — 플러그인 파라미터 상태를 JSON/JSONL로 덤프

import argparse
import json
import sys

from audioman.cli.output import print_error, print_json, print_warning, output_console
from audioman.core.registry import get_registry
from audioman.core.engine import parse_params
from audioman.i18n import _
from audioman.plugins.vst3 import VST3PluginWrapper


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "dump",
        help=_("Dump plugin parameter state to JSON/JSONL"),
    )
    # 단일 모드: 플러그인 이름 지정
    parser.add_argument("plugin", nargs="?", default=None, help=_("Plugin name (omit for --all)"))
    parser.add_argument("--param", action="append", default=[], help=_("Set parameter before dump (key=value)"))
    parser.add_argument("--preset", help=_("Preset name (apply before dump)"))
    parser.add_argument("--save-preset", metavar="NAME", help=_("Save dump as preset"))
    # 배치 모드
    parser.add_argument("--all", action="store_true", help=_("Dump all plugins as JSONL"))
    parser.add_argument("--filter", metavar="KEYWORD", help=_("Plugin name filter (with --all)"))
    parser.add_argument("--format-filter", choices=["vst3", "au"], help=_("Format filter (with --all)"))
    parser.add_argument("--output-file", "-o", metavar="PATH", help=_("JSONL output file (default: stdout)"))
    parser.set_defaults(func=run)


def _dump_plugin_state(wrapper: VST3PluginWrapper, meta) -> dict:
    """플러그인의 전체 파라미터 상태를 dict로 추출"""
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
        print_error("플러그인 이름 또는 --all 플래그가 필요합니다")


def _run_single(args: argparse.Namespace) -> None:
    registry = get_registry()
    meta = registry.get(args.plugin)
    if not meta:
        print_error(f"플러그인을 찾을 수 없습니다: '{args.plugin}'")

    wrapper = VST3PluginWrapper(meta.path)
    wrapper.load()

    # 프리셋 적용
    if args.preset:
        from audioman.core.preset_manager import PresetManager
        manager = PresetManager()
        try:
            preset = manager.load(args.preset, plugin=meta.short_name)
            wrapper.set_parameters(preset.parameters)
        except FileNotFoundError as e:
            print_error(str(e))

    # CLI 파라미터 적용
    if args.param:
        params = parse_params(args.param)
        wrapper.set_parameters(params)

    state = _dump_plugin_state(wrapper, meta)

    # 프리셋으로 저장
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

    print_json({"command": "dump", **state})


def _run_batch(args: argparse.Namespace) -> None:
    """모든 플러그인의 기본 파라미터 상태를 JSONL로 덤프"""
    registry = get_registry()
    plugins = registry.list(fmt=args.format_filter)

    # 키워드 필터
    if args.filter:
        keyword = args.filter.lower()
        plugins = [p for p in plugins if keyword in p.name.lower() or keyword in p.short_name]

    if not plugins:
        print_error("조건에 맞는 플러그인이 없습니다")

    # 출력 대상
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
                line = json.dumps(state, ensure_ascii=False, default=str)
                out_file.write(line + "\n")
                ok += 1

                if out_file is not sys.stdout:
                    output_console.print(f"  [{i+1}/{len(plugins)}] {meta.short_name} ({state['parameter_count']} params)")

            except Exception as e:
                fail += 1
                # 실패해도 JSONL에 에러 레코드 기록
                err_line = json.dumps({
                    "plugin": meta.name,
                    "short_name": meta.short_name,
                    "path": meta.path,
                    "error": str(e),
                }, ensure_ascii=False)
                out_file.write(err_line + "\n")

                if out_file is not sys.stdout:
                    print_warning(f"  [{i+1}/{len(plugins)}] {meta.short_name}: {e}")

    finally:
        if out_file is not sys.stdout:
            out_file.close()

    if out_file is not sys.stdout or args.output_file:
        output_console.print(f"\n덤프 완료: {ok} 성공, {fail} 실패 / {len(plugins)} 전체")
        if args.output_file:
            output_console.print(f"출력: {args.output_file}")
