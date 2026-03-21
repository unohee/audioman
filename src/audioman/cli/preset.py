# Created: 2026-03-21
# Purpose: audioman preset 서브커맨드

import argparse

from audioman.cli.output import print_error, print_json, print_success, print_table, output_console
from audioman.config.paths import ensure_app_dirs
from audioman.core.engine import parse_params
from audioman.core.preset_manager import PresetManager


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("preset", help="프리셋 관리")
    preset_sub = parser.add_subparsers(dest="preset_command")

    # save
    save_p = preset_sub.add_parser("save", help="프리셋 저장")
    save_p.add_argument("name", help="프리셋 이름")
    save_p.add_argument("--plugin", "-p", required=True, help="플러그인 이름")
    save_p.add_argument("--param", action="append", default=[], help="파라미터 (key=value)")
    save_p.add_argument("--description", "-d", default="", help="설명")
    save_p.set_defaults(func=run_save)

    # load
    load_p = preset_sub.add_parser("load", help="프리셋 정보 표시")
    load_p.add_argument("name", help="프리셋 이름")
    load_p.add_argument("--plugin", "-p", help="플러그인 이름 (선택)")
    load_p.set_defaults(func=run_load)

    # list
    list_p = preset_sub.add_parser("list", help="프리셋 목록")
    list_p.add_argument("--plugin", "-p", help="플러그인 필터")
    list_p.set_defaults(func=run_list)

    # delete
    del_p = preset_sub.add_parser("delete", help="프리셋 삭제")
    del_p.add_argument("name", help="프리셋 이름")
    del_p.add_argument("--plugin", "-p", help="플러그인 이름 (선택)")
    del_p.set_defaults(func=run_delete)

    parser.set_defaults(func=lambda args: parser.print_help())


def run_save(args: argparse.Namespace) -> None:
    ensure_app_dirs()
    manager = PresetManager()
    params = parse_params(args.param) if args.param else {}

    path = manager.save(
        name=args.name,
        plugin=args.plugin,
        params=params,
        description=args.description,
    )

    if args.json:
        print_json({"command": "preset save", "name": args.name, "path": str(path)})
    else:
        print_success(f"프리셋 저장: {path}")


def run_load(args: argparse.Namespace) -> None:
    manager = PresetManager()
    try:
        preset = manager.load(args.name, plugin=args.plugin)
    except FileNotFoundError as e:
        print_error(str(e))

    if args.json:
        print_json({"command": "preset load", **preset.to_dict()})
    else:
        output_console.print(f"\n[bold]{preset.name}[/bold] ({preset.plugin})")
        if preset.description:
            output_console.print(f"  {preset.description}")
        output_console.print(f"  Created: {preset.created}")
        for k, v in preset.parameters.items():
            output_console.print(f"  {k}: {v}")


def run_list(args: argparse.Namespace) -> None:
    manager = PresetManager()
    presets = manager.list(plugin=args.plugin)

    if args.json:
        print_json({
            "command": "preset list",
            "count": len(presets),
            "presets": [p.to_dict() for p in presets],
        })
        return

    if not presets:
        output_console.print("저장된 프리셋 없음")
        return

    rows = []
    for p in presets:
        param_count = str(len(p.parameters))
        rows.append([p.name, p.plugin, param_count, p.description or "-"])

    print_table("프리셋 목록", ["Name", "Plugin", "Params", "Description"], rows)


def run_delete(args: argparse.Namespace) -> None:
    manager = PresetManager()
    try:
        manager.delete(args.name, plugin=args.plugin)
    except FileNotFoundError as e:
        print_error(str(e))

    if args.json:
        print_json({"command": "preset delete", "name": args.name})
    else:
        print_success(f"프리셋 삭제: {args.name}")
