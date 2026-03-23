# Created: 2026-03-23
# Purpose: audioman preset-dump — 범용 프리셋 파일 파싱 및 JSONL 덤프

import argparse
import json
import sys

from audioman.cli.output import print_error, print_json, print_warning, output_console
from audioman.core.preset_parser import (
    PresetData,
    SUPPORTED_EXTENSIONS,
    find_presets,
    parse_auto,
    resolve_param_names,
)
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "preset-dump",
        help="프리셋 파일 파싱 → JSON/JSONL 덤프 (FXP/FXB, vstpreset, aupreset)",
    )
    parser.add_argument(
        "input",
        help="프리셋 파일 또는 디렉토리 경로",
    )
    parser.add_argument(
        "--output-file", "-o",
        metavar="PATH",
        help="JSONL 출력 파일 (기본: stdout)",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="하위 디렉토리 포함 (디렉토리 입력 시)",
    )
    parser.add_argument(
        "--include-chunk",
        action="store_true",
        help="opaque chunk 데이터를 hex로 포함",
    )
    parser.add_argument(
        "--resolve-names",
        action="store_true",
        help="DawDreamer로 플러그인 로딩 후 실제 파라미터 이름 매핑",
    )
    parser.add_argument(
        "--plugin", "-p",
        help="--resolve-names 사용 시 플러그인 경로 또는 이름",
    )
    parser.add_argument(
        "--filter",
        metavar="KEYWORD",
        help="파일명 키워드 필터 (디렉토리 모드)",
    )
    parser.set_defaults(func=run)


def _resolve_parameter_names(
    preset: PresetData, plugin_path: str
) -> dict[str, float]:
    """DawDreamer로 프리셋을 로딩하여 실제 파라미터 이름 매핑"""
    try:
        import dawdreamer as daw
    except ImportError:
        print_warning("dawdreamer 미설치 — --resolve-names 무시 (pip install dawdreamer)")
        return preset.parameters

    engine = daw.RenderEngine(44100, 512)
    plugin = engine.make_plugin_processor("resolver", plugin_path)

    # 프리셋 파일 로딩 시도
    ext = Path(preset.file_path).suffix.lower()
    try:
        if ext in (".fxp", ".fxb"):
            plugin.load_preset(preset.file_path)
        elif ext == ".vstpreset":
            plugin.load_vst3_preset(preset.file_path)
        else:
            print_warning(f"DawDreamer 프리셋 로딩 미지원 포맷: {ext}")
            return preset.parameters
    except Exception as e:
        print_warning(f"프리셋 로딩 실패: {e}")
        return preset.parameters

    # 파라미터 이름과 값 추출
    desc = plugin.get_plugin_parameter_size()
    resolved = {}
    for i in range(desc):
        name = plugin.get_parameter_name(i)
        value = plugin.get_parameter(i)
        resolved[name] = round(float(value), 6)

    return resolved


def _dump_single(
    path: Path,
    include_chunk: bool = False,
    resolve_names: bool = False,
    plugin_path: str | None = None,
) -> dict | None:
    """단일 프리셋 파싱 → dict"""
    try:
        preset = parse_auto(path)

        # 파라미터 이름 매핑 (DawDreamer 자동 감지 또는 --resolve-names)
        if resolve_names and plugin_path:
            preset.parameters = _resolve_parameter_names(preset, plugin_path)
            preset.num_params = len(preset.parameters)
        else:
            # 알려진 플러그인이면 자동으로 이름 매핑 시도
            resolve_param_names(preset, plugin_path)

        return preset.to_dict(include_chunk=include_chunk)

    except Exception as e:
        return {
            "file_path": str(path),
            "error": str(e),
        }


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)

    if not input_path.exists():
        print_error(f"경로 없음: {input_path}")

    # 플러그인 경로 해석 (--resolve-names)
    plugin_path = None
    if args.resolve_names:
        if not args.plugin:
            print_error("--resolve-names 사용 시 --plugin 경로 필수")
        plugin_path = args.plugin
        # 레지스트리에서 검색 시도
        if not Path(plugin_path).exists():
            try:
                from audioman.core.registry import get_registry
                registry = get_registry()
                meta = registry.get(plugin_path)
                if meta:
                    plugin_path = meta.path
                else:
                    print_error(f"플러그인을 찾을 수 없습니다: {plugin_path}")
            except Exception:
                print_error(f"플러그인 경로가 올바르지 않습니다: {plugin_path}")

    # 단일 파일 모드
    if input_path.is_file():
        result = _dump_single(
            input_path,
            include_chunk=args.include_chunk,
            resolve_names=args.resolve_names,
            plugin_path=plugin_path,
        )
        if result:
            print_json({"command": "preset-dump", **result})
        return

    # 디렉토리 모드 → JSONL
    presets = find_presets(input_path, recursive=args.recursive)

    # 키워드 필터
    if args.filter:
        keyword = args.filter.lower()
        presets = [p for p in presets if keyword in p.name.lower()]

    if not presets:
        print_error(f"프리셋 파일 없음: {input_path} (지원: {', '.join(sorted(SUPPORTED_EXTENSIONS))})")

    # 출력 대상
    if args.output_file:
        out_file = open(args.output_file, "w")
    else:
        out_file = sys.stdout

    ok, fail = 0, 0
    try:
        for i, preset_path in enumerate(presets):
            result = _dump_single(
                preset_path,
                include_chunk=args.include_chunk,
                resolve_names=args.resolve_names,
                plugin_path=plugin_path,
            )
            if result:
                line = json.dumps(result, ensure_ascii=False, default=str)
                out_file.write(line + "\n")

                if "error" in result:
                    fail += 1
                    if out_file is not sys.stdout:
                        print_warning(f"  [{i+1}/{len(presets)}] {preset_path.name}: {result['error']}")
                else:
                    ok += 1
                    if out_file is not sys.stdout:
                        name = result.get("preset_name", preset_path.stem)
                        n = result.get("num_params", 0)
                        output_console.print(f"  [{i+1}/{len(presets)}] {name} ({n} params)")

    finally:
        if out_file is not sys.stdout:
            out_file.close()

    if out_file is not sys.stdout or args.output_file:
        output_console.print(f"\n덤프 완료: {ok} 성공, {fail} 실패 / {len(presets)} 전체")
        if args.output_file:
            output_console.print(f"출력: {args.output_file}")
