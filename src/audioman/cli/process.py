# Created: 2026-03-21
# Purpose: audioman process 서브커맨드 (단일 + 배치)

import argparse
import json
import sys

from audioman.cli.output import print_error, print_json, print_success, print_warning, output_console
from audioman.core.engine import parse_params, process_file
from audioman.core.batch import collect_audio_files, resolve_output_path
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("process", help="단일 플러그인으로 오디오 처리")
    # 입력: 파일 또는 디렉토리
    parser.add_argument("input", help="입력 오디오 파일 또는 디렉토리")
    parser.add_argument("--plugin", "-p", required=True, help="플러그인 이름")
    parser.add_argument("--param", action="append", default=[], help="파라미터 (key=value)")
    parser.add_argument("--output", "-o", required=True, help="출력 파일 또는 디렉토리")
    parser.add_argument("--passes", type=int, default=1, help="처리 횟수 (2=adaptive 학습용 멀티패스)")
    parser.add_argument("--recursive", "-r", action="store_true", help="하위 디렉토리 포함 (배치)")
    parser.add_argument("--suffix", default="", help="출력 파일명 접미사 (배치)")
    parser.add_argument("--dry-run", action="store_true", help="실행하지 않고 계획만 표시")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    params = parse_params(args.param) if args.param else {}
    input_path = Path(args.input)

    # 배치 모드 판정: 입력이 디렉토리이면 배치
    if input_path.is_dir():
        _run_batch(args, params, input_path)
    else:
        _run_single(args, params)


def _run_single(args: argparse.Namespace, params: dict) -> None:
    if args.dry_run:
        plan = {
            "command": "process",
            "dry_run": True,
            "input": args.input,
            "output": args.output,
            "plugin": args.plugin,
            "params": params,
        }
        if args.json:
            print_json(plan)
        else:
            output_console.print(f"[dry-run] {args.input} → [{args.plugin}] → {args.output}")
            if params:
                output_console.print(f"  params: {params}")
        return

    try:
        result = process_file(
            input_path=args.input,
            output_path=args.output,
            plugin_name=args.plugin,
            params=params,
            passes=args.passes,
        )
    except (FileNotFoundError, ValueError) as e:
        print_error(str(e))
    except Exception as e:
        print_error(f"처리 실패: {e}")

    if args.json:
        print_json({"command": "process", **result.to_dict()})
        return

    output_console.print(f"\n[bold]처리 완료[/bold]")
    output_console.print(f"  Plugin: {result.plugin_name}")
    output_console.print(f"  Input:  {result.input_path}")
    output_console.print(f"  Output: {result.output_path}")
    output_console.print(f"  Time:   {result.duration_seconds}s")
    input_s = result.input_stats
    output_s = result.output_stats
    output_console.print(f"  RMS:    {input_s['rms']:.4f} → {output_s['rms']:.4f}")
    output_console.print(f"  Peak:   {input_s['peak']:.4f} → {output_s['peak']:.4f}")
    print_success("완료")


def _run_batch(args: argparse.Namespace, params: dict, input_dir: Path) -> None:
    output_dir = Path(args.output)
    files = collect_audio_files(input_dir, recursive=args.recursive)

    if not files:
        print_error(f"오디오 파일이 없습니다: {input_dir}")

    if args.dry_run:
        plan = {
            "command": "process",
            "dry_run": True,
            "batch": True,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "file_count": len(files),
            "plugin": args.plugin,
            "params": params,
            "files": [str(f) for f in files],
        }
        if args.json:
            print_json(plan)
        else:
            output_console.print(f"[dry-run] 배치: {len(files)}개 파일 → [{args.plugin}] → {output_dir}")
        return

    ok, fail = 0, 0
    for i, fpath in enumerate(files):
        out_path = resolve_output_path(fpath, input_dir, output_dir, suffix=args.suffix)

        try:
            result = process_file(
                input_path=fpath,
                output_path=out_path,
                plugin_name=args.plugin,
                params=params,
                passes=args.passes,
            )
            ok += 1

            if args.json:
                # JSONL: 한 줄씩 출력
                print(json.dumps({"command": "process", **result.to_dict()}, ensure_ascii=False, default=str))
            else:
                output_console.print(
                    f"  [{i+1}/{len(files)}] {fpath.name} → {out_path.name} "
                    f"({result.duration_seconds}s)"
                )

        except Exception as e:
            fail += 1
            if args.json:
                print(json.dumps({"command": "process", "input": str(fpath), "error": str(e)}, ensure_ascii=False))
            else:
                print_warning(f"  [{i+1}/{len(files)}] {fpath.name}: {e}")

    if not args.json:
        print_success(f"배치 완료: {ok} 성공, {fail} 실패 / {len(files)} 전체")
