# Created: 2026-03-21
# Purpose: audioman chain 서브커맨드 (단일 + 배치)

import argparse
import json

from audioman.cli.output import print_error, print_json, print_success, print_warning, output_console
from audioman.core.pipeline import parse_chain_string, run_pipeline
from audioman.core.batch import collect_audio_files, resolve_output_path
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("chain", help="다중 플러그인 순차 처리")
    parser.add_argument("input", help="입력 오디오 파일 또는 디렉토리")
    parser.add_argument(
        "--steps", "-s", required=True,
        help="처리 체인 (예: 'dehum:notch_frequency=60,declick,denoise:noise_reduction_db=15')",
    )
    parser.add_argument("--output", "-o", required=True, help="출력 파일 또는 디렉토리")
    parser.add_argument("--recursive", "-r", action="store_true", help="하위 디렉토리 포함 (배치)")
    parser.add_argument("--suffix", default="", help="출력 파일명 접미사 (배치)")
    parser.add_argument("--dry-run", action="store_true", help="실행하지 않고 계획만 표시")
    parser.add_argument("--workers", "-w", type=int, default=1, help="병렬 처리 워커 수 (기본: 1)")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    steps = parse_chain_string(args.steps)
    if not steps:
        print_error("처리 단계가 비어있습니다")

    input_path = Path(args.input)

    if input_path.is_dir():
        _run_batch(args, steps, input_path)
    else:
        _run_single(args, steps)


def _run_single(args: argparse.Namespace, steps) -> None:
    if args.dry_run:
        plan = {
            "command": "chain",
            "dry_run": True,
            "input": args.input,
            "output": args.output,
            "steps": [s.to_dict() for s in steps],
        }
        if args.json:
            print_json(plan)
        else:
            output_console.print(f"[dry-run] {args.input}")
            for i, s in enumerate(steps, 1):
                params_str = f" ({s.params})" if s.params else ""
                output_console.print(f"  → [{s.plugin_name}{params_str}]")
            output_console.print(f"  → {args.output}")
        return

    try:
        result = run_pipeline(
            input_path=args.input,
            output_path=args.output,
            steps=steps,
        )
    except (FileNotFoundError, ValueError) as e:
        print_error(str(e))
    except Exception as e:
        print_error(f"체인 처리 실패: {e}")

    if args.json:
        print_json({"command": "chain", **result.to_dict()})
        return

    output_console.print(f"\n[bold]체인 처리 완료[/bold]")
    output_console.print(f"  Steps: {len(result.steps)}")
    for i, s in enumerate(result.steps, 1):
        output_console.print(f"    {i}. {s['plugin']}")
    output_console.print(f"  Input:  {result.input_path}")
    output_console.print(f"  Output: {result.output_path}")
    output_console.print(f"  Time:   {result.duration_seconds}s")
    print_success("완료")


def _run_batch(args: argparse.Namespace, steps, input_dir: Path) -> None:
    output_dir = Path(args.output)
    files = collect_audio_files(input_dir, recursive=args.recursive)

    if not files:
        print_error(f"오디오 파일이 없습니다: {input_dir}")

    step_names = " → ".join(s.plugin_name for s in steps)

    if args.dry_run:
        plan = {
            "command": "chain",
            "dry_run": True,
            "batch": True,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "file_count": len(files),
            "steps": [s.to_dict() for s in steps],
            "files": [str(f) for f in files],
        }
        if args.json:
            print_json(plan)
        else:
            output_console.print(f"[dry-run] 배치: {len(files)}개 파일 → [{step_names}] → {output_dir}")
        return

    jobs = []
    # steps를 직렬화 (multiprocessing 전달용)
    steps_dicts = [s.to_dict() for s in steps]
    for fpath in files:
        out_path = resolve_output_path(fpath, input_dir, output_dir, suffix=args.suffix)
        jobs.append((str(fpath), str(out_path), steps_dicts))

    if args.workers > 1:
        _run_chain_parallel(args, jobs, len(files))
    else:
        _run_chain_sequential(args, jobs, steps, len(files))


def _chain_one(job_args):
    """체인 멀티프로세싱 워커"""
    from audioman.core.pipeline import parse_chain_string, run_pipeline, ChainStep
    fpath, out_path, steps_dicts = job_args
    steps = [ChainStep(**d) for d in steps_dicts]
    try:
        result = run_pipeline(input_path=fpath, output_path=out_path, steps=steps)
        return {"ok": True, "result": result.to_dict(), "input": fpath}
    except Exception as e:
        return {"ok": False, "error": str(e), "input": fpath}


def _run_chain_sequential(args, jobs, steps, total):
    ok, fail = 0, 0
    for i, (fpath, out_path, _) in enumerate(jobs):
        try:
            result = run_pipeline(input_path=fpath, output_path=out_path, steps=steps)
            ok += 1
            if args.json:
                print(json.dumps({"command": "chain", **result.to_dict()}, ensure_ascii=False, default=str))
            else:
                output_console.print(f"  [{i+1}/{total}] {Path(fpath).name} → {Path(out_path).name} ({result.duration_seconds}s)")
        except Exception as e:
            fail += 1
            if args.json:
                print(json.dumps({"command": "chain", "input": fpath, "error": str(e)}, ensure_ascii=False))
            else:
                print_warning(f"  [{i+1}/{total}] {Path(fpath).name}: {e}")
    if not args.json:
        print_success(f"배치 완료: {ok} 성공, {fail} 실패 / {total} 전체")


def _run_chain_parallel(args, jobs, total):
    from multiprocessing import Pool
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

    ok, fail = 0, 0
    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        console=output_console,
    ) as progress:
        task_id = progress.add_task(f"체인 ({args.workers} workers)", total=total)
        with Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(_chain_one, jobs):
                if r["ok"]:
                    ok += 1
                    if args.json:
                        print(json.dumps({"command": "chain", **r["result"]}, ensure_ascii=False, default=str))
                else:
                    fail += 1
                    if args.json:
                        print(json.dumps({"command": "chain", "input": r["input"], "error": r["error"]}, ensure_ascii=False))
                progress.update(task_id, advance=1, description=f"[{ok+fail}/{total}] {Path(r['input']).name}")
    if not args.json:
        print_success(f"배치 완료: {ok} 성공, {fail} 실패 / {total} 전체 ({args.workers} workers)")
