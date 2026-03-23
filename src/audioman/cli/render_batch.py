# Created: 2026-03-23
# Purpose: audioman render-batch — 프리셋 배치 렌더링 → 데이터셋 패키징

import argparse
import time

from audioman.cli.output import print_error, print_json, print_success, print_info, output_console
from audioman.core.instrument import parse_note


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "render-batch",
        help="프리셋 디렉토리 배치 렌더링 → 데이터셋 (npy + jsonl + manifest)",
    )

    # 필수
    parser.add_argument(
        "--plugin", "-p",
        required=True,
        help="VST3 인스트루먼트 플러그인 (이름 또는 경로)",
    )
    parser.add_argument(
        "--preset-dir",
        default=None,
        help="프리셋 디렉토리 경로 (--program-scan 사용 시 불필요)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="출력 데이터셋 디렉토리",
    )

    # 렌더링 옵션
    parser.add_argument(
        "--note", "-n",
        action="append",
        default=None,
        help="MIDI 노트 (기본: C4). 여러 개 가능: -n C4 -n C3",
    )
    parser.add_argument(
        "--velocity", "-vel",
        type=int,
        default=100,
        help="벨로시티 (기본: 100)",
    )
    parser.add_argument(
        "--velocity-layers",
        type=str,
        default=None,
        help="벨로시티 레이어 (쉼표 구분, 예: 25,50,75,100,127)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=2.0,
        help="노트 지속 시간 - 초 (기본: 2.0)",
    )
    parser.add_argument(
        "--tail",
        type=float,
        default=1.0,
        help="노트 이후 추가 렌더링 시간 (기본: 1.0)",
    )
    parser.add_argument(
        "--fadeout",
        type=float,
        default=0.5,
        help="끝부분 페이드아웃 (기본: 0.5)",
    )
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=44100,
        help="샘플레이트 (기본: 44100)",
    )

    # 출력 포맷
    parser.add_argument(
        "--format", "-f",
        choices=["npy", "wav", "both"],
        default="npy",
        help="오디오 출력 포맷 (기본: npy)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="오디오 정규화 비활성화",
    )

    # Mel-spectrogram
    parser.add_argument(
        "--mel",
        action="store_true",
        help="Mel-spectrogram 추출 (mel.npy)",
    )
    parser.add_argument(
        "--mel-bands",
        type=int,
        default=128,
        help="Mel 밴드 수 (기본: 128)",
    )
    parser.add_argument(
        "--mel-fmax",
        type=int,
        default=8000,
        help="Mel 최대 주파수 (기본: 8000)",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="STFT hop 길이 (기본: 512)",
    )

    # 프로그램 스캔 모드 (내장 프리셋 플러그인용)
    parser.add_argument(
        "--program-scan",
        type=int,
        metavar="N",
        default=None,
        help="프리셋 파일 없이 플러그인 내장 프로그램 N개 순회 (KORG, Roland 등)",
    )

    # 기타
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="하위 디렉토리 포함 (기본: True)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="하위 디렉토리 제외",
    )

    parser.set_defaults(func=run)


def _resolve_plugin_path(plugin_arg: str) -> str:
    """플러그인 인자 → 실제 경로 해석"""
    from pathlib import Path

    p = Path(plugin_arg)
    if p.exists() and p.suffix.lower() in (".vst3", ".component"):
        return str(p)

    from audioman.core.registry import get_registry

    registry = get_registry()
    meta = registry.get(plugin_arg)
    if meta:
        return meta.path

    print_error(f"플러그인을 찾을 수 없습니다: '{plugin_arg}'")


def run(args: argparse.Namespace) -> None:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

    plugin_path = _resolve_plugin_path(args.plugin)

    # 노트 파싱
    if args.note:
        notes = [parse_note(n) for n in args.note]
    else:
        notes = [60]  # C4

    # --- 프로그램 스캔 모드 (내장 프리셋) ---
    if args.program_scan is not None:
        from audioman.core.batch_render import batch_render_programs
        return _run_program_scan(args, plugin_path, notes)

    from audioman.core.batch_render import BatchConfig, batch_render

    if not args.preset_dir:
        print_error("--preset-dir 필수 (내장 프리셋은 --program-scan N 사용)")

    # velocity 레이어 파싱
    vel_layers = []
    if args.velocity_layers:
        vel_layers = [int(v.strip()) for v in args.velocity_layers.split(",")]

    config = BatchConfig(
        plugin_path=plugin_path,
        preset_dir=args.preset_dir,
        output_dir=args.output,
        notes=notes,
        velocity=args.velocity,
        velocity_layers=vel_layers,
        duration=args.duration,
        tail=args.tail,
        fadeout=args.fadeout,
        sample_rate=args.sample_rate,
        buffer_size=512,
        recursive=not args.no_recursive,
        format=args.format,
        mel=args.mel,
        mel_bands=args.mel_bands,
        mel_fmax=args.mel_fmax,
        hop_length=args.hop_length,
        normalize_audio=not args.no_normalize,
    )

    # Rich 진행률 표시
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=output_console,
    ) as progress:
        task_id = progress.add_task("렌더링", total=0)

        def on_progress(current, total, entry):
            if progress.tasks[task_id].total == 0:
                progress.update(task_id, total=total)
            status = "OK" if not entry.error else f"FAIL: {entry.error}"
            progress.update(
                task_id,
                completed=current,
                description=f"[{current}/{total}] {entry.preset_name[:30]}",
            )

        result = batch_render(config, progress_callback=on_progress)

    # 결과 출력
    if getattr(args, "json", False):
        print_json({
            "command": "render-batch",
            "total": result.total,
            "success": result.success,
            "failed": result.failed,
            "duration_seconds": result.duration_seconds,
            "output_dir": result.output_dir,
        })
    else:
        output_console.print()
        print_success(f"배치 렌더링 완료: {result.success}/{result.total} 성공 ({result.duration_seconds:.1f}s)")
        if result.failed:
            output_console.print(f"  [yellow]실패: {result.failed}[/yellow]")
        output_console.print(f"  출력: {result.output_dir}/")

        from pathlib import Path
        out = Path(result.output_dir)
        for f in sorted(out.iterdir()):
            if f.is_file():
                size = f.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f}MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                output_console.print(f"    {f.name:20s} {size_str:>10s}")


def _run_program_scan(args, plugin_path: str, notes: list[int]) -> None:
    """내장 프로그램 스캔 모드"""
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from audioman.core.batch_render import batch_render_programs

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=output_console,
    ) as progress:
        task_id = progress.add_task("프로그램 스캔", total=args.program_scan)

        def on_progress(current, total, entry):
            progress.update(
                task_id,
                completed=current,
                total=total,
                description=f"[{current}/{total}] {entry.preset_name[:30]}",
            )

        result = batch_render_programs(
            plugin_path=plugin_path,
            output_dir=args.output,
            num_programs=args.program_scan,
            notes=notes,
            velocity=args.velocity,
            duration=args.duration,
            tail=args.tail,
            fadeout=args.fadeout,
            sample_rate=args.sample_rate,
            mel=args.mel,
            mel_bands=args.mel_bands,
            mel_fmax=args.mel_fmax,
            hop_length=args.hop_length,
            normalize_audio=not args.no_normalize,
            progress_callback=on_progress,
        )

    if getattr(args, "json", False):
        print_json({
            "command": "render-batch",
            "mode": "program_scan",
            "total": result.total,
            "success": result.success,
            "failed": result.failed,
            "duration_seconds": result.duration_seconds,
            "output_dir": result.output_dir,
        })
    else:
        output_console.print()
        print_success(f"프로그램 스캔 완료: {result.success} 사운드 / {result.total} 프로그램 ({result.duration_seconds:.1f}s)")
        output_console.print(f"  출력: {result.output_dir}/")

        from pathlib import Path
        out = Path(result.output_dir)
        for f in sorted(out.iterdir()):
            if f.is_file():
                size = f.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f}MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                output_console.print(f"    {f.name:20s} {size_str:>10s}")
