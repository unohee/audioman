# Created: 2026-03-21
# Purpose: audioman analyze 서브커맨드 — 오디오 분석

import argparse
import json
from pathlib import Path

from audioman.cli.output import print_error, print_json, print_table, print_success, output_console
from audioman.core.audio_file import read_audio, get_audio_stats
from audioman.core.analysis import compute_frame_metrics, compute_summary, detect_silence
from audioman.core.batch import collect_audio_files


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("analyze", help="오디오 분석 (RMS, spectral entropy, silence 감지 등)")
    parser.add_argument("input", help="입력 오디오 파일 또는 디렉토리")
    parser.add_argument("--frames", action="store_true", help="프레임 단위 상세 출력")
    parser.add_argument("--frame-size", type=int, default=2048, help="프레임 크기 (기본: 2048)")
    parser.add_argument("--hop", type=int, default=512, help="홉 크기 (기본: 512)")
    parser.add_argument("--silence-threshold", type=float, default=-40.0, help="Silence 감지 임계값 dB (기본: -40)")
    parser.add_argument("--recursive", "-r", action="store_true", help="하위 디렉토리 포함")
    parser.set_defaults(func=run)


def _analyze_file(
    path: Path, frame_size: int, hop: int, silence_threshold: float, frames_mode: bool
) -> dict:
    audio, sr = read_audio(path)
    stats = get_audio_stats(audio, sr)

    metrics = compute_frame_metrics(audio, sr, frame_size=frame_size, hop_size=hop)
    summary = compute_summary(metrics)
    silence = detect_silence(audio, sr, threshold_db=silence_threshold)

    result = {
        "file": str(path),
        "sample_rate": sr,
        "channels": stats.channels,
        "duration": round(stats.duration, 4),
        "frames": stats.frames,
        "rms": round(stats.rms, 6),
        "peak": round(stats.peak, 6),
        "summary": summary,
        "silence_regions": [s.to_dict() for s in silence],
        "silence_total_sec": round(sum(s.duration_sec for s in silence), 4),
    }

    if frames_mode:
        result["frame_metrics"] = {
            "frame_size": frame_size,
            "hop_size": hop,
            "n_frames": len(metrics.rms),
            "rms": [round(v, 6) for v in metrics.rms],
            "peak": [round(v, 6) for v in metrics.peak],
            "spectral_centroid": [round(v, 2) for v in metrics.spectral_centroid],
            "spectral_entropy": [round(v, 4) for v in metrics.spectral_entropy],
            "zero_crossing_rate": [round(v, 6) for v in metrics.zero_crossing_rate],
        }

    return result


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)

    if input_path.is_dir():
        _run_batch(args, input_path)
    else:
        _run_single(args, input_path)


def _run_single(args: argparse.Namespace, path: Path) -> None:
    try:
        result = _analyze_file(
            path, args.frame_size, args.hop, args.silence_threshold, args.frames
        )
    except FileNotFoundError as e:
        print_error(str(e))

    if args.json:
        print_json({"command": "analyze", **result})
        return

    # human-readable 출력
    output_console.print(f"\n[bold]{result['file']}[/bold]")
    output_console.print(f"  Duration: {result['duration']}s | SR: {result['sample_rate']}Hz | CH: {result['channels']}")
    output_console.print(f"  RMS: {result['rms']:.4f} | Peak: {result['peak']:.4f}")
    output_console.print()

    # summary 테이블
    rows = []
    for metric, stats in result["summary"].items():
        rows.append([
            metric,
            f"{stats['mean']:.4f}",
            f"{stats['min']:.4f}",
            f"{stats['max']:.4f}",
            f"{stats['std']:.4f}",
        ])
    print_table("Summary", ["Metric", "Mean", "Min", "Max", "Std"], rows)

    # silence
    if result["silence_regions"]:
        output_console.print(f"\n  Silence regions: {len(result['silence_regions'])} ({result['silence_total_sec']}s total)")
        for i, s in enumerate(result["silence_regions"][:10]):
            output_console.print(f"    {i+1}. {s['start_sample']}–{s['end_sample']} ({s['duration_sec']:.3f}s)")
        if len(result["silence_regions"]) > 10:
            output_console.print(f"    ... +{len(result['silence_regions']) - 10} more")
    else:
        output_console.print("\n  Silence regions: none")


def _run_batch(args: argparse.Namespace, input_dir: Path) -> None:
    files = collect_audio_files(input_dir, recursive=args.recursive)
    if not files:
        print_error(f"오디오 파일이 없습니다: {input_dir}")

    for i, fpath in enumerate(files):
        try:
            result = _analyze_file(
                fpath, args.frame_size, args.hop, args.silence_threshold, args.frames
            )
            if args.json:
                print(json.dumps({"command": "analyze", **result}, ensure_ascii=False, default=str))
            else:
                output_console.print(
                    f"  [{i+1}/{len(files)}] {fpath.name}: "
                    f"RMS={result['rms']:.4f} Peak={result['peak']:.4f} "
                    f"Silence={result['silence_total_sec']}s "
                    f"Centroid={result['summary']['spectral_centroid']['mean']:.0f}Hz"
                )
        except Exception as e:
            if args.json:
                print(json.dumps({"command": "analyze", "file": str(fpath), "error": str(e)}, ensure_ascii=False))
            else:
                output_console.print(f"  [{i+1}/{len(files)}] {fpath.name}: ERROR {e}")

    if not args.json:
        print_success(f"분석 완료: {len(files)}개 파일")
