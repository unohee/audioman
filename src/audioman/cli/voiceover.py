# Created: 2026-04-27
# Purpose: audioman vo 서브명령 — 보이스오버 분석/디노이즈/레벨링 일괄 처리

from __future__ import annotations

import argparse
from pathlib import Path

from audioman.cli.output import (
    output_console,
    print_error,
    print_json,
)
from audioman.core import voiceover
from audioman.core.engine import parse_params


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "vo",
        help="Voiceover workflow (VAD + denoise + per-utterance LUFS leveling)",
    )
    sub = parser.add_subparsers(dest="vo_command", required=True)

    # vo analyze
    p_analyze = sub.add_parser(
        "analyze",
        help="VAD + 통계만 — 편집하지 않음",
    )
    p_analyze.add_argument("input", help="입력 오디오 파일")
    _add_vad_args(p_analyze)
    p_analyze.add_argument(
        "--segments", action="store_true",
        help="JSON 모드에서 모든 segment 상세 출력 (기본은 요약)",
    )
    p_analyze.set_defaults(func=_run_analyze)

    # vo process
    p_proc = sub.add_parser(
        "process",
        help="일괄 처리: VAD → denoise → utterance LUFS leveling",
    )
    p_proc.add_argument("input", help="입력 오디오 파일")
    p_proc.add_argument("--output", "-o", required=True, help="출력 파일")
    p_proc.add_argument(
        "--target-lufs", type=float, default=-20.0,
        help="발화 단위 목표 LUFS (default: -20)",
    )
    p_proc.add_argument(
        "--max-true-peak", type=float, default=-1.0,
        help="True Peak 천장 dBTP (default: -1)",
    )
    p_proc.add_argument(
        "--noise-attenuation", type=float, default=-12.0,
        help="비음성 구간 추가 감쇠 dB (default: -12)",
    )
    p_proc.add_argument(
        "--denoise-plugin", default="voice-de-noise",
        help='RX denoise 플러그인 short name (default: voice-de-noise, "none"이면 skip)',
    )
    p_proc.add_argument(
        "--denoise-param", action="append", default=[],
        help="디노이즈 플러그인 파라미터 (key=value, 반복 가능)",
    )
    _add_vad_args(p_proc)
    p_proc.set_defaults(func=_run_process)


def _add_vad_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="Silero VAD confidence threshold (default: 0.5)")
    parser.add_argument("--min-speech-ms", type=int, default=250,
                        help="이보다 짧은 음성 구간은 무시 (default: 250)")
    parser.add_argument("--min-silence-ms", type=int, default=200,
                        help="이보다 짧은 무음은 발화 분할 안 함 (default: 200)")
    parser.add_argument("--speech-pad-ms", type=int, default=80,
                        help="검출 구간 양쪽 padding (default: 80)")


def _run_analyze(args: argparse.Namespace) -> None:
    if not Path(args.input).exists():
        print_error(f"파일이 없습니다: {args.input}")

    try:
        result = voiceover.analyze(
            args.input,
            vad_threshold=args.vad_threshold,
            min_speech_ms=args.min_speech_ms,
            min_silence_ms=args.min_silence_ms,
        )
    except Exception as e:
        print_error(f"분석 실패: {e}")
        return

    if args.json:
        if not args.segments:
            result.pop("speech_segments", None)
            result.pop("noise_segments", None)
        print_json(result)
        return

    output_console.print(f"\n[bold]보이스오버 분석[/bold]: {result['input']}")
    output_console.print(f"  Duration: {result['duration_sec']}s @ {result['sample_rate']}Hz")
    output_console.print(f"  Speech segments: {result['n_speech_segments']}")
    output_console.print(
        f"  Speech: {result['speech_total_sec']}s "
        f"({result['speech_ratio']*100:.1f}%) | "
        f"Noise: {result['noise_total_sec']}s"
    )
    loud = result["loudness"]
    output_console.print(
        f"  Integrated LUFS: {loud.get('integrated_lufs')} | "
        f"True Peak: {loud.get('true_peak_dbtp')} dBTP | "
        f"LRA: {loud.get('loudness_range_lu')} LU"
    )
    if args.segments:
        output_console.print("\n  [dim]Speech segments:[/dim]")
        for s in result["speech_segments"][:30]:
            output_console.print(
                f"    {s['start_sec']:>7.2f}s - {s['end_sec']:>7.2f}s  ({s['duration_sec']:.2f}s)"
            )
        more = len(result["speech_segments"]) - 30
        if more > 0:
            output_console.print(f"    [dim]... +{more} more[/dim]")


def _run_process(args: argparse.Namespace) -> None:
    if not Path(args.input).exists():
        print_error(f"파일이 없습니다: {args.input}")

    denoise_plugin = args.denoise_plugin
    if denoise_plugin and denoise_plugin.lower() == "none":
        denoise_plugin = None

    denoise_params = parse_params(args.denoise_param) if args.denoise_param else None

    try:
        result = voiceover.process(
            input_path=args.input,
            output_path=args.output,
            target_lufs=args.target_lufs,
            max_true_peak_dbtp=args.max_true_peak,
            noise_attenuation_db=args.noise_attenuation,
            denoise_plugin=denoise_plugin,
            denoise_params=denoise_params,
            vad_threshold=args.vad_threshold,
            min_speech_ms=args.min_speech_ms,
            min_silence_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
        )
    except Exception as e:
        print_error(f"처리 실패: {e}")
        return

    data = result.to_dict()

    if args.json:
        # per_segment는 길어서 요약만
        leveling = data.get("leveling") or {}
        if "per_segment" in leveling:
            leveling["per_segment_count"] = len(leveling["per_segment"])
            leveling.pop("per_segment", None)
        print_json(data)
        return

    output_console.print(f"\n[bold green]Voiceover 처리 완료[/bold green]")
    output_console.print(f"  Input:  {data['input']}")
    output_console.print(f"  Output: {data['output']}")
    output_console.print(
        f"  Speech: {data['n_speech_segments']} segments "
        f"({data['speech_total_sec']}s / {data['duration_sec']}s)"
    )
    if data["denoise_plugin"]:
        output_console.print(f"  Denoise: {data['denoise_plugin']}")
    leveling = data.get("leveling") or {}
    output_console.print(
        f"  Target: {leveling.get('target_lufs')} LUFS | "
        f"Noise atten: {leveling.get('noise_attenuation_db')} dB | "
        f"TP ceil: {leveling.get('max_true_peak_dbtp')} dBTP"
    )
    mi = data["measured_in"]; mo = data["measured_out"]
    output_console.print(
        f"  Loudness  in:  {mi.get('integrated_lufs')} LUFS, "
        f"TP {mi.get('true_peak_dbtp')} dBTP, LRA {mi.get('loudness_range_lu')} LU"
    )
    output_console.print(
        f"  Loudness  out: {mo.get('integrated_lufs')} LUFS, "
        f"TP {mo.get('true_peak_dbtp')} dBTP, LRA {mo.get('loudness_range_lu')} LU"
    )
