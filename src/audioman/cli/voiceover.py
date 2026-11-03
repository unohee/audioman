# Created: 2026-04-27
# Purpose: audioman vo subcommand - voiceover analysis/denoise/leveling batch workflow

from __future__ import annotations

import argparse
from pathlib import Path

from audioman.cli.output import (
    output_console,
    print_error,
    print_json,
    print_success,
    print_markup,
)
from audioman.core import voiceover
from audioman.core.engine import parse_params
from audioman.core.findings import json_envelope, schema_uri


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "vo",
        help="Voiceover workflow (VAD + denoise + per-utterance LUFS leveling)",
    )
    sub = parser.add_subparsers(dest="vo_command", required=True)

    # vo analyze
    p_analyze = sub.add_parser(
        "analyze",
        help="VAD + statistics only - does not edit audio",
    )
    p_analyze.add_argument("input", help="Input audio file")
    _add_vad_args(p_analyze)
    p_analyze.add_argument(
        "--segments", action="store_true",
        help="In JSON mode print every segment in detail (default: summary)",
    )
    p_analyze.set_defaults(func=_run_analyze)

    # vo process
    p_proc = sub.add_parser(
        "process",
        help="Batch workflow: VAD -> denoise -> per-utterance LUFS leveling",
    )
    p_proc.add_argument("input", help="Input audio file")
    p_proc.add_argument("--output", "-o", required=True, help="Output file")
    p_proc.add_argument(
        "--target-lufs", type=float, default=-20.0,
        help="Target LUFS per utterance (default: -20)",
    )
    p_proc.add_argument(
        "--max-true-peak", type=float, default=-1.0,
        help="True peak ceiling in dBTP (default: -1)",
    )
    p_proc.add_argument(
        "--noise-attenuation", type=float, default=-12.0,
        help="Extra attenuation for non-speech regions in dB (default: -12)",
    )
    p_proc.add_argument(
        "--denoise-plugin", default="voice-de-noise",
        help='RX denoise plugin short name (default: voice-de-noise, "none" to skip)',
    )
    p_proc.add_argument(
        "--denoise-param", action="append", default=[],
        help="Denoise plugin parameters (key=value, repeatable)",
    )
    _add_vad_args(p_proc)
    p_proc.set_defaults(func=_run_process)


def _add_vad_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="Silero VAD confidence threshold (default: 0.5)")
    parser.add_argument("--min-speech-ms", type=int, default=250,
                        help="Speech regions shorter than this are ignored (default: 250)")
    parser.add_argument("--min-silence-ms", type=int, default=200,
                        help="Silence shorter than this does not split an utterance (default: 200)")
    parser.add_argument("--speech-pad-ms", type=int, default=80,
                        help="Padding on both sides of a detected region (default: 80)")


def _run_analyze(args: argparse.Namespace) -> None:
    if not Path(args.input).exists():
        print_error(f"File not found: {args.input}")

    try:
        result = voiceover.analyze(
            args.input,
            vad_threshold=args.vad_threshold,
            min_speech_ms=args.min_speech_ms,
            min_silence_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
        )
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        return

    if args.json:
        if not args.segments:
            result.pop("speech_segments", None)
            result.pop("noise_segments", None)
        print_json(json_envelope("vo analyze", result, schema=schema_uri("voiceover")))
        return

    print_markup(f"\n[bold]Voiceover analysis[/bold]: {result['input']}")
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
        print_markup("\n  [dim]Speech segments:[/dim]")
        for s in result["speech_segments"][:30]:
            output_console.print(
                f"    {s['start_sec']:>7.2f}s - {s['end_sec']:>7.2f}s  ({s['duration_sec']:.2f}s)"
            )
        more = len(result["speech_segments"]) - 30
        if more > 0:
            print_markup(f"    [dim]... +{more} more[/dim]")


def _run_process(args: argparse.Namespace) -> None:
    if not Path(args.input).exists():
        print_error(f"File not found: {args.input}")

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
        print_error(f"Processing failed: {e}")
        return

    data = result.to_dict()

    if args.json:
        # per_segment is long, so keep only the summary counts
        leveling = data.get("leveling") or {}
        if "per_segment" in leveling:
            leveling["per_segment_count"] = len(leveling["per_segment"])
            leveling.pop("per_segment", None)
        print_json(json_envelope("vo process", data, schema=schema_uri("voiceover")))
        return

    print_success("Voiceover complete")
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
