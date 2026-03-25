# Created: 2026-03-21
# Purpose: audioman fx 서브커맨드 — 내장 DSP 이펙트

import argparse
import json
import time
from pathlib import Path

import numpy as np

from audioman.cli.output import print_error, print_json, print_success, print_warning, output_console
from audioman.core.audio_file import read_audio, write_audio, get_audio_stats
from audioman.core.batch import collect_audio_files, resolve_output_path
from audioman.core import dsp
from audioman.i18n import _


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("fx", help=_("Built-in DSP effects (fade, trim, normalize, gate, gain)"))
    parser.add_argument("input", help=_("Input audio file or directory"))

    fx_sub = parser.add_subparsers(dest="effect", help=_("Effect type"))

    # fade-in
    fi = fx_sub.add_parser("fade-in", help=_("Linear fade in"))
    fi.add_argument("--samples", type=int, default=None, help=_("Fade length (samples)"))
    fi.add_argument("--duration", type=float, default=None, help=_("Fade length (seconds)"))
    fi.add_argument("--output", "-o", required=True, help=_("Output path"))
    fi.add_argument("--recursive", "-r", action="store_true")
    fi.add_argument("--suffix", default="")

    # fade-out
    fo = fx_sub.add_parser("fade-out", help=_("Linear fade out"))
    fo.add_argument("--samples", type=int, default=None, help=_("Fade length (samples)"))
    fo.add_argument("--duration", type=float, default=None, help=_("Fade length (seconds)"))
    fo.add_argument("--output", "-o", required=True, help=_("Output path"))
    fo.add_argument("--recursive", "-r", action="store_true")
    fo.add_argument("--suffix", default="")

    # trim
    tr = fx_sub.add_parser("trim", help=_("Trim by samples/time"))
    tr.add_argument("--start", type=int, default=0, help=_("Start sample"))
    tr.add_argument("--end", type=int, default=None, help=_("End sample"))
    tr.add_argument("--start-sec", type=float, default=None, help=_("Start (seconds)"))
    tr.add_argument("--end-sec", type=float, default=None, help=_("End (seconds)"))
    tr.add_argument("--output", "-o", required=True, help=_("Output path"))
    tr.add_argument("--recursive", "-r", action="store_true")
    tr.add_argument("--suffix", default="")

    # trim-silence
    ts = fx_sub.add_parser("trim-silence", help=_("Trim leading/trailing silence"))
    ts.add_argument("--threshold", type=float, default=-40.0, help=_("Threshold dB (default: -40)"))
    ts.add_argument("--pad", type=int, default=0, help=_("Silence boundary padding samples"))
    ts.add_argument("--output", "-o", required=True, help=_("Output path"))
    ts.add_argument("--recursive", "-r", action="store_true")
    ts.add_argument("--suffix", default="")

    # normalize
    nm = fx_sub.add_parser("normalize", help=_("Normalize (peak or RMS)"))
    nm.add_argument("--peak", type=float, default=None, help=_("Peak target dB (e.g. -1)"))
    nm.add_argument("--target-rms", type=float, default=None, help=_("RMS target dB (e.g. -20)"))
    nm.add_argument("--output", "-o", required=True, help=_("Output path"))
    nm.add_argument("--recursive", "-r", action="store_true")
    nm.add_argument("--suffix", default="")

    # gate
    gt = fx_sub.add_parser("gate", help=_("Noise gate (RMS-based)"))
    gt.add_argument("--threshold", type=float, default=-50.0, help=_("Threshold dB (default: -50)"))
    gt.add_argument("--attack", type=float, default=0.01, help=_("Attack time (seconds)"))
    gt.add_argument("--release", type=float, default=0.05, help=_("Release time (seconds)"))
    gt.add_argument("--output", "-o", required=True, help=_("Output path"))
    gt.add_argument("--recursive", "-r", action="store_true")
    gt.add_argument("--suffix", default="")

    # gain
    gn = fx_sub.add_parser("gain", help=_("dB gain"))
    gn.add_argument("--db", type=float, required=True, help=_("Gain (dB)"))
    gn.add_argument("--output", "-o", required=True, help=_("Output path"))
    gn.add_argument("--recursive", "-r", action="store_true")
    gn.add_argument("--suffix", default="")

    parser.set_defaults(func=run)


def _apply_effect(audio: np.ndarray, sr: int, args: argparse.Namespace) -> np.ndarray:
    """이펙트 적용"""
    effect = args.effect

    if effect == "fade-in":
        samples = args.samples or (int(args.duration * sr) if args.duration else sr // 10)
        return dsp.fade_in(audio, samples)

    elif effect == "fade-out":
        samples = args.samples or (int(args.duration * sr) if args.duration else sr // 10)
        return dsp.fade_out(audio, samples)

    elif effect == "trim":
        start = args.start
        end = args.end
        if args.start_sec is not None:
            start = int(args.start_sec * sr)
        if args.end_sec is not None:
            end = int(args.end_sec * sr)
        return dsp.trim(audio, start=start, end=end)

    elif effect == "trim-silence":
        return dsp.trim_silence(audio, sr, threshold_db=args.threshold, pad_samples=args.pad)

    elif effect == "normalize":
        if args.peak is None and args.target_rms is None:
            # 기본: peak -1dB
            return dsp.normalize(audio, peak_db=-1.0)
        return dsp.normalize(audio, peak_db=args.peak, target_rms_db=args.target_rms)

    elif effect == "gate":
        return dsp.gate(audio, sr, threshold_db=args.threshold, attack_sec=args.attack, release_sec=args.release)

    elif effect == "gain":
        return dsp.gain(audio, args.db)

    else:
        raise ValueError(f"알 수 없는 이펙트: {effect}")


def run(args: argparse.Namespace) -> None:
    if not args.effect:
        print_error("이펙트를 지정해주세요. (fade-in, fade-out, trim, trim-silence, normalize, gate, gain)")

    input_path = Path(args.input)

    if input_path.is_dir():
        _run_batch(args, input_path)
    else:
        _run_single(args, input_path)


def _run_single(args: argparse.Namespace, input_path: Path) -> None:
    start_time = time.monotonic()

    try:
        audio, sr = read_audio(input_path)
    except FileNotFoundError as e:
        print_error(str(e))

    input_stats = get_audio_stats(audio, sr)
    result = _apply_effect(audio, sr, args)
    output_stats = get_audio_stats(result, sr)

    write_audio(args.output, result, sr)
    elapsed = round(time.monotonic() - start_time, 3)

    if args.json:
        print_json({
            "command": "fx",
            "effect": args.effect,
            "input": str(input_path),
            "output": args.output,
            "input_stats": {"rms": round(input_stats.rms, 6), "peak": round(input_stats.peak, 6),
                           "duration": round(input_stats.duration, 4), "frames": input_stats.frames},
            "output_stats": {"rms": round(output_stats.rms, 6), "peak": round(output_stats.peak, 6),
                            "duration": round(output_stats.duration, 4), "frames": output_stats.frames},
            "time_seconds": elapsed,
        })
        return

    output_console.print(f"\n[bold]{args.effect}[/bold] 완료")
    output_console.print(f"  Input:  {input_path} ({input_stats.duration:.2f}s)")
    output_console.print(f"  Output: {args.output} ({output_stats.duration:.2f}s)")
    output_console.print(f"  RMS: {input_stats.rms:.4f} → {output_stats.rms:.4f}")
    output_console.print(f"  Peak: {input_stats.peak:.4f} → {output_stats.peak:.4f}")
    output_console.print(f"  Time: {elapsed}s")


def _run_batch(args: argparse.Namespace, input_dir: Path) -> None:
    output_dir = Path(args.output)
    files = collect_audio_files(input_dir, recursive=getattr(args, "recursive", False))

    if not files:
        print_error(f"오디오 파일이 없습니다: {input_dir}")

    ok, fail = 0, 0
    for i, fpath in enumerate(files):
        out_path = resolve_output_path(fpath, input_dir, output_dir, suffix=getattr(args, "suffix", ""))

        try:
            audio, sr = read_audio(fpath)
            result = _apply_effect(audio, sr, args)
            write_audio(out_path, result, sr)
            ok += 1

            if args.json:
                output_stats = get_audio_stats(result, sr)
                print(json.dumps({
                    "command": "fx", "effect": args.effect,
                    "input": str(fpath), "output": str(out_path),
                    "output_rms": round(output_stats.rms, 6),
                    "output_peak": round(output_stats.peak, 6),
                }, ensure_ascii=False))
            else:
                output_console.print(f"  [{i+1}/{len(files)}] {fpath.name} → {out_path.name}")

        except Exception as e:
            fail += 1
            if args.json:
                print(json.dumps({"command": "fx", "input": str(fpath), "error": str(e)}, ensure_ascii=False))
            else:
                print_warning(f"  [{i+1}/{len(files)}] {fpath.name}: {e}")

    if not args.json:
        print_success(f"배치 완료: {ok} 성공, {fail} 실패 / {len(files)} 전체")
