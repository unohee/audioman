# Created: 2026-03-21
# Purpose: audioman analyze 서브커맨드 — 오디오 분석

import argparse
import json
from pathlib import Path

from audioman.cli.output import print_error, print_json, print_table, print_success, output_console
from audioman.core.audio_file import read_audio, get_audio_stats
from audioman.core.analysis import (
    compute_frame_metrics,
    compute_summary,
    detect_silence,
    spectrum_diagnostics,
)
from audioman.core.batch import collect_audio_files
from audioman.core.waveform import render_waveform, render_envelope, render_spectral_envelope
from audioman.i18n import _


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("analyze", help=_("Audio analysis (RMS, spectral entropy, silence detection, etc.)"))
    parser.add_argument("input", help=_("Input audio file or directory"))
    parser.add_argument("--frames", action="store_true", help=_("Per-frame detailed output"))
    parser.add_argument("--frame-size", type=int, default=2048, help=_("Frame size (default: 2048)"))
    parser.add_argument("--hop", type=int, default=512, help=_("Hop size (default: 512)"))
    parser.add_argument("--silence-threshold", type=float, default=-40.0, help=_("Silence detection threshold dB (default: -40)"))
    parser.add_argument("--waveform", "-w", action="store_true", help=_("Show ASCII waveform"))
    parser.add_argument("--waveform-width", type=int, default=80, help=_("Waveform width (default: 80)"))
    parser.add_argument("--waveform-height", type=int, default=16, help=_("Waveform height (default: 16)"))
    parser.add_argument("--waveform-mode", choices=["rms", "peak"], default="peak", help=_("Waveform mode (default: peak)"))
    parser.add_argument("--recursive", "-r", action="store_true", help=_("Include subdirectories (batch)"))
    parser.add_argument("--spectrum", action="store_true",
                        help=_("Add long-term FFT diagnostics (band energy, dominant frequencies, hum, hf slope)"))
    parser.add_argument("--spectrum-fft", type=int, default=16384,
                        help=_("FFT size for spectrum diagnostics (default: 16384)"))
    parser.add_argument("--spectrum-min-rms", type=float, default=0.01,
                        help=_("Skip frames below this RMS when averaging spectrum (default: 0.01)"))
    parser.set_defaults(func=run)


def _analyze_file(
    path: Path, frame_size: int, hop: int, silence_threshold: float, frames_mode: bool,
    spectrum: bool = False, spectrum_fft: int = 16384, spectrum_min_rms: float = 0.01,
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

    if spectrum:
        result["spectrum"] = spectrum_diagnostics(
            audio, sr, fft_size=spectrum_fft, min_rms=spectrum_min_rms
        )

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
            path, args.frame_size, args.hop, args.silence_threshold, args.frames,
            spectrum=args.spectrum, spectrum_fft=args.spectrum_fft,
            spectrum_min_rms=args.spectrum_min_rms,
        )
    except FileNotFoundError as e:
        print_error(str(e))

    # 웨이브폼 렌더링 (JSON 모드에서도 ascii_waveform 필드로 포함)
    waveform_text = None
    envelope_text = None
    spectral_text = None

    if args.waveform:
        audio, sr = read_audio(path)
        waveform_text = render_waveform(
            audio, sr,
            width=args.waveform_width,
            height=args.waveform_height,
            mode=args.waveform_mode,
        )
        envelope_text = render_envelope(audio, sr, width=args.waveform_width)

        metrics = compute_frame_metrics(audio, sr, frame_size=args.frame_size, hop_size=args.hop)
        spectral_text = render_spectral_envelope(
            metrics.spectral_centroid, metrics.spectral_entropy,
            sr, result["duration"], width=args.waveform_width,
        )

    if args.json:
        out = {"command": "analyze", **result}
        if waveform_text:
            out["ascii_waveform"] = waveform_text
            out["ascii_envelope"] = envelope_text
            out["ascii_spectral"] = spectral_text
        print_json(out)
        return

    # human-readable 출력
    output_console.print(f"\n[bold]{result['file']}[/bold]")
    output_console.print(f"  Duration: {result['duration']}s | SR: {result['sample_rate']}Hz | CH: {result['channels']}")
    output_console.print(f"  RMS: {result['rms']:.4f} | Peak: {result['peak']:.4f}")

    # 웨이브폼
    if waveform_text:
        output_console.print(f"\n[bold]Waveform[/bold]")
        output_console.print(waveform_text, highlight=False)
        output_console.print(f"\n[bold]RMS Envelope[/bold]")
        output_console.print(envelope_text, highlight=False)
        output_console.print(f"\n[bold]Spectral[/bold]")
        output_console.print(spectral_text, highlight=False)

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

    # spectrum diagnostics
    if "spectrum" in result:
        spec = result["spectrum"]
        output_console.print(f"\n[bold]Spectrum diagnostics[/bold] (FFT={spec['fft_size']}, frames={spec['frames_analyzed']})")
        band_rows = [[b["band"], f"{b['freq_low']:.0f}-{b['freq_high']:.0f}",
                      f"{b['percent']:.2f}%", f"{b['db_rel_total']:+.1f}"]
                     for b in spec["band_energy"]]
        print_table("Band energy", ["Band", "Hz", "%", "dB rel total"], band_rows)
        if spec["dominant_frequencies"]:
            output_console.print("  Dominant frequencies:")
            for d in spec["dominant_frequencies"][:6]:
                output_console.print(f"    {d['frequency_hz']:8.1f} Hz  {d['db_rel_peak']:+5.1f} dB rel peak")
        hum_flags = [h for h in spec["hum_check"] if h["is_hum"]]
        if hum_flags:
            for h in hum_flags:
                output_console.print(f"  [red]HUM detected[/red] @ {h['frequency_hz']} Hz (SNR {h['snr_db']:+.1f} dB)")
        else:
            output_console.print("  Mains hum: not detected")
        sl = spec["hf_slope"]
        if sl["slope_db"] is not None:
            output_console.print(f"  HF slope: mid={sl['mid_db']:+.1f} dB, high={sl['high_db']:+.1f} dB, slope={sl['slope_db']:+.1f} dB")

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
                fpath, args.frame_size, args.hop, args.silence_threshold, args.frames,
                spectrum=args.spectrum, spectrum_fft=args.spectrum_fft,
                spectrum_min_rms=args.spectrum_min_rms,
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
