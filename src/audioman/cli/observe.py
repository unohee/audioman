# Created: 2026-05-11
# Purpose: `audioman observe` — fault 관측 1급 명령.
# analyze/doctor가 산발적으로 만들던 결함 정보를 단일 Finding[] 스키마로 통합.
# LLM agent가 `audioman observe X --json | jq '.findings'`로 바로 소비할 수 있게 한다.

from __future__ import annotations

import argparse
from pathlib import Path

from audioman import __version__
from audioman.cli.output import (
    output_console,
    print_error,
    print_json,
    print_table,
)
from audioman.core.analysis import detect_silence, spectrum_diagnostics
from audioman.core.audio_file import get_audio_stats, read_audio
from audioman.core.batch import collect_audio_files
from audioman.core.detectors import (
    detect_signal_findings,
    silence_to_findings,
    spectrum_to_findings,
)
from audioman.core.findings import (
    Category,
    SCHEMA_URI,
    Severity,
    filter_findings,
)


_CATEGORY_CHOICES = [c.value for c in Category]
_SEVERITY_CHOICES = [s.value for s in Severity]


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "observe",
        help="Observe audio faults across categories (signal, spectral, plugin, container)",
        description=(
            "Single-shot audio fault observation. Emits a uniform finding[] array "
            "across signal/spectral/plugin/container categories with $schema metadata. "
            "Designed for LLM agents — pair with --plain and --json for clean piping."
        ),
    )
    parser.add_argument("input", help="Input audio file or directory")
    parser.add_argument(
        "--category",
        default=",".join(_CATEGORY_CHOICES),
        help=(
            f"Comma-separated categories to enable (default: all). "
            f"Choices: {','.join(_CATEGORY_CHOICES)}"
        ),
    )
    parser.add_argument(
        "--severity",
        choices=_SEVERITY_CHOICES,
        default="info",
        help="Minimum severity to report (info|warn|critical). Default: info",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=-40.0,
        help="Silence detection threshold dB (default: -40)",
    )
    parser.add_argument(
        "--spectrum-fft",
        type=int,
        default=16384,
        help="FFT size for spectral detectors (default: 16384)",
    )
    parser.add_argument(
        "--spectrum-min-rms",
        type=float,
        default=0.01,
        help="Skip frames below this RMS when averaging spectrum (default: 0.01)",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recurse into subdirectories (batch mode)",
    )
    parser.set_defaults(func=run)


def _observe_file(
    path: Path,
    *,
    categories: set[str],
    min_severity: Severity,
    silence_threshold: float,
    spectrum_fft: int,
    spectrum_min_rms: float,
) -> dict:
    audio, sr = read_audio(path)
    stats = get_audio_stats(audio, sr)
    audio_length = audio.shape[-1] if audio.ndim == 2 else audio.shape[0]

    all_findings = []

    if "signal" in categories:
        all_findings.extend(detect_signal_findings(audio, sr, file=str(path)))
        silence = detect_silence(audio, sr, threshold_db=silence_threshold)
        all_findings.extend(silence_to_findings(silence, audio_length, sr, file=str(path)))

    spectrum = None
    if "spectral" in categories:
        spectrum = spectrum_diagnostics(
            audio, sr, fft_size=spectrum_fft, min_rms=spectrum_min_rms
        )
        all_findings.extend(spectrum_to_findings(spectrum, file=str(path)))

    # plugin/container 카테고리는 Phase C에서 채움. 지금은 사용자가 명시적으로
    # 지정하면 빈 결과를 반환 (스키마 일관성 유지).

    filtered = filter_findings(
        all_findings,
        categories=categories,
        min_severity=min_severity,
    )

    payload = {
        "$schema": SCHEMA_URI,
        "audioman_version": __version__,
        "command": "observe",
        "file": str(path),
        "sample_rate": sr,
        "channels": stats.channels,
        "duration_sec": round(stats.duration, 6),
        "total_samples": int(stats.frames),
        "filter": {
            "categories": sorted(categories),
            "min_severity": min_severity.value,
        },
        "findings": [f.to_dict() for f in filtered],
        "summary": {
            "total": len(filtered),
            "by_severity": {
                "info": sum(1 for f in filtered if f.severity is Severity.INFO),
                "warn": sum(1 for f in filtered if f.severity is Severity.WARN),
                "critical": sum(1 for f in filtered if f.severity is Severity.CRITICAL),
            },
            "by_category": {
                cat: sum(1 for f in filtered if f.category.value == cat)
                for cat in _CATEGORY_CHOICES
            },
        },
    }
    return payload


def _parse_categories(raw: str) -> set[str]:
    categories = {c.strip() for c in raw.split(",") if c.strip()}
    invalid = categories - set(_CATEGORY_CHOICES)
    if invalid:
        print_error(f"unknown category: {','.join(sorted(invalid))}")
    return categories


def run(args: argparse.Namespace) -> None:
    categories = _parse_categories(args.category)
    min_severity = Severity(args.severity)
    input_path = Path(args.input)

    if input_path.is_dir():
        files = collect_audio_files(input_path, recursive=args.recursive)
        if not files:
            print_error(f"No audio files in: {input_path}")
        for fpath in files:
            payload = _observe_file(
                fpath,
                categories=categories,
                min_severity=min_severity,
                silence_threshold=args.silence_threshold,
                spectrum_fft=args.spectrum_fft,
                spectrum_min_rms=args.spectrum_min_rms,
            )
            if args.json:
                print_json(payload)
            else:
                _print_human(payload)
        return

    payload = _observe_file(
        input_path,
        categories=categories,
        min_severity=min_severity,
        silence_threshold=args.silence_threshold,
        spectrum_fft=args.spectrum_fft,
        spectrum_min_rms=args.spectrum_min_rms,
    )
    if args.json:
        print_json(payload)
        return
    _print_human(payload)


def _print_human(payload: dict) -> None:
    output_console.print(f"\n[bold]{payload['file']}[/bold]")
    output_console.print(
        f"  {payload['duration_sec']}s @ {payload['sample_rate']}Hz, "
        f"{payload['channels']} ch, {payload['total_samples']} samples"
    )
    summary = payload["summary"]
    output_console.print(
        f"  Findings: {summary['total']} "
        f"(critical={summary['by_severity']['critical']}, "
        f"warn={summary['by_severity']['warn']}, "
        f"info={summary['by_severity']['info']})"
    )

    if not payload["findings"]:
        output_console.print("  [green]No findings at requested severity.[/green]")
        return

    rows = []
    for f in payload["findings"]:
        where = f["where"]
        loc = ""
        if "start_sec" in where:
            loc = f"{where['start_sec']:.3f}s"
            if "end_sec" in where and where["end_sec"] != where["start_sec"]:
                loc += f"-{where['end_sec']:.3f}s"
        elif "frequency_hz" in where:
            loc = f"{where['frequency_hz']}Hz"
        rows.append([
            f["severity"].upper(),
            f["category"],
            f["code"],
            loc,
            f["hint"][:80],
        ])
    print_table("Findings", ["Sev", "Category", "Code", "Where", "Hint"], rows)
