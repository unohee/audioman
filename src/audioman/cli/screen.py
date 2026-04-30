# Created: 2026-04-29
# Purpose: Audio aesthetic issue screening CLI.

from __future__ import annotations

import argparse
from pathlib import Path

from audioman.cli.output import output_console, print_error, print_json, print_table, print_warning
from audioman.core.aesthetic import DEFAULT_ISSUES, screen_file
from audioman.core.batch import collect_audio_files
from audioman.i18n import _


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "screen",
        help=_("Screen audio for aesthetic issues such as clicks, hum, breaths, sibilance, and noise"),
    )
    parser.add_argument("input", help=_("Input audio file or directory"))
    parser.add_argument(
        "--issues",
        default=",".join(DEFAULT_ISSUES),
        help=_("Comma-separated issue list (default: click,hum,mouth_click,sibilance,breath,background_noise,rf_noise)"),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "essentia", "fallback"],
        default="auto",
        help=_("Detector backend (default: auto)"),
    )
    parser.add_argument("--recursive", "-r", action="store_true", help=_("Include subdirectories (batch)"))
    parser.set_defaults(func=run)


def _parse_issues(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    issues = _parse_issues(args.issues)
    if not issues:
        print_error("--issues must include at least one issue")

    if input_path.is_dir():
        files = collect_audio_files(input_path, recursive=args.recursive)
        reports = [_screen_one(path, issues, args.backend) for path in files]
        out = {"command": "screen", "input": str(input_path), "files": reports}
        if args.json:
            print_json(out)
            return
        _print_batch(reports)
        return

    report = _screen_one(input_path, issues, args.backend)
    if args.json:
        print_json({"command": "screen", **report})
        return
    _print_single(report)


def _screen_one(path: Path, issues: list[str], backend: str) -> dict:
    try:
        return screen_file(path, issues=issues, backend=backend)
    except FileNotFoundError as e:
        print_error(str(e))
    except ImportError as e:
        print_error(f"{e}. Install essentia or use --backend fallback.")
    except Exception as e:
        print_error(str(e))
    raise AssertionError("unreachable")


def _print_single(report: dict) -> None:
    output_console.print(f"\n[bold]{report['file']}[/bold]")
    output_console.print(
        f"  Duration: {report['duration']}s | SR: {report['sample_rate']}Hz | CH: {report['channels']}"
    )
    output_console.print(
        f"  Backend: {', '.join(f'{k}={v}' for k, v in report['backends'].items()) or 'none'}"
    )
    if report["unsupported_issues"]:
        print_warning(f"unsupported issues skipped: {', '.join(report['unsupported_issues'])}")

    rows = [
        [
            event["type"],
            f"{event['start_sec']:.3f}",
            f"{event['end_sec']:.3f}",
            event["severity"],
            event["backend"],
            _event_note(event),
        ]
        for event in report["events"]
    ]
    if rows:
        print_table("Aesthetic events", ["Type", "Start", "End", "Severity", "Backend", "Detail"], rows)
    else:
        output_console.print("  No events detected")


def _print_batch(reports: list[dict]) -> None:
    rows = []
    for report in reports:
        rows.append([
            report["file"],
            str(len(report["events"])),
            ", ".join(f"{k}:{v}" for k, v in report["summary"].items()),
        ])
    print_table("Aesthetic screening", ["File", "Events", "Summary"], rows)


def _event_note(event: dict) -> str:
    parts = []
    if "frequency_hz" in event:
        parts.append(f"{event['frequency_hz']} Hz")
    if "snr_db" in event and event["snr_db"] is not None:
        parts.append(f"SNR {event['snr_db']} dB")
    if "salience" in event:
        parts.append(f"salience {event['salience']}")
    if "ratio_db" in event:
        parts.append(f"ratio {event['ratio_db']} dB")
    if "rms_db" in event:
        parts.append(f"RMS {event['rms_db']} dB")
    if "tones" in event:
        tone_text = ", ".join(f"{t['frequency_hz']}Hz" for t in event["tones"][:4])
        parts.append(f"tones {tone_text}")
    return ", ".join(parts)
