# Created: 2026-04-05
# Purpose: audioman commit 서브커맨드 — destructive commit + delay compensation

import argparse

from audioman.cli.output import print_error, print_json, print_success, print_warning, output_console
from audioman.i18n import _


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("commit", help=_("Commit plugin chain to audio with auto delay compensation"))
    parser.add_argument("input", help=_("Input audio file"))
    parser.add_argument("--output", "-o", required=True, help=_("Output file path"))
    parser.add_argument(
        "--chain", "-s", required=True,
        help=_("Plugin chain (e.g. 'denoise:threshold=-20,dehum:freq=60')"),
    )
    parser.add_argument(
        "--no-compensation", action="store_true",
        help=_("Disable auto delay compensation"),
    )
    parser.add_argument(
        "--no-tail-trim", action="store_true",
        help=_("Keep plugin tail (don't trim to original length)"),
    )
    parser.add_argument("--dry-run", action="store_true", help=_("Measure latency only (no processing)"))
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from audioman.core.pipeline import parse_chain_string

    steps = parse_chain_string(args.chain)
    if not steps:
        print_error("처리 단계가 비어있습니다")
        return

    # Dry-run: 레이턴시 측정만
    if args.dry_run:
        from audioman.core.commit import dry_run_commit

        try:
            measurements, total = dry_run_commit(steps)
        except Exception as e:
            print_error(f"레이턴시 측정 실패: {e}")
            return

        if args.json:
            print_json({
                "command": "commit",
                "dry_run": True,
                "chain": [s.to_dict() for s in steps],
                "latency": [m.to_dict() for m in measurements],
                "total_latency_samples": total,
            })
        else:
            output_console.print(f"\n[bold]Latency Measurement (dry-run)[/bold]")
            for m in measurements:
                conf_str = f"{m.confidence:.0%}"
                match = "✓" if abs(m.reported_latency - m.measured_latency) <= 1 else "≠"
                output_console.print(
                    f"  {m.plugin_name}: {m.used_latency} samples "
                    f"(measured={m.measured_latency}, reported={m.reported_latency} {match}) "
                    f"confidence={conf_str}"
                )
            sr = 48000  # dry-run 기본 SR
            output_console.print(
                f"\n  [bold]Total: {total} samples ({total / sr * 1000:.1f}ms @ {sr}Hz)[/bold]"
            )
        return

    # 실제 commit 실행
    from audioman.core.commit import commit_file

    try:
        result = commit_file(
            input_path=args.input,
            output_path=args.output,
            steps=steps,
            compensate_latency=not args.no_compensation,
            tail_trim=not args.no_tail_trim,
        )
    except Exception as e:
        print_error(f"커밋 실패: {e}")
        return

    if args.json:
        print_json({"command": "commit", **result.to_dict()})
        return

    output_console.print(f"\n[bold]커밋 완료[/bold]")
    output_console.print(f"  Input:  {result.input_path}")
    output_console.print(f"  Output: {result.output_path}")
    output_console.print(f"  Chain:  {len(result.steps)} steps")
    for i, s in enumerate(result.steps, 1):
        output_console.print(f"    {i}. {s['plugin']}")

    if result.total_latency_samples > 0:
        output_console.print(f"\n  [bold]Delay Compensation[/bold]")
        for m in result.latency_compensation:
            output_console.print(
                f"    {m['plugin_name']}: {m['used_latency']} samples "
                f"(confidence={m['confidence']:.0%})"
            )
        output_console.print(f"    Total: {result.total_latency_samples} samples")
    else:
        output_console.print(f"  Latency: 0 samples (보상 불필요)")

    output_console.print(f"  Time:   {result.duration_seconds}s")
    print_success("완료")
