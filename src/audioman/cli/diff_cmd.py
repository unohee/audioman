# Created: 2026-11-04
# Purpose: audioman diff A B — what changed between two renders, and is it real.

from __future__ import annotations

import argparse
from pathlib import Path

from audioman.cli._detail import add_detail_arguments, shape_json
from audioman.cli.output import print_error, print_json, print_success, print_table
from audioman.core.findings import json_envelope, schema_uri


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "diff",
        help="Texture descriptor deltas between two files (aligned), with floor significance",
    )
    parser.add_argument("a", help="Reference file (before)")
    parser.add_argument("b", help="Compared file (after)")
    parser.add_argument("--segment-sec", type=float, default=1.0, help="Segment length (default: 1.0 s)")
    parser.add_argument("--no-align", action="store_true", help="Compare sample-for-sample without aligning B to A")
    parser.add_argument("--max-lag-ms", type=float, default=500.0, help="Alignment search window (default: 500 ms)")
    parser.add_argument("--png", help="Write a figure of the per-band and per-descriptor changes")
    add_detail_arguments(parser)
    parser.set_defaults(func=run)


def _read(path: str):
    from audioman.core.audio_file import read_audio

    try:
        return read_audio(Path(path))
    except (OSError, RuntimeError, ValueError) as exc:
        print_error(f"cannot read {path}: {exc}")  # exits with status 1


def run(args: argparse.Namespace) -> None:
    from audioman.core.diff import diff_audio

    a, sr_a = _read(args.a)
    b, sr_b = _read(args.b)
    if sr_a != sr_b:
        print_error(f"sample rates differ ({sr_a} vs {sr_b} Hz); resample one file first")
    result = diff_audio(a, b, sr_a, segment_sec=args.segment_sec, do_align=not args.no_align,
                        max_lag_ms=args.max_lag_ms)
    payload = {"a": str(Path(args.a).resolve()), "b": str(Path(args.b).resolve()), **result}
    if args.png:
        from audioman.core.diff_figure import render

        payload["figure"] = str(render(result, args.png, title=f"{Path(args.b).name} - {Path(args.a).name}").resolve())

    if getattr(args, "json", False):
        print_json(shape_json("diff", json_envelope("diff", payload, schema=schema_uri("diff")), args,
                              label=f"{Path(args.a).stem}-vs-{Path(args.b).stem}"))
        return

    from audioman.core.digest import digest_diff

    rows = [[c["descriptor"], c["direction"], f"{c['delta']} {c['unit']}", str(c["floor"]),
             str(c["ratio_to_floor"])] for c in result["changes"]]
    if rows:
        print_table("significant changes (B - A)", ["descriptor", "dir", "delta", "floor", "x floor"], rows)
    bands = [[nb["band"], f"{nb['delta_db']:+.2f}", f"{nb['floor_db']:.3f}", "yes" if nb["significant"] else ""]
             for nb in result["named_bands"]]
    print_table("bands (dB)", ["band", "delta", "floor", "significant"], bands)
    print_success(digest_diff(payload))
