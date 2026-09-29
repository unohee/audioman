# Created: 2026-04-27
# Purpose: audioman fader-compare — compare a fader-test ground truth against automix results.
#          Quantify how close the algorithm gets to the human decision + report the worst-off tracks.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from audioman.cli.output import (
    output_console,
    print_error,
    print_json,
    print_table,
    print_markup,
)
from audioman.core.findings import json_envelope, schema_uri


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "fader-compare",
        help="Compare automix recommendations against a fader-test ground truth",
    )
    parser.add_argument("ground_truth", help="fader-test gains JSON (ground truth)")
    parser.add_argument(
        "--target", default="archive_techno_standard",
        help="Automix target profile (default: archive_techno_standard)",
    )
    parser.add_argument(
        "--reference", default=None,
        help="Reference WAV (used when --target reference)",
    )
    parser.set_defaults(func=run)


def _load_ground_truth(gt_path: Path) -> Optional[dict]:
    """Read + validate the fader-test export. Returns None after print_error."""
    try:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        print_error(f"Cannot read ground truth JSON: {gt_path} ({e})")
        return None

    if not isinstance(data, dict):
        print_error(f"Top level of the ground truth JSON is not an object: {type(data).__name__}")
        return None

    source_dir = data.get("source_dir")
    if not isinstance(source_dir, str) or not Path(source_dir).is_dir():
        print_error(f"ground truth source_dir is not valid: {source_dir}")
        return None

    gt_gains = data.get("gains")
    if not isinstance(gt_gains, dict) or not gt_gains:
        print_error("ground truth JSON has no 'gains' field.")
        return None

    return {"source_dir": source_dir, "gains": gt_gains}


def run(args: argparse.Namespace) -> None:
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print_error(f"File not found: {gt_path}")
        return

    loaded = _load_ground_truth(gt_path)
    if loaded is None:
        return
    source_dir = loaded["source_dir"]
    gt_gains = loaded["gains"]

    # Compute the automix recommendations
    from pathlib import Path as _P
    from audioman.core.automix import automix as run_automix

    track_paths = sorted(_P(source_dir).glob("*.wav"))
    if not track_paths:
        print_error(f"No wav files in source_dir: {source_dir}")
        return

    try:
        result = run_automix(
            track_paths=[str(p) for p in track_paths],
            target=args.target,
            reference_path=args.reference,
        )
    except Exception as e:
        print_error(f"automix failed: {e}")
        return

    if len(result.gains_db) != len(track_paths):
        print_error(
            "automix returned a different number of gains than input tracks: "
            f"tracks={len(track_paths)}, gains={len(result.gains_db)}"
        )
        return

    # Map by track name (assuming alphabetical order — both fader-test and automix sort)
    rows: list[dict] = []
    for path, auto_db in zip(track_paths, result.gains_db):
        name = path.stem.strip()
        gt_db = gt_gains.get(name)
        if gt_db is None:
            # Try both the whitespace-stripped name and the raw name
            gt_db = gt_gains.get(path.stem)
        if gt_db is None:
            continue
        try:
            gt_db_f = float(gt_db)
        except (TypeError, ValueError):
            print_error(f"ground truth gain is not a number: {name}={gt_db!r}")
            return
        rows.append({
            "track": name,
            "ground_truth_db": gt_db_f,
            "automix_db": float(auto_db),
            "diff_db": float(auto_db) - gt_db_f,  # how much louder automix is than the ground truth
        })

    if not rows:
        print_error("No tracks matched — check that the track names agree.")
        return

    n = len(rows)
    diffs = [abs(r["diff_db"]) for r in rows]
    mean_abs_err = sum(diffs) / n
    max_abs_err = max(diffs)
    within_3 = sum(1 for d in diffs if d <= 3.0) / n * 100
    within_6 = sum(1 for d in diffs if d <= 6.0) / n * 100

    # JSON output
    if args.json:
        print_json(json_envelope(
            "fader-compare",
            {
                "ground_truth": str(gt_path),
                "automix_target": args.target,
                "n_tracks_matched": n,
                "summary": {
                    "mean_abs_error_db": round(mean_abs_err, 2),
                    "max_abs_error_db": round(max_abs_err, 2),
                    "within_3dB_pct": round(within_3, 1),
                    "within_6dB_pct": round(within_6, 1),
                },
                "tracks": [
                    {
                        "track": r["track"],
                        "ground_truth_db": round(r["ground_truth_db"], 2),
                        "automix_db": round(r["automix_db"], 2),
                        "diff_db": round(r["diff_db"], 2),
                    }
                    for r in rows
                ],
            },
            schema=schema_uri("fader-compare"),
        ))
        return

    # Human-readable
    print_markup(f"\n[bold]Fader-test vs Automix ({args.target})[/bold]")
    output_console.print(f"  matched tracks: {n}")
    output_console.print(f"  mean |error|:   {mean_abs_err:.2f} dB")
    output_console.print(f"  max |error|:    {max_abs_err:.2f} dB")
    output_console.print(f"  within ±3 dB:   {within_3:.0f}%")
    output_console.print(f"  within ±6 dB:   {within_6:.0f}%\n")

    # The most disagreeing tracks, top 15
    rows_sorted = sorted(rows, key=lambda r: -abs(r["diff_db"]))
    rows_table = []
    for r in rows_sorted[:15]:
        sign = "+" if r["diff_db"] > 0 else ""
        marker = "↑" if r["diff_db"] > 3 else ("↓" if r["diff_db"] < -3 else "·")
        rows_table.append([
            r["track"][:25],
            f"{r['ground_truth_db']:+.1f}",
            f"{r['automix_db']:+.1f}",
            f"{sign}{r['diff_db']:.1f}",
            marker,
        ])
    print_table(
        "Top 15 disagreements (|diff| desc)",
        ["Track", "Ground truth (you)", "Automix", "Diff", ""],
        rows_table,
    )

    # The closest matches
    rows_close = sorted(rows, key=lambda r: abs(r["diff_db"]))[:5]
    print_markup("\n[bold]Closest matches[/bold]")
    for r in rows_close:
        output_console.print(
            f"  {r['track']:<25s}  gt={r['ground_truth_db']:+.1f}  "
            f"auto={r['automix_db']:+.1f}  diff={r['diff_db']:+.2f}"
        )
