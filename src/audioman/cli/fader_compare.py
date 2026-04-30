# Created: 2026-04-27
# Purpose: audioman fader-compare — fader-test ground truth와 automix 결과를 비교.
#          자동 알고리즘이 본인 결정과 얼마나 가까운지 정량 평가 + 가장 어긋난 트랙 보고.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audioman.cli.output import (
    output_console,
    print_error,
    print_json,
    print_table,
)
from audioman.i18n import _


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "fader-compare",
        help=_("Compare automix recommendations against a fader-test ground truth"),
    )
    parser.add_argument("ground_truth", help=_("fader-test gains JSON (ground truth)"))
    parser.add_argument(
        "--target", default="archive_techno_standard",
        help=_("Automix target profile (default: archive_techno_standard)"),
    )
    parser.add_argument(
        "--reference", default=None,
        help=_("Reference WAV (used when --target reference)"),
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print_error(f"파일 없음: {gt_path}")

    data = json.loads(gt_path.read_text())
    source_dir = data.get("source_dir")
    if not source_dir or not Path(source_dir).is_dir():
        print_error(f"ground truth의 source_dir가 유효하지 않습니다: {source_dir}")

    gt_gains = data.get("gains") or {}
    if not gt_gains:
        print_error("ground truth JSON에 'gains' 필드가 없습니다.")

    # automix 권고값 계산
    from pathlib import Path as _P
    from audioman.core.automix import automix as run_automix

    track_paths = sorted(_P(source_dir).glob("*.wav"))
    if not track_paths:
        print_error(f"source_dir에 wav 없음: {source_dir}")

    try:
        result = run_automix(
            track_paths=[str(p) for p in track_paths],
            target=args.target,
            reference_path=args.reference,
        )
    except Exception as e:
        print_error(f"automix 실패: {e}")
        return

    # 트랙 이름으로 매핑 (alphabetical 순서 가정 — fader-test와 automix 둘 다 sorted)
    rows: list[dict] = []
    for path, auto_db in zip(track_paths, result.gains_db):
        name = path.stem.strip()
        gt_db = gt_gains.get(name)
        if gt_db is None:
            # whitespace stripped 이름과 raw 이름 둘 다 시도
            gt_db = gt_gains.get(path.stem)
        if gt_db is None:
            continue
        rows.append({
            "track": name,
            "ground_truth_db": float(gt_db),
            "automix_db": float(auto_db),
            "diff_db": float(auto_db) - float(gt_db),  # automix가 ground truth보다 얼마나 큰가
        })

    if not rows:
        print_error("매칭된 트랙이 없습니다 — 트랙명이 일치하는지 확인.")

    n = len(rows)
    diffs = [abs(r["diff_db"]) for r in rows]
    mean_abs_err = sum(diffs) / n
    max_abs_err = max(diffs)
    within_3 = sum(1 for d in diffs if d <= 3.0) / n * 100
    within_6 = sum(1 for d in diffs if d <= 6.0) / n * 100

    # JSON output
    if args.json:
        print_json({
            "command": "fader-compare",
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
        })
        return

    # Human-readable
    output_console.print(f"\n[bold]Fader-test vs Automix ({args.target})[/bold]")
    output_console.print(f"  matched tracks: {n}")
    output_console.print(f"  mean |error|:   {mean_abs_err:.2f} dB")
    output_console.print(f"  max |error|:    {max_abs_err:.2f} dB")
    output_console.print(f"  within ±3 dB:   {within_3:.0f}%")
    output_console.print(f"  within ±6 dB:   {within_6:.0f}%\n")

    # 가장 어긋난 트랙 top 10
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

    # 가장 일치한 트랙
    rows_close = sorted(rows, key=lambda r: abs(r["diff_db"]))[:5]
    output_console.print("\n[bold]Closest matches[/bold]")
    for r in rows_close:
        output_console.print(
            f"  {r['track']:<25s}  gt={r['ground_truth_db']:+.1f}  "
            f"auto={r['automix_db']:+.1f}  diff={r['diff_db']:+.2f}"
        )
