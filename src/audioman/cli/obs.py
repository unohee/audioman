# Created: 2026-05-07
# Purpose: audioman obs — automatic diagnosis of OBS multitrack video (dry-run)
# Guide: docs/obs-workflow.md

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audioman.cli.output import (
    output_console,
    print_error,
    print_info,
    print_json,
    print_table,
)
from audioman.core import obs as obs_core
from audioman.core.findings import json_envelope, schema_uri


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "obs",
        help="Automatic diagnosis of OBS multitrack video (dry-run) — see docs/obs-workflow.md",
    )
    sub = parser.add_subparsers(dest="obs_command", required=True)

    # obs probe — topology only, fast
    p_probe = sub.add_parser("probe", help="Identify track topology (separate/single/duplicated/silent)")
    p_probe.add_argument("input", help="Input video file or directory")
    p_probe.add_argument(
        "--probe-seconds", type=float, default=None,
        help="Per-track RMS measurement length in seconds. Defaults to scanning the whole video. "
             "Use ~15 for a quick check.",
    )
    p_probe.set_defaults(func=_run_probe)

    # obs dry-run — diagnosis + treatment plan
    p_dry = sub.add_parser(
        "dry-run",
        help="Classify active tracks + diagnose + build treatment plan (no actual processing)",
    )
    p_dry.add_argument("input", help="Input video file or directory")
    p_dry.add_argument(
        "--seconds", type=float, default=60.0,
        help="Analysis window length (default: 60s, taken from the middle of the video)",
    )
    p_dry.add_argument(
        "--start", type=float, default=None,
        help="Analysis start time in seconds. Defaults to the middle of the video",
    )
    p_dry.add_argument(
        "--out-dir", type=str, default=None,
        help="Directory to write JSON reports to (writes files instead of stdout when set)",
    )
    p_dry.set_defaults(func=_run_dry_run)


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def _iter_videos(path: Path):
    if path.is_dir():
        exts = {".mov", ".mp4", ".mkv", ".m4v"}
        seen_stems = set()
        for p in sorted(path.iterdir()):
            if p.suffix.lower() not in exts:
                continue
            # .mov wins when .mov and .mp4 share the same stem
            stem = p.with_suffix("")
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            yield p
    else:
        yield path


def _run_probe(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    if not inp.exists():
        print_error(f"File/folder not found: {inp}")
        return

    json_mode = getattr(args, "json", False)
    results: list[dict] = []
    for video in _iter_videos(inp):
        try:
            r = obs_core.probe_topology(
                video, probe_seconds=args.probe_seconds,
            )
        except Exception as e:
            print_error(f"{video}: {e}")
            continue
        results.append({"video": str(video), **r.to_dict()})

    if json_mode:
        print_json(json_envelope(
            "obs probe",
            {"input": str(inp), "count": len(results), "files": results},
            schema=schema_uri("obs"),
        ))
        return

    rows = []
    for r in results:
        active_str = ",".join(str(i) for i in r["active_indices"]) or "-"
        groups_str = " | ".join("[" + ",".join(str(i) for i in g) + "]"
                                for g in r["unique_signal_groups"]) or "-"
        rows.append([
            Path(r["video"]).name,
            r["topology"],
            f"{r['n_streams']}",
            active_str,
            groups_str,
            f"{r['duration_sec']:.1f}s",
        ])
    print_table(
        title=f"OBS topology — {len(results)} file(s)",
        columns=["file", "topology", "streams", "active", "groups", "duration"],
        rows=rows,
    )


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------


def _run_dry_run(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    if not inp.exists():
        print_error(f"File/folder not found: {inp}")
        return

    json_mode = getattr(args, "json", False)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[list[str]] = []
    all_reports: list[dict] = []

    for video in _iter_videos(inp):
        try:
            print_info(f"Analyzing: {video.name}")
            report = obs_core.dry_run_video(
                video,
                analysis_seconds=args.seconds,
                analysis_start_sec=args.start,
            )
        except Exception as e:
            print_error(f"{video}: {e}")
            continue

        d = report.to_dict()
        all_reports.append(d)

        if out_dir is not None:
            out_file = out_dir / f"{video.stem}.json"
            out_file.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str))

        # Summary rows (one per track)
        topo = d["topology"]["topology"]
        for tr in d["treatments"]:
            actions = ",".join(p["action"] for p in tr["plan"]) or "-"
            n_warn = sum(1 for p in tr["plan"] if p["severity"] == "warn")
            n_crit = sum(1 for p in tr["plan"] if p["severity"] == "critical")
            mirrors = ",".join(str(m) for m in tr.get("mirrors", []))
            summary_rows.append([
                video.name,
                topo,
                f"track {tr['track_index']}" + (f" (=>{mirrors})" if mirrors else ""),
                tr["kind"],
                actions,
                f"warn={n_warn} crit={n_crit}",
            ])
        if not d["treatments"]:
            summary_rows.append([video.name, topo, "-", "-", "skip", "—"])

    if json_mode:
        print_json(json_envelope(
            "obs dry-run",
            {"input": str(inp), "count": len(all_reports), "files": all_reports},
            schema=schema_uri("obs"),
        ))
        return

    if not summary_rows:
        print_info("Nothing to process")
        return

    print_table(
        title=f"OBS dry-run — {len(all_reports)} file(s)",
        columns=["file", "topology", "track", "kind", "actions", "issues"],
        rows=summary_rows,
    )
    if out_dir is not None:
        output_console.print(f"\n[dim]Detailed JSON: {out_dir}/[/dim]")
