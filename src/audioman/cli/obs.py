# Created: 2026-05-07
# Purpose: audioman obs — OBS 멀티트랙 영상 자동 진단 (dry-run)
# 사용 가이드: docs/obs-workflow.md

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


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "obs",
        help="OBS 멀티트랙 영상 자동 진단 (dry-run) — see docs/obs-workflow.md",
    )
    sub = parser.add_subparsers(dest="obs_command", required=True)

    # obs probe — 토폴로지만 빠르게
    p_probe = sub.add_parser("probe", help="트랙 토폴로지(분리/단일/복제/무음) 식별")
    p_probe.add_argument("input", help="입력 영상 파일 또는 디렉터리")
    p_probe.add_argument(
        "--probe-seconds", type=float, default=None,
        help="트랙별 RMS 측정 길이(초). 미지정 시 영상 전체를 스캔. "
             "빠른 확인이 필요하면 15 정도로 지정.",
    )
    p_probe.set_defaults(func=_run_probe)

    # obs dry-run — 진단 + 처치 계획
    p_dry = sub.add_parser(
        "dry-run",
        help="활성 트랙 분류 + 진단 + 처치 계획 생성 (실제 처리 없음)",
    )
    p_dry.add_argument("input", help="입력 영상 파일 또는 디렉터리")
    p_dry.add_argument(
        "--seconds", type=float, default=60.0,
        help="분석 구간 길이 (default: 60s, 영상 중간부에서 추출)",
    )
    p_dry.add_argument(
        "--start", type=float, default=None,
        help="분석 시작 시점(초). 미지정 시 영상 중간",
    )
    p_dry.add_argument(
        "--out-dir", type=str, default=None,
        help="JSON 리포트를 저장할 디렉터리 (지정 시 stdout 대신 파일로)",
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
            # .mov와 .mp4가 같은 stem이면 .mov 우선
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
        print_error(f"파일/폴더 없음: {inp}")
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
        print_json(results if inp.is_dir() else results[0] if results else {})
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
        print_error(f"파일/폴더 없음: {inp}")
        return

    json_mode = getattr(args, "json", False)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[list[str]] = []
    all_reports: list[dict] = []

    for video in _iter_videos(inp):
        try:
            print_info(f"분석 중: {video.name}")
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

        # 요약 행 (트랙별)
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
        print_json(all_reports if inp.is_dir() else (all_reports[0] if all_reports else {}))
        return

    if not summary_rows:
        print_info("처리할 항목 없음")
        return

    print_table(
        title=f"OBS dry-run — {len(all_reports)} file(s)",
        columns=["file", "topology", "track", "kind", "actions", "issues"],
        rows=summary_rows,
    )
    if out_dir is not None:
        output_console.print(f"\n[dim]상세 JSON: {out_dir}/[/dim]")
