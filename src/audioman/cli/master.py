# Created: 2026-04-26
# Purpose: audioman master 서브커맨드 — 마스터링 납품 워크플로
#
# 3개 서브커맨드:
#   prep   : remove_dc → pad → fade_in/out → loudness_normalize 한 번에 적용
#   qc     : 결과물 검수 리포트 (target profile별 PASS/WARN/FAIL)
#   verify : prep + qc를 연쇄 (납품 전 한 줄 검증)

from __future__ import annotations

import argparse
import time
from pathlib import Path

from audioman.cli.output import (
    output_console,
    print_error,
    print_json,
    print_success,
    print_warning,
    print_table,
)
from audioman.core import dsp, edl as edl_core, qc
from audioman.i18n import _


# 마스터링 프로파일별 권장 prep 파라미터
PREP_PROFILES = {
    "spotify": {
        "head_pad_ms": 200, "tail_pad_sec": 2.0,
        "fade_in_ms": 5, "fade_out_ms": 200,
        "fade_curve": "cosine",
        "target_lufs": -14.0, "max_true_peak_dbtp": -1.0,
    },
    "apple_music": {
        "head_pad_ms": 200, "tail_pad_sec": 2.0,
        "fade_in_ms": 5, "fade_out_ms": 200,
        "fade_curve": "cosine",
        "target_lufs": -16.0, "max_true_peak_dbtp": -1.0,
    },
    "youtube": {
        "head_pad_ms": 200, "tail_pad_sec": 2.0,
        "fade_in_ms": 5, "fade_out_ms": 200,
        "fade_curve": "cosine",
        "target_lufs": -14.0, "max_true_peak_dbtp": -1.0,
    },
    "broadcast_ebu_r128": {
        "head_pad_ms": 500, "tail_pad_sec": 2.0,
        "fade_in_ms": 10, "fade_out_ms": 300,
        "fade_curve": "cosine",
        "target_lufs": -23.0, "max_true_peak_dbtp": -1.0,
    },
    "cd_master": {
        "head_pad_ms": 0, "tail_pad_sec": 3.0,
        "fade_in_ms": 0, "fade_out_ms": 500,
        "fade_curve": "cosine",
        "target_lufs": None, "max_true_peak_dbtp": -0.3,  # CD는 LUFS norm 안 함
    },
}


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "master", help=_("Mastering delivery workflow (prep / qc / verify)")
    )
    sub = parser.add_subparsers(dest="action", help=_("Master action"))

    # prep
    p_prep = sub.add_parser("prep", help=_("Prepare master file (DC remove, pad, fade, loudness norm)"))
    p_prep.add_argument("input", help=_("Source audio file"))
    p_prep.add_argument("--output", "-o", required=True, help=_("Output file path"))
    p_prep.add_argument("--profile", choices=list(PREP_PROFILES.keys()), default="spotify",
                        help=_("Mastering profile (default: spotify)"))
    p_prep.add_argument("--head-pad-ms", type=float, default=None, help=_("Override head pad (ms)"))
    p_prep.add_argument("--tail-pad-sec", type=float, default=None, help=_("Override tail pad (seconds)"))
    p_prep.add_argument("--fade-in-ms", type=float, default=None, help=_("Override fade-in (ms)"))
    p_prep.add_argument("--fade-out-ms", type=float, default=None, help=_("Override fade-out (ms)"))
    p_prep.add_argument("--fade-curve", choices=list(dsp.FADE_CURVES), default=None,
                        help=_("Fade curve (default: cosine)"))
    p_prep.add_argument("--target-lufs", type=float, default=None, help=_("Target LUFS (None to skip norm)"))
    p_prep.add_argument("--max-tp", type=float, default=None, help=_("Max true peak dBTP"))
    p_prep.add_argument("--no-dc-remove", action="store_true", help=_("Skip DC offset removal"))
    p_prep.add_argument("--write-edl", action="store_true",
                        help=_("Also write the generated EDL into .audioman workspace"))
    p_prep.set_defaults(func=run_prep)

    # qc
    p_qc = sub.add_parser("qc", help=_("Run mastering QC report against a target profile"))
    p_qc.add_argument("input", help=_("Audio file to evaluate"))
    p_qc.add_argument("--target", choices=qc.list_targets(), default="spotify",
                      help=_("Target profile (default: spotify)"))
    p_qc.add_argument("--click-sensitivity", type=float, default=6.0,
                      help=_("Click detector sensitivity (default: 6.0). Lower = more sensitive"))
    p_qc.set_defaults(func=run_qc)

    # verify (prep + qc)
    p_verify = sub.add_parser("verify", help=_("Prep + QC in one shot"))
    p_verify.add_argument("input", help=_("Source audio file"))
    p_verify.add_argument("--output", "-o", required=True, help=_("Output file path"))
    p_verify.add_argument("--profile", choices=list(PREP_PROFILES.keys()), default="spotify")
    p_verify.add_argument("--target", choices=qc.list_targets(), default=None,
                          help=_("QC target profile (default: same as --profile)"))
    p_verify.set_defaults(func=run_verify)

    # list-profiles
    p_list = sub.add_parser("list-profiles", help=_("List available mastering profiles"))
    p_list.set_defaults(func=run_list_profiles)

    parser.set_defaults(func=lambda args: parser.print_help())


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


def _build_prep_params(args: argparse.Namespace) -> dict:
    """profile + CLI override 병합. 결과는 prep 단일 파라미터 dict."""
    base = dict(PREP_PROFILES[args.profile])
    overrides = {
        "head_pad_ms": args.head_pad_ms,
        "tail_pad_sec": args.tail_pad_sec,
        "fade_in_ms": args.fade_in_ms,
        "fade_out_ms": args.fade_out_ms,
        "fade_curve": args.fade_curve,
        "target_lufs": args.target_lufs,
        "max_true_peak_dbtp": args.max_tp,
    }
    for k, v in overrides.items():
        if v is not None:
            base[k] = v
    return base


def _build_prep_edl(source: Path, params: dict, skip_dc: bool) -> edl_core.EDL:
    """prep 시퀀스를 EDL ops로 표현. 동일 입력 → 동일 EDL 보장."""
    edl = edl_core.init_edl(source)
    if not skip_dc:
        edl_core.add_op(edl, {"type": "remove_dc"})
    edl_core.add_op(edl, {
        "type": "pad",
        "head_ms": params["head_pad_ms"],
        "tail_sec": params["tail_pad_sec"],
    })
    if params["fade_in_ms"] > 0:
        edl_core.add_op(edl, {
            "type": "fade_in",
            "duration_sec": params["fade_in_ms"] / 1000.0,
            "curve": params["fade_curve"],
        })
    if params["fade_out_ms"] > 0:
        edl_core.add_op(edl, {
            "type": "fade_out",
            "duration_sec": params["fade_out_ms"] / 1000.0,
            "curve": params["fade_curve"],
        })
    if params["target_lufs"] is not None:
        edl_core.add_op(edl, {
            "type": "loudness_normalize",
            "target_lufs": params["target_lufs"],
            "max_true_peak_dbtp": params["max_true_peak_dbtp"],
        })
    return edl


def run_prep(args: argparse.Namespace) -> None:
    src = Path(args.input).resolve()
    if not src.exists():
        print_error(f"파일 없음: {src}")

    params = _build_prep_params(args)
    edl = _build_prep_edl(src, params, skip_dc=args.no_dc_remove)

    if args.write_edl:
        edl_path = edl_core.edl_path(src)
        edl_core.workspace_dir(src).mkdir(parents=True, exist_ok=True)
        edl_core.save_edl(edl, edl_path)
        edl_core.snapshot_history(edl, src)

    start = time.monotonic()
    try:
        result = edl_core.render_edl(edl, args.output, edl_path=None)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print_error(str(e))
    elapsed = time.monotonic() - start

    if args.json:
        print_json({
            "command": "master prep",
            "profile": args.profile,
            "params": params,
            "input": str(src),
            "output": args.output,
            "input_duration_sec": result.input_duration_sec,
            "output_duration_sec": result.output_duration_sec,
            "ops_applied": [op["type"] for op in edl.ops],
            "elapsed_sec": round(elapsed, 3),
        })
        return

    print_success(f"master prep 완료 ({args.profile})")
    output_console.print(f"  Output:    {args.output}")
    output_console.print(f"  Duration:  {result.input_duration_sec:.2f}s → {result.output_duration_sec:.2f}s")
    output_console.print(f"  Ops:       {', '.join(op['type'] for op in edl.ops)}")
    output_console.print(f"  Time:      {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# qc
# ---------------------------------------------------------------------------


def _print_qc_human(report: dict) -> None:
    verdict = report["verdict"]
    color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(verdict, "white")
    output_console.print(
        f"\n[bold]QC Verdict:[/bold] [{color}]{verdict}[/{color}] "
        f"(target: {report['target_profile']['name']})"
    )
    output_console.print(
        f"  PASS: {report['summary']['n_pass']}, "
        f"WARN: {report['summary']['n_warn']}, "
        f"FAIL: {report['summary']['n_fail']}\n"
    )

    rows = []
    for c in report["checks"]:
        status = c["status"]
        sc = {"PASS": "[green]PASS[/green]",
              "WARN": "[yellow]WARN[/yellow]",
              "FAIL": "[red]FAIL[/red]"}.get(status, status)
        rows.append([
            c["category"],
            c["name"],
            str(c.get("value", "")),
            str(c.get("target", "")),
            sc,
        ])
    print_table("Checks", ["Category", "Check", "Value", "Target", "Status"], rows)

    # WARN/FAIL 항목의 detail
    for c in report["checks"]:
        if c["status"] in ("WARN", "FAIL") and "detail" in c:
            output_console.print(f"  [{c['status']}] {c['name']}: {c['detail']}")


def run_qc(args: argparse.Namespace) -> None:
    src = Path(args.input).resolve()
    if not src.exists():
        print_error(f"파일 없음: {src}")

    try:
        report = qc.evaluate_file(src, target=args.target, click_sensitivity=args.click_sensitivity)
    except ValueError as e:
        print_error(str(e))

    if args.json:
        print_json({"command": "master qc", "input": str(src), **report})
        return

    output_console.print(f"\n[bold]{src.name}[/bold]")
    if report.get("format"):
        f = report["format"]
        output_console.print(
            f"  Format: {f['sample_rate']} Hz / "
            f"{f['bit_depth']}-bit / {f['channels']} ch / "
            f"{f['duration_sec']:.2f}s"
        )
    _print_qc_human(report)


# ---------------------------------------------------------------------------
# verify (prep + qc)
# ---------------------------------------------------------------------------


def run_verify(args: argparse.Namespace) -> None:
    args.head_pad_ms = None
    args.tail_pad_sec = None
    args.fade_in_ms = None
    args.fade_out_ms = None
    args.fade_curve = None
    args.target_lufs = None
    args.max_tp = None
    args.no_dc_remove = False
    args.write_edl = False

    # prep
    src = Path(args.input).resolve()
    if not src.exists():
        print_error(f"파일 없음: {src}")
    params = _build_prep_params(args)
    edl = _build_prep_edl(src, params, skip_dc=False)

    start = time.monotonic()
    try:
        result = edl_core.render_edl(edl, args.output, edl_path=None)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print_error(str(e))
    prep_elapsed = time.monotonic() - start

    # qc
    qc_target = args.target or args.profile
    if qc_target not in qc.list_targets():
        print_warning(f"{qc_target!r}는 QC 프로파일에 없음. 'spotify'로 fallback")
        qc_target = "spotify"
    qc_report = qc.evaluate_file(args.output, target=qc_target)

    if args.json:
        print_json({
            "command": "master verify",
            "profile": args.profile,
            "qc_target": qc_target,
            "input": str(src),
            "output": args.output,
            "prep": {
                "params": params,
                "input_duration_sec": result.input_duration_sec,
                "output_duration_sec": result.output_duration_sec,
                "ops_applied": [op["type"] for op in edl.ops],
                "elapsed_sec": round(prep_elapsed, 3),
            },
            "qc": qc_report,
        })
        return

    print_success(f"prep 완료 ({args.profile}) — {result.input_duration_sec:.2f}s → {result.output_duration_sec:.2f}s in {prep_elapsed:.2f}s")
    output_console.print(f"\n[bold]{Path(args.output).name}[/bold]  (QC target: {qc_target})")
    if qc_report.get("format"):
        f = qc_report["format"]
        output_console.print(
            f"  Format: {f['sample_rate']} Hz / {f['bit_depth']}-bit / {f['channels']} ch"
        )
    _print_qc_human(qc_report)


# ---------------------------------------------------------------------------
# list-profiles
# ---------------------------------------------------------------------------


def run_list_profiles(args: argparse.Namespace) -> None:
    if args.json:
        print_json({
            "command": "master list-profiles",
            "prep_profiles": PREP_PROFILES,
            "qc_targets": qc.list_targets(),
        })
        return

    rows = []
    for name, p in PREP_PROFILES.items():
        rows.append([
            name,
            f"{p['head_pad_ms']}ms / {p['tail_pad_sec']}s",
            f"{p['fade_in_ms']}/{p['fade_out_ms']}ms ({p['fade_curve']})",
            str(p["target_lufs"]),
            str(p["max_true_peak_dbtp"]),
        ])
    print_table(
        "Mastering profiles",
        ["Profile", "Head/Tail pad", "Fade in/out", "Target LUFS", "Max TP dBTP"],
        rows,
    )
    output_console.print(f"\n  QC targets: {', '.join(qc.list_targets())}")
