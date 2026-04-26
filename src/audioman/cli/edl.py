# Created: 2026-04-26
# Purpose: audioman edl 서브커맨드 — 비파괴 편집 워크플로우

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audioman.cli.output import (
    output_console,
    print_error,
    print_json,
    print_success,
    print_warning,
    print_table,
)
from audioman.core import edl as edl_core
from audioman.i18n import _


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "edl", help=_("Non-destructive edit workflow (EDL)")
    )
    sub = parser.add_subparsers(dest="action", help=_("EDL action"))

    # init
    p_init = sub.add_parser("init", help=_("Initialize EDL workspace for an input file"))
    p_init.add_argument("input", help=_("Source audio file"))
    p_init.set_defaults(func=run_init)

    # add
    p_add = sub.add_parser("add", help=_("Append an op to the active EDL"))
    p_add.add_argument("--source", "-s", required=True, help=_("Source audio file"))
    p_add.add_argument("op_type", help=_(
        "Op type (cut_region, trim, trim_silence, splice, fade_in, fade_out, "
        "normalize, gain, gate, process, chain)"
    ))
    p_add.add_argument(
        "--param", "-p", action="append", default=[],
        help=_("Op parameter as key=value (repeat). Numbers/bools auto-detected; "
              "use json:<value> for arrays/objects."),
    )
    p_add.set_defaults(func=run_add)

    # list (= show ops)
    p_list = sub.add_parser("list", help=_("Show all ops in the active EDL"))
    p_list.add_argument("--source", "-s", required=True)
    p_list.set_defaults(func=run_list)

    # undo
    p_undo = sub.add_parser("undo", help=_("Undo the most recent op"))
    p_undo.add_argument("--source", "-s", required=True)
    p_undo.set_defaults(func=run_undo)

    # redo
    p_redo = sub.add_parser("redo", help=_("Redo the most recently undone op"))
    p_redo.add_argument("--source", "-s", required=True)
    p_redo.set_defaults(func=run_redo)

    # render
    p_render = sub.add_parser("render", help=_("Render the EDL to a final output file"))
    p_render.add_argument("--source", "-s", required=True)
    p_render.add_argument("--output", "-o", required=True, help=_("Output WAV path"))
    p_render.add_argument("--no-verify", action="store_true",
                          help=_("Skip source SHA-256 verification"))
    p_render.set_defaults(func=run_render)

    # status (= 워크스페이스 상태)
    p_status = sub.add_parser("status", help=_("Show workspace status"))
    p_status.add_argument("--source", "-s", required=True)
    p_status.set_defaults(func=run_status)

    # clear
    p_clear = sub.add_parser("clear", help=_("Remove all ops from active EDL (history kept)"))
    p_clear.add_argument("--source", "-s", required=True)
    p_clear.set_defaults(func=run_clear)

    parser.set_defaults(func=lambda args: parser.print_help())


# ---------------------------------------------------------------------------
# Param parsing
# ---------------------------------------------------------------------------


def _parse_value(raw: str):
    """문자열을 적절한 타입으로 변환. process plugin 같은 문자열은 그대로."""
    if raw.startswith("json:"):
        return json.loads(raw[5:])
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        if "." in raw or "e" in low:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _parse_params(raw_list: list[str]) -> dict:
    out: dict = {}
    for item in raw_list:
        if "=" not in item:
            print_error(f"--param 형식 오류 (key=value 필요): {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = _parse_value(v.strip())
    return out


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _ensure_workspace(source: Path) -> None:
    edl_core.workspace_dir(source).mkdir(parents=True, exist_ok=True)


def run_init(args: argparse.Namespace) -> None:
    src = Path(args.input).resolve()
    if not src.exists():
        print_error(f"파일 없음: {src}")

    edl_path = edl_core.edl_path(src)
    if edl_path.exists():
        print_warning(f"이미 초기화된 EDL이 있습니다: {edl_path}")

    edl = edl_core.init_edl(src)
    _ensure_workspace(src)
    edl_core.save_edl(edl, edl_path)
    edl_core.snapshot_history(edl, src)

    if args.json:
        print_json({
            "command": "edl init",
            "source": str(src),
            "edl_path": str(edl_path),
            "workspace": str(edl_core.workspace_dir(src)),
            "duration_sec": edl.duration_sec,
            "sample_rate": edl.sample_rate,
            "channels": edl.channels,
            "source_sha256": edl.source_sha256,
        })
        return

    print_success(f"EDL 초기화: {edl_path}")
    output_console.print(f"  Source:    {src}")
    output_console.print(f"  Duration:  {edl.duration_sec:.2f}s")
    output_console.print(f"  SR / CH:   {edl.sample_rate} Hz / {edl.channels} ch")
    output_console.print(f"  SHA-256:   {edl.source_sha256[:16]}…")


def run_add(args: argparse.Namespace) -> None:
    src = Path(args.source).resolve()
    edl_path = edl_core.edl_path(src)
    if not edl_path.exists():
        print_error(f"EDL이 초기화되지 않았습니다. 먼저 'audioman edl init {src}' 실행")

    edl = edl_core.load_edl(edl_path)
    params = _parse_params(args.param)
    op = {"type": args.op_type, **params}

    try:
        edl_core.add_op(edl, op)
    except ValueError as e:
        print_error(str(e))

    edl_core.save_edl(edl, edl_path)
    edl_core.snapshot_history(edl, src)  # clear_redo=True

    if args.json:
        print_json({
            "command": "edl add",
            "source": str(src),
            "op": op,
            "n_ops": len(edl.ops),
        })
        return

    print_success(f"op 추가: {op['type']} (총 {len(edl.ops)}개)")
    for k, v in op.items():
        if k != "type":
            output_console.print(f"  {k}: {v}")


def run_list(args: argparse.Namespace) -> None:
    src = Path(args.source).resolve()
    edl_path = edl_core.edl_path(src)
    if not edl_path.exists():
        print_error(f"EDL이 초기화되지 않았습니다: {src}")
    edl = edl_core.load_edl(edl_path)

    if args.json:
        print_json({"command": "edl list", "source": str(src), "ops": edl.ops,
                    "n_ops": len(edl.ops)})
        return

    if not edl.ops:
        output_console.print("  (ops 없음 — 'edl add'로 추가)")
        return
    rows = []
    for i, op in enumerate(edl.ops):
        params = ", ".join(f"{k}={v}" for k, v in op.items() if k != "type")
        rows.append([str(i + 1), op["type"], params])
    print_table(f"EDL ops ({len(edl.ops)})", ["#", "type", "params"], rows)


def run_undo(args: argparse.Namespace) -> None:
    src = Path(args.source).resolve()
    edl_path = edl_core.edl_path(src)
    if not edl_path.exists():
        print_error(f"EDL이 초기화되지 않았습니다: {src}")
    new_edl = edl_core.undo(src)
    if new_edl is None:
        if args.json:
            print_json({"command": "edl undo", "source": str(src), "undone": False,
                        "reason": "history empty"})
            return
        print_warning("되돌릴 op이 없습니다.")
        return
    if args.json:
        print_json({"command": "edl undo", "source": str(src), "undone": True,
                    "n_ops": len(new_edl.ops)})
        return
    print_success(f"undo 완료. 현재 op 수: {len(new_edl.ops)}")


def run_redo(args: argparse.Namespace) -> None:
    src = Path(args.source).resolve()
    new_edl = edl_core.redo(src)
    if new_edl is None:
        if args.json:
            print_json({"command": "edl redo", "source": str(src), "redone": False})
            return
        print_warning("redo할 op이 없습니다.")
        return
    if args.json:
        print_json({"command": "edl redo", "source": str(src), "redone": True,
                    "n_ops": len(new_edl.ops)})
        return
    print_success(f"redo 완료. 현재 op 수: {len(new_edl.ops)}")


def run_render(args: argparse.Namespace) -> None:
    src = Path(args.source).resolve()
    edl_path = edl_core.edl_path(src)
    if not edl_path.exists():
        print_error(f"EDL이 초기화되지 않았습니다: {src}")
    edl = edl_core.load_edl(edl_path)

    try:
        result = edl_core.render_edl(
            edl, args.output,
            edl_path=edl_path,
            verify_source=not args.no_verify,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print_error(str(e))

    if args.json:
        print_json({"command": "edl render", **result.to_dict()})
        return

    print_success(f"render 완료: {args.output}")
    output_console.print(f"  ops 적용:  {result.n_ops}")
    output_console.print(f"  Duration:  {result.input_duration_sec:.2f}s → {result.output_duration_sec:.2f}s")
    output_console.print(f"  Time:      {result.elapsed_sec:.2f}s")


def run_status(args: argparse.Namespace) -> None:
    src = Path(args.source).resolve()
    ws = edl_core.workspace_dir(src)
    edl_path = edl_core.edl_path(src)
    hist = edl_core.list_history(src)
    rd = edl_core.list_redo(src)

    if not edl_path.exists():
        if args.json:
            print_json({"command": "edl status", "source": str(src), "initialized": False})
            return
        output_console.print(f"  (초기화되지 않음. 'audioman edl init {src}')")
        return

    edl = edl_core.load_edl(edl_path)
    info = {
        "initialized": True,
        "source": str(src),
        "workspace": str(ws),
        "edl_path": str(edl_path),
        "n_ops": len(edl.ops),
        "history_depth": len(hist),
        "redo_depth": len(rd),
        "duration_sec": edl.duration_sec,
        "modified_at": edl.modified_at,
    }
    if args.json:
        print_json({"command": "edl status", **info})
        return
    output_console.print(f"  Source:        {src}")
    output_console.print(f"  Workspace:     {ws}")
    output_console.print(f"  Ops:           {len(edl.ops)}")
    output_console.print(f"  History depth: {len(hist)}")
    output_console.print(f"  Redo depth:    {len(rd)}")
    output_console.print(f"  Modified:      {edl.modified_at}")


def run_clear(args: argparse.Namespace) -> None:
    src = Path(args.source).resolve()
    edl_path = edl_core.edl_path(src)
    if not edl_path.exists():
        print_error(f"EDL이 초기화되지 않았습니다: {src}")
    edl = edl_core.load_edl(edl_path)
    edl.ops = []
    edl_core.save_edl(edl, edl_path)
    edl_core.snapshot_history(edl, src)
    if args.json:
        print_json({"command": "edl clear", "source": str(src), "n_ops": 0})
        return
    print_success("모든 op 삭제 (history는 유지)")
