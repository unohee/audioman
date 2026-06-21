# Created: 2026-05-11
# Purpose: `audioman schemas {list,show}` — JSONSchema 발행.
# LLM agent가 audioman --json 출력의 모양을 사전에 알 수 있게 한다.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audioman import __version__


def _schemas_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "schemas"


def _list_schemas() -> list[dict]:
    d = _schemas_dir()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out.append({
            "name": p.stem,
            "id": obj.get("$id", ""),
            "title": obj.get("title", ""),
            "path": str(p),
        })
    return out


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "schemas",
        help="Show audioman JSONSchemas (machine-readable contract for --json output)",
        description="LLM agents can use these schemas to validate audioman output without running it.",
    )
    sub = parser.add_subparsers(dest="schemas_action")

    p_list = sub.add_parser("list", help="List available schemas")
    p_list.set_defaults(func=_run_list)

    p_show = sub.add_parser("show", help="Show a schema body")
    p_show.add_argument("name", help="Schema name (e.g. finding.v1, observe.v1, analyze.v1)")
    p_show.set_defaults(func=_run_show)

    parser.set_defaults(func=_run_default)


def _run_default(args: argparse.Namespace) -> None:
    # 서브명령 미지정 시 list와 동일하게 동작
    _run_list(args)


def _run_list(args: argparse.Namespace) -> None:
    schemas = _list_schemas()
    if getattr(args, "json", False):
        print(json.dumps({
            "$schema": "audioman://schema/schemas.v1.json",
            "audioman_version": __version__,
            "command": "schemas",
            "schemas": schemas,
        }, indent=2, ensure_ascii=False))
        return
    for s in schemas:
        print(f"{s['name']}\t{s['id']}\t{s['title']}")


def _run_show(args: argparse.Namespace) -> None:
    name = args.name
    if not name.endswith(".json"):
        name = name + ".json"
    target = _schemas_dir() / name
    if not target.is_file():
        # fallback: name 기반 검색 ("finding" → "finding.v1.json")
        matches = sorted(_schemas_dir().glob(f"{args.name}*.json"))
        if not matches:
            print(f"error: schema not found: {args.name}", file=sys.stderr)
            sys.exit(1)
        target = matches[0]

    text = target.read_text(encoding="utf-8")
    # 그대로 stdout (JSONSchema 자체가 JSON이므로 --json과 무관하게 valid JSON)
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
