# Created: 2026-05-11
# Purpose: `audioman schemas {list,show}` — publish JSONSchemas.
# Lets LLM agents learn the shape of audioman --json output up front.

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from audioman.core.findings import json_envelope, schema_uri


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
    # With no subcommand given, behave the same as list
    _run_list(args)


def _run_list(args: argparse.Namespace) -> None:
    schemas = _list_schemas()
    if getattr(args, "json", False):
        print(json.dumps(
            json_envelope("schemas", {"schemas": schemas}, schema=schema_uri("schemas")),
            indent=2, ensure_ascii=False,
        ))
        return
    for s in schemas:
        print(f"{s['name']}\t{s['id']}\t{s['title']}")


def _run_show(args: argparse.Namespace) -> None:
    name = args.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        print(f"error: invalid schema name: {name}", file=sys.stderr)
        sys.exit(2)
    if not name.endswith(".json"):
        name = name + ".json"
    schema_dir = _schemas_dir().resolve()
    target = (schema_dir / name).resolve()
    if schema_dir not in target.parents or target.suffix != ".json":
        print(f"error: invalid schema name: {args.name}", file=sys.stderr)
        sys.exit(2)
    if not target.is_file():
        # fallback: search by name ("finding" → "finding.v1.json")
        matches = sorted(_schemas_dir().glob(f"{args.name}*.json"))
        if not matches:
            print(f"error: schema not found: {args.name}", file=sys.stderr)
            sys.exit(1)
        target = matches[0].resolve()

    text = target.read_text(encoding="utf-8")
    # Straight to stdout (a JSONSchema is itself JSON, so this is valid JSON regardless of --json)
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
