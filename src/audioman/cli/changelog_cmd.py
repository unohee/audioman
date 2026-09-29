# Created: 2026-05-11
# Purpose: `audioman changelog` — expose the change history to LLM agents.
# Addresses feedback #5: --version alone cannot tell which flags landed when.
# Parses CHANGELOG.md (Keep a Changelog format) into plain text or JSON.

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from audioman.core.findings import json_envelope, schema_uri


_HEADER_RE = re.compile(r"^##\s*\[(?P<version>[^\]]+)\](?:\s*-\s*(?P<date>\S+))?\s*$")
_SECTION_RE = re.compile(r"^###\s+(?P<name>.+?)\s*$")


def _find_changelog() -> Optional[Path]:
    """Find CHANGELOG.md, checking package install location, then repo root, then cwd."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent.parent / "CHANGELOG.md",  # src/audioman/cli → repo
        here.parent.parent.parent / "CHANGELOG.md",
        Path.cwd() / "CHANGELOG.md",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def parse_changelog(text: str) -> list[dict]:
    """Parse Keep-a-Changelog formatted text into entries[]."""
    entries: list[dict] = []
    current: Optional[dict] = None
    current_section: Optional[str] = None

    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            current = {
                "version": m.group("version"),
                "date": m.group("date"),
                "sections": {},
            }
            entries.append(current)
            current_section = None
            continue

        if current is None:
            continue

        ms = _SECTION_RE.match(line)
        if ms:
            current_section = ms.group("name").strip().lower()
            current["sections"].setdefault(current_section, [])
            continue

        if current_section is not None:
            stripped = line.strip()
            if stripped.startswith("- "):
                current["sections"][current_section].append(stripped[2:].strip())
            elif stripped and current["sections"][current_section]:
                # An indented line continues the previous bullet
                current["sections"][current_section][-1] += " " + stripped

    return entries


def _version_tuple(v: str) -> tuple:
    """Turn values like `0.2.0` or `unreleased` into a comparable tuple."""
    if v.lower() == "unreleased":
        return (1 << 30,)  # always the newest
    parts = []
    for p in re.split(r"[.\-+]", v):
        if p.isdigit():
            parts.append(int(p))
        else:
            parts.append(p)
    return tuple(parts)


def filter_since(entries: list[dict], since: str) -> list[dict]:
    since_t = _version_tuple(since)
    out = []
    for e in entries:
        try:
            v_t = _version_tuple(e["version"])
        except Exception:
            continue
        if v_t > since_t:
            out.append(e)
    return out


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "changelog",
        help="Show audioman changelog (LLM-friendly, parses CHANGELOG.md)",
        description=(
            "Surface the project CHANGELOG so LLM agents can tell which flags / "
            "commands exist in which version. Use --since X.Y.Z to filter."
        ),
    )
    parser.add_argument("--since", default=None, help="Only show entries newer than this version")
    parser.add_argument("--path", default=None, help="Explicit path to CHANGELOG.md")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    path: Optional[Path]
    if args.path:
        path = Path(args.path)
    else:
        path = _find_changelog()

    if path is None or not path.is_file():
        msg = "CHANGELOG.md not found"
        if args.json:
            print(json.dumps(json_envelope(
                "changelog", {"error": msg, "entries": []},
                schema=schema_uri("changelog"),
            )))
            sys.exit(1)
        print(f"error: {msg}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    entries = parse_changelog(text)
    if args.since:
        entries = filter_since(entries, args.since)

    if args.json:
        print(json.dumps(
            json_envelope("changelog", {"source": str(path), "entries": entries},
                          schema=schema_uri("changelog")),
            indent=2, ensure_ascii=False,
        ))
        return

    # plain text output (no rich — friendlier to LLM grep)
    for e in entries:
        header = f"## [{e['version']}]"
        if e.get("date"):
            header += f" - {e['date']}"
        print(header)
        for section, bullets in e["sections"].items():
            if not bullets:
                continue
            print(f"### {section}")
            for b in bullets:
                print(f"- {b}")
        print()
