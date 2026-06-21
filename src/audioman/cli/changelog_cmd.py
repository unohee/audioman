# Created: 2026-05-11
# Purpose: `audioman changelog` — LLM agent에게 변경 이력을 노출.
# 후기 #5 대응: --version만 있고 어떤 인자가 언제 들어왔는지 추적 불가.
# CHANGELOG.md(Keep a Changelog 형식)를 파싱해 plain text 또는 JSON으로.

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from audioman import __version__


_HEADER_RE = re.compile(r"^##\s*\[(?P<version>[^\]]+)\](?:\s*-\s*(?P<date>\S+))?\s*$")
_SECTION_RE = re.compile(r"^###\s+(?P<name>.+?)\s*$")


def _find_changelog() -> Optional[Path]:
    """패키지 설치 위치 → repo root → cwd 순으로 CHANGELOG.md를 찾는다."""
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
    """Keep-a-Changelog 형식의 텍스트를 entries[]로 파싱."""
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
                # 들여쓰기 줄 = 직전 bullet 연속
                current["sections"][current_section][-1] += " " + stripped

    return entries


def _version_tuple(v: str) -> tuple:
    """`0.2.0`, `unreleased` 같은 값을 비교 가능한 튜플로."""
    if v.lower() == "unreleased":
        return (1 << 30,)  # 항상 최신
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
            print(json.dumps({
                "$schema": "audioman://schema/changelog.v1.json",
                "audioman_version": __version__,
                "command": "changelog",
                "error": msg,
                "entries": [],
            }))
            sys.exit(1)
        print(f"error: {msg}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    entries = parse_changelog(text)
    if args.since:
        entries = filter_since(entries, args.since)

    if args.json:
        print(json.dumps({
            "$schema": "audioman://schema/changelog.v1.json",
            "audioman_version": __version__,
            "command": "changelog",
            "source": str(path),
            "entries": entries,
        }, indent=2, ensure_ascii=False))
        return

    # plain text 출력 (rich 미사용 — LLM grep 친화)
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
