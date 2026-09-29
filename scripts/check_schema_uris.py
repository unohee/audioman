#!/usr/bin/env python3
# Created: 2026-09-28
# Purpose: guard against dangling audioman://schema/*.json URIs.
# Run: .venv/bin/python scripts/check_schema_uris.py
"""Enumerate every schema URI reachable from src/ and assert a matching
published file exists under src/audioman/schemas/.

Two forms are collected:
  1. literal      audioman://schema/<name>.json
  2. constructed  schema_uri("<name>")  ->  audioman://schema/<name>.v1.json
                  SCHEMA_URI            ->  audioman://schema/finding.v1.json

Exit 0 when every URI resolves; exit 1 listing the dangling ones.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
SCHEMAS = SRC / "audioman" / "schemas"
PREFIX = "audioman://schema/"

LITERAL_RE = re.compile(r"audioman://schema/[A-Za-z0-9._-]+\.json")
CALL_RE = re.compile(r"schema_uri\(\s*[\"']([A-Za-z0-9._-]+)[\"']\s*\)")

found: dict[str, list[str]] = {}


def _record(uri: str, where: str) -> None:
    refs = found.setdefault(uri, [])
    if where not in refs:
        refs.append(where)


for path in sorted(SRC.rglob("*.py")):
    text = path.read_text(encoding="utf-8")
    rel = str(path.relative_to(REPO))
    for uri in set(LITERAL_RE.findall(text)):
        _record(uri, rel)
    for name in set(CALL_RE.findall(text)):
        _record(f"{PREFIX}{name}.v1.json", rel)

published = sorted(p.name for p in SCHEMAS.glob("*.json"))
print(f"schema URIs referenced in src/: {len(found)}")
print(f"published schema files:        {len(published)}")
print()

dangling = []
for uri in sorted(found):
    name = uri[len(PREFIX):]
    target = SCHEMAS / name
    ok = target.is_file()
    if not ok:
        dangling.append(uri)
    print(f"  [{'OK' if ok else 'DANGLING':8s}] {uri}")
    for where in found[uri]:
        print(f"               referenced by {where}")

unreferenced = [n for n in published if f"{PREFIX}{n}" not in found]
print()
print(f"published but not directly referenced from src/: {unreferenced}")

if dangling:
    print(f"\nFAIL: {len(dangling)} dangling schema URI(s): {dangling}")
    sys.exit(1)
print(f"\nOK: all {len(found)} referenced schema URIs resolve to a published file.")
