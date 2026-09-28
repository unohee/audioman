#!/usr/bin/env python3
"""Pre-commit echo of the constitution gates CI enforces (book.md Art. II, III).

Both checks are deterministic and finish in seconds, which is the bar for a hook that
runs on every commit. They are deliberately thin: the reference implementations are the
gate scripts in `unohee/ci-templates/scripts/gates/`, run by
`.github/workflows/constitution.yml`. This file only keeps the feedback loop local.

    python scripts/gates.py loc [--ceiling 1500]   # Art. III — code lines per file
    python scripts/gates.py bs                      # Art. II  — `cxt bs` criticals

Exit codes: 0 pass, 1 breach, 2 the gate itself could not run (Art. VI: not a pass).

`cxt` must be on PATH (`npm install -g @intrect/cxt`). It is invoked with `--json` and
its exit code is read as a value rather than acted on by a shell, because `cxt bs` exits
non-zero when it reports findings and that is a result, not a failure of the gate.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LOC_LINE = re.compile(r"^\s*(\S+)\s+([\d,]+)\s")
# Generated and scratch trees. `testing/` is gitignored, so `cxt loc` already skips it;
# listed here so the exclusion survives a change in how cxt enumerates files.
EXCLUDE_PREFIXES = ("outputs/", "testing/")


def run_cxt(*args: str) -> str:
    """Run cxt and return stdout, tolerating any exit code.

    `cxt bs` exits non-zero both when it reports findings and when its own analysis is
    only partial, so the exit code cannot distinguish "findings" from "the gate did not
    run". The JSON is read instead, exactly as the CI gate `cxt_bs.py` does — a non-zero
    exit is a value here, not a status the shell acts on.
    """
    try:
        proc = subprocess.run(
            ["cxt", *args], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        print("gate: `cxt` is not installed (npm i -g @intrect/cxt)", file=sys.stderr)
        sys.exit(2)
    return proc.stdout


def gate_loc(ceiling: int) -> int:
    out = ANSI.sub("", run_cxt("loc", "--no-blank", "--no-comments", "--ext", "py"))
    breaches: list[tuple[str, int]] = []
    for line in out.splitlines():
        m = LOC_LINE.match(line)
        if not m or m.group(1).startswith(EXCLUDE_PREFIXES) or m.group(1) == "Total":
            continue
        loc = int(m.group(2).replace(",", ""))
        if loc > ceiling:
            breaches.append((m.group(1), loc))
    for path, loc in breaches:
        print(f"Art. III: {path} has {loc} code lines (> {ceiling}); split it.")
    return 1 if breaches else 0


def gate_bs() -> int:
    try:
        report = json.loads(run_cxt("bs", "--json", "--dir", "src"))
    except json.JSONDecodeError as exc:
        print(f"gate: cxt bs produced no JSON: {exc}", file=sys.stderr)
        return 2
    if not report.get("filesScanned"):
        # A scan of nothing reports zero criticals and passes — the fake-gate shape
        # Art. VI names. Zero files scanned means the gate did not run.
        print("gate: cxt bs scanned 0 files; the gate did not run", file=sys.stderr)
        return 2
    criticals = [i for i in report.get("issues", []) if i.get("severity") == "critical"]
    for issue in criticals:
        print(
            f"Art. II: {issue['file']}:{issue['line']} [{issue['category']}] {issue['message']}"
        )
    return 1 if criticals else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="gate", required=True)
    loc = sub.add_parser("loc")
    loc.add_argument("--ceiling", type=int, default=1500)
    sub.add_parser("bs")
    args = ap.parse_args()
    return gate_loc(args.ceiling) if args.gate == "loc" else gate_bs()


if __name__ == "__main__":
    sys.exit(main())
