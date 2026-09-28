# Created: 2026-09-28
# Purpose: Guard against NameError in cli/*.py from using an output helper that was never imported.

"""Regression guard for a class of bug that shipped twice.

`cli/voiceover.py` and `cli/dump.py` both called `print_success(...)` without importing
it, so the human-readable success path raised `NameError` *after* the output file had
already been written. The failure hid because the only reachable path to those lines
requires a resolvable plugin, and this host has none.

A grep for the helper name matches the call site as well as the import, so it cannot
distinguish the two. This test uses the AST: for every module under `src/audioman/cli/`,
collect the names bound by import statements and the names actually loaded, and assert
that every `audioman.cli.output` helper is imported by the module that uses it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

CLI_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "audioman" / "cli"

# The public surface of audioman.cli.output. Keep in step with that module.
OUTPUT_HELPERS = frozenset(
    {
        "console",
        "output_console",
        "print_error",
        "print_info",
        "print_json",
        "print_literal",
        "print_markup",
        "print_success",
        "print_table",
        "print_warning",
        "set_plain",
    }
)


def _cli_modules() -> list[pathlib.Path]:
    return sorted(p for p in CLI_DIR.glob("*.py") if p.name != "output.py")


@pytest.mark.parametrize("path", _cli_modules(), ids=lambda p: p.name)
def test_output_helpers_are_imported_where_used(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update((a.asname or a.name).split(".")[0] for a in node.names)

    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }

    used_but_not_imported = (loaded & OUTPUT_HELPERS) - imported
    assert not used_but_not_imported, (
        f"{path.name} uses {sorted(used_but_not_imported)} but never imports "
        f"{'it' if len(used_but_not_imported) == 1 else 'them'} — that is a NameError "
        f"on the code path that calls it."
    )


def test_the_guard_actually_catches_a_missing_import() -> None:
    """The check must flag a real omission, not merely pass on clean files."""
    source = "def run():\n    print_success('done')\n"
    tree = ast.parse(source)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)

    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert (loaded & OUTPUT_HELPERS) - imported == {"print_success"}
