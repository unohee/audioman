# Created: 2026-11-04
# Purpose: `--detail` / `--spill-dir` options and the JSON shaping step shared by analysis commands.
# Dependencies: audioman.core.detail, audioman.core.digest

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from audioman.core.detail import DEFAULT_DETAIL, DETAIL_LEVELS, apply_detail
from audioman.core.digest import build_digest
from audioman.core.report_v2 import build_report


def add_detail_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--detail",
        choices=DETAIL_LEVELS,
        default=DEFAULT_DETAIL,
        help=(
            "JSON detail level: summary (<= 4 KiB), standard (<= 32 KiB, default) or full "
            "(everything inline). Long arrays and lists move to spill files referenced by path."
        ),
    )
    parser.add_argument(
        "--spill-dir",
        default=None,
        help="Directory for spilled arrays (default: <cache dir>/spill)",
    )


def default_spill_dir() -> Path:
    from audioman.config.settings import get_settings

    return Path(get_settings().cache_dir) / "spill"


def shape_json(command: str, payload: dict[str, Any], args: argparse.Namespace, *, label: str) -> dict[str, Any]:
    """Add the digest after the envelope keys and the report.v2 block, then shape for `args.detail`.

    Both are derived from the full payload, before any shaping.
    """
    head = {k: payload[k] for k in ("$schema", "audioman_version", "command") if k in payload}
    body = {k: v for k, v in payload.items() if k not in head}
    with_digest = {**head, "digest": build_digest(command, payload), **body, "report": build_report(command, payload)}
    spill_dir = Path(args.spill_dir) if getattr(args, "spill_dir", None) else default_spill_dir()
    level = getattr(args, "detail", DEFAULT_DETAIL)
    return apply_detail(with_digest, level, spill_dir=spill_dir, label=f"{command.replace(' ', '-')}-{label}")
