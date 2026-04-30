# Created: 2026-04-27
# Purpose: audioman fader-test — 멀티트랙 stem을 PyQt UI로 재생하며 fader로 mix balance 잡기.
#          본인이 들으며 정한 gain → ground truth JSON으로 export → automix 알고리즘 평가에 사용.

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from audioman.cli.output import print_error, print_success
from audioman.i18n import _

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "fader-test",
        help=_("Open a multitrack mixer GUI to set per-track gain balance (export as ground truth JSON)"),
    )
    parser.add_argument("input", help=_("Stem directory (folder of .wav files)"))
    parser.add_argument("--load", help=_("Load gains JSON at startup"))
    parser.add_argument("--block-size", type=int, default=1024,
                        help=_("Audio block size (default: 1024)"))
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        print_error(f"디렉터리 아님: {input_dir}")

    # PyQt + audio engine은 무거우니 lazy import
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print_error("PyQt6 미설치 — uv add PyQt6")

    from audioman.cli._fader_test_ui import FaderTestWindow
    from audioman.core.multitrack_player import MultitrackPlayer

    app = QApplication.instance() or QApplication(sys.argv)

    print(f"loading stems from {input_dir} ...")
    try:
        player = MultitrackPlayer.from_directory(input_dir, block_size=args.block_size)
    except Exception as e:
        print_error(f"로드 실패: {e}")
        return

    print_success(f"loaded {len(player.tracks)} tracks "
                  f"({player.duration_sec:.1f}s @ {player.sample_rate}Hz)")

    if args.load:
        with open(args.load) as f:
            data = json.load(f)
        gains = data.get("gains", data)  # 두 형식 다 지원
        if isinstance(gains, dict):
            n = player.import_gains(gains)
            print_success(f"loaded gains for {n} tracks from {args.load}")

    win = FaderTestWindow(player, source_dir=input_dir)
    win.show()
    sys.exit(app.exec())
