# Created: 2026-04-27
# Purpose: audioman fader-test — play multitrack stems in a PyQt UI and set the mix balance with faders.
#          The gains chosen by ear are exported as ground-truth JSON and used to evaluate the automix algorithm.

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from audioman.cli.output import print_error, print_success

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "fader-test",
        help="Open a multitrack mixer GUI to set per-track gain balance (export as ground truth JSON)",
        description=(
            "Interactive PyQt6 mixer GUI: play the stems and set per-track gain "
            "balance by ear, then export the result as ground-truth JSON for "
            "`audioman fader-compare`. The session is interactive, so this command "
            "emits no machine-readable payload of its own (no JSON output mode); "
            "the exported JSON file is the machine-readable artifact."
        ),
    )
    parser.add_argument("input", help="Stem directory (folder of .wav files)")
    parser.add_argument("--load", help="Load gains JSON at startup")
    parser.add_argument("--block-size", type=int, default=1024,
                        help="Audio block size (default: 1024)")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        print_error(f"Not a directory: {input_dir}")
        return

    # PyQt + audio engine are heavy, so import them lazily.
    #
    # `ImportError` alone is not enough to describe this failure: PyQt6 also fails
    # when its *native* libraries cannot be dlopened (`libEGL.so.1: cannot open
    # shared object file`), which is an OS-level gap rather than a missing package.
    # Reporting that as "PyQt6 not installed" sends the reader to the wrong fix.
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        native = "libEGL" in str(exc) or "cannot open shared object" in str(exc)
        if native:
            print_error(
                f"PyQt6 cannot load its native libraries: {exc} "
                f"(Debian/Ubuntu: sudo apt-get install libegl1 libgl1)"
            )
        else:
            print_error("PyQt6 not installed — run: uv add PyQt6")
        return

    # The audio backend needs the PortAudio *system* library, which is a separate OS
    # package from the `sounddevice` Python module. Without it `import sounddevice`
    # raises OSError — an uncaught traceback for a missing host dependency, the same
    # class of failure as the libEGL case above. `_fader_test_ui` imports the player
    # itself, so this must be resolved before that module is imported.
    try:
        from audioman.core.multitrack_player import MultitrackPlayer
    except OSError as exc:
        print_error(
            f"The audio backend is unavailable: {exc} "
            f"(Debian/Ubuntu: sudo apt-get install libportaudio2 portaudio19-dev)"
        )
        return

    from audioman.cli._fader_test_ui import FaderTestWindow

    app = QApplication.instance() or QApplication(sys.argv)

    print(f"loading stems from {input_dir} ...")
    try:
        player = MultitrackPlayer.from_directory(input_dir, block_size=args.block_size)
    except Exception as e:
        print_error(f"Load failed: {e}")
        return

    print_success(f"loaded {len(player.tracks)} tracks "
                  f"({player.duration_sec:.1f}s @ {player.sample_rate}Hz)")

    if args.load:
        with open(args.load) as f:
            data = json.load(f)
        gains = data.get("gains", data)  # both formats are supported
        if isinstance(gains, dict):
            n = player.import_gains(gains)
            print_success(f"loaded gains for {n} tracks from {args.load}")

    win = FaderTestWindow(player, source_dir=input_dir)
    win.show()
    sys.exit(app.exec())
