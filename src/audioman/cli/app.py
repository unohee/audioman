# Created: 2026-03-21
# Purpose: audioman CLI 메인 파서 (argparse)

import argparse
import logging
import os
import sys


def _early_plain_detect(argv: list[str] | None) -> bool:
    """parse_args 전에 --plain / AUDIOMAN_PLAIN을 감지.

    i18n._detect_lang()이 import 시점에 호출될 수 있으므로
    env를 먼저 세팅해야 한국어 카탈로그가 활성화되지 않는다.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    if "--plain" in args:
        os.environ["AUDIOMAN_PLAIN"] = "1"
        return True
    val = os.environ.get("AUDIOMAN_PLAIN", "").strip().lower()
    return val in ("1", "true", "yes", "on")


_PLAIN_EARLY = _early_plain_detect(None)

from audioman import __version__  # noqa: E402
from audioman.cli import scan, list_cmd, info, process, chain, preset, dump, analyze, fx, visualize, doctor, eq_profile, bounce, commit_cmd, mixdown, edl as edl_cli, master as master_cli, fader_test as fader_test_cli, fader_compare as fader_compare_cli, voiceover as voiceover_cli, screen as screen_cli, obs as obs_cli, observe as observe_cli, changelog_cmd, schemas_cmd, stream as stream_cli  # noqa: E402
from audioman.cli.output import set_plain  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audioman",
        description="Cross-platform CLI wrapper for VST3/AU audio plugins",
    )
    parser.add_argument("--version", action="version", version=f"audioman {__version__}")
    parser.add_argument("--json", action="store_true", help="JSON output mode")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="LLM-friendly output: no color, no rich tables, English help (also via AUDIOMAN_PLAIN=1)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    scan.add_parser(subparsers)
    list_cmd.add_parser(subparsers)
    info.add_parser(subparsers)
    process.add_parser(subparsers)
    chain.add_parser(subparsers)
    preset.add_parser(subparsers)
    dump.add_parser(subparsers)
    analyze.add_parser(subparsers)
    fx.add_parser(subparsers)
    visualize.add_parser(subparsers)
    doctor.add_parser(subparsers)
    eq_profile.add_parser(subparsers)
    bounce.add_parser(subparsers)
    commit_cmd.add_parser(subparsers)
    mixdown.add_parser(subparsers)
    edl_cli.add_parser(subparsers)
    master_cli.add_parser(subparsers)
    fader_test_cli.add_parser(subparsers)
    fader_compare_cli.add_parser(subparsers)
    voiceover_cli.add_parser(subparsers)
    screen_cli.add_parser(subparsers)
    obs_cli.add_parser(subparsers)
    observe_cli.add_parser(subparsers)
    changelog_cmd.add_parser(subparsers)
    schemas_cmd.add_parser(subparsers)
    stream_cli.add_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> None:
    # parse 전에 --plain 재감지 (argv가 명시적으로 전달된 경우)
    plain = _early_plain_detect(argv) or _PLAIN_EARLY
    if plain:
        set_plain(True)

    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "plain", False):
        set_plain(True)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)
