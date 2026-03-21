# Created: 2026-03-21
# Purpose: audioman CLI 메인 파서 (argparse)

import argparse
import logging
import sys

from audioman import __version__
from audioman.cli import scan, list_cmd, info, process, chain, preset, dump, analyze, fx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audioman",
        description="Cross-platform CLI wrapper for VST3/AU audio plugins",
    )
    parser.add_argument("--version", action="version", version=f"audioman {__version__}")
    parser.add_argument("--json", action="store_true", help="JSON 출력 모드")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로깅")

    subparsers = parser.add_subparsers(dest="command", help="사용 가능한 명령")

    scan.add_parser(subparsers)
    list_cmd.add_parser(subparsers)
    info.add_parser(subparsers)
    process.add_parser(subparsers)
    chain.add_parser(subparsers)
    preset.add_parser(subparsers)
    dump.add_parser(subparsers)
    analyze.add_parser(subparsers)
    fx.add_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)
