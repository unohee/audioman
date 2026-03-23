# Created: 2026-03-23
# Purpose: audioman render — MIDI 노트 → VST3 인스트루먼트 → 오디오 렌더링

import argparse

from audioman.cli.output import print_error, print_json, print_success, output_console
from audioman.core.engine import parse_params


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "render",
        help="MIDI 노트를 VST3 인스트루먼트로 렌더링 (DawDreamer 백엔드)",
    )

    # 플러그인
    parser.add_argument(
        "--plugin", "-p",
        required=True,
        help="VST3 인스트루먼트 플러그인 (이름 또는 경로)",
    )

    # 입력 모드: 노트 또는 MIDI 파일
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--note", "-n",
        action="append",
        help="MIDI 노트 (60, C4, A#3 등). 여러 개 가능: -n C4 -n E4 -n G4",
    )
    input_group.add_argument(
        "--midi", "-m",
        metavar="FILE",
        help="MIDI 파일 경로 (.mid)",
    )

    # 노트 파라미터
    parser.add_argument(
        "--velocity", "-vel",
        type=int,
        default=100,
        help="벨로시티 (기본: 100)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=2.0,
        help="노트 지속 시간 - 초 (기본: 2.0)",
    )
    parser.add_argument(
        "--start", "-s",
        type=float,
        default=0.0,
        help="첫 노트 시작 시간 - 초 (기본: 0.0)",
    )
    parser.add_argument(
        "--strum",
        type=float,
        default=0.0,
        help="여러 노트 간 시간 간격 - 초 (코드: 0, 아르페지오: 0.1 등)",
    )

    # 출력
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="출력 오디오 파일 경로",
    )
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=44100,
        help="샘플레이트 (기본: 44100)",
    )

    # 프리셋/파라미터
    parser.add_argument(
        "--preset",
        help="프리셋 파일 경로 (.fxp, .vstpreset)",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="플러그인 파라미터 (key=value)",
    )

    # 렌더링 옵션
    parser.add_argument(
        "--tail",
        type=float,
        default=2.0,
        help="마지막 노트 이후 추가 렌더링 시간 - 초 (기본: 2.0)",
    )
    parser.add_argument(
        "--render-duration",
        type=float,
        default=None,
        help="전체 렌더링 길이 강제 지정 (초)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=512,
        help="버퍼 크기 (기본: 512)",
    )
    parser.add_argument(
        "--fadeout",
        type=float,
        default=0.5,
        help="끝부분 페이드아웃 길이 - 초 (기본: 0.5, 0으로 비활성화)",
    )

    parser.set_defaults(func=run)


def _resolve_plugin_path(plugin_arg: str) -> str:
    """플러그인 인자 → 실제 경로 해석"""
    from pathlib import Path

    # 직접 경로인 경우
    p = Path(plugin_arg)
    if p.exists() and p.suffix.lower() in (".vst3", ".component"):
        return str(p)

    # 레지스트리에서 검색
    from audioman.core.registry import get_registry

    registry = get_registry()
    meta = registry.get(plugin_arg)
    if meta:
        return meta.path

    print_error(f"플러그인을 찾을 수 없습니다: '{plugin_arg}' (경로 또는 등록된 이름)")


def run(args: argparse.Namespace) -> None:
    from audioman.core.instrument import (
        MidiNote,
        parse_note,
        render_midi_file,
        render_notes,
    )

    plugin_path = _resolve_plugin_path(args.plugin)

    # 파라미터 파싱
    params = parse_params(args.param) if args.param else None

    # MIDI 파일 모드
    if args.midi:
        result = render_midi_file(
            plugin_path=plugin_path,
            midi_path=args.midi,
            output_path=args.output,
            sample_rate=args.sample_rate,
            buffer_size=args.buffer_size,
            preset_path=args.preset,
            params=params,
            tail=args.tail,
            fadeout=args.fadeout,
        )
    else:
        # 노트 모드
        notes = []
        for i, note_str in enumerate(args.note):
            midi_num = parse_note(note_str)
            notes.append(
                MidiNote(
                    note=midi_num,
                    velocity=args.velocity,
                    start=args.start + i * args.strum,
                    duration=args.duration,
                )
            )

        result = render_notes(
            plugin_path=plugin_path,
            notes=notes,
            output_path=args.output,
            sample_rate=args.sample_rate,
            buffer_size=args.buffer_size,
            duration=args.render_duration,
            preset_path=args.preset,
            params=params,
            tail=args.tail,
            fadeout=args.fadeout,
        )

    # 출력
    if getattr(args, "json", False):
        print_json({"command": "render", **result.to_dict()})
    else:
        output_console.print(f"[green]렌더링 완료[/green]: {result.output_path}")
        output_console.print(f"  플러그인: {result.plugin_path}")
        output_console.print(f"  {result.duration:.2f}s / {result.sample_rate}Hz / {result.channels}ch")
        output_console.print(f"  peak: {result.peak:.4f} / rms: {result.rms:.4f}")
        if result.notes:
            note_strs = [f"{n['note']}(vel={n['velocity']})" for n in result.notes]
            output_console.print(f"  노트: {', '.join(note_strs)}")
        if result.preset:
            output_console.print(f"  프리셋: {result.preset}")
