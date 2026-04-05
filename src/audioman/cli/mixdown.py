# Created: 2026-04-05
# Purpose: audioman mixdown 서브커맨드 — bounce + 마스터 체인

import argparse

from audioman.cli.output import print_error, print_json, print_success, print_warning, output_console
from audioman.i18n import _


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("mixdown", help=_("Mix tracks with master chain processing"))
    parser.add_argument("inputs", nargs="*", help=_("Input audio files"))
    parser.add_argument("--output", "-o", required=True, help=_("Output file path"))
    parser.add_argument(
        "--gain", default="",
        help=_("Comma-separated gain values in dB per track"),
    )
    parser.add_argument(
        "--pan", default="",
        help=_("Comma-separated pan values per track (-1.0 L ~ 0.0 C ~ 1.0 R)"),
    )
    parser.add_argument(
        "--chain", default="",
        help=_("Per-track plugin chains separated by '|'"),
    )
    parser.add_argument(
        "--master", default="",
        help=_("Master bus plugin chain (e.g. 'limiter:threshold=-1')"),
    )
    parser.add_argument("--session", help=_("Session file (YAML/JSON)"))
    parser.add_argument(
        "--no-compensation", action="store_true",
        help=_("Disable master chain delay compensation"),
    )
    parser.add_argument("--dry-run", action="store_true", help=_("Show plan without executing"))
    parser.set_defaults(func=run)


def _parse_float_list(s: str) -> list[float]:
    if not s.strip():
        return []
    return [float(v.strip()) for v in s.split(",")]


def run(args: argparse.Namespace) -> None:
    from audioman.core.mixer import TrackConfig, mixdown
    from audioman.core.pipeline import parse_chain_string

    # 세션 파일 모드
    if args.session:
        from audioman.core.session import load_session
        try:
            session = load_session(args.session)
        except Exception as e:
            print_error(f"세션 파일 로드 실패: {e}")
            return

        tracks = session.tracks
        output_path = args.output if args.output else session.output
        sample_rate = session.sample_rate
        subtype = session.subtype
        master_chain = session.master_chain
    else:
        if not args.inputs:
            print_error("입력 파일을 지정하세요 (또는 --session 사용)")
            return

        gains = _parse_float_list(args.gain)
        pans = _parse_float_list(args.pan)

        chains = []
        if args.chain.strip():
            for chain_str in args.chain.split("|"):
                chain_str = chain_str.strip()
                chains.append(parse_chain_string(chain_str) if chain_str else None)

        tracks = []
        for i, inp in enumerate(args.inputs):
            tracks.append(TrackConfig(
                path=inp,
                gain_db=gains[i] if i < len(gains) else 0.0,
                pan=pans[i] if i < len(pans) else 0.0,
                chain=chains[i] if i < len(chains) else None,
            ))

        output_path = args.output
        sample_rate = None
        subtype = "PCM_24"
        master_chain = parse_chain_string(args.master) if args.master.strip() else None

    # Dry-run
    if args.dry_run:
        plan = {
            "command": "mixdown",
            "dry_run": True,
            "output": output_path,
            "track_count": len(tracks),
            "tracks": [t.to_dict() for t in tracks],
            "master_chain": [s.to_dict() for s in master_chain] if master_chain else None,
        }
        if args.json:
            print_json(plan)
        else:
            output_console.print(f"\n[bold]Mixdown Plan[/bold] — {len(tracks)} tracks → {output_path}")
            for i, t in enumerate(tracks, 1):
                chain_str = f" → [{', '.join(s.plugin_name for s in t.chain)}]" if t.chain else ""
                output_console.print(
                    f"  {i}. {t.path}  gain={t.gain_db:+.1f}dB  pan={t.pan:+.1f}{chain_str}"
                )
            if master_chain:
                master_str = " → ".join(s.plugin_name for s in master_chain)
                output_console.print(f"\n  Master: [{master_str}]")
        return

    # 실행
    try:
        result = mixdown(
            tracks=tracks,
            output_path=output_path,
            master_chain=master_chain,
            sample_rate=sample_rate,
            subtype=subtype,
            compensate_latency=not args.no_compensation,
        )
    except Exception as e:
        print_error(f"믹스다운 실패: {e}")
        return

    if args.json:
        print_json({"command": "mixdown", **result.to_dict()})
        return

    output_console.print(f"\n[bold]믹스다운 완료[/bold]")
    output_console.print(f"  Tracks: {result.track_count}")
    output_console.print(f"  Output: {result.output_path}")
    output_console.print(f"  SR:     {result.sample_rate} Hz")

    if result.master_chain:
        output_console.print(f"  Master: {len(result.master_chain)} steps")
        if result.master_latency_samples > 0:
            output_console.print(f"  Master latency compensation: {result.master_latency_samples} samples")

    output_console.print(f"  Time:   {result.duration_seconds}s")
    if result.clipping_detected:
        print_warning("클리핑 감지 — 마스터 체인에 리미터 추가 또는 트랙 볼륨 조정 권장")
    print_success("완료")
