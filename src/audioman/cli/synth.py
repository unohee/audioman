# Created: 2026-03-24
# Purpose: audioman synth — SignalFlow 신스 엔진 CLI

import argparse

from audioman.cli.output import print_error, print_json, print_success, output_console


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "synth",
        help="SignalFlow 신스 엔진으로 사운드 생성 (MIT 라이선스)",
    )

    # 노트
    parser.add_argument("--note", "-n", default="C4", help="MIDI 노트 (C4, A#3, 60 등)")
    parser.add_argument("--velocity", "-vel", type=int, default=100)
    parser.add_argument("--duration", "-d", type=float, default=2.0, help="초")
    parser.add_argument("--tail", type=float, default=1.0)
    parser.add_argument("--output", "-o", default=None, help="출력 WAV 경로")
    parser.add_argument("--sample-rate", "-sr", type=int, default=44100)

    # 패치 소스 (상호 배타)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--patch", metavar="JSON", help="패치 JSON 파일 경로")
    source.add_argument("--text", metavar="DESC", help="자연어 설명 (CLAP 검색)")

    # 빠른 파라미터 오버라이드
    osc = parser.add_argument_group("oscillator")
    osc.add_argument("--osc1-mode", choices=["va", "wt", "fm"], default=None)
    osc.add_argument("--osc1-wave", choices=["saw", "square", "tri", "sine"], default=None)
    osc.add_argument("--osc1-vol", type=float, default=None)
    osc.add_argument("--osc2-wave", choices=["saw", "square", "tri", "sine"], default=None)
    osc.add_argument("--osc2-vol", type=float, default=None)
    osc.add_argument("--sub-vol", type=float, default=None)
    osc.add_argument("--noise-vol", type=float, default=None)

    fm = parser.add_argument_group("fm")
    fm.add_argument("--fm-ratio", type=float, default=None)
    fm.add_argument("--fm-depth", type=float, default=None)

    filt = parser.add_argument_group("filter")
    filt.add_argument("--filter-cutoff", type=float, default=None)
    filt.add_argument("--filter-reso", type=float, default=None)
    filt.add_argument("--filter-type", choices=["lp", "bp", "hp"], default=None)

    env = parser.add_argument_group("envelope")
    env.add_argument("--env1-attack", type=float, default=None)
    env.add_argument("--env1-decay", type=float, default=None)
    env.add_argument("--env1-sustain", type=float, default=None)
    env.add_argument("--env1-release", type=float, default=None)
    env.add_argument("--env1-lag", type=float, default=None)

    fx = parser.add_argument_group("fx")
    fx.add_argument("--drive", type=float, default=None)
    fx.add_argument("--delay-wet", type=float, default=None)
    fx.add_argument("--delay-time", type=float, default=None)

    parser.add_argument("--master-vol", type=float, default=None)
    parser.add_argument("--dump-patch", action="store_true", help="패치 JSON 출력 후 종료")

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    from audioman.core.synth_engine import SynthPatch, render_patch
    from audioman.core.instrument import parse_note
    from audioman.core.audio_file import write_audio
    from pathlib import Path

    midi_note = parse_note(args.note)

    # 패치 로드
    if args.patch:
        patch = SynthPatch.from_json(args.patch)
    elif args.text:
        # CLAP text-to-sound
        from audioman.core.synth_engine import text_to_sound
        audio = text_to_sound(
            args.text, note=midi_note,
            duration=args.duration, sample_rate=args.sample_rate,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_audio(out, audio, args.sample_rate)
        output_console.print(f"[green]생성 완료[/green]: {out}")
        return
    else:
        patch = SynthPatch()

    # CLI 오버라이드 적용
    wave_map = {"saw": 0.0, "square": 0.25, "tri": 0.5, "sine": 0.75}
    mode_map = {"va": 0.0, "wt": 0.5, "fm": 1.0}
    ftype_map = {"lp": 0.0, "bp": 0.33, "hp": 0.66}

    if args.osc1_mode is not None: patch.osc1_mode = mode_map[args.osc1_mode]
    if args.osc1_wave is not None: patch.osc1_waveform = wave_map[args.osc1_wave]
    if args.osc1_vol is not None: patch.osc1_volume = args.osc1_vol
    if args.osc2_wave is not None: patch.osc2_waveform = wave_map[args.osc2_wave]
    if args.osc2_vol is not None: patch.osc2_volume = args.osc2_vol
    if args.sub_vol is not None: patch.sub_volume = args.sub_vol
    if args.noise_vol is not None: patch.noise_volume = args.noise_vol
    if args.fm_ratio is not None: patch.fm_ratio = args.fm_ratio
    if args.fm_depth is not None: patch.fm_depth = args.fm_depth
    if args.filter_cutoff is not None: patch.filter_cutoff = args.filter_cutoff
    if args.filter_reso is not None: patch.filter_resonance = args.filter_reso
    if args.filter_type is not None: patch.filter_type = ftype_map[args.filter_type]
    if args.env1_attack is not None: patch.env1_attack = args.env1_attack
    if args.env1_decay is not None: patch.env1_decay = args.env1_decay
    if args.env1_sustain is not None: patch.env1_sustain = args.env1_sustain
    if args.env1_release is not None: patch.env1_release = args.env1_release
    if args.env1_lag is not None: patch.env1_lag = args.env1_lag
    if args.drive is not None: patch.drive = args.drive
    if args.delay_wet is not None: patch.delay_wet = args.delay_wet
    if args.delay_time is not None: patch.delay_time = args.delay_time
    if args.master_vol is not None: patch.master_volume = args.master_vol

    # 덤프 모드
    if args.dump_patch:
        print_json({"command": "synth", "patch": patch.to_dict()})
        return

    # 출력 경로 필수 (dump-patch 제외)
    if not args.output:
        print_error("--output 필수 (--dump-patch 사용 시 불필요)")

    # 렌더링
    audio = render_patch(
        patch, note=midi_note, velocity=args.velocity,
        duration=args.duration, tail=args.tail,
        sample_rate=args.sample_rate,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_audio(out, audio, args.sample_rate)

    if getattr(args, "json", False):
        import numpy as np
        print_json({
            "command": "synth", "output": str(out),
            "note": midi_note, "duration": args.duration + args.tail,
            "peak": float(np.max(np.abs(audio))),
            "rms": float(np.sqrt(np.mean(audio**2))),
        })
    else:
        import numpy as np
        output_console.print(f"[green]생성 완료[/green]: {out}")
        output_console.print(f"  {args.duration + args.tail:.1f}s / {args.sample_rate}Hz / 2ch")
        output_console.print(f"  peak: {np.max(np.abs(audio)):.4f} / rms: {np.sqrt(np.mean(audio**2)):.4f}")
