# Created: 2026-05-31
# Purpose: audioman stream — DAW 실시간 블록 처리 재현 + 벤치마크/triage.
#
# 서브커맨드:
#   bench    : 블록 크기별 실시간 CPU 부하 측정 (RT factor, xrun, est tracks)
#   triage   : 블록 스트리밍 출력의 클릭/불연속 검출 + offline 대비 null test
#   compare  : 여러 블록 크기의 출력을 서로/오프라인과 비교 (block-size 의존 버그)
#   play     : 실제 오디오 디바이스로 플러그인 통과 신호 재생 (실청 모니터링)

import argparse
from pathlib import Path

from audioman.cli.output import print_error, print_json, print_info, print_table, print_success


DEFAULT_BLOCK_SIZES = [64, 128, 256, 512, 1024]


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "stream",
        help="Reproduce DAW real-time block processing — benchmark & triage plugin clicks/dropouts",
    )
    sub = parser.add_subparsers(dest="stream_command", help="stream subcommands")

    # --- bench ---
    p_bench = sub.add_parser("bench", help="Real-time CPU load across block sizes")
    p_bench.add_argument("input", help="Audio file (or 'sine'/'impulse' for synthetic)")
    p_bench.add_argument("--plugin", "-p", required=True, help="Plugin name/path, or 'chain:...'")
    p_bench.add_argument("--param", action="append", default=[], help="Parameter key=value")
    p_bench.add_argument("--blocks", help="Comma block sizes (default 64,128,256,512,1024)")
    p_bench.add_argument("--sample-rate", "-sr", type=int, default=48000, help="For synthetic input")
    p_bench.add_argument("--duration", type=float, default=5.0, help="Synthetic input seconds")
    p_bench.set_defaults(func=run_bench)

    # --- triage ---
    p_tri = sub.add_parser("triage", help="Detect clicks/discontinuities in block streaming")
    p_tri.add_argument("input", help="Audio file (or 'sine'/'impulse')")
    p_tri.add_argument("--plugin", "-p", required=True, help="Plugin name/path, or 'chain:...'")
    p_tri.add_argument("--param", action="append", default=[], help="Parameter key=value")
    p_tri.add_argument("--block-size", "-b", type=int, default=512)
    p_tri.add_argument("--sample-rate", "-sr", type=int, default=48000)
    p_tri.add_argument("--duration", type=float, default=2.0)
    p_tri.add_argument("--reset-per-block", action="store_true",
                       help="Reset plugin every block (simulate a broken streaming host)")
    p_tri.add_argument("--no-reset-first", action="store_true",
                       help="Start without resetting (carry stale tail → start-of-playback click)")
    p_tri.add_argument("--output", "-o", metavar="FILE", help="Write streamed output WAV")
    p_tri.set_defaults(func=run_triage)

    # --- compare ---
    p_cmp = sub.add_parser("compare", help="Compare outputs across block sizes vs offline render")
    p_cmp.add_argument("input", help="Audio file (or 'sine'/'impulse')")
    p_cmp.add_argument("--plugin", "-p", required=True, help="Plugin name/path, or 'chain:...'")
    p_cmp.add_argument("--param", action="append", default=[], help="Parameter key=value")
    p_cmp.add_argument("--blocks", help="Comma block sizes (default 64,128,256,512,1024)")
    p_cmp.add_argument("--sample-rate", "-sr", type=int, default=48000)
    p_cmp.add_argument("--duration", type=float, default=2.0)
    p_cmp.set_defaults(func=run_compare)

    # --- play ---
    p_play = sub.add_parser("play", help="Play plugin-processed signal through audio device")
    p_play.add_argument("input", help="Audio file (or 'sine'/'impulse')")
    p_play.add_argument("--plugin", "-p", required=True, help="Plugin name/path, or 'chain:...'")
    p_play.add_argument("--param", action="append", default=[], help="Parameter key=value")
    p_play.add_argument("--block-size", "-b", type=int, default=512)
    p_play.add_argument("--sample-rate", "-sr", type=int, default=48000)
    p_play.add_argument("--duration", type=float, default=5.0)
    p_play.add_argument("--reset-per-block", action="store_true")
    p_play.set_defaults(func=run_play)

    parser.set_defaults(func=_no_subcommand)


def _no_subcommand(args: argparse.Namespace) -> None:
    print_error("stream requires a subcommand: bench | triage | compare | play")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_blocks(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_BLOCK_SIZES)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _load_input(args: argparse.Namespace):
    """파일 경로 또는 합성 신호('sine'/'impulse')를 (audio (ch,n), sr)로."""
    import numpy as np

    spec = args.input
    sr = args.sample_rate
    if spec in ("sine", "impulse", "two-tone"):
        from audioman.core import test_signal as ts
        dur = getattr(args, "duration", 2.0)
        if spec == "sine":
            audio = ts.generate_sine(440.0, sr, dur, level_db=-6.0, channels=2)
        elif spec == "impulse":
            audio = ts.generate_impulse(sr, dur, channels=2)
        else:
            audio = ts.generate_two_tone(sample_rate=sr, duration_sec=dur, channels=2)
        return audio.astype(np.float32), sr

    from audioman.core.audio_file import read_audio
    p = Path(spec)
    if not p.exists():
        print_error(f"input not found (and not 'sine'/'impulse'): {spec}")
    audio, file_sr = read_audio(p)
    return audio, file_sr


def _build_process_fn(args: argparse.Namespace, sample_rate: int):
    """--plugin 인자를 pedalboard 또는 VST3 체인으로 해석해 ProcessFn 반환.

    'chain:reverb,delay' 또는 'builtin:reverb' 형식이면 pedalboard 빌트인 사용
    (테스트/데모용). 그 외에는 registry에서 VST3 플러그인 해석.
    """
    from audioman.core.engine import parse_params
    from audioman.core.streaming import make_pedalboard_process_fn, make_wrapper_process_fn

    spec = args.plugin
    params = parse_params(args.param) if args.param else {}

    # builtin pedalboard 이펙트 (의존성 없이 테스트 가능)
    if spec.startswith("builtin:"):
        from pedalboard import Pedalboard, Reverb, Compressor, Delay, Chorus, Gain, Distortion
        registry = {"reverb": Reverb, "compressor": Compressor, "delay": Delay,
                    "chorus": Chorus, "gain": Gain, "distortion": Distortion}
        names = spec.split(":", 1)[1].split(",")
        fx = []
        for nm in names:
            nm = nm.strip().lower()
            if nm not in registry:
                print_error(f"unknown builtin effect: {nm} (have {sorted(registry)})")
            fx.append(registry[nm]())
        board = Pedalboard(fx)
        return make_pedalboard_process_fn(board)

    # VST3 플러그인 (registry 해석)
    from audioman.core.registry import get_registry
    reg = get_registry()
    meta = reg.get(spec)
    if not meta:
        print_error(f"plugin not found: '{spec}' (use 'builtin:reverb' for a dependency-free test)")
    from audioman.plugins.vst3 import VST3PluginWrapper
    wrapper = VST3PluginWrapper(meta.path)
    wrapper.load()
    if params:
        wrapper.set_parameters(params)
    return make_wrapper_process_fn(wrapper)


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

def run_bench(args: argparse.Namespace) -> None:
    from audioman.core.streaming import render_streamed
    from audioman.core.rt_bench import benchmark

    audio, sr = _load_input(args)
    blocks = _parse_blocks(args.blocks)
    fn = _build_process_fn(args, sr)

    reports = []
    for bs in blocks:
        result = render_streamed(audio, sr, fn, block_size=bs, reset_per_block=False)
        reports.append(benchmark(result))

    payload = {
        "input": args.input,
        "plugin": args.plugin,
        "sample_rate": sr,
        "reports": [r.to_dict() for r in reports],
    }

    if getattr(args, "json", False):
        print_json(payload)
        return

    rows = []
    for r in reports:
        rows.append([
            str(r.block_size),
            f"{r.deadline_ms:.2f}",
            f"{r.block_ms_mean:.3f}",
            f"{r.block_ms_max:.3f}",
            f"{r.rt_factor_p50:.4f}",
            f"{r.rt_factor_p99:.4f}",
            f"{r.rt_factor_max:.4f}",
            str(r.xruns),
            str(r.est_max_tracks),
        ])
    print_table(
        f"RT bench — {args.plugin} @ {sr}Hz",
        ["block", "deadline_ms", "proc_ms_avg", "proc_ms_max",
         "rt_p50", "rt_p99", "rt_max", "xruns", "est_tracks"],
        rows,
    )
    worst = max(reports, key=lambda r: r.rt_factor_max)
    if worst.xruns > 0:
        print_info(f"xruns at block={worst.block_size}: {worst.xruns} blocks missed deadline")
    else:
        print_success(f"no xruns; smallest safe est_tracks = {min(r.est_max_tracks for r in reports)}")


# ---------------------------------------------------------------------------
# triage
# ---------------------------------------------------------------------------

def run_triage(args: argparse.Namespace) -> None:
    from audioman.core.streaming import render_offline, render_streamed
    from audioman.core.discontinuity import detect_discontinuities, detect_nonfinite, null_test
    from audioman.core.findings import envelope, Severity

    audio, sr = _load_input(args)
    fn = _build_process_fn(args, sr)
    bs = args.block_size

    offline = render_offline(audio, sr, fn)
    # process_fn은 상태를 들고 있으므로 streamed는 새 fn으로 다시 만들어 오염 방지
    fn2 = _build_process_fn(args, sr)
    streamed = render_streamed(
        audio, sr, fn2,
        block_size=bs,
        reset_per_block=args.reset_per_block,
        reset_first=not args.no_reset_first,
    )

    findings = []
    findings += detect_nonfinite(streamed.audio, sr, file=args.input)
    findings += detect_discontinuities(streamed.audio, sr, block_size=bs, file=args.input)
    findings += null_test(offline.audio, streamed.audio, sr, file=args.input)

    if args.output:
        from audioman.core.audio_file import write_audio
        write_audio(args.output, streamed.audio, sr)

    env = envelope(findings, file=args.input, extra={
        "stream": {
            "block_size": bs,
            "reset_per_block": args.reset_per_block,
            "reset_first": not args.no_reset_first,
            "offline_summary": offline.to_summary(),
            "streamed_summary": streamed.to_summary(),
        }
    })

    if getattr(args, "json", False):
        print_json(env)
        return

    n_crit = env["summary"]["by_severity"]["critical"]
    if not findings:
        print_success(f"clean — no clicks/discontinuities at block_size={bs}")
        return
    rows = []
    for f in findings:
        m = f.measurement
        rows.append([
            f.code.value,
            f.severity.value,
            str(f.where.start_sample),
            f"{f.where.start_sec:.4f}" if f.where.start_sec is not None else "-",
            str(m.get("block_aligned", "-")),
            (str(m.get("peak_jump_db")) if "peak_jump_db" in m
             else str(m.get("max_diff_db", "-"))),
        ])
    print_table(
        f"triage — {args.plugin} block={bs}",
        ["code", "severity", "sample", "sec", "block_aligned", "level_db"],
        rows,
    )
    if n_crit:
        print_info(f"{n_crit} critical finding(s) — block-aligned discontinuities are streaming bugs, "
                   f"not source clicks.")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def run_compare(args: argparse.Namespace) -> None:
    import numpy as np
    from audioman.core.streaming import render_offline, render_streamed

    audio, sr = _load_input(args)
    blocks = _parse_blocks(args.blocks)

    offline = render_offline(audio, sr, _build_process_fn(args, sr))
    ref = offline.audio.mean(axis=0) if offline.audio.ndim == 2 else offline.audio

    def diff_db(a, b):
        n = min(len(a), len(b))
        return float(20 * np.log10(np.max(np.abs(a[:n] - b[:n])) + 1e-30))

    streamed_mono = {}
    rows = []
    for bs in blocks:
        st = render_streamed(audio, sr, _build_process_fn(args, sr), block_size=bs)
        mono = st.audio.mean(axis=0) if st.audio.ndim == 2 else st.audio
        streamed_mono[bs] = mono
        rows.append([str(bs), f"{diff_db(ref, mono):.2f}"])

    # 블록 크기 간 상호 비교 (인접)
    cross = []
    for i in range(len(blocks) - 1):
        a, b = blocks[i], blocks[i + 1]
        cross.append([f"{a} vs {b}", f"{diff_db(streamed_mono[a], streamed_mono[b]):.2f}"])

    payload = {
        "input": args.input, "plugin": args.plugin, "sample_rate": sr,
        "vs_offline_db": {str(b): round(diff_db(ref, streamed_mono[b]), 2) for b in blocks},
        "cross_block_db": {f"{blocks[i]}_vs_{blocks[i+1]}":
                           round(diff_db(streamed_mono[blocks[i]], streamed_mono[blocks[i+1]]), 2)
                           for i in range(len(blocks) - 1)},
    }
    if getattr(args, "json", False):
        print_json(payload)
        return

    print_table(f"streamed vs offline — {args.plugin}",
                ["block", "max_diff_db"], rows)
    print_table("cross block-size diff", ["pair", "max_diff_db"], cross)
    worst = max(payload["vs_offline_db"].values())
    if worst > -60.0:
        print_info(f"block streaming diverges from offline by up to {worst:.1f} dB — "
                   f"this plugin is block-size sensitive.")
    else:
        print_success("all block sizes match offline render (≤ -60 dB) — deterministic.")


# ---------------------------------------------------------------------------
# play
# ---------------------------------------------------------------------------

def run_play(args: argparse.Namespace) -> None:
    import numpy as np
    audio, sr = _load_input(args)
    fn = _build_process_fn(args, sr)
    bs = args.block_size

    try:
        import sounddevice as sd
    except Exception as e:  # pragma: no cover
        print_error(f"sounddevice unavailable: {e}")

    # 블록 단위로 처리하면서 실시간 재생. callback이 마감을 못 맞추면 PortAudio가
    # underflow status를 올린다 — 실제 DAW xrun과 동일한 신호.
    n_ch = audio.shape[0]
    pos = {"i": 0}
    reset_first = {"done": False}
    underflows = {"n": 0}

    def callback(outdata, frames, time_info, status):
        if status:
            underflows["n"] += 1
        i = pos["i"]
        block = audio[:, i:i + frames]
        if block.shape[1] == 0:
            raise sd.CallbackStop()
        reset = args.reset_per_block or (not reset_first["done"])
        reset_first["done"] = True
        out = fn(block, sr, reset)
        out = np.asarray(out, dtype=np.float32)
        m = out.shape[1]
        # stereo 출력 정규화
        if out.shape[0] == 1:
            outdata[:m, 0] = out[0]
            outdata[:m, 1] = out[0]
        else:
            outdata[:m, 0] = out[0]
            outdata[:m, 1] = out[1]
        if m < frames:
            outdata[m:].fill(0.0)
        pos["i"] += frames

    print_info(f"playing {args.input} through {args.plugin} (block={bs}, sr={sr}) — Ctrl-C to stop")
    import threading
    done = threading.Event()
    stream = sd.OutputStream(samplerate=sr, channels=2, dtype="float32",
                             blocksize=bs, callback=callback,
                             finished_callback=done.set)
    try:
        with stream:
            done.wait(timeout=args.duration + audio.shape[1] / sr + 1.0)
    except KeyboardInterrupt:
        pass
    if underflows["n"]:
        print_info(f"{underflows['n']} PortAudio underflow(s) — real xruns at block={bs}")
    else:
        print_success("playback complete, no underflows")
