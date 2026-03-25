# Created: 2026-03-25
# Purpose: audioman doctor — 플러그인 분석 (PluginDoctor 스타일)

import argparse
import json

from audioman.cli.output import print_error, print_json, print_success, print_info, output_console


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="플러그인 분석 — frequency response, THD, dynamics, waveshaper, performance",
    )
    parser.add_argument("--plugin", "-p", required=True, help="플러그인 이름 또는 경로")
    parser.add_argument("--param", action="append", default=[], help="파라미터 (key=value)")

    # 분석 모드
    parser.add_argument(
        "--mode", "-m",
        choices=["linear", "thd", "imd", "sweep", "dynamics", "attack-release",
                 "waveshaper", "performance", "all"],
        default="all",
        help="분석 모드 (기본: all)",
    )

    # 옵션
    parser.add_argument("--frequency", "-f", type=float, default=1000.0, help="테스트 주파수 Hz")
    parser.add_argument("--level", type=float, default=-6.0, help="입력 레벨 dB")
    parser.add_argument("--sample-rate", "-sr", type=int, default=44100)
    parser.add_argument("--fft-size", type=int, default=16384)
    parser.add_argument("--mid-side", action="store_true", help="M/S 모드")

    # 비교 모드
    parser.add_argument("--compare", metavar="PLUGIN2", help="2번째 플러그인과 비교")
    parser.add_argument("--compare-param", action="append", default=[], help="2번째 플러그인 파라미터")

    # 출력
    parser.add_argument("--output", "-o", metavar="FILE", help="결과 JSON 파일 저장")

    parser.set_defaults(func=run)


def _resolve_plugin(plugin_arg: str) -> str:
    from pathlib import Path
    p = Path(plugin_arg)
    if p.exists() and p.suffix.lower() in (".vst3", ".component"):
        return str(p)
    from audioman.core.registry import get_registry
    registry = get_registry()
    meta = registry.get(plugin_arg)
    if meta:
        return meta.path
    print_error(f"플러그인 없음: '{plugin_arg}'")


def run(args: argparse.Namespace) -> None:
    from audioman.core.engine import parse_params
    from audioman.core import plugin_analysis as pa
    from dataclasses import asdict

    plugin_path = _resolve_plugin(args.plugin)
    params = parse_params(args.param) if args.param else None

    results = {"command": "doctor", "plugin": plugin_path, "mode": args.mode}
    modes = [args.mode] if args.mode != "all" else [
        "linear", "thd", "imd", "sweep", "dynamics", "attack-release",
        "waveshaper", "performance",
    ]

    for mode in modes:
        if not args.json:
            output_console.print(f"\n[bold cyan]{mode}[/bold cyan] 분석 중...", highlight=False)

        try:
            if mode == "linear":
                r = pa.measure_linear(plugin_path, params, args.sample_rate, args.fft_size,
                                      level_db=args.level)
                results["linear"] = {
                    "method": r.method, "fft_size": r.fft_size,
                    "freq_count": len(r.frequencies),
                    "magnitude_range_db": [round(min(r.magnitude_db), 1), round(max(r.magnitude_db), 1)],
                }
                if not args.json:
                    output_console.print(f"  주파수 응답: {len(r.frequencies)} bins, "
                        f"range {min(r.magnitude_db):.1f} ~ {max(r.magnitude_db):.1f} dB")

            elif mode == "thd":
                r = pa.measure_thd(plugin_path, params, args.frequency, args.level,
                                   args.sample_rate, args.fft_size)
                results["thd"] = asdict(r)
                if not args.json:
                    output_console.print(f"  THD: {r.thd_percent:.4f}%  THD+N: {r.thd_plus_n_percent:.4f}%")
                    output_console.print(f"  Fundamental: {r.fundamental_freq}Hz @ {r.fundamental_db:.1f}dB")
                    for h in r.harmonics[:5]:
                        output_console.print(f"    {h['order']}차: {h['freq']}Hz @ {h['db']:.1f}dB")

            elif mode == "imd":
                r = pa.measure_imd(plugin_path, params, sample_rate=args.sample_rate, fft_size=args.fft_size)
                results["imd"] = {"imd_percent": r.imd_percent, "harmonics": r.harmonics[:10]}
                if not args.json:
                    output_console.print(f"  IMD: {r.imd_percent:.4f}%")

            elif mode == "sweep":
                r = pa.measure_sweep(plugin_path, params, sample_rate=args.sample_rate,
                                     level_db=args.level)
                results["sweep"] = {
                    "freq_count": len(r.frequencies),
                    "thd_range": [round(min(r.thd_per_freq) if r.thd_per_freq else 0, 4),
                                  round(max(r.thd_per_freq) if r.thd_per_freq else 0, 4)],
                    "gain_range": [round(min(r.gain_per_freq) if r.gain_per_freq else 0, 1),
                                   round(max(r.gain_per_freq) if r.gain_per_freq else 0, 1)],
                }
                if not args.json:
                    output_console.print(f"  스윕: {len(r.frequencies)} points")
                    if r.thd_per_freq:
                        output_console.print(f"  THD range: {min(r.thd_per_freq):.4f}% ~ {max(r.thd_per_freq):.4f}%")

            elif mode == "dynamics":
                r = pa.measure_dynamics_ramp(plugin_path, params, args.frequency, args.sample_rate)
                results["dynamics"] = {
                    "input_range": [r.input_levels_db[0], r.input_levels_db[-1]],
                    "output_range": [r.output_levels_db[0], r.output_levels_db[-1]],
                    "max_gain_reduction": round(min(r.gain_reduction_db), 2),
                    "io_curve": list(zip(r.input_levels_db, r.output_levels_db)),
                }
                if not args.json:
                    output_console.print(f"  I/O: {r.input_levels_db[0]}~{r.input_levels_db[-1]} dB")
                    output_console.print(f"  Max gain reduction: {min(r.gain_reduction_db):.1f} dB")

            elif mode == "attack-release":
                r = pa.measure_dynamics_ar(plugin_path, params, args.frequency, args.sample_rate)
                results["attack_release"] = {
                    "input_levels": r.input_levels_db,
                    "envelope_points": len(r.output_levels_db),
                }
                if not args.json:
                    output_console.print(f"  Envelope: {len(r.output_levels_db)} points")

            elif mode == "waveshaper":
                r = pa.measure_waveshaper(plugin_path, params, args.frequency, args.level, args.sample_rate)
                # 선형성 체크
                ws_in = r.waveshaper_input
                ws_out = r.waveshaper_output
                if len(ws_in) > 2:
                    linearity = float(np.corrcoef(ws_in, ws_out)[0, 1])
                else:
                    linearity = 1.0
                results["waveshaper"] = {
                    "points": len(ws_in),
                    "linearity": round(linearity, 6),
                    "is_linear": linearity > 0.999,
                }
                if not args.json:
                    output_console.print(f"  Waveshaper: {len(ws_in)} points, linearity={linearity:.4f}")
                    if linearity > 0.999:
                        output_console.print(f"  [green]선형 플러그인[/green]")
                    else:
                        output_console.print(f"  [yellow]비선형 ({(1-linearity)*100:.2f}% 왜곡)[/yellow]")

            elif mode == "performance":
                r = pa.measure_performance(plugin_path, params, args.sample_rate)
                results["performance"] = {
                    "buffer_sizes": r.buffer_sizes,
                    "process_times_ms": r.process_times_ms,
                    "realtime_ratio": r.realtime_ratio,
                }
                if not args.json:
                    output_console.print(f"  {'Buffer':>8s} {'Time(ms)':>10s} {'RT ratio':>10s}")
                    for bs, t, rt in zip(r.buffer_sizes, r.process_times_ms, r.realtime_ratio):
                        output_console.print(f"  {bs:>8d} {t:>10.3f} {rt:>10.1f}x")

        except Exception as e:
            results[mode] = {"error": str(e)}
            if not args.json:
                output_console.print(f"  [red]에러: {e}[/red]")

    # 비교 모드
    if args.compare:
        plugin2_path = _resolve_plugin(args.compare)
        params2 = parse_params(args.compare_param) if args.compare_param else None
        if not args.json:
            output_console.print(f"\n[bold]비교: {args.plugin} vs {args.compare}[/bold]")
        try:
            diff = pa.compare_linear(plugin_path, plugin2_path, params, params2, args.sample_rate)
            max_diff = max(abs(d) for d in diff["diff_magnitude_db"])
            results["compare"] = {"max_diff_db": round(max_diff, 2)}
            if not args.json:
                output_console.print(f"  최대 차이: {max_diff:.2f} dB")
        except Exception as e:
            results["compare"] = {"error": str(e)}

    # 출력
    if args.json:
        print_json(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        if not args.json:
            print_success(f"결과 저장: {args.output}")

    if not args.json and not args.output:
        print_success("분석 완료")


# numpy import (waveshaper linearity 계산용)
import numpy as np
