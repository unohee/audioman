# Created: 2026-03-25
# Purpose: audioman doctor — 플러그인 분석 (PluginDoctor 스타일)

import argparse
import json

from audioman.cli.output import print_error, print_json, print_success, print_info, output_console
from audioman.i18n import _


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help=_("Plugin analysis — frequency response, THD, dynamics, waveshaper, performance"),
    )
    parser.add_argument("--plugin", "-p", required=True, help=_("Plugin name or path"))
    parser.add_argument("--param", action="append", default=[], help=_("Parameter (key=value)"))

    # 분석 모드
    parser.add_argument(
        "--mode", "-m",
        choices=["linear", "thd", "imd", "sweep", "dynamics", "attack-release",
                 "waveshaper", "performance", "all"],
        default="all",
        help=_("Analysis mode (default: all)"),
    )

    # 옵션
    parser.add_argument("--frequency", "-f", type=float, default=1000.0, help=_("Test frequency Hz"))
    parser.add_argument("--level", type=float, default=-6.0, help=_("Input level dB"))
    parser.add_argument("--sample-rate", "-sr", type=int, default=44100)
    parser.add_argument("--fft-size", type=int, default=16384)
    parser.add_argument("--mid-side", action="store_true", help=_("M/S mode"))

    # 비교 모드
    parser.add_argument("--compare", metavar="PLUGIN2", help=_("Compare with second plugin"))
    parser.add_argument("--compare-param", action="append", default=[], help=_("Second plugin parameters"))

    # CLAP 임베딩
    parser.add_argument("--clap", action="store_true", help=_("CLAP embedding profiling (per-parameter saturation fingerprint)"))
    parser.add_argument("--clap-sweep", metavar="PARAM=v1,v2,...", action="append", default=[],
                        help=_("CLAP sweep parameters (e.g. --clap-sweep drive=0,25,50,75,100)"))
    parser.add_argument("--clap-output", metavar="NPY", help=_("CLAP embedding npy save path"))

    # waveshaper v2 옵션
    parser.add_argument("--legacy-waveshaper", action="store_true",
                        help=_("Use legacy waveshaper (single level, single cycle)"))
    parser.add_argument("--ws-levels", metavar="dB", type=float, nargs="+",
                        default=None,
                        help=_("Waveshaper v2 measurement levels in dBFS (default: -24 -18 -12 -6 -3 -1 0)"))
    parser.add_argument("--ws-points", type=int, default=256,
                        help=_("Waveshaper v2 resampling points (default: 256)"))

    # 출력
    parser.add_argument("--output", "-o", metavar="FILE", help=_("Save result JSON file"))

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
                if args.legacy_waveshaper:
                    # 레거시: 단일 레벨, 단일 주기
                    r = pa.measure_waveshaper(plugin_path, params, args.frequency, args.level, args.sample_rate)
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
                        "version": "v1",
                    }
                    if not args.json:
                        output_console.print(f"  Waveshaper (legacy): {len(ws_in)} points, linearity={linearity:.4f}")
                        if linearity > 0.999:
                            output_console.print(f"  [green]선형 플러그인[/green]")
                        else:
                            output_console.print(f"  [yellow]비선형 ({(1-linearity)*100:.2f}% 왜곡)[/yellow]")
                else:
                    # v2: 다중 진폭 레벨 + 복수 주기 평균 + 리샘플링
                    r = pa.measure_waveshaper_v2(
                        plugin_path, params,
                        frequency=args.frequency,
                        sample_rate=args.sample_rate,
                        levels_db=args.ws_levels,
                        n_points=args.ws_points,
                    )
                    # 선형성 체크
                    if r.n_points > 2:
                        linearity = float(np.corrcoef(r.input_values, r.output_values)[0, 1])
                    else:
                        linearity = 1.0
                    results["waveshaper"] = {
                        "input_values": r.input_values.tolist(),
                        "output_values": r.output_values.tolist(),
                        "points": r.n_points,
                        "levels_db": r.levels_db,
                        "input_coverage": r.input_coverage,
                        "is_symmetric": r.is_symmetric,
                        "linearity": round(linearity, 6),
                        "is_linear": linearity > 0.999,
                        "version": "v2",
                    }
                    if not args.json:
                        output_console.print(
                            f"  Waveshaper v2: {r.n_points} points, "
                            f"{len(r.levels_db)} levels, "
                            f"coverage={r.input_coverage:.1%}"
                        )
                        output_console.print(
                            f"  대칭: {'예 (홀수 하모닉)' if r.is_symmetric else '아니오'}, "
                            f"linearity={linearity:.4f}"
                        )
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

    # CLAP 임베딩 프로파일링
    if args.clap or args.clap_sweep:
        if not args.json:
            output_console.print(f"\n[bold cyan]CLAP[/bold cyan] 임베딩 프로파일링...", highlight=False)

        # 스윕 파라미터 파싱
        sweeps = {}
        for sweep_str in args.clap_sweep:
            if '=' not in sweep_str:
                continue
            key, vals = sweep_str.split('=', 1)
            values = []
            for v in vals.split(','):
                v = v.strip()
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(v)  # enum 문자열
            sweeps[key.strip()] = values

        # 스윕이 없으면 기본 drive 스윕
        if not sweeps:
            sweeps = {"drive": [0, 25, 50, 75, 100]}

        try:
            r = pa.measure_clap_profile(
                plugin_path, sweeps, params,
                sample_rate=args.sample_rate,
                test_frequency=args.frequency,
                test_level_db=args.level,
            )
            results["clap"] = {
                "n_settings": r["n_settings"],
                "embedding_dim": r["embedding_dim"],
                "labels": r["labels"],
            }
            if not args.json:
                output_console.print(f"  {r['n_settings']}개 설정 × {r['embedding_dim']}dim 임베딩")
                for label in r["labels"][:5]:
                    output_console.print(f"    {label}")
                if len(r["labels"]) > 5:
                    output_console.print(f"    ... +{len(r['labels'])-5} more")

            # npy 저장
            if args.clap_output:
                np.save(args.clap_output, r["embeddings_npy"])
                if not args.json:
                    print_success(f"CLAP 임베딩 저장: {args.clap_output} ({r['embeddings_npy'].shape})")

                # 라벨 JSON도 같이 저장
                import json as _json
                label_path = args.clap_output.replace('.npy', '_labels.json')
                with open(label_path, 'w') as f:
                    _json.dump({"labels": r["labels"], "params": r["params"]}, f, indent=2, default=str)

        except ImportError as e:
            results["clap"] = {"error": "laion-clap 미설치: pip install laion-clap"}
            if not args.json:
                output_console.print(f"  [yellow]laion-clap 미설치[/yellow]")
        except Exception as e:
            results["clap"] = {"error": str(e)}
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
