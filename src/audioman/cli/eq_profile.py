# Created: 2026-03-31
# Purpose: audioman eq-profile — EQ 플러그인 주파수/위상/비선형성 프로파일링

import argparse
import json

import numpy as np

from audioman.cli.output import print_error, print_json, print_success, output_console
from audioman.i18n import _


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "eq-profile",
        help=_("EQ plugin profiling — frequency response, phase, group delay, nonlinearity"),
    )
    parser.add_argument("--plugin", "-p", required=True, help=_("Plugin name or path"))
    parser.add_argument("--param", action="append", default=[], help=_("EQ parameter (key=value)"))
    parser.add_argument("--bypass-param", action="append", default=[],
                        help=_("Bypass state parameter (key=value)"))

    # 분석 모드
    parser.add_argument(
        "--mode", "-m",
        choices=["response", "sweep", "nonlinear", "all"],
        default="all",
        help=_("Analysis mode (default: all)"),
    )

    # 스윕 파라미터
    parser.add_argument(
        "--sweep-param", action="append", default=[],
        metavar="NAME=v1,v2,...",
        help=_("Parameter sweep (e.g. --sweep-param band1_gain=-12,-6,0,6,12)"),
    )
    parser.add_argument(
        "--sweep-fixed", action="append", default=[],
        metavar="KEY=VALUE",
        help=_("Fixed parameters during sweep (e.g. --sweep-fixed band1_freq=1000)"),
    )

    # 비선형성 레벨
    parser.add_argument(
        "--levels", type=float, nargs="+",
        default=None,
        help=_("Input levels for nonlinearity test (dBFS, default: -36 -24 -18 -12 -6 -3 0)"),
    )

    # 공통 옵션
    parser.add_argument("--sample-rate", "-sr", type=int, default=44100)
    parser.add_argument("--fft-size", type=int, default=32768)
    parser.add_argument("--level", type=float, default=-12.0, help=_("Input level dB"))
    parser.add_argument("--sweep-duration", type=float, default=6.0,
                        help=_("Log sweep duration in seconds"))

    # 출력
    parser.add_argument("--output", "-o", metavar="FILE", help=_("Save result JSON file"))
    parser.add_argument("--save-npy", metavar="DIR",
                        help=_("Save frequency/phase/delay curves as .npy files"))

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


def _parse_params(param_list: list[str]) -> dict | None:
    if not param_list:
        return None
    from audioman.core.engine import parse_params
    return parse_params(param_list)


def _parse_sweep_config(sweep_params: list[str], fixed_params: list[str]) -> dict:
    """--sweep-param와 --sweep-fixed를 sweep_config dict로 변환"""
    fixed = {}
    for fp in fixed_params:
        if "=" not in fp:
            continue
        k, v = fp.split("=", 1)
        try:
            fixed[k.strip()] = float(v.strip())
        except ValueError:
            fixed[k.strip()] = v.strip()

    config = {}
    for sp in sweep_params:
        if "=" not in sp:
            continue
        name, vals_str = sp.split("=", 1)
        name = name.strip()
        values = []
        for v in vals_str.split(","):
            v = v.strip()
            try:
                values.append(float(v))
            except ValueError:
                values.append(v)

        config[f"{name}_sweep"] = {
            "param": name,
            "values": values,
            "fixed": dict(fixed),
        }

    return config


def run(args: argparse.Namespace) -> None:
    from audioman.core import plugin_analysis as pa
    from dataclasses import asdict

    plugin_path = _resolve_plugin(args.plugin)
    params = _parse_params(args.param)
    bypass_params = _parse_params(args.bypass_param)

    results = {
        "command": "eq-profile",
        "plugin": plugin_path,
        "plugin_type": "eq",
        "mode": args.mode,
    }

    modes = [args.mode] if args.mode != "all" else ["response", "sweep", "nonlinear"]

    all_eq_results = []

    for mode in modes:
        if not args.json:
            output_console.print(f"\n[bold cyan]{mode}[/bold cyan] 분석 중...", highlight=False)

        try:
            if mode == "response":
                r = pa.measure_eq_response(
                    plugin_path, params, bypass_params,
                    sample_rate=args.sample_rate,
                    fft_size=args.fft_size,
                    sweep_duration=args.sweep_duration,
                    level_db=args.level,
                )
                all_eq_results.append(r)

                # 주요 통계
                mag = np.array(r.magnitude_db)
                # 20Hz~20kHz 범위만
                freq_arr = np.array(r.frequencies)
                mask = (freq_arr >= 20) & (freq_arr <= 20000)
                mag_range = mag[mask]

                results["response"] = {
                    "params": r.params,
                    "magnitude_range_db": [round(float(np.min(mag_range)), 2),
                                           round(float(np.max(mag_range)), 2)],
                    "is_minimum_phase": r.is_minimum_phase,
                    "thd_at_1k": r.thd_at_1k,
                    "fft_size": r.fft_size,
                }

                if not args.json:
                    output_console.print(
                        f"  주파수 응답: {float(np.min(mag_range)):.1f} ~ "
                        f"{float(np.max(mag_range)):.1f} dB"
                    )
                    output_console.print(
                        f"  최소위상: {'예' if r.is_minimum_phase else '아니오'}, "
                        f"THD@1kHz: {r.thd_at_1k:.4f}%"
                    )

            elif mode == "sweep":
                sweep_config = _parse_sweep_config(args.sweep_param, args.sweep_fixed)
                if not sweep_config:
                    results["sweep"] = {"error": "스윕 파라미터 미지정 (--sweep-param 필요)"}
                    if not args.json:
                        output_console.print("  [yellow]--sweep-param 미지정[/yellow]")
                    continue

                eq_results = pa.measure_eq_parameter_sweep(
                    plugin_path, sweep_config, bypass_params,
                    sample_rate=args.sample_rate,
                    fft_size=args.fft_size,
                    level_db=args.level,
                )
                all_eq_results.extend(eq_results)

                sweep_summary = []
                for r in eq_results:
                    mag = np.array(r.magnitude_db)
                    freq_arr = np.array(r.frequencies)
                    mask = (freq_arr >= 20) & (freq_arr <= 20000)
                    mag_range = mag[mask]
                    sweep_summary.append({
                        "params": r.params,
                        "peak_db": round(float(np.max(mag_range)), 2),
                        "min_db": round(float(np.min(mag_range)), 2),
                        "is_minimum_phase": r.is_minimum_phase,
                        "thd_at_1k": r.thd_at_1k,
                    })

                results["sweep"] = {
                    "n_settings": len(eq_results),
                    "measurements": sweep_summary,
                }

                if not args.json:
                    output_console.print(f"  {len(eq_results)}개 설정 측정 완료")
                    for s in sweep_summary[:5]:
                        p_str = ", ".join(f"{k}={v}" for k, v in s["params"].items())
                        output_console.print(
                            f"    {p_str}: {s['min_db']:.1f}~{s['peak_db']:.1f} dB"
                        )
                    if len(sweep_summary) > 5:
                        output_console.print(f"    ... +{len(sweep_summary)-5} more")

            elif mode == "nonlinear":
                eq_results = pa.measure_eq_nonlinearity(
                    plugin_path, params, bypass_params,
                    levels_db=args.levels,
                    sample_rate=args.sample_rate,
                    fft_size=args.fft_size,
                )
                all_eq_results.extend(eq_results)

                # 레벨 간 편차 계산
                thd_values = [r.thd_at_1k for r in eq_results]
                levels_tested = [r.params.get("_input_level_db", 0) for r in eq_results]

                is_level_dependent = False
                max_deviation = 0.0
                if len(eq_results) >= 2:
                    ref = np.array(eq_results[0].magnitude_db)
                    for r in eq_results[1:]:
                        diff = np.max(np.abs(np.array(r.magnitude_db) - ref))
                        max_deviation = max(max_deviation, diff)
                    is_level_dependent = max_deviation > 0.5

                results["nonlinear"] = {
                    "n_levels": len(eq_results),
                    "levels_db": levels_tested,
                    "thd_per_level": [round(t, 4) for t in thd_values],
                    "is_level_dependent": is_level_dependent,
                    "max_response_deviation_db": round(max_deviation, 2),
                }

                if not args.json:
                    output_console.print(
                        f"  {len(eq_results)}개 레벨 측정, "
                        f"최대 편차: {max_deviation:.2f} dB"
                    )
                    label = "[yellow]비선형 (레벨 의존)[/yellow]" if is_level_dependent \
                        else "[green]선형 (레벨 무관)[/green]"
                    output_console.print(f"  {label}")
                    for lv, thd in zip(levels_tested, thd_values):
                        output_console.print(f"    {lv:>6.0f} dBFS: THD={thd:.4f}%")

        except Exception as e:
            results[mode] = {"error": str(e)}
            if not args.json:
                output_console.print(f"  [red]에러: {e}[/red]")

    # .npy 저장
    if args.save_npy and all_eq_results:
        from pathlib import Path
        npy_dir = Path(args.save_npy)
        npy_dir.mkdir(parents=True, exist_ok=True)

        freq_curves = np.array([r.magnitude_db for r in all_eq_results], dtype=np.float32)
        phase_curves = np.array([r.phase_deg for r in all_eq_results], dtype=np.float32)
        delay_curves = np.array([r.group_delay_ms for r in all_eq_results], dtype=np.float32)
        freq_axis = np.array(all_eq_results[0].frequencies, dtype=np.float32)

        np.save(str(npy_dir / "frequency_response_curves.npy"), freq_curves)
        np.save(str(npy_dir / "phase_response_curves.npy"), phase_curves)
        np.save(str(npy_dir / "group_delay_curves.npy"), delay_curves)
        np.save(str(npy_dir / "frequency_axis.npy"), freq_axis)

        if not args.json:
            print_success(
                f"곡선 저장: {npy_dir}/ "
                f"({freq_curves.shape[0]} settings × {freq_curves.shape[1]} bins)"
            )

        # 파라미터 라벨 저장
        labels = [r.params for r in all_eq_results]
        label_path = npy_dir / "settings_labels.json"
        with open(label_path, "w") as f:
            json.dump(labels, f, indent=2, ensure_ascii=False, default=str)

    # JSON 출력
    if args.json:
        print_json(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        if not args.json:
            print_success(f"결과 저장: {args.output}")

    if not args.json and not args.output:
        print_success("EQ 프로파일링 완료")
