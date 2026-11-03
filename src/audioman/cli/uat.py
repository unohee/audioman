# Created: 2026-11-04
# Purpose: audioman uat — plugin/host acceptance gates (compare, smoke, pdc) as report.v2.
#
#   compare  real-time take vs offline render vs source (+ optional bypass take)
#   smoke    render a probe through a plugin at several rates, or judge a host's render
#   pdc      measured (steady-state cross-correlation) vs reported latency
#
# Every subcommand prints a report.v2 block; the exit status is 1 when the verdict
# is FAIL, so scripts and CI can gate on it directly.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from audioman.cli.output import print_error, print_json, print_success, print_table
from audioman.core.digest import digest_report
from audioman.core.findings import json_envelope, schema_uri
from audioman.core.report_v2 import Gate, Measurement, Report, require

_BUILTINS = ("reverb", "compressor", "delay", "chorus", "gain", "distortion")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("uat", help="Plugin/host acceptance gates (compare, smoke, pdc) as report.v2")
    sub = parser.add_subparsers(dest="uat_command")
    parser.set_defaults(func=lambda args: print_error("uat requires a subcommand: compare | smoke | pdc"))

    p_cmp = sub.add_parser("compare", help="Real-time take vs offline render vs source")
    p_cmp.add_argument("--source", required=True, help="Dry input file")
    p_cmp.add_argument("--offline", required=True, help="Host offline render of the plugin")
    p_cmp.add_argument("--realtime", required=True, help="Host real-time recording of the plugin output")
    p_cmp.add_argument("--bypassed", help="Real-time recording with the plugin bypassed (adds bypass_equals_source)")
    p_cmp.add_argument("--render-seconds", type=float, help="Offline render wall time (adds faster_than_real_time)")
    p_cmp.add_argument("--match-dbfs", type=float, default=-90.0,
                       help="Residual peak counted as identical (default: -90 dBFS)")
    p_cmp.add_argument("--transparent", action="store_true",
                       help="The plugin is expected to leave the signal unchanged (drops the 'processed' gate)")
    p_cmp.set_defaults(func=run_compare)

    p_smoke = sub.add_parser("smoke", help="Does the plugin load, process and stay finite and audible?")
    p_smoke.add_argument("--plugin", help="Plugin name/path, or builtin:<effect>[,<effect>]")
    p_smoke.add_argument("--param", nargs="*", help="Parameters key=value")
    p_smoke.add_argument("--rates", default="44100,48000", help="Comma-separated sample rates (default: 44100,48000)")
    p_smoke.add_argument("--duration", type=float, default=2.0, help="Probe length in seconds (default: 2)")
    p_smoke.add_argument("--source", help="With --check-output: the probe the host processed")
    p_smoke.add_argument("--check-output", help="Judge a render made by another host instead of rendering")
    p_smoke.set_defaults(func=run_smoke)

    p_pdc = sub.add_parser("pdc", help="Measured vs reported latency (steady-state cross-correlation)")
    p_pdc.add_argument("--reference", help="Dry input file")
    p_pdc.add_argument("--processed", help="Plugin output file")
    p_pdc.add_argument("--reported", type=int, help="Reported latency in samples (files mode)")
    p_pdc.add_argument("--plugin", help="Render broadband noise through this plugin and read its reported latency")
    p_pdc.add_argument("--param", nargs="*", help="Parameters key=value")
    p_pdc.add_argument("--sample-rate", type=int, default=48000, help="Probe rate in plugin mode (default: 48000)")
    p_pdc.add_argument("--tolerance-ms", type=float, default=2.0, help="Allowed |measured - reported| (default: 2 ms)")
    p_pdc.add_argument("--min-ncc", type=float, default=0.70, help="Minimum correlation (default: 0.70)")
    p_pdc.set_defaults(func=run_pdc)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read(path: str) -> tuple[np.ndarray, int]:
    from audioman.core.audio_file import read_audio

    try:
        return read_audio(Path(path))
    except (OSError, RuntimeError, ValueError) as exc:
        print_error(f"cannot read {path}: {exc}")  # exits with status 1


class _Plugin:
    """A processor resolved from --plugin: pedalboard builtins or a VST3/AU wrapper."""

    def __init__(self, spec: str, params: dict):
        self.n_parameters: int | None = None
        self._wrapper = None
        if spec.startswith("builtin:"):
            import pedalboard

            classes = {name: getattr(pedalboard, name.capitalize()) for name in _BUILTINS}
            names = [n.strip().lower() for n in spec.split(":", 1)[1].split(",")]
            unknown = [n for n in names if n not in classes]
            if unknown:
                print_error(f"unknown builtin effect(s): {unknown} (have {list(_BUILTINS)})")
            self._board = pedalboard.Pedalboard([classes[n]() for n in names])
            return
        from audioman.core.registry import get_registry
        from audioman.plugins.vst3 import VST3PluginWrapper

        meta = get_registry().get(spec)
        if not meta:
            print_error(f"plugin not found: '{spec}' (use builtin:gain for a dependency-free run)")
        self._wrapper = VST3PluginWrapper(meta.path)
        self._wrapper.load()
        if params:
            self._wrapper.set_parameters(params)
        self.n_parameters = len(self._wrapper.get_parameters())

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self._wrapper is None:
            return self._board(audio, sample_rate)
        return self._wrapper.process(audio, sample_rate)

    def reported_latency(self) -> int:
        if self._wrapper is None:
            return 0
        from audioman.core.latency import _get_reported_latency

        return _get_reported_latency(self._wrapper)


def _probe(sample_rate: int, duration: float) -> np.ndarray:
    """440 Hz tone plus seeded broadband noise, stereo: tonal and broadband content."""
    rng = np.random.default_rng(1729)
    t = np.arange(int(sample_rate * duration)) / sample_rate
    mono = 0.25 * np.sin(2 * np.pi * 440.0 * t) + 0.05 * rng.standard_normal(len(t))
    return np.stack([mono, mono]).astype(np.float32)


def _emit(args: argparse.Namespace, command: str, report: Report, inputs: dict) -> None:
    data = report.to_dict()
    payload = json_envelope(command, {"digest": digest_report(data), "inputs": inputs, "report": data},
                            schema=schema_uri("uat"))
    if getattr(args, "json", False):
        print_json(payload)
    else:
        rows = [[g["name"], g["status"], str(g.get("threshold", "")), g.get("reason", "")] for g in data["gates"]]
        print_table(f"{command} — {data['verdict']}", ["gate", "status", "threshold", "reason"], rows)
        if data["verdict"] != "FAIL":
            print_success(payload["digest"])
    if data["verdict"] == "FAIL":
        sys.exit(1)


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def run_compare(args: argparse.Namespace) -> None:
    from audioman.core.uat_gates import TakeSet, compare_takes

    source, sr = _read(args.source)
    takes = {"offline": args.offline, "realtime": args.realtime, "bypassed": args.bypassed}
    loaded = {}
    for name, path in takes.items():
        if path is None:
            continue
        audio, rate = _read(path)
        if rate != sr:
            print_error(f"--{name} is {rate} Hz but --source is {sr} Hz; render both at one rate")
        loaded[name] = audio
    report = compare_takes(
        TakeSet(source, loaded["offline"], loaded["realtime"], sr, loaded.get("bypassed"), args.render_seconds),
        match_dbfs=args.match_dbfs,
        expect_processing=not args.transparent,
    )
    report.subject["file"] = str(Path(args.source).resolve())
    _emit(args, "uat compare", report, {k: v for k, v in vars(args).items()
                                        if k in ("source", "offline", "realtime", "bypassed", "render_seconds")})


def run_smoke(args: argparse.Namespace) -> None:
    from audioman.core.engine import parse_params
    from audioman.core.uat_gates import smoke_check

    if args.check_output:
        if not args.source:
            print_error("--check-output needs --source (the probe the host processed)")
        source, sr = _read(args.source)
        output, out_sr = _read(args.check_output)
        gates = smoke_check(source, output, sr)
        if out_sr != sr:
            gates.append(Gate("sample_rate_matches", "FAIL", threshold=sr, reason=f"output is {out_sr} Hz"))
        subject = {"file": str(Path(args.check_output).resolve()), "sample_rate": sr,
                   "channels": int(np.asarray(output).shape[0]) if np.asarray(output).ndim == 2 else 1,
                   "duration_sec": round(np.asarray(output).shape[-1] / out_sr, 6)}
        report = Report("uat smoke", subject, [], [], require(gates, ["shape_matches", "finite", "second_half_audible"]))
        _emit(args, "uat smoke", report, {"source": args.source, "check_output": args.check_output})
        return

    if not args.plugin:
        print_error("uat smoke needs --plugin, or --check-output with --source")
    rates = [int(r) for r in str(args.rates).split(",") if r.strip()]
    params = parse_params(args.param) if args.param else {}
    gates: list[Gate] = []
    ms: list[Measurement] = []
    required: list[str] = []
    for sr in rates:
        plugin = _Plugin(args.plugin, params)  # fresh instance per rate, as a host would
        probe = _probe(sr, args.duration)
        try:
            output = np.asarray(plugin.process(probe, sr))
        except Exception as exc:  # a plugin that raises has failed the smoke test, not the tool
            gates.append(Gate(f"{sr}.processed", "FAIL", reason=str(exc)))
            required += [f"{sr}.shape_matches", f"{sr}.finite", f"{sr}.second_half_audible"]
            continue
        rate_gates = smoke_check(probe, output, sr, n_parameters=plugin.n_parameters, label=str(sr))
        gates += rate_gates
        required += [g.name for g in rate_gates]
    subject = {"file": args.plugin, "sample_rate": None, "channels": 2, "duration_sec": args.duration}
    report = Report("uat smoke", subject, ms, [], require(gates, required))
    _emit(args, "uat smoke", report, {"plugin": args.plugin, "rates": rates, "duration": args.duration})


def run_pdc(args: argparse.Namespace) -> None:
    from audioman.core.engine import parse_params
    from audioman.core.pdc import estimate_lag, judge_pdc

    if args.plugin:
        sr = args.sample_rate
        rng = np.random.default_rng(7)
        reference = (0.1 * rng.standard_normal((2, sr * 3))).astype(np.float32)
        plugin = _Plugin(args.plugin, parse_params(args.param) if args.param else {})
        processed = np.asarray(plugin.process(reference, sr))
        reported = plugin.reported_latency() if args.reported is None else args.reported
        inputs = {"plugin": args.plugin, "sample_rate": sr}
        subject_file = args.plugin
    else:
        if not (args.reference and args.processed) or args.reported is None:
            print_error("uat pdc needs --plugin, or --reference, --processed and --reported")
        reference, sr = _read(args.reference)
        processed, rate = _read(args.processed)
        if rate != sr:
            print_error(f"--processed is {rate} Hz but --reference is {sr} Hz")
        reported = args.reported
        inputs = {"reference": args.reference, "processed": args.processed, "reported": reported}
        subject_file = str(Path(args.processed).resolve())

    estimate = estimate_lag(reference, processed, sr)
    verdict = judge_pdc(estimate, reported, sr, tolerance_ms=args.tolerance_ms, min_ncc=args.min_ncc)
    one = 1.0 / sr
    ms = [
        Measurement("measured_latency_samples", verdict.measured_samples, "samples", 0.0, "exact", "uat.pdc"),
        Measurement("reported_latency_samples", verdict.reported_samples, "samples", 0.0, "exact", "uat.pdc"),
        Measurement("residual_ms", verdict.residual_ms, "ms", one * 1000.0, "resolution", "uat.pdc"),
        Measurement("ncc", verdict.ncc, "ratio", 0.5e-6, "rounding", "uat.pdc"),
    ]
    gate = Gate("latency_matches_reported", verdict.status, measurement="residual_ms",
                threshold=f"|residual| <= {args.tolerance_ms} ms, ncc >= {args.min_ncc}, not at search edge",
                reason=verdict.reason)
    subject = {"file": subject_file, "sample_rate": sr, "channels": None,
               "duration_sec": round(np.asarray(reference).shape[-1] / sr, 6)}
    report = Report("uat pdc", subject, ms, [], require([gate], ["latency_matches_reported"]))
    _emit(args, "uat pdc", report, inputs)
