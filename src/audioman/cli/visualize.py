# Created: 2026-03-22
# Purpose: audioman visualize 서브커맨드 — Vamp 플러그인 + 내장 분석 → SVL export
# Dependencies: core.svl, core.vamp_host, core.analysis

import argparse
from pathlib import Path

import numpy as np

from audioman.cli.output import console, print_error, print_success, print_info, output_console
from audioman.core.audio_file import read_audio
from audioman.core.svl import (
    write_time_instants,
    write_time_values,
    write_notes,
    write_dense3d,
)
from audioman.i18n import _


# 내장 분석 타입과 설명
BUILTIN_TYPES = {
    "spectral-centroid": "스펙트럼 무게중심 주파수 (Hz)",
    "spectral-entropy": "스펙트럼 엔트로피 (bits)",
    "rms": "RMS 에너지",
    "peak": "피크 진폭",
    "zcr": "제로 크로싱 레이트",
    "spectrogram": "STFT 파워 스펙트로그램",
}


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "visualize",
        help=_("Vamp plugin or built-in analysis -> Sonic Visualiser SVL file"),
    )
    parser.add_argument("input", help=_("Input audio file"))

    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--plugin", "-p",
        help=_("Vamp plugin ID (e.g. qm-vamp-plugins:qm-chromagram)"),
    )
    source.add_argument(
        "--builtin", "-b",
        choices=list(BUILTIN_TYPES.keys()),
        help=f"{_('Built-in analysis type')}: {', '.join(BUILTIN_TYPES.keys())}",
    )

    parser.add_argument("-o", "--output", help=_("Output SVL file path (default: auto)"))
    parser.add_argument("--output-name", help=_("Vamp plugin output name (for multiple outputs)"))
    parser.add_argument("--frame-size", type=int, default=2048, help=_("FFT frame size (default: 2048)"))
    parser.add_argument("--hop", type=int, default=512, help=_("Hop size (default: 512)"))
    parser.add_argument("--list-plugins", action="store_true", help=_("List installed Vamp plugins"))
    parser.add_argument("--plugin-info", help=_("Query plugin output info"))
    parser.add_argument("--open", action="store_true", help=_("Open in Sonic Visualiser after creation"))

    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    if args.list_plugins:
        _list_plugins()
        return

    if args.plugin_info:
        _plugin_info(args.plugin_info)
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"파일을 찾을 수 없습니다: {input_path}")

    if args.plugin:
        _run_vamp(args, input_path)
    elif args.builtin:
        _run_builtin(args, input_path)
    else:
        # 기본: spectrogram
        args.builtin = "spectrogram"
        _run_builtin(args, input_path)


def _list_plugins() -> None:
    from audioman.core.vamp_host import list_plugins

    plugins = list_plugins()
    if not plugins:
        print_error("설치된 Vamp 플러그인이 없습니다.")

    output_console.print(f"\n[bold]설치된 Vamp 플러그인 ({len(plugins)}개)[/bold]\n")
    for p in plugins:
        output_console.print(f"  {p}")
    output_console.print()


def _plugin_info(plugin_id: str) -> None:
    from audioman.core.vamp_host import get_plugin_outputs

    outputs = get_plugin_outputs(plugin_id)
    output_console.print(f"\n[bold]{plugin_id}[/bold]\n")
    for name, info in outputs.items():
        output_console.print(f"  {name}: {info}")
    output_console.print()


def _resolve_output_path(args: argparse.Namespace, input_path: Path, suffix: str) -> Path:
    """출력 경로 결정"""
    if args.output:
        return Path(args.output)
    stem = input_path.stem
    return input_path.parent / f"{stem}_{suffix}.svl"


def _run_vamp(args: argparse.Namespace, input_path: Path) -> None:
    from audioman.core.vamp_host import (
        run_plugin,
        result_to_frames_and_values,
        result_to_instants,
        result_to_matrix,
    )

    audio, sr = read_audio(input_path)
    console.print(f"[dim]Vamp 플러그인 실행: {args.plugin}[/dim]")

    result = run_plugin(
        audio, sr, args.plugin,
        output=args.output_name or "",
        step_size=args.hop,
        block_size=args.frame_size,
    )

    console.print(f"[dim]결과 형태: {result.shape}[/dim]")

    # 플러그인 이름에서 suffix 생성
    plugin_suffix = args.plugin.replace(":", "_").replace("-", "")

    if result.shape == "matrix":
        matrix, hop_samples = result_to_matrix(result)
        out_path = _resolve_output_path(args, input_path, plugin_suffix)
        write_dense3d(
            out_path, matrix,
            sample_rate=sr,
            window_size=args.frame_size,
            hop_size=hop_samples,
        )

    elif result.shape == "vector":
        frames, values = result_to_frames_and_values(result, sr, hop_size=args.hop)
        out_path = _resolve_output_path(args, input_path, plugin_suffix)
        # 단위 추정
        units = _guess_units(args.plugin)
        write_time_values(
            out_path, frames, values,
            units=units, name=args.plugin,
            sample_rate=sr, resolution=args.hop,
        )

    elif result.shape == "list":
        events = result.data["list"]
        # duration 필드가 있으면 notes, 없으면 instants
        has_duration = any(
            ev.get("duration") and float(ev["duration"]) > 0
            for ev in events[:10]
        )

        if has_duration:
            frames, values = result_to_frames_and_values(result, sr)
            durations = []
            levels = []
            labels = []
            for ev in events:
                dur = ev.get("duration", 0)
                durations.append(int(round(float(dur) * sr)))
                vals = ev.get("values", [])
                levels.append(float(vals[0]) if vals else 1.0)
                labels.append(str(ev.get("label", "")))

            frame_list = []
            pitch_list = []
            for ev in events:
                t = ev.get("timestamp", ev.get("time", 0))
                frame_list.append(int(round(float(t) * sr)))
                vals = ev.get("values", [])
                pitch_list.append(float(vals[0]) if vals else 0.0)

            out_path = _resolve_output_path(args, input_path, plugin_suffix)
            write_notes(
                out_path, frame_list, pitch_list, durations,
                levels=levels, labels=labels,
                sample_rate=sr, resolution=args.hop,
            )
        else:
            frames, labels = result_to_instants(result, sr)
            out_path = _resolve_output_path(args, input_path, plugin_suffix)
            write_time_instants(
                out_path, frames, labels,
                sample_rate=sr, resolution=args.hop,
            )
    else:
        print_error(f"알 수 없는 결과 형태: {result.shape}")

    print_success(f"SVL 생성: {out_path}")

    if args.open:
        _open_in_sv(out_path)


def _run_builtin(args: argparse.Namespace, input_path: Path) -> None:
    from audioman.core.analysis import compute_frame_metrics

    audio, sr = read_audio(input_path)
    builtin = args.builtin
    frame_size = args.frame_size
    hop = args.hop

    console.print(f"[dim]내장 분석: {builtin} (frame={frame_size}, hop={hop})[/dim]")

    if builtin == "spectrogram":
        matrix = _compute_spectrogram(audio, sr, frame_size, hop)
        out_path = _resolve_output_path(args, input_path, "spectrogram")

        # bin 이름 생성 (주파수 범위)
        n_bins = matrix.shape[1]
        freq_per_bin = (sr / 2) / n_bins
        bin_names = [
            f"{freq_per_bin * i:.0f}-{freq_per_bin * (i + 1):.0f}Hz"
            for i in range(n_bins)
        ]

        write_dense3d(
            out_path, matrix,
            sample_rate=sr,
            window_size=frame_size,
            hop_size=hop,
            bin_names=bin_names,
        )

    else:
        # 프레임 단위 메트릭 → time values
        metrics = compute_frame_metrics(audio, sr, frame_size=frame_size, hop_size=hop)

        metric_map = {
            "spectral-centroid": ("spectral_centroid", "Hz"),
            "spectral-entropy": ("spectral_entropy", "bits"),
            "rms": ("rms", ""),
            "peak": ("peak", ""),
            "zcr": ("zero_crossing_rate", ""),
        }

        attr, units = metric_map[builtin]
        values = getattr(metrics, attr)
        frames = [i * hop for i in range(len(values))]

        out_path = _resolve_output_path(args, input_path, builtin.replace("-", "_"))
        write_time_values(
            out_path, frames, values,
            units=units, name=builtin,
            sample_rate=sr, resolution=hop,
        )

    print_success(f"SVL 생성: {out_path}")

    if args.open:
        _open_in_sv(out_path)


def _compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> np.ndarray:
    """STFT 파워 스펙트로그램 (dB 스케일)

    Returns:
        (n_frames, n_bins) 배열, dB 스케일
    """
    # mono 변환
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    n_samples = len(mono)
    window = np.hanning(frame_size)
    frames_list = []

    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = mono[start:start + frame_size]
        spectrum = np.abs(np.fft.rfft(frame * window))
        # 파워 → dB (Sonic Visualiser 호환)
        power = spectrum ** 2
        power_db = 10 * np.log10(np.maximum(power, 1e-10))
        frames_list.append(power_db)

    return np.array(frames_list)


def _guess_units(plugin_id: str) -> str:
    """플러그인 이름에서 단위 추정"""
    pid = plugin_id.lower()
    if "centroid" in pid or "pitch" in pid or "frequency" in pid:
        return "Hz"
    if "energy" in pid or "amplitude" in pid or "rms" in pid:
        return "dB"
    if "tempo" in pid or "bpm" in pid:
        return "bpm"
    return ""


def _open_in_sv(path: Path) -> None:
    """Sonic Visualiser로 SVL 파일 열기 (macOS)"""
    import subprocess
    try:
        subprocess.Popen(["open", "-a", "Sonic Visualiser", str(path)])
        console.print("[dim]Sonic Visualiser로 열기 시도...[/dim]")
    except FileNotFoundError:
        console.print("[yellow]Sonic Visualiser를 찾을 수 없습니다. 수동으로 열어주세요.[/yellow]")
