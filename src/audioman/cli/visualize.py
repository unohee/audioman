# Created: 2026-03-22
# Purpose: audioman visualize subcommand - Vamp plugin + built-in analysis -> SVL export
# Dependencies: core.svl, core.vamp_host, core.analysis

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from audioman.cli.output import print_error, print_info, print_markup, print_success, print_warning, output_console
from audioman.core.audio_file import read_audio
from audioman.core.svl import (
    write_time_instants,
    write_time_values,
    write_notes,
    write_dense3d,
)


# Guidance when no launcher is available (print this instead of a traceback)
_NO_SV_LAUNCHER = "Sonic Visualiser was not found. Please open the file manually."

# Built-in analysis types and their descriptions
BUILTIN_TYPES = {
    "spectral-centroid": "Spectral centroid frequency (Hz)",
    "spectral-entropy": "Spectral entropy (bits)",
    "rms": "RMS energy",
    "peak": "Peak amplitude",
    "zcr": "Zero crossing rate",
    "spectrogram": "STFT power spectrogram",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "visualize",
        help="Vamp plugin or built-in analysis -> Sonic Visualiser SVL file",
    )
    parser.add_argument("input", help="Input audio file")

    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--plugin", "-p",
        help="Vamp plugin ID (e.g. qm-vamp-plugins:qm-chromagram)",
    )
    source.add_argument(
        "--builtin", "-b",
        choices=list(BUILTIN_TYPES.keys()),
        help=f"{'Built-in analysis type'}: {', '.join(BUILTIN_TYPES.keys())}",
    )

    parser.add_argument("-o", "--output", help="Output SVL file path (default: auto)")
    parser.add_argument("--output-name", help="Vamp plugin output name (for multiple outputs)")
    parser.add_argument("--frame-size", type=_positive_int, default=2048, help="FFT frame size (default: 2048)")
    parser.add_argument("--hop", type=_positive_int, default=512, help="Hop size (default: 512)")
    parser.add_argument("--list-plugins", action="store_true", help="List installed Vamp plugins")
    parser.add_argument("--plugin-info", help="Query plugin output info")
    parser.add_argument("--open", action="store_true", help="Open in Sonic Visualiser after creation")
    parser.add_argument("--png", help="Also write a PNG spectrogram image (multimodal-friendly) to this path")
    parser.add_argument("--png-only", action="store_true", help="Write only PNG, skip SVL output")
    parser.add_argument("--png-width", type=int, default=1600, help="PNG width in pixels (default: 1600)")
    parser.add_argument("--png-height", type=int, default=600, help="PNG height in pixels (default: 600)")
    parser.add_argument("--png-db-min", type=float, default=-90.0, help="PNG color floor in dB (default: -90)")
    parser.add_argument("--png-db-max", type=float, default=0.0, help="PNG color ceiling in dB (default: 0)")
    parser.add_argument("--png-fmax", type=float, default=None, help="PNG max display frequency Hz (default: Nyquist)")

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
        print_error(f"File not found: {input_path}")

    if args.plugin:
        _run_vamp(args, input_path)
    elif args.builtin:
        _run_builtin(args, input_path)
    else:
        # Default: spectrogram
        args.builtin = "spectrogram"
        _run_builtin(args, input_path)


def _list_plugins() -> None:
    from audioman.core.vamp_host import list_plugins

    plugins = list_plugins()
    if not plugins:
        print_error("No Vamp plugins installed.")

    print_markup(f"\n[bold]Installed Vamp plugins ({len(plugins)})[/bold]\n")
    for p in plugins:
        output_console.print(f"  {p}")
    output_console.print()


def _plugin_info(plugin_id: str) -> None:
    from audioman.core.vamp_host import get_plugin_outputs

    outputs = get_plugin_outputs(plugin_id)
    print_markup(f"\n[bold]{plugin_id}[/bold]\n")
    for name, info in outputs.items():
        output_console.print(f"  {name}: {info}")
    output_console.print()


def _resolve_output_path(args: argparse.Namespace, input_path: Path, suffix: str) -> Path:
    """Resolve the output path."""
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

    try:
        audio, sr = read_audio(input_path)
    except (OSError, RuntimeError, ValueError) as e:
        # `soundfile.LibsndfileError` derives from RuntimeError, not OSError.
        print_error(str(e))
        return
    print_info(f"Running Vamp plugin: {args.plugin}")

    result = run_plugin(
        audio, sr, args.plugin,
        output=args.output_name or "",
        step_size=args.hop,
        block_size=args.frame_size,
    )

    print_info(f"Result shape: {result.shape}")

    # Build a suffix from the plugin name
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
        # Guess the units
        units = _guess_units(args.plugin)
        write_time_values(
            out_path, frames, values,
            units=units, name=args.plugin,
            sample_rate=sr, resolution=args.hop,
        )

    elif result.shape == "list":
        events = result.data["list"]
        # Use notes when a duration field is present, otherwise instants
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
        print_error(f"Unknown result shape: {result.shape}")

    print_success(f"SVL written: {out_path}")

    if args.open:
        _open_in_sv(out_path)


def _run_builtin(args: argparse.Namespace, input_path: Path) -> None:
    from audioman.core.analysis import compute_frame_metrics

    try:
        audio, sr = read_audio(input_path)
    except (OSError, RuntimeError, ValueError) as e:
        # `soundfile.LibsndfileError` derives from RuntimeError, not OSError.
        print_error(str(e))
        return
    builtin = args.builtin
    frame_size = args.frame_size
    hop = args.hop

    print_info(f"Built-in analysis: {builtin} (frame={frame_size}, hop={hop})")

    if builtin == "spectrogram":
        if audio.shape[-1] < frame_size:
            print_error(
                f"Input audio is shorter than the spectrogram frame size: "
                f"samples={audio.shape[-1]}, frame-size={frame_size}"
            )
        matrix = _compute_spectrogram(audio, sr, frame_size, hop)

        if args.png or args.png_only:
            png_path = Path(args.png) if args.png else _resolve_output_path(args, input_path, "spectrogram").with_suffix(".png")
            _write_spectrogram_png(
                matrix, sr, hop, frame_size,
                png_path,
                width=args.png_width, height=args.png_height,
                db_min=args.png_db_min, db_max=args.png_db_max,
                fmax=args.png_fmax,
                title=input_path.name,
            )
            print_success(f"PNG written: {png_path}")

        if args.png_only:
            if args.open:
                _open_in_sv(png_path)
            return

        out_path = _resolve_output_path(args, input_path, "spectrogram")

        # Build bin names from the frequency range
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
        # Per-frame metrics -> time values
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

    print_success(f"SVL written: {out_path}")

    if args.open:
        _open_in_sv(out_path)


def _compute_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> np.ndarray:
    """STFT power spectrogram in dB scale.

    Returns:
        (n_frames, n_bins) array in dB scale.
    """
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")

    # Convert to mono
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    n_samples = len(mono)
    if n_samples < frame_size:
        raise ValueError(
            f"audio has {n_samples} samples, shorter than frame_size {frame_size}"
        )
    window = np.hanning(frame_size)
    frames_list = []

    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = mono[start:start + frame_size]
        spectrum = np.abs(np.fft.rfft(frame * window))
        # Power -> dB (Sonic Visualiser compatible)
        power = spectrum ** 2
        power_db = 10 * np.log10(np.maximum(power, 1e-10))
        frames_list.append(power_db)

    return np.array(frames_list)


def _write_spectrogram_png(
    matrix: np.ndarray,
    sample_rate: int,
    hop_size: int,
    frame_size: int,
    out_path: Path,
    width: int = 1600,
    height: int = 600,
    db_min: float = -90.0,
    db_max: float = 0.0,
    fmax: float | None = None,
    title: str = "",
) -> None:
    """Write a matplotlib PNG spectrogram; matrix shape is (n_frames, n_bins) in dB."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print_error("matplotlib is required: uv add matplotlib")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames, n_bins = matrix.shape
    duration_sec = n_frames * hop_size / sample_rate
    nyquist = sample_rate / 2.0
    if fmax is None or fmax > nyquist:
        fmax = nyquist
    bin_cutoff = int(round(fmax / nyquist * n_bins))
    display = matrix[:, :bin_cutoff].T  # (n_bins, n_frames)

    dpi = 100
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(
        display,
        origin="lower",
        aspect="auto",
        extent=[0, duration_sec, 0, fmax],
        vmin=db_min, vmax=db_max,
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectrogram: {title}  (FFT={frame_size}, hop={hop_size}, fs={sample_rate})")
    cbar = fig.colorbar(im, ax=ax, label="Power (dB)")
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _guess_units(plugin_id: str) -> str:
    """Guess the units from the plugin name."""
    pid = plugin_id.lower()
    if "centroid" in pid or "pitch" in pid or "frequency" in pid:
        return "Hz"
    if "energy" in pid or "amplitude" in pid or "rms" in pid:
        return "dB"
    if "tempo" in pid or "bpm" in pid:
        return "bpm"
    return ""


def _sv_launcher(path: Path) -> list[str] | None:
    """Return Sonic Visualiser launcher arguments for the current platform.

    macOS uses `open -a`, Linux uses `xdg-open`. Other platforms have no way to
    open the file, so they return None (this CLI does not support a Windows
    launcher).
    """
    if sys.platform == "darwin":
        return ["open", "-a", "Sonic Visualiser", str(path)]
    if sys.platform.startswith("linux"):
        return ["xdg-open", str(path)]
    return None


def _open_in_sv(path: Path) -> None:
    """Open an SVL file in Sonic Visualiser (platform-specific launcher)."""
    command = _sv_launcher(path)
    if command is None:
        print_warning(_NO_SV_LAUNCHER)
        return

    try:
        subprocess.Popen(command)
    except OSError:
        # The launcher is missing or cannot be executed (includes FileNotFoundError)
        print_warning(_NO_SV_LAUNCHER)
    else:
        print_info("Opening in Sonic Visualiser...")
