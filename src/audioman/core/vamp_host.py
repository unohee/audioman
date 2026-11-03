# Created: 2026-03-22
# Purpose: Vamp plugin host wrapper
# Dependencies: vamp (optional)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _import_vamp():
    """Lazy import of the vamp package"""
    try:
        import vamp
        return vamp
    except ImportError:
        raise ImportError(
            "The 'vamp' package is required. Install: pip install vamp\n"
            "Vamp plugins must also be installed on the system.\n"
            "  macOS: brew install vamp-plugin-sdk qm-vamp-plugins"
        )


@dataclass
class VampResult:
    """Result of running a Vamp plugin"""
    plugin_id: str
    output: str
    shape: str  # "list", "vector", "matrix"
    sample_rate: int
    data: Any  # dict from vamp.collect()


def list_plugins() -> list[str]:
    """List of installed Vamp plugins"""
    vamp = _import_vamp()
    return sorted(vamp.list_plugins())


def get_plugin_outputs(plugin_id: str) -> dict:
    """Query a plugin's output information"""
    vamp = _import_vamp()
    return vamp.get_outputs_of(plugin_id)


def run_plugin(
    audio: np.ndarray,
    sample_rate: int,
    plugin_id: str,
    output: str = "",
    parameters: dict[str, float] | None = None,
    block_size: int = 0,
    step_size: int = 0,
) -> VampResult:
    """Run a Vamp plugin

    Args:
        audio: (channels, samples) or (samples,) — converted to mono
        sample_rate: sample rate
        plugin_id: "library:plugin" or "library:plugin:output" form
        output: output name (empty string → default output)
        parameters: plugin parameters {name: value}
        block_size: FFT block size (0 → the plugin's default)
        step_size: hop size (0 → the plugin's default)

    Returns:
        VampResult
    """
    vamp = _import_vamp()

    # mono conversion (vamp expects a 1-D float32 array)
    if audio.ndim == 2:
        mono = audio.mean(axis=0).astype(np.float32)
    else:
        mono = audio.astype(np.float32)

    # split output off plugin_id ("lib:plugin:output" form)
    parts = plugin_id.split(":")
    if len(parts) == 3 and not output:
        plugin_id = f"{parts[0]}:{parts[1]}"
        output = parts[2]

    kwargs: dict[str, Any] = {}
    if output:
        kwargs["output"] = output
    if parameters:
        kwargs["parameters"] = parameters
    if block_size > 0:
        kwargs["block_size"] = block_size
    if step_size > 0:
        kwargs["step_size"] = step_size

    result = vamp.collect(mono, sample_rate, plugin_id, **kwargs)

    # determine the result shape
    if "matrix" in result:
        shape = "matrix"
    elif "vector" in result:
        shape = "vector"
    elif "list" in result:
        shape = "list"
    else:
        shape = "unknown"

    return VampResult(
        plugin_id=plugin_id,
        output=output,
        shape=shape,
        sample_rate=sample_rate,
        data=result,
    )


def result_to_frames_and_values(
    result: VampResult,
    sample_rate: int,
    hop_size: int = 512,
) -> tuple[list[int], list[float]]:
    """Convert a vector/list result into a (frames, values) pair

    Returns:
        (frame_numbers, values) — for SVL time values
    """
    if result.shape == "vector":
        step, values = result.data["vector"]
        step_samples = int(round(float(step) * sample_rate))
        if step_samples == 0:
            step_samples = hop_size
        frames = [i * step_samples for i in range(len(values))]
        return frames, [float(v) for v in values]

    elif result.shape == "list":
        events = result.data["list"]
        frames = []
        values = []
        for ev in events:
            t = ev.get("timestamp", ev.get("time", 0))
            frame = int(round(float(t) * sample_rate))
            frames.append(frame)
            vals = ev.get("values", [])
            values.append(float(vals[0]) if len(vals) > 0 else 0.0)
        return frames, values

    else:
        raise ValueError(f"cannot convert vector/list result: shape={result.shape}")


def result_to_instants(
    result: VampResult,
    sample_rate: int,
) -> tuple[list[int], list[str]]:
    """Convert a list result into a (frames, labels) pair

    Returns:
        (frame_numbers, labels) — for SVL time instants
    """
    if result.shape != "list":
        raise ValueError(f"time instants conversion requires a list result: shape={result.shape}")

    events = result.data["list"]
    frames = []
    labels = []
    for ev in events:
        t = ev.get("timestamp", ev.get("time", 0))
        frame = int(round(float(t) * sample_rate))
        frames.append(frame)
        labels.append(str(ev.get("label", "")))
    return frames, labels


def result_to_matrix(
    result: VampResult,
) -> tuple[np.ndarray, int]:
    """Convert a matrix result into a (matrix, hop_samples) pair

    Returns:
        (matrix[n_frames, n_bins], hop_size_in_samples) — for SVL dense 3D
    """
    if result.shape != "matrix":
        raise ValueError(f"cannot convert matrix result: shape={result.shape}")

    step, matrix = result.data["matrix"]
    hop_samples = int(round(float(step) * result.sample_rate))
    return np.array(matrix), hop_samples
