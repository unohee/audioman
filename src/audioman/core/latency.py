# Created: 2026-04-05
# Purpose: plugin latency measurement + auto delay compensation

import logging
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from audioman.core.test_signal import generate_impulse
from audioman.core.registry import get_registry
from audioman.plugins.vst3 import VST3PluginWrapper

logger = logging.getLogger(__name__)


@dataclass
class LatencyMeasurement:
    """Plugin latency measurement result."""
    plugin_name: str
    reported_latency: int       # value reported by pedalboard (samples)
    measured_latency: int       # impulse round-trip measurement (samples)
    confidence: float           # measurement confidence (0.0 ~ 1.0)
    used_latency: int           # value actually used

    def to_dict(self) -> dict:
        return asdict(self)


def _get_reported_latency(wrapper: VST3PluginWrapper) -> int:
    """Reported plugin latency in samples, 0 when the plugin does not report one.

    Only the two failures the introspection itself can produce are absorbed:
    AttributeError when this plugin class has no `latency_samples` (pedalboard exposes
    it on some types only — verified against VST3Plugin, which raises
    "'super' object has no attribute 'latency_samples'") and TypeError when the
    attribute exists but is not int-convertible. Anything else is a real error: an
    unknown error silently returning 0 here would zero the delay compensation the host
    relies on, with no signal that it happened.
    """
    try:
        # access pedalboard's internal attribute
        plugin = wrapper._plugin
        if hasattr(plugin, "latency_samples"):
            return int(plugin.latency_samples)
    except (AttributeError, TypeError) as exc:
        logger.debug(f"{wrapper.name}: reported latency unavailable ({exc}); using 0")
    return 0


def measure_plugin_latency(
    wrapper: VST3PluginWrapper,
    sample_rate: int = 48000,
    channels: int = 2,
    test_duration_sec: float = 1.0,
) -> LatencyMeasurement:
    """Measure a single plugin's latency via impulse round trip.

    Algorithm:
        1. Generate a delta impulse at sample[0]
        2. Reset plugin state, then pass the signal through
        3. Peak position in the output = latency (samples)
        4. Confidence from the peak-to-noise-floor ratio
    """
    wrapper.load()
    wrapper.reset()

    # generate the impulse (1.0 at sample 0)
    impulse = generate_impulse(
        sample_rate=sample_rate,
        duration_sec=test_duration_sec,
        channels=channels,
    )

    # run through the plugin
    output = wrapper.process(impulse, sample_rate)

    # sum to mono for analysis
    if output.ndim == 2:
        mono = np.mean(output, axis=0)
    else:
        mono = output

    abs_mono = np.abs(mono)

    # peak position = latency
    peak_idx = int(np.argmax(abs_mono))
    peak_val = float(abs_mono[peak_idx])

    # estimate the noise floor (excluding ±100 samples around the peak)
    mask = np.ones(len(abs_mono), dtype=bool)
    exclude_start = max(0, peak_idx - 100)
    exclude_end = min(len(abs_mono), peak_idx + 100)
    mask[exclude_start:exclude_end] = False

    if np.any(mask):
        noise_floor = float(np.mean(abs_mono[mask]))
    else:
        noise_floor = 0.0

    # confidence: peak-to-noise-floor ratio
    if noise_floor > 0:
        snr = peak_val / noise_floor
        # SNR 20 or more -> confidence 1.0, 1 or less -> 0.0
        confidence = float(np.clip((snr - 1.0) / 19.0, 0.0, 1.0))
    elif peak_val > 1e-6:
        confidence = 1.0
    else:
        confidence = 0.0

    # value reported by pedalboard
    reported = _get_reported_latency(wrapper)

    # decide the final latency
    if confidence >= 0.5:
        used = peak_idx
    elif reported > 0:
        used = reported
        logger.warning(
            f"{wrapper.name}: low confidence in impulse measurement "
            f"(confidence={confidence:.2f}), using reported latency: {reported} samples"
        )
    else:
        used = peak_idx
        logger.warning(
            f"{wrapper.name}: latency measurement uncertain (confidence={confidence:.2f}), "
            f"using measured value: {peak_idx} samples"
        )

    # warn when reported and measured disagree
    if reported > 0 and abs(reported - peak_idx) > 1:
        logger.info(
            f"{wrapper.name}: reported latency ({reported}) != measured latency ({peak_idx}), "
            f"using measured value (confidence={confidence:.2f})"
        )

    measurement = LatencyMeasurement(
        plugin_name=wrapper.name,
        reported_latency=reported,
        measured_latency=peak_idx,
        confidence=round(confidence, 4),
        used_latency=used,
    )

    logger.debug(
        f"latency measurement: {wrapper.name} -> "
        f"measured={peak_idx}, reported={reported}, "
        f"confidence={confidence:.2f}, used={used}"
    )

    return measurement


def measure_chain_latency(
    steps: list[dict[str, Any]],
    sample_rate: int = 48000,
) -> tuple[list[LatencyMeasurement], int]:
    """Measure the latency of each plugin in a chain and sum the total.

    Args:
        steps: [{"plugin_name": str, "params": dict}, ...]
            PipelineStep.to_dict() format, or PipelineStep objects
        sample_rate: sample rate to measure at

    Returns:
        (measurements, total_latency_samples)
    """
    from audioman.core.pipeline import PipelineStep

    registry = get_registry()
    measurements = []
    total = 0

    for step in steps:
        # accept both PipelineStep objects and dicts
        if isinstance(step, PipelineStep):
            plugin_name = step.plugin_name
            params = step.params
        else:
            plugin_name = step.get("plugin", step.get("plugin_name", ""))
            params = step.get("params", {})

        meta = registry.get(plugin_name)
        if not meta:
            raise ValueError(f"plugin not found: '{plugin_name}'")

        wrapper = VST3PluginWrapper(meta.path)
        wrapper.load()

        if params:
            wrapper.set_parameters(params)

        measurement = measure_plugin_latency(wrapper, sample_rate)
        measurement.plugin_name = meta.short_name
        measurements.append(measurement)
        total += measurement.used_latency

    logger.info(f"chain total latency: {total} samples ({total / sample_rate * 1000:.1f}ms)")
    return measurements, total


def apply_delay_compensation(
    audio: np.ndarray,
    total_latency_samples: int,
) -> np.ndarray:
    """Drop the leading latency samples and zero-pad the tail (original length kept).

    Args:
        audio: (channels, samples) or (samples,)
        total_latency_samples: latency to compensate (samples)

    Returns:
        Compensated audio (same shape as the input)
    """
    if total_latency_samples <= 0:
        return audio

    if audio.ndim == 1:
        n = len(audio)
        if total_latency_samples >= n:
            logger.warning(
                f"latency ({total_latency_samples}) exceeds audio length ({n}), "
                f"returning silence"
            )
            return np.zeros_like(audio)
        # drop the head + zero-pad the tail
        compensated = np.zeros(n, dtype=audio.dtype)
        remaining = n - total_latency_samples
        compensated[:remaining] = audio[total_latency_samples:]
        return compensated
    else:
        channels, n = audio.shape
        if total_latency_samples >= n:
            logger.warning(
                f"latency ({total_latency_samples}) exceeds audio length ({n}), "
                f"returning silence"
            )
            return np.zeros_like(audio)
        compensated = np.zeros((channels, n), dtype=audio.dtype)
        remaining = n - total_latency_samples
        compensated[:, :remaining] = audio[:, total_latency_samples:]
        return compensated
