# Created: 2026-04-05
# Purpose: Destructive commit — apply a plugin chain + auto delay compensation

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from audioman.core.audio_file import read_audio, write_audio, get_audio_stats
from audioman.core.latency import (
    LatencyMeasurement,
    apply_delay_compensation,
    measure_chain_latency,
)
from audioman.core.pipeline import PipelineStep
from audioman.core.registry import get_registry
from audioman.plugins.vst3 import VST3PluginWrapper

logger = logging.getLogger(__name__)


@dataclass
class CommitResult:
    """Commit result"""
    input_path: str
    output_path: str
    steps: list[dict]
    latency_compensation: list[dict]
    total_latency_samples: int
    input_stats: dict
    output_stats: dict
    duration_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def commit_file(
    input_path: str | Path,
    output_path: str | Path,
    steps: list[PipelineStep],
    compensate_latency: bool = True,
    tail_trim: bool = True,
) -> CommitResult:
    """Apply a plugin chain to a single file + delay compensation

    Unlike pipeline.run_pipeline():
    - measures each plugin's latency up front
    - compensates the accumulated latency in the final output (drop the front + zero-pad)
    - restores the original length with tail_trim

    Args:
        input_path: input audio file
        output_path: output file path
        steps: plugin chain (list of PipelineStep)
        compensate_latency: whether to apply delay compensation
        tail_trim: trim the tail a plugin added back to the original length
    """
    start = time.monotonic()
    registry = get_registry()

    # read the audio
    audio, sr = read_audio(input_path)
    input_stats = get_audio_stats(audio, sr)
    original_length = audio.shape[-1]

    # measure latency
    measurements: list[LatencyMeasurement] = []
    total_latency = 0

    if compensate_latency:
        measurements, total_latency = measure_chain_latency(steps, sample_rate=sr)
        if total_latency > 0:
            logger.info(
                f"Total latency: {total_latency} samples "
                f"({total_latency / sr * 1000:.1f}ms) — compensation will be applied"
            )

    # process the plugin chain sequentially (in memory)
    for i, step in enumerate(steps):
        meta = registry.get(step.plugin_name)
        if not meta:
            raise ValueError(f"Plugin not found: '{step.plugin_name}' (step {i+1})")

        wrapper = VST3PluginWrapper(meta.path)
        wrapper.load()
        if step.params:
            wrapper.set_parameters(step.params)

        logger.info(f"Step {i+1}/{len(steps)}: {meta.short_name}")
        audio = wrapper.process(audio, sr)

    # Delay compensation
    if compensate_latency and total_latency > 0:
        audio = apply_delay_compensation(audio, total_latency)

    # Tail trim — restore the original length
    if tail_trim and audio.shape[-1] > original_length:
        if audio.ndim == 1:
            audio = audio[:original_length]
        else:
            audio = audio[:, :original_length]

    output_stats = get_audio_stats(audio, sr)
    write_audio(output_path, audio, sr)

    elapsed = time.monotonic() - start
    return CommitResult(
        input_path=str(input_path),
        output_path=str(output_path),
        steps=[s.to_dict() for s in steps],
        latency_compensation=[m.to_dict() for m in measurements],
        total_latency_samples=total_latency,
        input_stats=asdict(input_stats),
        output_stats=asdict(output_stats),
        duration_seconds=round(elapsed, 3),
    )


def dry_run_commit(
    steps: list[PipelineStep],
    sample_rate: int = 48000,
) -> tuple[list[LatencyMeasurement], int]:
    """Measure latency only, without processing (dry-run)

    Returns:
        (measurements, total_latency_samples)
    """
    return measure_chain_latency(steps, sample_rate=sample_rate)
