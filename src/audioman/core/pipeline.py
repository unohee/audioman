# Created: 2026-03-21
# Purpose: Multi-plugin chain processing pipeline

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from audioman.core.audio_file import get_audio_stats, read_audio, write_audio
from audioman.core.registry import get_registry
from audioman.plugins.vst3 import VST3PluginWrapper

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """Pipeline step"""
    plugin_name: str
    params: dict[str, Any]

    def to_dict(self) -> dict:
        return {"plugin": self.plugin_name, "params": self.params}


@dataclass
class PipelineResult:
    """Pipeline execution result"""
    input_path: str
    output_path: str
    steps: list[dict]
    input_stats: dict
    output_stats: dict
    duration_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def parse_chain_string(chain_str: str) -> list[PipelineStep]:
    """Parse a chain string
    "denoise:threshold=-20,dehum:freq=60,declick" →
    [PipelineStep("denoise", {"threshold": -20}), PipelineStep("dehum", {"freq": 60}), ...]
    """
    steps = []
    for segment in chain_str.split(","):
        segment = segment.strip()
        if not segment:
            continue

        if ":" in segment:
            plugin_name, params_str = segment.split(":", 1)
            params = {}
            for kv in params_str.split(";"):
                kv = kv.strip()
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        params[k.strip()] = float(v.strip())
                    except ValueError:
                        if v.strip().lower() in ("true", "false"):
                            params[k.strip()] = v.strip().lower() == "true"
                        else:
                            params[k.strip()] = v.strip()
        else:
            plugin_name = segment
            params = {}

        steps.append(PipelineStep(plugin_name=plugin_name.strip(), params=params))

    return steps


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    steps: list[PipelineStep],
) -> PipelineResult:
    """Process multiple plugins sequentially"""
    start = time.monotonic()
    registry = get_registry()

    # read the audio
    audio, sr = read_audio(input_path)
    input_stats = get_audio_stats(audio, sr)

    # run each step sequentially
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

    # write the output
    output_stats = get_audio_stats(audio, sr)
    write_audio(output_path, audio, sr)

    elapsed = time.monotonic() - start
    return PipelineResult(
        input_path=str(input_path),
        output_path=str(output_path),
        steps=[s.to_dict() for s in steps],
        input_stats=asdict(input_stats),
        output_stats=asdict(output_stats),
        duration_seconds=round(elapsed, 3),
    )
