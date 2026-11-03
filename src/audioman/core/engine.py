# Created: 2026-03-21
# Purpose: Single-plugin audio processing engine

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from audioman.config.settings import get_settings
from audioman.core.audio_file import get_audio_stats, get_file_info, read_audio, stream_process, write_audio
from audioman.core.registry import get_registry
from audioman.plugins.vst3 import VST3PluginWrapper

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """Processing result"""
    input_path: str
    output_path: str
    plugin_name: str
    params_applied: dict
    input_stats: dict
    output_stats: dict
    duration_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def parse_params(param_strings: list[str]) -> dict[str, Any]:
    """Parse CLI parameter strings: ["threshold=-20", "reduction=12"] → dict

    A value wrapped in quotes (`key="4.00"` or `key='4.00'`) is forced to
    stay a string. This is needed for plugins like UAD whose enum labels are
    two-decimal strings such as `"4.00"`, `"0.97"` — converting to float
    turns `"4.00"` into `"4.0"` and fails to match the enum list.
    """
    params = {}
    for s in param_strings:
        if "=" not in s:
            raise ValueError(f"Invalid parameter format (expected key=value): '{s}'")
        key, value = s.split("=", 1)
        key = key.strip()

        # explicit string: preserve the original when wrapped in quotes.
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            params[key] = value[1:-1]
            continue

        # type inference
        if value.lower() in ("true", "false"):
            params[key] = value.lower() == "true"
        else:
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value  # string (enum, etc.)

    return params


def _stream_threshold_mb() -> int:
    """Auto-streaming threshold in MB (settings.large_file_threshold_mb)."""
    try:
        return get_settings().large_file_threshold_mb
    except Exception:
        return 500


def _auto_stream_enabled() -> bool:
    """Whether large files may switch to streaming automatically (settings.auto_stream)."""
    try:
        return get_settings().auto_stream
    except Exception:
        return True


def _stream_chunk_seconds(sample_rate: int) -> float:
    """Streaming chunk length in seconds, from settings.default_chunk_size.

    ``default_chunk_size`` is a frame count (441000 ≈ 10 s at 44.1 kHz), so the
    chunk length depends on the file's sample rate.
    """
    try:
        chunk_frames = get_settings().default_chunk_size
    except Exception:
        return 10.0
    if sample_rate <= 0 or chunk_frames <= 0:
        return 10.0
    return chunk_frames / sample_rate


def process_file(
    input_path: str | Path,
    output_path: str | Path,
    plugin_name: str,
    params: Optional[dict[str, Any]] = None,
    passes: int = 1,
    stream: bool | None = None,
) -> ProcessResult:
    """Process an audio file with a single plugin

    Args:
        passes: number of processing passes. With 2 or more, the first pass is for
                learning and only the last pass is written out. Useful for learning
                the noise profile on plugins in adaptive mode.
    """
    start = time.monotonic()

    # look up the plugin
    registry = get_registry()
    meta = registry.get(plugin_name)
    if not meta:
        raise ValueError(f"Plugin not found: '{plugin_name}'")

    # large file → automatic streaming
    if stream is None:
        if _auto_stream_enabled():
            try:
                info = get_file_info(input_path)
                stream = info["file_size_mb"] > _stream_threshold_mb()
            except Exception:
                stream = False
        else:
            stream = False

    if stream:
        return _process_file_streaming(input_path, output_path, meta, params, start)

    # read the audio
    audio, sr = read_audio(input_path)
    input_stats = get_audio_stats(audio, sr)

    # load the plugin + set parameters
    wrapper = VST3PluginWrapper(meta.path)
    wrapper.load()

    if params:
        wrapper.set_parameters(params)

    # multi-pass processing
    output = audio
    for i in range(passes):
        logger.info(f"Pass {i+1}/{passes}")
        output = wrapper.process(audio, sr)

    output_stats = get_audio_stats(output, sr)

    # write the output
    write_audio(output_path, output, sr)

    elapsed = time.monotonic() - start
    return ProcessResult(
        input_path=str(input_path),
        output_path=str(output_path),
        plugin_name=meta.short_name,
        params_applied=params or {},
        input_stats=asdict(input_stats),
        output_stats=asdict(output_stats),
        duration_seconds=round(elapsed, 3),
    )


def _process_file_streaming(input_path, output_path, meta, params, start) -> ProcessResult:
    """Stream a large file — do not load the whole thing into memory"""
    wrapper = VST3PluginWrapper(meta.path)
    wrapper.load()
    if params:
        wrapper.set_parameters(params)

    def process_chunk(chunk, sr):
        return wrapper.process(chunk, sr)

    info = get_file_info(input_path)
    result = stream_process(
        input_path, output_path, process_chunk,
        chunk_seconds=_stream_chunk_seconds(info["sample_rate"]),
    )

    elapsed = time.monotonic() - start
    logger.info(f"streaming finished: {result['chunks']} chunks, {elapsed:.1f}s")

    return ProcessResult(
        input_path=str(input_path),
        output_path=str(output_path),
        plugin_name=meta.short_name,
        params_applied=params or {},
        input_stats={"duration": info["duration"], "sample_rate": info["sample_rate"],
                      "channels": info["channels"], "frames": info["frames"]},
        output_stats={"frames_processed": result["frames_processed"],
                       "chunks": result["chunks"], "streamed": True},
        duration_seconds=round(elapsed, 3),
    )
