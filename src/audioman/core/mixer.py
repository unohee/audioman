# Created: 2026-04-05
# Purpose: Multitrack mixing engine — bounce / mixdown

import logging
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from audioman.core.audio_file import read_audio, write_audio, get_audio_stats, AudioStats
from audioman.core.dsp import gain as apply_gain
from audioman.core.pipeline import PipelineStep, parse_chain_string
from audioman.core.registry import get_registry
from audioman.plugins.vst3 import VST3PluginWrapper

logger = logging.getLogger(__name__)


@dataclass
class TrackConfig:
    """Settings for a single track"""
    path: str
    gain_db: float = 0.0
    pan: float = 0.0              # -1.0 (L) ~ 0.0 (C) ~ 1.0 (R)
    mute: bool = False
    solo: bool = False
    chain: Optional[list[PipelineStep]] = None
    offset_samples: int = 0

    def to_dict(self) -> dict:
        d = {
            "path": self.path,
            "gain_db": self.gain_db,
            "pan": self.pan,
        }
        if self.mute:
            d["mute"] = True
        if self.solo:
            d["solo"] = True
        if self.chain:
            d["chain"] = [s.to_dict() for s in self.chain]
        if self.offset_samples:
            d["offset_samples"] = self.offset_samples
        return d


@dataclass
class BounceResult:
    """Bounce result"""
    output_path: str
    track_count: int
    tracks: list[dict]
    output_stats: dict
    sample_rate: int
    duration_seconds: float
    clipping_detected: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MixdownResult:
    """Mixdown result (bounce + master chain)"""
    output_path: str
    track_count: int
    tracks: list[dict]
    master_chain: Optional[list[dict]]
    master_latency_samples: int
    output_stats: dict
    sample_rate: int
    duration_seconds: float
    clipping_detected: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def apply_pan(audio_stereo: np.ndarray, pan: float) -> np.ndarray:
    """Apply the Equal Power Pan Law

    Args:
        audio_stereo: (2, samples) stereo audio
        pan: -1.0 (L) ~ 0.0 (C) ~ 1.0 (R)

    Returns:
        (2, samples) stereo with panning applied
    """
    pan = float(np.clip(pan, -1.0, 1.0))
    # map the pan value to an angle in 0~pi/2
    angle = (pan + 1.0) * 0.25 * np.pi
    gain_l = float(np.cos(angle))
    gain_r = float(np.sin(angle))

    out = audio_stereo.copy()
    out[0] *= gain_l
    out[1] *= gain_r
    return out


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Convert mono to stereo; return it unchanged if already stereo"""
    if audio.ndim == 1:
        return np.stack([audio, audio])
    if audio.shape[0] == 1:
        return np.concatenate([audio, audio], axis=0)
    if audio.shape[0] == 2:
        return audio
    # 3 or more channels → use only the first 2 channels
    logger.warning(f"{audio.shape[0]}ch audio → using only the first 2 channels")
    return audio[:2]


def _resample_if_needed(
    audio: np.ndarray,
    current_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample with soxr on a sample-rate mismatch"""
    if current_sr == target_sr:
        return audio

    import soxr

    # soxr expects (samples, channels)
    if audio.ndim == 2:
        data = audio.T  # (channels, samples) → (samples, channels)
        resampled = soxr.resample(data, current_sr, target_sr, quality="HQ")
        return resampled.T  # → (channels, samples)
    else:
        return soxr.resample(audio.reshape(-1, 1), current_sr, target_sr, quality="HQ").flatten()


def _apply_track_chain(
    audio: np.ndarray,
    sr: int,
    chain: list[PipelineStep],
) -> np.ndarray:
    """Apply a per-track plugin chain (in memory, no file I/O)"""
    registry = get_registry()

    for step in chain:
        meta = registry.get(step.plugin_name)
        if not meta:
            raise ValueError(f"Plugin not found: '{step.plugin_name}'")

        wrapper = VST3PluginWrapper(meta.path)
        wrapper.load()
        if step.params:
            wrapper.set_parameters(step.params)
        audio = wrapper.process(audio, sr)

    return audio


def mix_tracks(
    tracks: list[TrackConfig],
    sample_rate: Optional[int] = None,
    apply_chain: bool = True,
) -> tuple[np.ndarray, int]:
    """Mix several tracks into stereo

    Args:
        tracks: list of track settings
        sample_rate: target sample rate (None → use the first track's)
        apply_chain: whether to apply each track's plugin chain

    Returns:
        (audio (2, samples), sample_rate)
    """
    if not tracks:
        raise ValueError("No tracks provided")

    # Solo filtering: if any track is soloed, play only the soloed tracks
    has_solo = any(t.solo for t in tracks)
    active_tracks = []
    for t in tracks:
        if t.mute:
            continue
        if has_solo and not t.solo:
            continue
        active_tracks.append(t)

    if not active_tracks:
        logger.warning("All tracks are muted; outputting silence")
        # build empty audio using the first track's information
        sr = sample_rate or 48000
        return np.zeros((2, sr), dtype=np.float32), sr

    # load each track
    loaded: list[tuple[np.ndarray, int, TrackConfig]] = []
    for t in active_tracks:
        audio, sr = read_audio(t.path)
        loaded.append((audio, sr, t))

    # determine the target sample rate
    if sample_rate is None:
        sample_rate = loaded[0][1]

    # process each track
    processed: list[np.ndarray] = []
    for audio, sr, track_cfg in loaded:
        # resampling
        audio = _resample_if_needed(audio, sr, sample_rate)

        # per-track plugin chain
        if apply_chain and track_cfg.chain:
            audio = _apply_track_chain(audio, sample_rate, track_cfg.chain)

        # convert to stereo
        audio = _ensure_stereo(audio)

        # apply gain
        if track_cfg.gain_db != 0.0:
            audio = apply_gain(audio, track_cfg.gain_db)

        # apply panning (even at center the equal power law gives ~0.707 gain)
        audio = apply_pan(audio, track_cfg.pan)

        # apply offset (insert silence at the front)
        if track_cfg.offset_samples > 0:
            pad = np.zeros((2, track_cfg.offset_samples), dtype=audio.dtype)
            audio = np.concatenate([pad, audio], axis=1)

        processed.append(audio)

    # align lengths (zero-pad to the longest track)
    max_len = max(a.shape[1] for a in processed)
    aligned = []
    for a in processed:
        if a.shape[1] < max_len:
            pad_len = max_len - a.shape[1]
            a = np.pad(a, ((0, 0), (0, pad_len)), mode="constant")
        aligned.append(a)

    # sum
    mix = np.sum(np.stack(aligned), axis=0).astype(np.float32)

    # clipping check
    peak = float(np.max(np.abs(mix)))
    if peak > 1.0:
        logger.warning(
            f"Clipping detected: peak={peak:.3f} ({20 * np.log10(peak):.1f} dBFS). "
            f"Add a limiter to the master chain or lower the track volumes."
        )

    return mix, sample_rate


def bounce(
    tracks: list[TrackConfig],
    output_path: str | Path,
    sample_rate: Optional[int] = None,
    subtype: str = "PCM_24",
) -> BounceResult:
    """Multitrack bounce — sum several tracks into one stereo file

    Args:
        tracks: list of track settings
        output_path: output file path
        sample_rate: target sample rate (None → use the first track's)
        subtype: output format (PCM_16, PCM_24, FLOAT, etc.)

    Returns:
        BounceResult
    """
    start = time.monotonic()

    mix, sr = mix_tracks(tracks, sample_rate)
    peak = float(np.max(np.abs(mix)))

    write_audio(output_path, mix, sr, subtype=subtype)
    output_stats = get_audio_stats(mix, sr)

    elapsed = time.monotonic() - start
    return BounceResult(
        output_path=str(output_path),
        track_count=len(tracks),
        tracks=[t.to_dict() for t in tracks],
        output_stats=asdict(output_stats),
        sample_rate=sr,
        duration_seconds=round(elapsed, 3),
        clipping_detected=peak > 1.0,
    )


def mixdown(
    tracks: list[TrackConfig],
    output_path: str | Path,
    master_chain: Optional[list[PipelineStep]] = None,
    sample_rate: Optional[int] = None,
    subtype: str = "PCM_24",
    compensate_latency: bool = True,
) -> MixdownResult:
    """Multitrack mixdown — bounce + master chain applied

    Args:
        tracks: list of track settings
        output_path: output file path
        master_chain: master bus plugin chain
        sample_rate: target sample rate
        subtype: output format
        compensate_latency: whether to apply delay compensation for the master chain
    """
    start = time.monotonic()

    mix, sr = mix_tracks(tracks, sample_rate)
    master_latency = 0

    if master_chain:
        # apply the master chain
        if compensate_latency:
            from audioman.core.latency import measure_chain_latency, apply_delay_compensation
            measurements, master_latency = measure_chain_latency(
                master_chain, sample_rate=sr,
            )

        mix = _apply_track_chain(mix, sr, master_chain)

        if compensate_latency and master_latency > 0:
            mix = apply_delay_compensation(mix, master_latency)

    peak = float(np.max(np.abs(mix)))
    write_audio(output_path, mix, sr, subtype=subtype)
    output_stats = get_audio_stats(mix, sr)

    elapsed = time.monotonic() - start
    return MixdownResult(
        output_path=str(output_path),
        track_count=len(tracks),
        tracks=[t.to_dict() for t in tracks],
        master_chain=[s.to_dict() for s in master_chain] if master_chain else None,
        master_latency_samples=master_latency,
        output_stats=asdict(output_stats),
        sample_rate=sr,
        duration_seconds=round(elapsed, 3),
        clipping_detected=peak > 1.0,
    )
