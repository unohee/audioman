# Created: 2026-03-21
# Purpose: Audio file I/O abstraction

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class AudioStats:
    """Audio file statistics"""
    duration: float  # seconds
    sample_rate: int
    channels: int
    frames: int
    peak: float
    rms: float
    format: str


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file. Returns: (audio shape (channels, samples), sample_rate)"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile: (samples, channels) → (channels, samples) for pedalboard
    audio = data.T
    logger.debug(f"read: {path.name} ({audio.shape[0]}ch, {sr}Hz, {audio.shape[1]} samples)")
    return audio, sr


def write_audio(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    subtype: str = "PCM_24",
) -> None:
    """Write an audio file. audio shape: (channels, samples)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # (channels, samples) → (samples, channels) for soundfile
    if audio.ndim == 1:
        data = audio
    else:
        data = audio.T

    sf.write(str(path), data, sample_rate, subtype=subtype)
    logger.debug(f"write: {path.name} ({sample_rate}Hz, {subtype})")


def get_file_info(path: str | Path) -> dict:
    """Read file metadata only, quickly (without loading the audio)"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    info = sf.info(str(path))
    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "format": info.format,
        "subtype": info.subtype,
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
    }


def stream_process(
    input_path: str | Path,
    output_path: str | Path,
    process_fn,
    chunk_seconds: float = 10.0,
    subtype: str = "PCM_24",
) -> dict:
    """Stream a large file through in chunks

    Args:
        process_fn: (audio_chunk: ndarray, sr: int) → ndarray processing function
        chunk_seconds: chunk size (seconds)

    Returns: {"frames_processed", "duration", "chunks"}
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds must be positive, got {chunk_seconds}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = sf.info(str(input_path))
    chunk_frames = int(chunk_seconds * info.samplerate)
    if chunk_frames <= 0:
        raise ValueError(
            f"chunk_seconds is too small for sample rate {info.samplerate}: {chunk_seconds}"
        )
    total_frames = info.frames
    frames_done = 0
    chunks = 0

    with sf.SoundFile(str(input_path), 'r') as infile:
        with sf.SoundFile(
            str(output_path), 'w',
            samplerate=info.samplerate,
            channels=info.channels,
            subtype=subtype,
        ) as outfile:
            while frames_done < total_frames:
                n_read = min(chunk_frames, total_frames - frames_done)
                data = infile.read(n_read, dtype="float32", always_2d=True)
                if len(data) == 0:
                    break

                # (samples, channels) → (channels, samples) conversion
                chunk = data.T
                processed = np.asarray(process_fn(chunk, info.samplerate))
                if processed.ndim == 1:
                    processed = processed.reshape(1, -1)
                if processed.shape != chunk.shape:
                    raise ValueError(
                        "process_fn must return audio with the input chunk shape: "
                        f"expected {chunk.shape}, got {processed.shape}"
                    )

                # (channels, samples) → (samples, channels) for writing
                outfile.write(processed.T)

                frames_done += len(data)
                chunks += 1

    logger.debug(f"streamed: {chunks} chunks, {frames_done} frames")
    return {
        "frames_processed": frames_done,
        "duration": frames_done / info.samplerate,
        "chunks": chunks,
        "sample_rate": info.samplerate,
    }


def get_audio_stats(audio: np.ndarray, sample_rate: int) -> AudioStats:
    """Compute audio data statistics"""
    if audio.ndim == 1:
        channels = 1
        samples = len(audio)
    else:
        channels, samples = audio.shape

    return AudioStats(
        duration=samples / sample_rate,
        sample_rate=sample_rate,
        channels=channels,
        frames=samples,
        peak=float(np.max(np.abs(audio))),
        rms=float(np.sqrt(np.mean(audio**2))),
        format="float32",
    )
