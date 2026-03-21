# Created: 2026-03-21
# Purpose: 오디오 파일 I/O 추상화

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class AudioStats:
    """오디오 파일 통계"""
    duration: float  # 초
    sample_rate: int
    channels: int
    frames: int
    peak: float
    rms: float
    format: str


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """오디오 파일 읽기. 반환: (audio shape (channels, samples), sample_rate)"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile: (samples, channels) → (channels, samples) for pedalboard
    audio = data.T
    logger.debug(f"읽기: {path.name} ({audio.shape[0]}ch, {sr}Hz, {audio.shape[1]} samples)")
    return audio, sr


def write_audio(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    subtype: str = "PCM_24",
) -> None:
    """오디오 파일 쓰기. audio shape: (channels, samples)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # (channels, samples) → (samples, channels) for soundfile
    if audio.ndim == 1:
        data = audio
    else:
        data = audio.T

    sf.write(str(path), data, sample_rate, subtype=subtype)
    logger.debug(f"쓰기: {path.name} ({sample_rate}Hz, {subtype})")


def get_audio_stats(audio: np.ndarray, sample_rate: int) -> AudioStats:
    """오디오 데이터 통계 계산"""
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
