# Created: 2026-04-05
# Purpose: 멀티트랙 믹싱 엔진 — bounce / mixdown

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
    """개별 트랙 설정"""
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
    """바운스 결과"""
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
    """믹스다운 결과 (bounce + 마스터 체인)"""
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
    """Equal Power Pan Law 적용

    Args:
        audio_stereo: (2, samples) 스테레오 오디오
        pan: -1.0 (L) ~ 0.0 (C) ~ 1.0 (R)

    Returns:
        (2, samples) 패닝 적용된 스테레오
    """
    pan = float(np.clip(pan, -1.0, 1.0))
    # pan 값을 0~pi/2 각도로 변환
    angle = (pan + 1.0) * 0.25 * np.pi
    gain_l = float(np.cos(angle))
    gain_r = float(np.sin(angle))

    out = audio_stereo.copy()
    out[0] *= gain_l
    out[1] *= gain_r
    return out


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """모노를 스테레오로 변환, 이미 스테레오면 그대로"""
    if audio.ndim == 1:
        return np.stack([audio, audio])
    if audio.shape[0] == 1:
        return np.concatenate([audio, audio], axis=0)
    if audio.shape[0] == 2:
        return audio
    # 3ch 이상 → 처음 2채널만 사용
    logger.warning(f"{audio.shape[0]}ch 오디오 → 처음 2채널만 사용")
    return audio[:2]


def _resample_if_needed(
    audio: np.ndarray,
    current_sr: int,
    target_sr: int,
) -> np.ndarray:
    """샘플레이트 불일치 시 soxr로 리샘플링"""
    if current_sr == target_sr:
        return audio

    import soxr

    # soxr는 (samples, channels) 형태를 기대
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
    """트랙별 플러그인 체인 적용 (인메모리, 파일 I/O 없음)"""
    registry = get_registry()

    for step in chain:
        meta = registry.get(step.plugin_name)
        if not meta:
            raise ValueError(f"플러그인을 찾을 수 없습니다: '{step.plugin_name}'")

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
    """여러 트랙을 스테레오로 믹스

    Args:
        tracks: 트랙 설정 리스트
        sample_rate: 목표 샘플레이트 (None이면 첫 트랙 기준)
        apply_chain: 트랙별 플러그인 체인 적용 여부

    Returns:
        (audio (2, samples), sample_rate)
    """
    if not tracks:
        raise ValueError("트랙이 없습니다")

    # Solo 필터링: solo 트랙이 하나라도 있으면 solo만 재생
    has_solo = any(t.solo for t in tracks)
    active_tracks = []
    for t in tracks:
        if t.mute:
            continue
        if has_solo and not t.solo:
            continue
        active_tracks.append(t)

    if not active_tracks:
        logger.warning("모든 트랙이 mute 상태, silence 출력")
        # 첫 트랙의 정보로 빈 오디오 생성
        sr = sample_rate or 48000
        return np.zeros((2, sr), dtype=np.float32), sr

    # 각 트랙 로드
    loaded: list[tuple[np.ndarray, int, TrackConfig]] = []
    for t in active_tracks:
        audio, sr = read_audio(t.path)
        loaded.append((audio, sr, t))

    # 목표 SR 결정
    if sample_rate is None:
        sample_rate = loaded[0][1]

    # 각 트랙 처리
    processed: list[np.ndarray] = []
    for audio, sr, track_cfg in loaded:
        # 리샘플링
        audio = _resample_if_needed(audio, sr, sample_rate)

        # 트랙별 플러그인 체인
        if apply_chain and track_cfg.chain:
            audio = _apply_track_chain(audio, sample_rate, track_cfg.chain)

        # 스테레오 변환
        audio = _ensure_stereo(audio)

        # 게인 적용
        if track_cfg.gain_db != 0.0:
            audio = apply_gain(audio, track_cfg.gain_db)

        # 패닝 적용 (center에서도 equal power law로 ~0.707 gain)
        audio = apply_pan(audio, track_cfg.pan)

        # 오프셋 적용 (앞에 silence 삽입)
        if track_cfg.offset_samples > 0:
            pad = np.zeros((2, track_cfg.offset_samples), dtype=audio.dtype)
            audio = np.concatenate([pad, audio], axis=1)

        processed.append(audio)

    # 길이 정렬 (가장 긴 트랙 기준 zero-pad)
    max_len = max(a.shape[1] for a in processed)
    aligned = []
    for a in processed:
        if a.shape[1] < max_len:
            pad_len = max_len - a.shape[1]
            a = np.pad(a, ((0, 0), (0, pad_len)), mode="constant")
        aligned.append(a)

    # 합산
    mix = np.sum(np.stack(aligned), axis=0).astype(np.float32)

    # 클리핑 체크
    peak = float(np.max(np.abs(mix)))
    if peak > 1.0:
        logger.warning(
            f"클리핑 감지: peak={peak:.3f} ({20 * np.log10(peak):.1f} dBFS). "
            f"마스터 체인에 리미터를 추가하거나 트랙 볼륨을 낮추세요."
        )

    return mix, sample_rate


def bounce(
    tracks: list[TrackConfig],
    output_path: str | Path,
    sample_rate: Optional[int] = None,
    subtype: str = "PCM_24",
) -> BounceResult:
    """멀티트랙 바운스 — 여러 트랙을 하나의 스테레오 파일로 합산

    Args:
        tracks: 트랙 설정 리스트
        output_path: 출력 파일 경로
        sample_rate: 목표 SR (None이면 첫 트랙 기준)
        subtype: 출력 포맷 (PCM_16, PCM_24, FLOAT 등)

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
    """멀티트랙 믹스다운 — bounce + 마스터 체인 적용

    Args:
        tracks: 트랙 설정 리스트
        output_path: 출력 파일 경로
        master_chain: 마스터 버스 플러그인 체인
        sample_rate: 목표 SR
        subtype: 출력 포맷
        compensate_latency: 마스터 체인의 delay compensation 적용 여부
    """
    start = time.monotonic()

    mix, sr = mix_tracks(tracks, sample_rate)
    master_latency = 0

    if master_chain:
        # 마스터 체인 적용
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
