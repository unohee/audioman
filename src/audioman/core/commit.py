# Created: 2026-04-05
# Purpose: Destructive commit — 플러그인 체인 적용 + auto delay compensation

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
    """커밋 결과"""
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
    """단일 파일에 플러그인 체인 적용 + delay compensation

    기존 pipeline.run_pipeline()과 달리:
    - 각 플러그인의 레이턴시를 사전 측정
    - 최종 출력에서 누적 레이턴시만큼 보상 (앞부분 제거 + zero-pad)
    - tail_trim으로 원본 길이 복원

    Args:
        input_path: 입력 오디오 파일
        output_path: 출력 파일 경로
        steps: 플러그인 체인 (PipelineStep 리스트)
        compensate_latency: delay compensation 적용 여부
        tail_trim: 플러그인이 추가한 tail을 원본 길이로 자르기
    """
    start = time.monotonic()
    registry = get_registry()

    # 오디오 읽기
    audio, sr = read_audio(input_path)
    input_stats = get_audio_stats(audio, sr)
    original_length = audio.shape[-1]

    # 레이턴시 측정
    measurements: list[LatencyMeasurement] = []
    total_latency = 0

    if compensate_latency:
        measurements, total_latency = measure_chain_latency(steps, sample_rate=sr)
        if total_latency > 0:
            logger.info(
                f"총 레이턴시: {total_latency} samples "
                f"({total_latency / sr * 1000:.1f}ms) — compensation 적용 예정"
            )

    # 플러그인 체인 순차 처리 (인메모리)
    for i, step in enumerate(steps):
        meta = registry.get(step.plugin_name)
        if not meta:
            raise ValueError(f"플러그인을 찾을 수 없습니다: '{step.plugin_name}' (step {i+1})")

        wrapper = VST3PluginWrapper(meta.path)
        wrapper.load()
        if step.params:
            wrapper.set_parameters(step.params)

        logger.info(f"Step {i+1}/{len(steps)}: {meta.short_name}")
        audio = wrapper.process(audio, sr)

    # Delay compensation
    if compensate_latency and total_latency > 0:
        audio = apply_delay_compensation(audio, total_latency)

    # Tail trim — 원본 길이로 복원
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
    """처리 없이 레이턴시 측정만 수행 (dry-run)

    Returns:
        (measurements, total_latency_samples)
    """
    return measure_chain_latency(steps, sample_rate=sample_rate)
