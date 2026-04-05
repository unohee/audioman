# Created: 2026-04-05
# Purpose: 플러그인 레이턴시 측정 + auto delay compensation

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
    """플러그인 레이턴시 측정 결과"""
    plugin_name: str
    reported_latency: int       # pedalboard 보고값 (samples)
    measured_latency: int       # 임펄스 라운드트립 측정값 (samples)
    confidence: float           # 측정 신뢰도 (0.0 ~ 1.0)
    used_latency: int           # 최종 사용할 값

    def to_dict(self) -> dict:
        return asdict(self)


def _get_reported_latency(wrapper: VST3PluginWrapper) -> int:
    """pedalboard 플러그인의 보고된 레이턴시 읽기"""
    try:
        # pedalboard의 내부 속성 접근
        plugin = wrapper._plugin
        if hasattr(plugin, "latency_samples"):
            return int(plugin.latency_samples)
    except Exception:
        pass
    return 0


def measure_plugin_latency(
    wrapper: VST3PluginWrapper,
    sample_rate: int = 48000,
    channels: int = 2,
    test_duration_sec: float = 1.0,
) -> LatencyMeasurement:
    """단일 플러그인의 레이턴시를 임펄스 라운드트립으로 측정

    알고리즘:
        1. sample[0]에 delta 임펄스 생성
        2. 플러그인 상태 리셋 후 통과
        3. 출력에서 피크 위치 = 레이턴시 (samples)
        4. 피크/노이즈플로어 비율로 confidence 계산
    """
    wrapper.load()
    wrapper.reset()

    # 임펄스 생성 (sample 0에 1.0)
    impulse = generate_impulse(
        sample_rate=sample_rate,
        duration_sec=test_duration_sec,
        channels=channels,
    )

    # 플러그인 통과
    output = wrapper.process(impulse, sample_rate)

    # 모노로 합산하여 분석
    if output.ndim == 2:
        mono = np.mean(output, axis=0)
    else:
        mono = output

    abs_mono = np.abs(mono)

    # 피크 위치 = 레이턴시
    peak_idx = int(np.argmax(abs_mono))
    peak_val = float(abs_mono[peak_idx])

    # 노이즈 플로어 추정 (피크 주변 ±100 샘플 제외)
    mask = np.ones(len(abs_mono), dtype=bool)
    exclude_start = max(0, peak_idx - 100)
    exclude_end = min(len(abs_mono), peak_idx + 100)
    mask[exclude_start:exclude_end] = False

    if np.any(mask):
        noise_floor = float(np.mean(abs_mono[mask]))
    else:
        noise_floor = 0.0

    # confidence: 피크 대비 노이즈 플로어 비율
    if noise_floor > 0:
        snr = peak_val / noise_floor
        # SNR 20 이상이면 confidence 1.0, 1 이하면 0.0
        confidence = float(np.clip((snr - 1.0) / 19.0, 0.0, 1.0))
    elif peak_val > 1e-6:
        confidence = 1.0
    else:
        confidence = 0.0

    # pedalboard 보고값
    reported = _get_reported_latency(wrapper)

    # 최종 레이턴시 결정
    if confidence >= 0.5:
        used = peak_idx
    elif reported > 0:
        used = reported
        logger.warning(
            f"{wrapper.name}: 임펄스 측정 신뢰도 낮음 (confidence={confidence:.2f}), "
            f"보고된 레이턴시 사용: {reported} samples"
        )
    else:
        used = peak_idx
        logger.warning(
            f"{wrapper.name}: 레이턴시 측정 불확실 (confidence={confidence:.2f}), "
            f"측정값 사용: {peak_idx} samples"
        )

    # reported vs measured 불일치 경고
    if reported > 0 and abs(reported - peak_idx) > 1:
        logger.info(
            f"{wrapper.name}: 보고 레이턴시({reported}) ≠ 측정 레이턴시({peak_idx}), "
            f"측정값 사용 (confidence={confidence:.2f})"
        )

    measurement = LatencyMeasurement(
        plugin_name=wrapper.name,
        reported_latency=reported,
        measured_latency=peak_idx,
        confidence=round(confidence, 4),
        used_latency=used,
    )

    logger.debug(
        f"레이턴시 측정: {wrapper.name} → "
        f"measured={peak_idx}, reported={reported}, "
        f"confidence={confidence:.2f}, used={used}"
    )

    return measurement


def measure_chain_latency(
    steps: list[dict[str, Any]],
    sample_rate: int = 48000,
) -> tuple[list[LatencyMeasurement], int]:
    """체인의 각 플러그인 레이턴시 측정 + 총 합산

    Args:
        steps: [{"plugin_name": str, "params": dict}, ...]
            PipelineStep.to_dict() 형식 또는 PipelineStep 객체
        sample_rate: 측정 시 사용할 샘플레이트

    Returns:
        (measurements, total_latency_samples)
    """
    from audioman.core.pipeline import PipelineStep

    registry = get_registry()
    measurements = []
    total = 0

    for step in steps:
        # PipelineStep 객체 또는 dict 모두 지원
        if isinstance(step, PipelineStep):
            plugin_name = step.plugin_name
            params = step.params
        else:
            plugin_name = step.get("plugin", step.get("plugin_name", ""))
            params = step.get("params", {})

        meta = registry.get(plugin_name)
        if not meta:
            raise ValueError(f"플러그인을 찾을 수 없습니다: '{plugin_name}'")

        wrapper = VST3PluginWrapper(meta.path)
        wrapper.load()

        if params:
            wrapper.set_parameters(params)

        measurement = measure_plugin_latency(wrapper, sample_rate)
        measurement.plugin_name = meta.short_name
        measurements.append(measurement)
        total += measurement.used_latency

    logger.info(f"체인 총 레이턴시: {total} samples ({total / sample_rate * 1000:.1f}ms)")
    return measurements, total


def apply_delay_compensation(
    audio: np.ndarray,
    total_latency_samples: int,
) -> np.ndarray:
    """레이턴시만큼 앞부분 제거 + 뒤에 zero-pad (원본 길이 유지)

    Args:
        audio: (channels, samples) 또는 (samples,) 형태
        total_latency_samples: 보상할 레이턴시 (samples)

    Returns:
        보상된 오디오 (원본과 동일 shape)
    """
    if total_latency_samples <= 0:
        return audio

    if audio.ndim == 1:
        n = len(audio)
        if total_latency_samples >= n:
            logger.warning(
                f"레이턴시({total_latency_samples})가 오디오 길이({n})보다 큼, "
                f"전체 silence 반환"
            )
            return np.zeros_like(audio)
        # 앞부분 제거 + 뒤에 zero-pad
        compensated = np.zeros(n, dtype=audio.dtype)
        remaining = n - total_latency_samples
        compensated[:remaining] = audio[total_latency_samples:]
        return compensated
    else:
        channels, n = audio.shape
        if total_latency_samples >= n:
            logger.warning(
                f"레이턴시({total_latency_samples})가 오디오 길이({n})보다 큼, "
                f"전체 silence 반환"
            )
            return np.zeros_like(audio)
        compensated = np.zeros((channels, n), dtype=audio.dtype)
        remaining = n - total_latency_samples
        compensated[:, :remaining] = audio[:, total_latency_samples:]
        return compensated
