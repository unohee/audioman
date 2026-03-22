# Created: 2026-03-22
# Purpose: Vamp 플러그인 호스트 래퍼
# Dependencies: vamp (optional)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _import_vamp():
    """vamp 패키지 lazy import"""
    try:
        import vamp
        return vamp
    except ImportError:
        raise ImportError(
            "vamp 패키지가 필요합니다. 설치: pip install vamp\n"
            "Vamp 플러그인도 시스템에 설치되어 있어야 합니다.\n"
            "  macOS: brew install vamp-plugin-sdk qm-vamp-plugins"
        )


@dataclass
class VampResult:
    """Vamp 플러그인 실행 결과"""
    plugin_id: str
    output: str
    shape: str  # "list", "vector", "matrix"
    sample_rate: int
    data: Any  # dict from vamp.collect()


def list_plugins() -> list[str]:
    """설치된 Vamp 플러그인 목록"""
    vamp = _import_vamp()
    return sorted(vamp.list_plugins())


def get_plugin_outputs(plugin_id: str) -> dict:
    """플러그인의 출력 정보 조회"""
    vamp = _import_vamp()
    return vamp.get_outputs_of(plugin_id)


def run_plugin(
    audio: np.ndarray,
    sample_rate: int,
    plugin_id: str,
    output: str = "",
    parameters: dict[str, float] | None = None,
    block_size: int = 0,
    step_size: int = 0,
) -> VampResult:
    """Vamp 플러그인 실행

    Args:
        audio: (channels, samples) 또는 (samples,) — mono로 변환됨
        sample_rate: 샘플 레이트
        plugin_id: "library:plugin" 또는 "library:plugin:output" 형식
        output: 출력 이름 (빈 문자열이면 기본 출력)
        parameters: 플러그인 파라미터 {name: value}
        block_size: FFT 블록 크기 (0이면 플러그인 기본값)
        step_size: 홉 크기 (0이면 플러그인 기본값)

    Returns:
        VampResult
    """
    vamp = _import_vamp()

    # mono 변환 (vamp은 1D float32 배열 기대)
    if audio.ndim == 2:
        mono = audio.mean(axis=0).astype(np.float32)
    else:
        mono = audio.astype(np.float32)

    # plugin_id에서 output 분리 ("lib:plugin:output" 형식)
    parts = plugin_id.split(":")
    if len(parts) == 3 and not output:
        plugin_id = f"{parts[0]}:{parts[1]}"
        output = parts[2]

    kwargs: dict[str, Any] = {}
    if output:
        kwargs["output"] = output
    if parameters:
        kwargs["parameters"] = parameters
    if block_size > 0:
        kwargs["block_size"] = block_size
    if step_size > 0:
        kwargs["step_size"] = step_size

    result = vamp.collect(mono, sample_rate, plugin_id, **kwargs)

    # 결과 형태 판별
    if "matrix" in result:
        shape = "matrix"
    elif "vector" in result:
        shape = "vector"
    elif "list" in result:
        shape = "list"
    else:
        shape = "unknown"

    return VampResult(
        plugin_id=plugin_id,
        output=output,
        shape=shape,
        sample_rate=sample_rate,
        data=result,
    )


def result_to_frames_and_values(
    result: VampResult,
    sample_rate: int,
    hop_size: int = 512,
) -> tuple[list[int], list[float]]:
    """vector/list 결과를 (frames, values) 쌍으로 변환

    Returns:
        (frame_numbers, values) — SVL time values용
    """
    if result.shape == "vector":
        step, values = result.data["vector"]
        step_samples = int(round(float(step) * sample_rate))
        if step_samples == 0:
            step_samples = hop_size
        frames = [i * step_samples for i in range(len(values))]
        return frames, [float(v) for v in values]

    elif result.shape == "list":
        events = result.data["list"]
        frames = []
        values = []
        for ev in events:
            t = ev.get("timestamp", ev.get("time", 0))
            frame = int(round(float(t) * sample_rate))
            frames.append(frame)
            vals = ev.get("values", [])
            values.append(float(vals[0]) if len(vals) > 0 else 0.0)
        return frames, values

    else:
        raise ValueError(f"vector/list 변환 불가: shape={result.shape}")


def result_to_instants(
    result: VampResult,
    sample_rate: int,
) -> tuple[list[int], list[str]]:
    """list 결과를 (frames, labels) 쌍으로 변환

    Returns:
        (frame_numbers, labels) — SVL time instants용
    """
    if result.shape != "list":
        raise ValueError(f"time instants 변환은 list 결과만 지원: shape={result.shape}")

    events = result.data["list"]
    frames = []
    labels = []
    for ev in events:
        t = ev.get("timestamp", ev.get("time", 0))
        frame = int(round(float(t) * sample_rate))
        frames.append(frame)
        labels.append(str(ev.get("label", "")))
    return frames, labels


def result_to_matrix(
    result: VampResult,
) -> tuple[np.ndarray, int]:
    """matrix 결과를 (matrix, hop_samples) 쌍으로 변환

    Returns:
        (matrix[n_frames, n_bins], hop_size_in_samples) — SVL dense 3D용
    """
    if result.shape != "matrix":
        raise ValueError(f"matrix 변환 불가: shape={result.shape}")

    step, matrix = result.data["matrix"]
    hop_samples = int(round(float(step) * result.sample_rate))
    return np.array(matrix), hop_samples
