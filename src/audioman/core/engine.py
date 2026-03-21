# Created: 2026-03-21
# Purpose: 단일 플러그인 오디오 처리 엔진

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from audioman.core.audio_file import AudioStats, get_audio_stats, read_audio, write_audio
from audioman.core.registry import get_registry
from audioman.plugins.vst3 import VST3PluginWrapper

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """처리 결과"""
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
    """CLI 파라미터 문자열 파싱: ["threshold=-20", "reduction=12"] → dict"""
    params = {}
    for s in param_strings:
        if "=" not in s:
            raise ValueError(f"잘못된 파라미터 형식 (key=value 필요): '{s}'")
        key, value = s.split("=", 1)
        key = key.strip()

        # 타입 추론
        if value.lower() in ("true", "false"):
            params[key] = value.lower() == "true"
        else:
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value  # 문자열 (enum 등)

    return params


def process_file(
    input_path: str | Path,
    output_path: str | Path,
    plugin_name: str,
    params: Optional[dict[str, Any]] = None,
    passes: int = 1,
) -> ProcessResult:
    """단일 플러그인으로 오디오 파일 처리

    Args:
        passes: 처리 횟수. 2 이상이면 첫 패스는 학습용, 마지막 패스만 출력.
                adaptive 모드 플러그인에서 노이즈 프로파일 학습에 유용.
    """
    start = time.monotonic()

    # 플러그인 검색
    registry = get_registry()
    meta = registry.get(plugin_name)
    if not meta:
        raise ValueError(f"플러그인을 찾을 수 없습니다: '{plugin_name}'")

    # 오디오 읽기
    audio, sr = read_audio(input_path)
    input_stats = get_audio_stats(audio, sr)

    # 플러그인 로드 + 파라미터 설정
    wrapper = VST3PluginWrapper(meta.path)
    wrapper.load()

    if params:
        wrapper.set_parameters(params)

    # 멀티패스 처리
    output = audio
    for i in range(passes):
        logger.info(f"Pass {i+1}/{passes}")
        output = wrapper.process(audio, sr)

    output_stats = get_audio_stats(output, sr)

    # 출력 저장
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
