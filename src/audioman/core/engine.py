# Created: 2026-03-21
# Purpose: 단일 플러그인 오디오 처리 엔진

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from audioman.core.audio_file import AudioStats, get_audio_stats, get_file_info, read_audio, stream_process, write_audio
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
    """CLI 파라미터 문자열 파싱: ["threshold=-20", "reduction=12"] → dict

    Value가 따옴표로 감싸져 있으면(`key="4.00"` 또는 `key='4.00'`) 강제로
    문자열로 유지한다. UAD 플러그인처럼 enum 라벨이 `"4.00"`, `"0.97"` 같은
    2-자리 소수 문자열인 경우에 필요하다 — float 변환하면 `"4.0"`이 되어
    enum 리스트와 매칭 실패.
    """
    params = {}
    for s in param_strings:
        if "=" not in s:
            raise ValueError(f"잘못된 파라미터 형식 (key=value 필요): '{s}'")
        key, value = s.split("=", 1)
        key = key.strip()

        # 명시적 문자열: 따옴표 감싸면 원본 보존.
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            params[key] = value[1:-1]
            continue

        # 타입 추론
        if value.lower() in ("true", "false"):
            params[key] = value.lower() == "true"
        else:
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value  # 문자열 (enum 등)

    return params


STREAM_THRESHOLD_MB = 500  # 이 크기 이상이면 자동 스트리밍


def process_file(
    input_path: str | Path,
    output_path: str | Path,
    plugin_name: str,
    params: Optional[dict[str, Any]] = None,
    passes: int = 1,
    stream: bool | None = None,
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

    # 대용량 파일 → 자동 스트리밍
    if stream is None:
        try:
            info = get_file_info(input_path)
            stream = info["file_size_mb"] > STREAM_THRESHOLD_MB
        except Exception:
            stream = False

    if stream:
        return _process_file_streaming(input_path, output_path, meta, params, start)

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


def _process_file_streaming(input_path, output_path, meta, params, start) -> ProcessResult:
    """대용량 파일 스트리밍 처리 — 메모리에 전체 로드하지 않음"""
    wrapper = VST3PluginWrapper(meta.path)
    wrapper.load()
    if params:
        wrapper.set_parameters(params)

    def process_chunk(chunk, sr):
        return wrapper.process(chunk, sr)

    info = get_file_info(input_path)
    result = stream_process(input_path, output_path, process_chunk)

    elapsed = time.monotonic() - start
    logger.info(f"스트리밍 처리 완료: {result['chunks']} chunks, {elapsed:.1f}s")

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
