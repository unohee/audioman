# Created: 2026-05-31
# Purpose: DAW 재생 환경을 재현하는 블록 단위 결정적 처리 엔진.
#
# 실제 DAW(Ableton 등)는 오디오를 고정 블록 크기(128/256/512/1024 samples)로
# 콜백마다 plugin.process()를 호출하며, 블록 사이에 플러그인 내부 상태
# (필터 히스토리, lookahead 버퍼, 파라미터 스무딩)가 연속 유지된다.
#
# audioman의 기존 process_file은 전체 버퍼를 한 번에 통과시킨다(오프라인 렌더).
# 이 모듈은 그 둘의 차이를 노출한다:
#   - render_offline:  whole-buffer 1회 process (ground truth)
#   - render_streamed: 고정 블록으로 연속 process(reset=False) — 올바른 DAW 재현
#   - reset_per_block=True 옵션: 매 블록 reset (상태 끊김 시뮬레이션 = 클릭 재현)
#
# pedalboard 실측(0.9.22): reset=False 연속 호출은 내부 상태를 유지하며,
# Reverb 등은 whole-buffer와 비트 단위로 일치(-600dB). 그러나 Delay/Chorus/
# Compressor는 블록 처리 자체가 오프라인과 미세하게 달라질 수 있어, 이 차이를
# discontinuity 모듈이 triage 한다.

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# DAW에서 흔히 쓰이는 블록 크기. 벤치마크 sweep 기본값.
COMMON_BLOCK_SIZES = (64, 128, 256, 512, 1024, 2048)


@dataclass
class BlockTiming:
    """단일 블록의 처리 시간 기록."""
    index: int
    n_samples: int
    process_sec: float           # 이 블록을 처리하는 데 걸린 실시간
    deadline_sec: float          # n_samples / sample_rate (실시간 마감)

    @property
    def rt_factor(self) -> float:
        """처리시간 / 마감. >1.0 이면 실시간 추종 실패(xrun)."""
        return self.process_sec / self.deadline_sec if self.deadline_sec > 0 else float("inf")

    @property
    def is_xrun(self) -> bool:
        return self.rt_factor > 1.0


@dataclass
class StreamResult:
    """블록 스트리밍 처리 결과."""
    audio: np.ndarray                       # (channels, samples) 처리된 출력
    sample_rate: int
    block_size: int
    reset_per_block: bool
    timings: list[BlockTiming] = field(default_factory=list)
    total_process_sec: float = 0.0

    # 처리 모드 식별 (디버깅/리포트용)
    mode: str = "streamed"                  # "offline" | "streamed"

    def to_summary(self) -> dict[str, Any]:
        """오디오 데이터를 뺀 메트릭 요약 (JSON 직렬화용)."""
        n = self.audio.shape[1] if self.audio.ndim == 2 else len(self.audio)
        audio_sec = n / self.sample_rate if self.sample_rate else 0.0
        return {
            "mode": self.mode,
            "block_size": self.block_size,
            "reset_per_block": self.reset_per_block,
            "sample_rate": self.sample_rate,
            "blocks": len(self.timings),
            "audio_seconds": round(audio_sec, 4),
            "total_process_sec": round(self.total_process_sec, 6),
            "realtime_factor": round(self.total_process_sec / audio_sec, 4) if audio_sec > 0 else None,
        }


# 처리 함수 시그니처: (block (channels, n), sample_rate, reset: bool) -> block
ProcessFn = Callable[[np.ndarray, int, bool], np.ndarray]


def _as_2d(audio: np.ndarray) -> np.ndarray:
    """(samples,) → (1, samples), float32 보장."""
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio


def render_offline(
    audio: np.ndarray,
    sample_rate: int,
    process_fn: ProcessFn,
) -> StreamResult:
    """전체 버퍼를 한 번에 처리 — DAW 'freeze/bounce'에 해당하는 ground truth.

    process_fn은 (audio, sr, reset)을 받는다. 오프라인은 reset=True 1회.
    """
    audio = _as_2d(audio)
    start = time.perf_counter()
    out = process_fn(audio, sample_rate, True)
    elapsed = time.perf_counter() - start
    out = _as_2d(np.asarray(out))
    return StreamResult(
        audio=out,
        sample_rate=sample_rate,
        block_size=audio.shape[1],
        reset_per_block=False,
        timings=[BlockTiming(0, audio.shape[1], elapsed, audio.shape[1] / sample_rate)],
        total_process_sec=elapsed,
        mode="offline",
    )


def render_streamed(
    audio: np.ndarray,
    sample_rate: int,
    process_fn: ProcessFn,
    block_size: int = 512,
    reset_per_block: bool = False,
    reset_first: bool = True,
) -> StreamResult:
    """고정 블록 크기로 연속 처리 — DAW 실시간 콜백 재현.

    Args:
        block_size: 블록당 샘플 수 (DAW 버퍼 크기).
        reset_per_block: True면 매 블록 plugin reset — 상태 단절 버그 시뮬레이션.
                         실제 DAW는 False(연속)지만, 일부 잘못 구현된 플러그인/
                         호스트는 블록마다 상태가 끊겨 경계 클릭을 낸다.
        reset_first: True(기본)면 첫 블록을 reset 상태에서 시작 — DAW가 재생 시작 시
                     플러그인을 reset하는 동작 재현. False면 process_fn에 남아 있는
                     이전 상태(tail)를 물고 시작 → 재생 시작 지점 클릭 재현.
                     단, reset_per_block=True면 이 값과 무관하게 매 블록 reset.

    각 블록의 처리 시간을 perf_counter로 측정해 RT factor / xrun을 산출한다.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    audio = _as_2d(audio)
    n_ch, n = audio.shape

    out_chunks: list[np.ndarray] = []
    timings: list[BlockTiming] = []
    total = 0.0

    idx = 0
    pos = 0
    while pos < n:
        block = audio[:, pos:pos + block_size]
        bn = block.shape[1]
        # 첫 블록만 reset_first 적용, 이후는 reset_per_block 따름
        reset = reset_per_block or (idx == 0 and reset_first)
        start = time.perf_counter()
        processed = process_fn(block, sample_rate, reset)
        elapsed = time.perf_counter() - start

        processed = _as_2d(np.asarray(processed))
        out_chunks.append(processed)
        timings.append(BlockTiming(
            index=idx,
            n_samples=bn,
            process_sec=elapsed,
            deadline_sec=bn / sample_rate,
        ))
        total += elapsed
        pos += block_size
        idx += 1

    out = np.concatenate(out_chunks, axis=1) if out_chunks else np.zeros((n_ch, 0), dtype=np.float32)

    return StreamResult(
        audio=out,
        sample_rate=sample_rate,
        block_size=block_size,
        reset_per_block=reset_per_block,
        timings=timings,
        total_process_sec=total,
        mode="streamed",
    )


def make_pedalboard_process_fn(board) -> ProcessFn:
    """pedalboard.Pedalboard 또는 단일 Plugin을 ProcessFn으로 감싼다.

    pedalboard의 process(audio, sr, reset=...)를 그대로 호출한다. reset=False면
    이전 블록의 내부 상태를 유지하므로 올바른 스트리밍이 된다.
    """
    def fn(block: np.ndarray, sr: int, reset: bool) -> np.ndarray:
        return board.process(block, sr, reset=reset)
    return fn


def make_wrapper_process_fn(wrapper) -> ProcessFn:
    """VST3PluginWrapper를 ProcessFn으로 감싼다.

    reset 플래그를 wrapper.process로 그대로 전달한다. 스트리밍(reset=False)에서는
    블록 사이 상태가 연속 유지되고, reset=True면 매 호출마다 상태가 끊긴다.
    """
    def fn(block: np.ndarray, sr: int, reset: bool) -> np.ndarray:
        return wrapper.process(block, sr, reset=reset)
    return fn
