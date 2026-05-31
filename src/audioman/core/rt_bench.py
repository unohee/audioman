# Created: 2026-05-31
# Purpose: 블록 스트리밍의 실시간 CPU 부하 벤치마크.
#
# DAW에서 플러그인이 "무겁다"는 건 블록당 처리시간이 그 블록의 실시간 길이
# (block_size / sample_rate)를 잡아먹는다는 뜻이다. 처리시간 > 마감이면 xrun
# (오디오 드롭아웃/클릭)이 난다. 이 모듈은 core/streaming의 BlockTiming을 받아
# RT factor 분포, p50/p95/p99/max, xrun 수, 추정 동시 트랙 수를 산출한다.
#
# 단일 블록 timing은 OS 스케줄링 지터에 민감하므로, percentile과 worst-case를
# 함께 본다 (DAW에서 중요한 건 평균이 아니라 worst-case — 한 블록만 늦어도 클릭).

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from audioman.core.streaming import StreamResult


@dataclass
class RTBenchReport:
    """단일 (플러그인, 블록 크기) 조합의 실시간 성능 리포트."""
    block_size: int
    sample_rate: int
    blocks: int
    audio_seconds: float

    rt_factor_mean: float       # 전체 처리시간 / 전체 오디오 길이
    rt_factor_p50: float        # 블록별 RT factor 중앙값
    rt_factor_p95: float
    rt_factor_p99: float
    rt_factor_max: float        # worst-case — DAW 클릭 여부를 좌우

    block_ms_mean: float        # 블록당 처리시간 평균 (ms)
    block_ms_max: float
    deadline_ms: float          # block_size / sr * 1000

    xruns: int                  # rt_factor > 1.0 인 블록 수
    xrun_ratio: float           # xruns / blocks
    est_max_tracks: int         # 1/p99_rt_factor — 동시에 돌릴 수 있는 트랙 추정

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark(result: StreamResult, *, warmup_blocks: int = 1) -> RTBenchReport:
    """StreamResult의 블록 타이밍을 RT 성능 리포트로 집계한다.

    Args:
        warmup_blocks: 첫 N개 블록을 제외(첫 블록은 JIT/캐시 워밍업으로 느려
                       worst-case를 오염시킨다). 블록 수가 충분할 때만 적용.
    """
    timings = result.timings
    if warmup_blocks > 0 and len(timings) > warmup_blocks * 2:
        timings = timings[warmup_blocks:]
    if not timings:
        raise ValueError("StreamResult has no block timings to benchmark")

    rt = np.array([t.rt_factor for t in timings], dtype=np.float64)
    proc_ms = np.array([t.process_sec * 1000.0 for t in timings], dtype=np.float64)
    deadline_ms = result.block_size / result.sample_rate * 1000.0

    n = result.audio.shape[1] if result.audio.ndim == 2 else len(result.audio)
    audio_sec = n / result.sample_rate if result.sample_rate else 0.0
    total_proc = sum(t.process_sec for t in result.timings)

    p99 = float(np.percentile(rt, 99))
    xruns = int(np.sum(rt > 1.0))

    return RTBenchReport(
        block_size=result.block_size,
        sample_rate=result.sample_rate,
        blocks=len(timings),
        audio_seconds=round(audio_sec, 4),
        rt_factor_mean=round(total_proc / audio_sec, 6) if audio_sec > 0 else float("inf"),
        rt_factor_p50=round(float(np.percentile(rt, 50)), 6),
        rt_factor_p95=round(float(np.percentile(rt, 95)), 6),
        rt_factor_p99=round(p99, 6),
        rt_factor_max=round(float(np.max(rt)), 6),
        block_ms_mean=round(float(np.mean(proc_ms)), 4),
        block_ms_max=round(float(np.max(proc_ms)), 4),
        deadline_ms=round(deadline_ms, 4),
        xruns=xruns,
        xrun_ratio=round(xruns / len(timings), 4),
        # p99 기준으로 안전하게 돌릴 수 있는 동시 트랙 수 (worst-case 여유)
        est_max_tracks=int(1.0 / p99) if p99 > 0 else 0,
    )
