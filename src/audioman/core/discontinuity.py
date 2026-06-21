# Created: 2026-05-31
# Purpose: 블록 스트리밍 출력에서 클릭/불연속을 검출하고 triage 한다.
#
# 동기: 실제 DAW(Ableton 등)에서 클릭이 나는데, audioman의 단순 블록 스트리밍
# 비교만으로는 어디서 왜 나는지 triage가 안 됐다. 이 모듈은 두 방향으로 잡는다.
#
#   1) 출력 신호 자체의 불연속 검출 (기준 신호 없이) — sample diff spike.
#      그 위치가 블록 경계(block_size 배수)에 정렬되면 "스트리밍 상태 단절"로
#      분류한다. 이것이 reset-per-block / lookahead 끊김 / 파라미터 스무딩 끊김의
#      서명이다.
#
#   2) null test — offline(ground truth) vs streamed 출력의 차이. 차이가 큰
#      구간 = 블록 스트리밍이 오프라인 렌더와 갈리는 지점.
#
# 모든 결과는 core/findings.Finding 으로 반환되어 기존 envelope/스키마에 합류한다.

from __future__ import annotations

from typing import Optional

import numpy as np

from audioman.core.findings import (
    Category,
    Code,
    Finding,
    FixHint,
    Severity,
    Where,
)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def detect_nonfinite(
    audio: np.ndarray,
    sample_rate: int,
    *,
    file: Optional[str] = None,
) -> list[Finding]:
    """NaN/Inf 샘플 검출 — 플러그인이 발산했거나 미초기화 버퍼를 낸 경우."""
    bad = ~np.isfinite(audio)
    if not bad.any():
        return []
    flat = bad.any(axis=0) if audio.ndim == 2 else bad
    idx = np.where(flat)[0]
    return [Finding(
        code=Code.NONFINITE_SAMPLES,
        category=Category.PLUGIN,
        severity=Severity.CRITICAL,
        where=Where(
            file=file,
            start_sample=int(idx[0]),
            end_sample=int(idx[-1]) + 1,
            start_sec=round(float(idx[0]) / sample_rate, 6),
            end_sec=round(float(idx[-1] + 1) / sample_rate, 6),
        ),
        measurement={"nonfinite_samples": int(flat.sum())},
        hint=(f"{int(flat.sum())} non-finite (NaN/Inf) sample(s). Plugin diverged or "
              f"emitted uninitialized buffer — likely state/reset bug at a block edge."),
        fix_hint=FixHint(kind="manual", args=[], note="Check plugin reset/init at stream start"),
    )]


def detect_discontinuities(
    audio: np.ndarray,
    sample_rate: int,
    *,
    block_size: Optional[int] = None,
    file: Optional[str] = None,
    sigma: float = 8.0,
    min_jump: float = 0.02,
    edge_tolerance: int = 1,
) -> list[Finding]:
    """샘플 간 1차 차분 spike(클릭)를 검출하고 블록 경계 정렬 여부로 분류한다.

    알고리즘:
        d[n] = x[n] - x[n-1]. 클릭은 광대역 임펄스라 |d|가 국소적으로 급증한다.
        threshold = max(min_jump, sigma * MAD-based-std(|d|)). robust 통계로
        음악 신호의 정상 트랜지언트와 구분(MAD는 outlier에 둔감).

        block_size가 주어지면 각 spike 위치 n이 block_size 배수 ± edge_tolerance에
        들어가는지 본다. 정렬되면 BLOCK_ALIGNED → 스트리밍 상태 단절 서명.
        정렬 안 되면 콘텐츠 자체의 클릭(소스 결함)일 가능성.

    Args:
        sigma: robust std 대비 임계 배수. 높을수록 보수적(오검 적음).
        min_jump: 절대 최소 점프(선형 진폭). 조용한 구간 false positive 방지.
        edge_tolerance: 블록 경계 정렬 판정 허용 오차(samples).
    """
    mono = _to_mono(audio).astype(np.float64)
    if mono.size < 4:
        return []

    d = np.abs(np.diff(mono))
    # MAD 기반 robust scale (정상 트랜지언트에 둔감)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    robust_std = 1.4826 * mad if mad > 0 else float(np.std(d))
    threshold = max(min_jump, med + sigma * robust_std)

    spike_idx = np.where(d > threshold)[0] + 1  # x[n] - x[n-1] → 위치 n
    if spike_idx.size == 0:
        return []

    # 인접 spike(같은 클릭의 여러 샘플)를 하나의 이벤트로 묶기
    events: list[tuple[int, int, float]] = []  # (start, end, peak_jump)
    run_start = spike_idx[0]
    prev = spike_idx[0]
    peak = d[spike_idx[0] - 1]
    for s in spike_idx[1:]:
        if s - prev <= 2:
            peak = max(peak, d[s - 1])
            prev = s
        else:
            events.append((int(run_start), int(prev), float(peak)))
            run_start = s
            prev = s
            peak = d[s - 1]
    events.append((int(run_start), int(prev), float(peak)))

    findings: list[Finding] = []
    for start, end, peak_jump in events:
        aligned = False
        nearest_edge = None
        if block_size:
            r = start % block_size
            dist = min(r, block_size - r)
            if dist <= edge_tolerance:
                aligned = True
                nearest_edge = int(round(start / block_size)) * block_size

        peak_db = 20.0 * np.log10(peak_jump + 1e-30)
        if aligned:
            sev = Severity.CRITICAL
            hint = (f"Discontinuity at sample {start} aligns to block edge "
                    f"(block_size={block_size}, edge≈{nearest_edge}). Signature of a "
                    f"streaming state break — plugin reset/lookahead/parameter-smoothing "
                    f"discontinuity at the buffer boundary. This is the click you hear in the DAW.")
        else:
            sev = Severity.WARN
            hint = (f"Discontinuity (jump {peak_jump:.4f}, {peak_db:+.1f} dB) at sample {start}, "
                    f"not block-aligned — likely a click in the source content, not a streaming bug.")

        findings.append(Finding(
            code=Code.CLICK_DENSITY,
            category=Category.SIGNAL,
            severity=sev,
            where=Where(
                file=file,
                start_sample=start,
                end_sample=end + 1,
                start_sec=round(start / sample_rate, 6),
                end_sec=round((end + 1) / sample_rate, 6),
            ),
            measurement={
                "peak_jump": round(peak_jump, 6),
                "peak_jump_db": round(peak_db, 2),
                "threshold": round(threshold, 6),
                "block_aligned": aligned,
                "block_size": block_size,
                "nearest_block_edge": nearest_edge,
            },
            hint=hint,
            fix_hint=FixHint(
                kind="manual",
                args=[],
                note=("Stream with reset_per_block=False / verify PDC; ensure host resets "
                      "plugin only at transport start" if aligned else "Inspect source audio"),
            ),
        ))

    return findings


def null_test(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    latency_samples: int = 0,
    file: Optional[str] = None,
    threshold_db: float = -60.0,
) -> list[Finding]:
    """reference(오프라인 렌더) 대비 candidate(스트리밍) 차이를 측정한다.

    latency_samples만큼 candidate를 당겨 정렬(PDC 보상)한 뒤 차이를 본다.
    최대 차이가 threshold_db를 넘으면 finding. 넘지 않으면 빈 리스트(= 일치).

    Returns: 차이가 유의미하면 길이 1, 아니면 0.
    """
    ref = _to_mono(reference).astype(np.float64)
    cand = _to_mono(candidate).astype(np.float64)

    if latency_samples > 0:
        cand = cand[latency_samples:]
    n = min(len(ref), len(cand))
    if n == 0:
        return []
    diff = ref[:n] - cand[:n]

    max_abs = float(np.max(np.abs(diff)))
    max_db = 20.0 * np.log10(max_abs + 1e-30)
    rms = float(np.sqrt(np.mean(diff ** 2)))
    rms_db = 20.0 * np.log10(rms + 1e-30)

    if max_db < threshold_db:
        return []

    peak_idx = int(np.argmax(np.abs(diff)))
    sev = Severity.CRITICAL if max_db > -20.0 else Severity.WARN
    return [Finding(
        code=Code.SAMPLE_DROPOUT,
        category=Category.PLUGIN,
        severity=sev,
        where=Where(
            file=file,
            start_sample=peak_idx,
            start_sec=round(peak_idx / sample_rate, 6),
        ),
        measurement={
            "max_diff_db": round(max_db, 2),
            "rms_diff_db": round(rms_db, 2),
            "latency_compensated_samples": latency_samples,
            "compared_samples": n,
        },
        hint=(f"Streamed output diverges from offline render by up to {max_db:+.1f} dB "
              f"(RMS {rms_db:+.1f} dB) at sample {peak_idx}. Block streaming is NOT "
              f"bit-equivalent to the offline bounce — investigate state continuity / PDC."),
        fix_hint=FixHint(kind="manual", args=[], note="Compare block sizes; check latency reporting"),
    )]
