# Created: 2026-04-27
# Purpose: Voice Activity Detection — Silero-VAD wrapper with sample-rate adaptation
# Dependencies: silero-vad, torch, soxr (resample to 16k for detection)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    """샘플 인덱스 기반 구간. end는 exclusive."""
    start: int
    end: int
    kind: str  # "speech" | "noise"

    @property
    def duration_samples(self) -> int:
        return self.end - self.start

    def to_dict(self, sample_rate: int) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "start_sec": round(self.start / sample_rate, 3),
            "end_sec": round(self.end / sample_rate, 3),
            "duration_sec": round(self.duration_samples / sample_rate, 3),
            "kind": self.kind,
        }


_VAD_SR = 16000  # silero-vad 권장: 16kHz mono


def _to_mono_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """audioman (channels, samples) → mono 16kHz float32."""
    import soxr

    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio
    mono = mono.astype(np.float32, copy=False)
    if sample_rate != _VAD_SR:
        mono = soxr.resample(mono, sample_rate, _VAD_SR).astype(np.float32, copy=False)
    return mono


def detect_speech(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 200,
    speech_pad_ms: int = 80,
) -> list[Segment]:
    """Silero-VAD로 음성 구간 검출.

    Args:
        audio: (channels, samples) 또는 (samples,) — float32 권장
        sample_rate: 원본 SR. 16k가 아니면 내부적으로 다운샘플
        threshold: silero confidence (0~1). 0.5가 기본
        min_speech_ms: 이보다 짧은 음성 구간은 무시
        min_silence_ms: 이보다 짧은 무음은 음성 구간을 끊지 않음 (병합)
        speech_pad_ms: 검출된 구간 양쪽에 추가하는 padding

    Returns:
        kind="speech" Segment 목록 (원본 sample_rate 기준 인덱스).
        시간순 정렬, 겹치지 않음.
    """
    from silero_vad import load_silero_vad, get_speech_timestamps
    import torch

    mono16k = _to_mono_16k(audio, sample_rate)
    tensor = torch.from_numpy(mono16k)

    model = load_silero_vad()
    timestamps = get_speech_timestamps(
        tensor,
        model,
        threshold=threshold,
        sampling_rate=_VAD_SR,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )

    # 16k → 원본 SR 인덱스 변환
    n_samples = audio.shape[-1]
    ratio = sample_rate / _VAD_SR
    segments: list[Segment] = []
    for ts in timestamps:
        start = max(0, int(round(ts["start"] * ratio)))
        end = min(n_samples, int(round(ts["end"] * ratio)))
        if end > start:
            segments.append(Segment(start=start, end=end, kind="speech"))
    return segments


def invert_to_noise(
    speech_segments: list[Segment],
    total_samples: int,
) -> list[Segment]:
    """speech 구간의 보집합 → noise 구간."""
    out: list[Segment] = []
    cursor = 0
    for seg in speech_segments:
        if seg.start > cursor:
            out.append(Segment(start=cursor, end=seg.start, kind="noise"))
        cursor = max(cursor, seg.end)
    if cursor < total_samples:
        out.append(Segment(start=cursor, end=total_samples, kind="noise"))
    return out


def merge_segments(
    speech: list[Segment],
    noise: list[Segment],
) -> list[Segment]:
    """speech + noise를 시간순으로 정렬해 하나의 timeline으로."""
    all_segs = list(speech) + list(noise)
    all_segs.sort(key=lambda s: s.start)
    return all_segs
