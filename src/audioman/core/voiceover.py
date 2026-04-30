# Created: 2026-04-27
# Purpose: 보이스오버 일괄 처리 — VAD → RX denoise → utterance LUFS leveling
# Dependencies: vad.py, loudness.py, RX 10 Voice De-noise (VST3)

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audioman.core.audio_file import read_audio, write_audio
from audioman.core.loudness import level_utterances, measure
from audioman.core.registry import get_registry
from audioman.core.vad import Segment, detect_speech, invert_to_noise
from audioman.plugins.vst3 import VST3PluginWrapper

logger = logging.getLogger(__name__)


@dataclass
class VoiceoverResult:
    input_path: str
    output_path: str | None
    sample_rate: int
    duration_sec: float
    n_speech_segments: int
    speech_total_sec: float
    noise_total_sec: float
    denoise_plugin: str | None
    leveling_meta: dict | None
    measured_in: dict
    measured_out: dict | None

    def to_dict(self) -> dict:
        return {
            "input": self.input_path,
            "output": self.output_path,
            "sample_rate": self.sample_rate,
            "duration_sec": round(self.duration_sec, 3),
            "n_speech_segments": self.n_speech_segments,
            "speech_total_sec": round(self.speech_total_sec, 3),
            "noise_total_sec": round(self.noise_total_sec, 3),
            "denoise_plugin": self.denoise_plugin,
            "leveling": self.leveling_meta,
            "measured_in": self.measured_in,
            "measured_out": self.measured_out,
        }


def analyze(
    input_path: str | Path,
    vad_threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 200,
) -> dict:
    """파일을 읽어 VAD만 실행, 통계와 segment 목록 반환 (편집 안 함)."""
    audio, sr = read_audio(input_path)
    speech = detect_speech(
        audio, sr,
        threshold=vad_threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
    )
    noise = invert_to_noise(speech, audio.shape[-1])
    speech_sec = sum(s.duration_samples for s in speech) / sr
    noise_sec = sum(s.duration_samples for s in noise) / sr
    measured = measure(audio, sr)
    return {
        "input": str(input_path),
        "sample_rate": sr,
        "duration_sec": round(audio.shape[-1] / sr, 3),
        "n_speech_segments": len(speech),
        "speech_total_sec": round(speech_sec, 3),
        "noise_total_sec": round(noise_sec, 3),
        "speech_ratio": round(speech_sec / max(speech_sec + noise_sec, 1e-9), 3),
        "speech_segments": [s.to_dict(sr) for s in speech],
        "noise_segments": [s.to_dict(sr) for s in noise],
        "loudness": measured.to_dict(),
    }


def _apply_denoise(
    audio: np.ndarray,
    sample_rate: int,
    plugin_short_name: str,
    params: dict[str, Any] | None,
) -> tuple[np.ndarray, str]:
    """RX denoise 플러그인을 전체 오디오에 적용. (output, plugin_full_name) 반환."""
    registry = get_registry()
    meta = registry.get(plugin_short_name)
    if not meta:
        raise ValueError(f"플러그인을 찾을 수 없습니다: '{plugin_short_name}'")
    wrapper = VST3PluginWrapper(meta.path)
    wrapper.load()
    if params:
        wrapper.set_parameters(params)
    out = wrapper.process(audio, sample_rate)
    return out, meta.name


def process(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_lufs: float = -20.0,
    max_true_peak_dbtp: float = -1.0,
    noise_attenuation_db: float = -12.0,
    denoise_plugin: str | None = "voice-de-noise",
    denoise_params: dict[str, Any] | None = None,
    vad_threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 200,
    speech_pad_ms: int = 80,
) -> VoiceoverResult:
    """보이스오버 일괄 처리.

    1) VAD로 음성/노이즈 구간 식별 (길이 유지)
    2) 전체 오디오에 RX denoise 적용 (denoise_plugin이 None이면 skip)
    3) 발화 단위 LUFS 레벨링 + 노이즈 구간 manual attenuation
    4) TP 천장 보정 후 저장
    """
    audio, sr = read_audio(input_path)
    measured_in = measure(audio, sr)

    # 1) VAD
    speech = detect_speech(
        audio, sr,
        threshold=vad_threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    noise = invert_to_noise(speech, audio.shape[-1])
    speech_sec = sum(s.duration_samples for s in speech) / sr
    noise_sec = sum(s.duration_samples for s in noise) / sr
    logger.info(
        "VAD: %d speech segments (%.1fs speech / %.1fs noise)",
        len(speech), speech_sec, noise_sec,
    )

    # 2) Denoise (전체 오디오)
    plugin_full_name: str | None = None
    if denoise_plugin:
        logger.info("Applying denoise: %s", denoise_plugin)
        audio, plugin_full_name = _apply_denoise(audio, sr, denoise_plugin, denoise_params)

    # 3) Per-utterance LUFS leveling + noise attenuation (길이 유지)
    leveled, leveling_meta = level_utterances(
        audio, sr,
        speech_segments=speech,
        target_lufs=target_lufs,
        max_true_peak_dbtp=max_true_peak_dbtp,
        noise_attenuation_db=noise_attenuation_db,
    )

    # 4) Write
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_audio(output_path, leveled, sr)
    measured_out = measure(leveled, sr)

    return VoiceoverResult(
        input_path=str(input_path),
        output_path=str(output_path),
        sample_rate=sr,
        duration_sec=audio.shape[-1] / sr,
        n_speech_segments=len(speech),
        speech_total_sec=speech_sec,
        noise_total_sec=noise_sec,
        denoise_plugin=plugin_full_name,
        leveling_meta=leveling_meta,
        measured_in=measured_in.to_dict(),
        measured_out=measured_out.to_dict(),
    )
