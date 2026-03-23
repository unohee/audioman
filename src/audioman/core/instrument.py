# Created: 2026-03-23
# Purpose: DawDreamer 기반 VST3 인스트루먼트 엔진 (MIDI → 오디오 렌더링)
# Dependencies: dawdreamer (선택적)

import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# MIDI 노트 이름 → 번호 매핑
NOTE_NAMES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOTE_PATTERN = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)$")


def note_name_to_midi(name: str) -> int:
    """노트 이름을 MIDI 번호로 변환 (예: C4 → 60, A#3 → 58, Bb3 → 58)"""
    m = NOTE_PATTERN.match(name.strip())
    if not m:
        raise ValueError(f"잘못된 노트 이름: '{name}' (예: C4, A#3, Bb2)")

    letter, accidental, octave = m.groups()
    midi = NOTE_NAMES[letter.upper()] + (int(octave) + 1) * 12

    if accidental == "#":
        midi += 1
    elif accidental == "b":
        midi -= 1

    if not 0 <= midi <= 127:
        raise ValueError(f"MIDI 범위 초과: {name} → {midi} (0-127)")

    return midi


def parse_note(note_str: str) -> int:
    """MIDI 노트 파싱: 숫자 또는 이름 (60, C4, A#3 등)"""
    try:
        n = int(note_str)
        if 0 <= n <= 127:
            return n
        raise ValueError(f"MIDI 범위 초과: {n} (0-127)")
    except ValueError:
        pass

    return note_name_to_midi(note_str)


@dataclass
class MidiNote:
    """MIDI 노트 이벤트"""

    note: int  # MIDI 번호 (0-127)
    velocity: int = 100  # 벨로시티 (0-127)
    start: float = 0.0  # 시작 시간 (초)
    duration: float = 1.0  # 지속 시간 (초)


@dataclass
class RenderResult:
    """렌더링 결과"""

    output_path: str
    plugin_path: str
    sample_rate: int
    duration: float
    channels: int
    frames: int
    peak: float
    rms: float
    notes: list[dict]
    preset: Optional[str] = None
    params_applied: dict = None

    def to_dict(self) -> dict:
        return asdict(self)


def _ensure_dawdreamer():
    """DawDreamer 임포트 확인"""
    try:
        import dawdreamer

        return dawdreamer
    except ImportError:
        raise ImportError(
            "dawdreamer 필요: pip install dawdreamer\n"
            "또는: uv add dawdreamer --optional instrument"
        )


def render_notes(
    plugin_path: str | Path,
    notes: list[MidiNote],
    output_path: str | Path,
    sample_rate: int = 44100,
    buffer_size: int = 512,
    duration: Optional[float] = None,
    preset_path: Optional[str | Path] = None,
    params: Optional[dict[str, float]] = None,
    tail: float = 1.0,
    fadeout: float = 0.5,
) -> RenderResult:
    """MIDI 노트 → VST3 인스트루먼트 → 오디오 파일 렌더링

    Args:
        plugin_path: VST3 플러그인 경로
        notes: MIDI 노트 리스트
        output_path: 출력 오디오 파일 경로
        sample_rate: 샘플레이트
        buffer_size: 버퍼 크기
        duration: 렌더링 길이 (초). None이면 노트 끝 + tail로 자동 계산
        preset_path: 프리셋 파일 경로 (.fxp, .vstpreset 등)
        params: 파라미터 딕셔너리
        tail: 마지막 노트 이후 추가 렌더링 시간 (초)
        fadeout: 끝부분 페이드아웃 길이 (초, 기본 0.5)
    """
    daw = _ensure_dawdreamer()
    plugin_path = Path(plugin_path)
    output_path = Path(output_path)

    if not plugin_path.exists():
        raise FileNotFoundError(f"플러그인 없음: {plugin_path}")

    # 렌더 엔진 생성
    engine = daw.RenderEngine(sample_rate, buffer_size)
    synth = engine.make_plugin_processor("instrument", str(plugin_path))

    # 프리셋 로딩
    if preset_path:
        preset_path = Path(preset_path)
        if not preset_path.exists():
            raise FileNotFoundError(f"프리셋 없음: {preset_path}")

        ext = preset_path.suffix.lower()
        if ext in (".fxp", ".fxb", ".h2p", ".vital", ".phaseplant"):
            synth.load_preset(str(preset_path))
        elif ext == ".vstpreset":
            synth.load_vst3_preset(str(preset_path))
        else:
            raise ValueError(f"DawDreamer 프리셋 로딩 미지원: {ext}")

        logger.debug(f"프리셋 로딩: {preset_path}")

    # 파라미터 설정
    applied_params = {}
    if params:
        param_count = synth.get_plugin_parameter_size()
        # 이름 → 인덱스 매핑 구축
        name_to_idx = {}
        for i in range(param_count):
            name = synth.get_parameter_name(i)
            name_to_idx[name] = i
            name_to_idx[name.lower()] = i
            # 공백 → 언더스코어 변환도 매핑
            name_to_idx[name.replace(" ", "_").lower()] = i

        for key, value in params.items():
            idx = name_to_idx.get(key) or name_to_idx.get(key.lower())
            if idx is not None:
                synth.set_parameter(idx, float(value))
                applied_params[key] = value
                logger.debug(f"파라미터: {key} (idx={idx}) = {value}")
            else:
                logger.warning(f"파라미터 없음: {key}")

    # MIDI 노트 추가
    for n in notes:
        synth.add_midi_note(n.note, n.velocity, n.start, n.duration)

    # 렌더링 길이 계산
    if duration is None:
        if notes:
            last_end = max(n.start + n.duration for n in notes)
            duration = last_end + tail
        else:
            duration = tail

    # 렌더링
    engine.load_graph([(synth, [])])
    engine.render(duration)
    audio = engine.get_audio()  # shape: (channels, samples)

    # float32 보장
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # 끝부분 페이드아웃 적용 (뚝 끊김 방지)
    audio = _apply_fadeout(audio, sample_rate, fadeout)

    # 출력 저장
    from audioman.core.audio_file import write_audio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_audio(output_path, audio, sample_rate)

    channels, frames = audio.shape if audio.ndim == 2 else (1, len(audio))

    return RenderResult(
        output_path=str(output_path),
        plugin_path=str(plugin_path),
        sample_rate=sample_rate,
        duration=duration,
        channels=channels,
        frames=frames,
        peak=float(np.max(np.abs(audio))),
        rms=float(np.sqrt(np.mean(audio**2))),
        notes=[asdict(n) for n in notes],
        preset=str(preset_path) if preset_path else None,
        params_applied=applied_params or None,
    )


def _apply_fadeout(audio: np.ndarray, sample_rate: int, fadeout: float) -> np.ndarray:
    """오디오 끝부분에 선형 페이드아웃 적용"""
    if fadeout <= 0:
        return audio

    fade_samples = int(fadeout * sample_rate)
    total_samples = audio.shape[-1]

    if fade_samples >= total_samples:
        fade_samples = total_samples

    fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    if audio.ndim == 2:
        audio[:, -fade_samples:] *= fade_curve
    else:
        audio[-fade_samples:] *= fade_curve

    return audio


def render_midi_file(
    plugin_path: str | Path,
    midi_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 44100,
    buffer_size: int = 512,
    preset_path: Optional[str | Path] = None,
    params: Optional[dict[str, float]] = None,
    tail: float = 2.0,
    fadeout: float = 0.5,
) -> RenderResult:
    """MIDI 파일 → VST3 인스트루먼트 → 오디오 파일 렌더링"""
    daw = _ensure_dawdreamer()
    plugin_path = Path(plugin_path)
    midi_path = Path(midi_path)
    output_path = Path(output_path)

    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI 파일 없음: {midi_path}")
    if not plugin_path.exists():
        raise FileNotFoundError(f"플러그인 없음: {plugin_path}")

    engine = daw.RenderEngine(sample_rate, buffer_size)
    synth = engine.make_plugin_processor("instrument", str(plugin_path))

    # 프리셋 로딩
    if preset_path:
        preset_path = Path(preset_path)
        ext = preset_path.suffix.lower()
        if ext in (".fxp", ".fxb"):
            synth.load_preset(str(preset_path))
        elif ext == ".vstpreset":
            synth.load_vst3_preset(str(preset_path))

    # 파라미터 설정
    applied_params = {}
    if params:
        param_count = synth.get_plugin_parameter_size()
        name_to_idx = {}
        for i in range(param_count):
            name = synth.get_parameter_name(i)
            name_to_idx[name.lower()] = i
            name_to_idx[name.replace(" ", "_").lower()] = i

        for key, value in params.items():
            idx = name_to_idx.get(key.lower())
            if idx is not None:
                synth.set_parameter(idx, float(value))
                applied_params[key] = value

    # MIDI 파일 로딩
    synth.load_midi(str(midi_path))

    # MIDI 파일 길이 추출
    try:
        import mido

        mid = mido.MidiFile(str(midi_path))
        midi_duration = mid.length + tail
    except ImportError:
        # mido 없으면 기본값
        midi_duration = 30.0 + tail
        logger.warning("mido 미설치 — MIDI 길이 추정 불가, 30초로 설정")

    engine.load_graph([(synth, [])])
    engine.render(midi_duration)
    audio = engine.get_audio()

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # 끝부분 페이드아웃 적용
    audio = _apply_fadeout(audio, sample_rate, fadeout)

    from audioman.core.audio_file import write_audio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_audio(output_path, audio, sample_rate)

    channels, frames = audio.shape if audio.ndim == 2 else (1, len(audio))

    return RenderResult(
        output_path=str(output_path),
        plugin_path=str(plugin_path),
        sample_rate=sample_rate,
        duration=midi_duration,
        channels=channels,
        frames=frames,
        peak=float(np.max(np.abs(audio))),
        rms=float(np.sqrt(np.mean(audio**2))),
        notes=[],  # MIDI 파일 모드에서는 개별 노트 미추적
        preset=str(preset_path) if preset_path else None,
        params_applied=applied_params or None,
    )
