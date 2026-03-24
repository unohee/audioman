# Created: 2026-03-23
# Purpose: 프리셋 배치 렌더링 → 데이터셋 패키징 (oneshot + params + embeddings)
# Dependencies: dawdreamer, numpy, soundfile

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

from audioman.core.audio_file import write_audio
from audioman.core.instrument import MidiNote, render_notes, parse_note
from audioman.core.preset_parser import (
    find_presets,
    parse_auto,
    resolve_param_names,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """배치 렌더링 설정"""

    plugin_path: str
    preset_dir: str
    output_dir: str
    notes: list[int] = field(default_factory=lambda: [60])  # MIDI 노트 번호
    velocity: int = 100
    velocity_layers: list[int] = field(default_factory=list)  # 벨로시티 레이어 (빈 리스트면 단일 velocity 사용)
    duration: float = 2.0
    tail: float = 1.0
    fadeout: float = 0.5
    sample_rate: int = 44100
    buffer_size: int = 512
    recursive: bool = True
    format: str = "npy"  # "npy" | "wav" | "both"
    mel: bool = False  # mel-spectrogram 추출
    mel_bands: int = 128
    mel_fmax: int = 8000
    hop_length: int = 512
    normalize_audio: bool = True
    analyze: bool = True  # 스펙트럼 분석 (features.jsonl)


@dataclass
class RenderEntry:
    """개별 렌더링 결과 메타데이터"""

    index: int
    preset_name: str
    preset_path: str
    duration: float
    sample_rate: int
    channels: int
    frames: int
    peak: float
    rms: float
    parameters: dict
    velocity: int = 100
    notes: list[int] = field(default_factory=list)
    wav_path: Optional[str] = None
    features: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class BatchResult:
    """배치 렌더링 전체 결과"""

    total: int
    success: int
    failed: int
    duration_seconds: float
    output_dir: str
    plugin_path: str
    config: dict
    entries: list[RenderEntry] = field(default_factory=list)


def _compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 128,
    fmax: int = 8000,
    hop_length: int = 512,
) -> np.ndarray:
    """Mel-spectrogram 계산 (순수 numpy, librosa 미의존)

    Returns: (n_mels, time_frames) float32
    """
    # 모노 변환
    if audio.ndim == 2:
        mono = np.mean(audio, axis=0)
    else:
        mono = audio

    # STFT
    n_fft = 2048
    window = np.hanning(n_fft).astype(np.float32)
    num_frames = 1 + (len(mono) - n_fft) // hop_length

    if num_frames <= 0:
        return np.zeros((n_mels, 1), dtype=np.float32)

    stft = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        frame = mono[start : start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        stft[:, i] = np.abs(spectrum).astype(np.float32)

    # Power spectrum
    power = stft**2

    # Mel filterbank
    mel_filter = _mel_filterbank(sr, n_fft, n_mels, fmax)
    mel_spec = mel_filter @ power

    # Log scale (안전한 log)
    mel_spec = np.log(np.maximum(mel_spec, 1e-10))

    return mel_spec.astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmax: int) -> np.ndarray:
    """Mel filterbank 행렬 생성"""
    fmin = 0.0
    # Hz → Mel
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    # FFT bin에 매핑
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for m in range(n_mels):
        f_left = bin_points[m]
        f_center = bin_points[m + 1]
        f_right = bin_points[m + 2]

        for k in range(f_left, f_center):
            if f_center > f_left:
                filterbank[m, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                filterbank[m, k] = (f_right - k) / (f_right - f_center)

    return filterbank


def batch_render(config: BatchConfig, progress_callback=None) -> BatchResult:
    """프리셋 디렉토리 전체를 배치 렌더링 → 데이터셋 패키징

    출력 구조:
        output_dir/
        ├── manifest.json       # 전체 메타데이터
        ├── params.jsonl        # 프리셋별 파라미터 (이름 매핑됨)
        ├── audio.npy           # (N, channels, samples) 또는 (N, samples)
        ├── mel.npy             # (N, n_mels, time_frames) — --mel 시
        ├── labels.json         # 프리셋 이름 → 인덱스 매핑
        └── wav/                # --format wav|both 시 개별 WAV
            ├── 000_preset_name.wav
            └── ...
    """
    start_time = time.monotonic()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preset_dir = Path(config.preset_dir)
    presets = find_presets(preset_dir, recursive=config.recursive)

    if not presets:
        raise FileNotFoundError(
            f"프리셋 없음: {preset_dir} (지원: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )

    # WAV 출력 디렉토리
    wav_dir = None
    if config.format in ("wav", "both"):
        wav_dir = output_dir / "wav"
        wav_dir.mkdir(exist_ok=True)

    # 결과 저장소
    entries: list[RenderEntry] = []
    audio_list: list[np.ndarray] = []
    mel_list: list[np.ndarray] = []
    labels: dict[str, int] = {}
    max_samples = 0

    # velocity 레이어 결정
    velocities = config.velocity_layers if config.velocity_layers else [config.velocity]

    # 전체 작업 수: presets × notes × velocities
    total_jobs = len(presets) * len(config.notes) * len(velocities)

    success, failed = 0, 0
    job_idx = 0

    # 싱글턴 DawDreamer 엔진 (플러그인 초기화 1회만)
    from audioman.core.instrument import _ensure_dawdreamer, _apply_fadeout
    from audioman.core.audio_file import write_audio, read_audio
    daw = _ensure_dawdreamer()

    render_duration = config.duration + config.tail

    for preset_idx, preset_path in enumerate(presets):
        try:
            # 프리셋 파싱 (프리셋당 1회)
            preset_data = parse_auto(preset_path)
            resolve_param_names(preset_data)
        except Exception as e:
            logger.warning(f"[{preset_idx + 1}/{len(presets)}] {preset_path.name}: 파싱 실패 - {e}")
            for note in config.notes:
                for vel in velocities:
                    job_idx += 1
                    failed += 1
            if progress_callback:
                entry = RenderEntry(
                    index=job_idx - 1, preset_name=preset_path.stem,
                    preset_path=str(preset_path), notes=[], duration=0,
                    sample_rate=config.sample_rate, channels=0, frames=0,
                    peak=0, rms=0, parameters={}, error=str(e),
                )
                progress_callback(job_idx, total_jobs, entry)
            continue

        for note in config.notes:
            for vel in velocities:
                try:
                    safe_name = _safe_filename(preset_data.preset_name or preset_path.stem)
                    suffix = f"_n{note}_v{vel}" if (len(config.notes) > 1 or len(velocities) > 1) else ""

                    if config.format in ("wav", "both"):
                        wav_path = wav_dir / f"{job_idx:04d}_{safe_name}{suffix}.wav"
                    else:
                        wav_path = output_dir / f"_temp_{job_idx:04d}.wav"

                    # 싱글턴 엔진으로 렌더링 (매번 새 엔진+플러그인 생성 안 함)
                    engine = daw.RenderEngine(config.sample_rate, config.buffer_size)
                    synth = engine.make_plugin_processor("instrument", config.plugin_path)

                    # 프리셋 로딩
                    ext = preset_path.suffix.lower()
                    if ext in (".fxp", ".fxb", ".h2p", ".vital", ".phaseplant"):
                        synth.load_preset(str(preset_path))
                    elif ext == ".vstpreset":
                        synth.load_vst3_preset(str(preset_path))

                    # MIDI 노트
                    synth.add_midi_note(note, vel, 0.0, config.duration)
                    engine.load_graph([(synth, [])])
                    engine.render(render_duration)
                    audio = engine.get_audio()

                    if audio.dtype != np.float32:
                        audio = audio.astype(np.float32)
                    audio = _apply_fadeout(audio, config.sample_rate, config.fadeout)

                    # WAV 저장
                    wav_path.parent.mkdir(parents=True, exist_ok=True)
                    write_audio(wav_path, audio, config.sample_rate)

                    # 다시 읽기 (soundfile 정규화 보장)
                    audio, sr = read_audio(wav_path)

                    # 스펙트럼 분석 (정규화 전)
                    audio_features = None
                    if config.analyze:
                        from audioman.core.analysis import compute_audio_features
                        audio_features = compute_audio_features(audio, sr)

                    if config.normalize_audio:
                        peak = np.max(np.abs(audio))
                        if peak > 0:
                            audio = audio / peak

                    audio_list.append(audio)
                    max_samples = max(max_samples, audio.shape[-1])

                    if config.mel:
                        mel_spec = _compute_mel_spectrogram(
                            audio, sr, config.mel_bands, config.mel_fmax, config.hop_length
                        )
                        mel_list.append(mel_spec)

                    if config.format == "npy" and wav_path.exists():
                        wav_path.unlink()

                    channels = audio.shape[0] if audio.ndim == 2 else 1
                    frames = audio.shape[-1]

                    entry = RenderEntry(
                        index=job_idx,
                        preset_name=preset_data.preset_name or preset_path.stem,
                        preset_path=str(preset_path),
                        velocity=vel,
                        notes=[note],
                        duration=render_duration,
                        sample_rate=config.sample_rate,
                        channels=channels,
                        frames=frames,
                        peak=float(np.max(np.abs(audio))),
                        rms=float(np.sqrt(np.mean(audio**2))),
                        parameters=preset_data.to_dict(include_chunk=False).get("parameters", {}),
                        wav_path=str(wav_path) if config.format in ("wav", "both") else None,
                        features=audio_features,
                    )
                    entries.append(entry)
                    label_key = f"{entry.preset_name}_n{note}_v{vel}" if suffix else entry.preset_name
                    labels[label_key] = job_idx
                    success += 1

                except Exception as e:
                    logger.warning(f"[{job_idx + 1}/{total_jobs}] {preset_path.name} n{note} v{vel}: {e}")
                    entry = RenderEntry(
                        index=job_idx, preset_name=preset_path.stem,
                        preset_path=str(preset_path), velocity=vel, notes=[note],
                        duration=0, sample_rate=config.sample_rate, channels=0,
                        frames=0, peak=0, rms=0, parameters={}, error=str(e),
                    )
                    entries.append(entry)
                    failed += 1

                job_idx += 1

                if progress_callback:
                    progress_callback(job_idx, total_jobs, entry)

    # --- 패키징 ---

    # params.jsonl
    params_path = output_dir / "params.jsonl"
    with open(params_path, "w") as f:
        for entry in entries:
            if entry.error:
                continue
            record = {
                "index": entry.index,
                "preset_name": entry.preset_name,
                "preset_path": entry.preset_path,
                "note": entry.notes[0] if entry.notes else None,
                "velocity": entry.velocity,
                "parameters": entry.parameters,
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # features.jsonl (스펙트럼 분석)
    if config.analyze:
        features_path = output_dir / "features.jsonl"
        with open(features_path, "w") as f:
            for entry in entries:
                if entry.error or not entry.features:
                    continue
                record = {
                    "index": entry.index,
                    "preset_name": entry.preset_name,
                    "note": entry.notes[0] if entry.notes else None,
                    "velocity": entry.velocity,
                    **entry.features,
                }
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # labels.json
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # audio.npy — zero-padding으로 길이 통일
    if audio_list and config.format in ("npy", "both"):
        channels = audio_list[0].shape[0] if audio_list[0].ndim == 2 else 1
        padded = np.zeros((len(audio_list), channels, max_samples), dtype=np.float32)
        for i, a in enumerate(audio_list):
            if a.ndim == 1:
                padded[i, 0, : len(a)] = a
            else:
                padded[i, :, : a.shape[-1]] = a

        audio_npy_path = output_dir / "audio.npy"
        np.save(str(audio_npy_path), padded)
        logger.info(f"audio.npy: {padded.shape} ({padded.nbytes / 1024 / 1024:.1f}MB)")

    # mel.npy
    if mel_list and config.mel:
        max_mel_frames = max(m.shape[1] for m in mel_list)
        mel_padded = np.zeros(
            (len(mel_list), config.mel_bands, max_mel_frames), dtype=np.float32
        )
        for i, m in enumerate(mel_list):
            mel_padded[i, :, : m.shape[1]] = m

        mel_npy_path = output_dir / "mel.npy"
        np.save(str(mel_npy_path), mel_padded)
        logger.info(f"mel.npy: {mel_padded.shape} ({mel_padded.nbytes / 1024 / 1024:.1f}MB)")

    # manifest.json
    elapsed = time.monotonic() - start_time
    manifest = {
        "version": "1.0",
        "plugin": config.plugin_path,
        "preset_dir": str(preset_dir),
        "total_presets": len(presets),
        "total_jobs": total_jobs,
        "velocity_layers": velocities,
        "success": success,
        "failed": failed,
        "notes": config.notes,
        "velocity": config.velocity,
        "duration": config.duration,
        "tail": config.tail,
        "fadeout": config.fadeout,
        "sample_rate": config.sample_rate,
        "normalize": config.normalize_audio,
        "format": config.format,
        "mel": config.mel,
        "mel_bands": config.mel_bands if config.mel else None,
        "mel_fmax": config.mel_fmax if config.mel else None,
        "hop_length": config.hop_length if config.mel else None,
        "max_samples": max_samples,
        "channels": channels if audio_list else 0,
        "audio_shape": list(padded.shape) if audio_list and config.format in ("npy", "both") else None,
        "mel_shape": list(mel_padded.shape) if mel_list and config.mel else None,
        "elapsed_seconds": round(elapsed, 2),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return BatchResult(
        total=len(presets),
        success=success,
        failed=failed,
        duration_seconds=round(elapsed, 2),
        output_dir=str(output_dir),
        plugin_path=config.plugin_path,
        config=asdict(config),
        entries=entries,
    )


def _safe_filename(name: str) -> str:
    """프리셋 이름을 안전한 파일명으로 변환"""
    import re

    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    safe = safe.strip(". ")
    return safe[:100] if safe else "unnamed"


# =============================================================================
# 내장 프리셋 플러그인 배치 렌더링 (KORG M1, TRITON, SRX 등)
# =============================================================================


def batch_render_programs(
    plugin_path: str,
    output_dir: str,
    num_programs: int | None = None,
    notes: list[int] | None = None,
    velocity: int = 100,
    duration: float = 2.0,
    tail: float = 1.0,
    fadeout: float = 0.5,
    sample_rate: int = 44100,
    buffer_size: int = 512,
    mel: bool = False,
    mel_bands: int = 128,
    mel_fmax: int = 8000,
    hop_length: int = 512,
    normalize_audio: bool = True,
    progress_callback=None,
) -> BatchResult:
    """플러그인 내장 프로그램을 순회하며 배치 렌더링

    프리셋 파일이 없고 플러그인 내부에 프로그램이 내장된 경우 사용.
    DawDreamer의 program change로 각 프로그램을 순회.
    """
    from audioman.core.instrument import _ensure_dawdreamer, _apply_fadeout

    start_time = time.monotonic()
    daw = _ensure_dawdreamer()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if notes is None:
        notes = [60]

    # 플러그인 로딩 + 프로그램 수 확인
    engine = daw.RenderEngine(sample_rate, buffer_size)
    synth = engine.make_plugin_processor("instrument", plugin_path)

    if num_programs is None:
        # 프로그램 수 자동 감지 시도
        try:
            num_programs = synth.get_num_programs() if hasattr(synth, 'get_num_programs') else 128
        except Exception:
            num_programs = 128

    # 파라미터 이름 캐시
    param_count = synth.get_plugin_parameter_size()
    param_names = [synth.get_parameter_name(i) for i in range(param_count)]

    entries: list[RenderEntry] = []
    audio_list: list[np.ndarray] = []
    mel_list: list[np.ndarray] = []
    labels: dict[str, int] = {}
    max_samples = 0
    success, failed = 0, 0

    render_duration = duration + tail

    for idx in range(num_programs):
        try:
            # 프로그램 변경
            engine2 = daw.RenderEngine(sample_rate, buffer_size)
            synth2 = engine2.make_plugin_processor("instrument", plugin_path)

            # program change via set_parameter or load_program
            if hasattr(synth2, 'set_program'):
                synth2.set_program(idx)
            else:
                # MIDI program change 메시지
                synth2.add_midi_note(notes[0], 0, 0.0, 0.001)  # 더미
                synth2.clear_midi()

            # 프로그램 이름 추출 시도
            program_name = f"Program_{idx:04d}"
            if hasattr(synth2, 'get_program_name'):
                try:
                    pname = synth2.get_program_name()
                    if pname and pname.strip():
                        program_name = pname.strip()
                except Exception:
                    pass

            # 파라미터 값 추출
            params = {}
            for i in range(min(param_count, len(param_names))):
                val = synth2.get_parameter(i)
                params[param_names[i]] = round(float(val), 6)

            # MIDI 노트 추가 + 렌더링
            for n in notes:
                synth2.add_midi_note(n, velocity, 0.0, duration)

            engine2.load_graph([(synth2, [])])
            engine2.render(render_duration)
            audio = engine2.get_audio()

            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            audio = _apply_fadeout(audio, sample_rate, fadeout)

            # 무음 감지 — 실제 소리가 없는 프로그램은 스킵
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < 0.0001:
                logger.debug(f"[{idx}] {program_name}: 무음 — 스킵")
                continue

            # 스펙트럼 분석 (정규화 전)
            audio_features = None
            try:
                from audioman.core.analysis import compute_audio_features
                audio_features = compute_audio_features(audio, sample_rate)
            except Exception:
                pass

            # 정규화
            if normalize_audio:
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak

            audio_list.append(audio)
            max_samples = max(max_samples, audio.shape[-1])

            if mel:
                mel_spec = _compute_mel_spectrogram(audio, sample_rate, mel_bands, mel_fmax, hop_length)
                mel_list.append(mel_spec)

            entry = RenderEntry(
                index=len(entries),
                preset_name=program_name,
                preset_path=f"program://{idx}",
                notes=notes,
                duration=render_duration,
                sample_rate=sample_rate,
                channels=audio.shape[0] if audio.ndim == 2 else 1,
                frames=audio.shape[-1],
                peak=float(np.max(np.abs(audio))),
                rms=rms,
                parameters=params,
                features=audio_features,
            )
            entries.append(entry)
            labels[program_name] = entry.index
            success += 1

        except Exception as e:
            logger.warning(f"[{idx}/{num_programs}] Program {idx}: {e}")
            failed += 1

        if progress_callback:
            dummy_entry = RenderEntry(
                index=idx, preset_name=program_name if 'program_name' in dir() else f"Program_{idx}",
                preset_path="", notes=notes, duration=0, sample_rate=sample_rate,
                channels=0, frames=0, peak=0, rms=0, parameters={},
                error=str(e) if failed and 'e' in dir() else None,
            )
            progress_callback(idx + 1, num_programs, dummy_entry)

    # --- 패키징 (batch_render와 동일) ---

    # features.jsonl
    features_path = output_path / "features.jsonl"
    with open(features_path, "w") as f:
        for entry in entries:
            if not entry.features:
                continue
            record = {
                "index": entry.index,
                "preset_name": entry.preset_name,
                "note": entry.notes[0] if entry.notes else None,
                "velocity": entry.velocity,
                **entry.features,
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    params_path = output_path / "params.jsonl"
    with open(params_path, "w") as f:
        for entry in entries:
            record = {
                "index": entry.index,
                "preset_name": entry.preset_name,
                "preset_path": entry.preset_path,
                "parameters": entry.parameters,
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    labels_path = output_path / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    if audio_list:
        channels = audio_list[0].shape[0] if audio_list[0].ndim == 2 else 1
        padded = np.zeros((len(audio_list), channels, max_samples), dtype=np.float32)
        for i, a in enumerate(audio_list):
            if a.ndim == 1:
                padded[i, 0, : len(a)] = a
            else:
                padded[i, :, : a.shape[-1]] = a

        np.save(str(output_path / "audio.npy"), padded)

    if mel_list and mel:
        max_mel_frames = max(m.shape[1] for m in mel_list)
        mel_padded = np.zeros((len(mel_list), mel_bands, max_mel_frames), dtype=np.float32)
        for i, m in enumerate(mel_list):
            mel_padded[i, :, : m.shape[1]] = m
        np.save(str(output_path / "mel.npy"), mel_padded)

    elapsed = time.monotonic() - start_time

    manifest = {
        "version": "1.0",
        "plugin": plugin_path,
        "mode": "program_scan",
        "total_programs_scanned": num_programs,
        "total_with_sound": success,
        "skipped_silent": num_programs - success - failed,
        "failed": failed,
        "notes": notes,
        "velocity": velocity,
        "duration": duration,
        "sample_rate": sample_rate,
        "normalize": normalize_audio,
        "mel": mel,
        "audio_shape": list(padded.shape) if audio_list else None,
        "mel_shape": list(mel_padded.shape) if mel_list and mel else None,
        "elapsed_seconds": round(elapsed, 2),
    }

    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return BatchResult(
        total=num_programs,
        success=success,
        failed=failed,
        duration_seconds=round(elapsed, 2),
        output_dir=str(output_path),
        plugin_path=plugin_path,
        config={"mode": "program_scan", "num_programs": num_programs},
        entries=entries,
    )
