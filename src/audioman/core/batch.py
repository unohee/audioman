# Created: 2026-03-21
# Purpose: 배치 처리 유틸리티

import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

from audioman.core.audio_file import read_audio, write_audio, get_audio_stats
from audioman.core.registry import get_registry
from audioman.plugins.vst3 import VST3PluginWrapper

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".aiff", ".aif", ".ogg", ".opus", ".m4a", ".wma"}


def collect_audio_files(path: str | Path, recursive: bool = False) -> list[Path]:
    """디렉토리에서 오디오 파일 수집"""
    path = Path(path)
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {path}")

    glob_pattern = "**/*" if recursive else "*"
    files = sorted(
        f for f in path.glob(glob_pattern)
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    return files


def resolve_output_path(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
    suffix: str = "",
    ext: str = ".wav",
) -> Path:
    """입력 파일에 대응하는 출력 경로 생성 (디렉토리 구조 유지)"""
    relative = input_path.relative_to(input_dir) if input_dir != input_path else input_path.name
    stem = Path(relative).stem
    parent = Path(relative).parent
    out_name = f"{stem}{suffix}{ext}"
    out_path = output_dir / parent / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path
