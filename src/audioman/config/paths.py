# Created: 2026-03-21
# Purpose: 플랫폼별 VST3/AU 플러그인 경로 및 앱 디렉토리 해석

import platform
import sys
from pathlib import Path


def get_app_dir() -> Path:
    """~/.audioman 앱 설정 디렉토리"""
    return Path.home() / ".audioman"


def get_cache_dir() -> Path:
    return get_app_dir() / "cache"


def get_preset_dir() -> Path:
    return get_app_dir() / "presets"


def ensure_app_dirs() -> None:
    """앱 디렉토리 구조 생성"""
    for d in [get_app_dir(), get_cache_dir(), get_preset_dir()]:
        d.mkdir(parents=True, exist_ok=True)


def get_vst3_search_paths() -> list[Path]:
    """플랫폼별 VST3 플러그인 기본 검색 경로"""
    system = platform.system()

    if system == "Darwin":
        return [
            Path("/Library/Audio/Plug-Ins/VST3"),
            Path.home() / "Library" / "Audio" / "Plug-Ins" / "VST3",
        ]
    elif system == "Linux":
        return [
            Path("/usr/lib/vst3"),
            Path("/usr/local/lib/vst3"),
            Path.home() / ".vst3",
        ]
    elif system == "Windows":
        program_files = Path("C:/Program Files/Common Files/VST3")
        return [program_files]
    else:
        return []


def get_au_search_paths() -> list[Path]:
    """macOS AU 플러그인 기본 검색 경로"""
    if platform.system() != "Darwin":
        return []

    return [
        Path("/Library/Audio/Plug-Ins/Components"),
        Path.home() / "Library" / "Audio" / "Plug-Ins" / "Components",
    ]
