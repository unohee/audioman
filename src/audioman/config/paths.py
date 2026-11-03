# Created: 2026-03-21
# Purpose: Platform-specific VST3/AU plugin paths and app directory resolution

import platform
from pathlib import Path


def get_app_dir() -> Path:
    """~/.audioman app settings directory"""
    return Path.home() / ".audioman"


def get_cache_dir() -> Path:
    return get_app_dir() / "cache"


def get_preset_dir() -> Path:
    return get_app_dir() / "presets"


def ensure_app_dirs() -> None:
    """Create the app directory structure"""
    for d in [get_app_dir(), get_cache_dir(), get_preset_dir()]:
        d.mkdir(parents=True, exist_ok=True)


def get_vst3_search_paths() -> list[Path]:
    """Default VST3 plugin search paths per platform"""
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
    """Default macOS AU plugin search paths"""
    if platform.system() != "Darwin":
        return []

    return [
        Path("/Library/Audio/Plug-Ins/Components"),
        Path.home() / "Library" / "Audio" / "Plug-Ins" / "Components",
    ]
