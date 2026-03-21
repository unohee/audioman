# Created: 2026-03-21
# Purpose: 앱 설정 관리 (pydantic-settings)

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from audioman.config.paths import get_app_dir, get_cache_dir, get_preset_dir


class AudiomanSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUDIOMAN_",
        toml_file=str(get_app_dir() / "config.toml"),
    )

    # 일반
    default_output_format: str = "wav"
    default_sample_rate: int = 44100
    json_output: bool = False
    verbose: bool = False

    # 경로
    extra_vst3_paths: list[str] = Field(default_factory=list)
    extra_au_paths: list[str] = Field(default_factory=list)
    preset_dir: str = str(get_preset_dir())
    cache_dir: str = str(get_cache_dir())

    # 처리
    default_chunk_size: int = 441000  # ~10s @ 44.1kHz
    large_file_threshold_mb: int = 500
    auto_stream: bool = True

    # GPU (Phase 2)
    gpu_enabled: bool = False
    gpu_device: str = "auto"


_settings: Optional[AudiomanSettings] = None


def get_settings() -> AudiomanSettings:
    """설정 싱글턴"""
    global _settings
    if _settings is None:
        _settings = AudiomanSettings()
    return _settings


def reset_settings() -> None:
    """설정 초기화 (테스트용)"""
    global _settings
    _settings = None
