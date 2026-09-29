# Created: 2026-03-21
# Purpose: App settings management (pydantic-settings)

import tomllib
from pathlib import Path
from typing import Any, ClassVar, Optional

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from audioman.config.paths import get_app_dir, get_cache_dir, get_preset_dir


class _TomlSettingsSource(PydanticBaseSettingsSource):
    """Python 3.12 stdlib TOML source, compatible with pydantic-settings 2.x."""

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        data = self()
        return data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        path = self.settings_cls.config_file
        if not path.is_file():
            return {}
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        # Unreachable in practice, kept as the boundary's diagnostic: `tomllib.load`
        # returns the document table and rejects a non-table root with TOMLDecodeError
        # while parsing, so `data` is always a dict. The premise is pinned by
        # TestTomlNonTableRootIsDead in tests/unit/cli_extra2/test_residual_lines.py.
        if not isinstance(data, dict):  # pragma: no cover - load returns a table
            raise ValueError(f"Audioman TOML settings must be a table: {path}")
        return data


class AudiomanSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUDIOMAN_",
    )
    config_file: ClassVar[Path] = get_app_dir() / "config.toml"

    # General
    default_output_format: str = "wav"
    default_sample_rate: int = 44100
    json_output: bool = False
    verbose: bool = False

    # Paths
    extra_vst3_paths: list[str] = Field(default_factory=list)
    extra_au_paths: list[str] = Field(default_factory=list)
    preset_dir: str = str(get_preset_dir())
    cache_dir: str = str(get_cache_dir())

    # Processing
    default_chunk_size: int = 441000  # ~10s @ 44.1kHz
    large_file_threshold_mb: int = 500
    auto_stream: bool = True

    # GPU (Phase 2)
    gpu_enabled: bool = False
    gpu_device: str = "auto"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            _TomlSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


_settings: Optional[AudiomanSettings] = None


def get_settings() -> AudiomanSettings:
    """Settings singleton"""
    global _settings
    if _settings is None:
        _settings = AudiomanSettings()
    return _settings


def reset_settings() -> None:
    """Reset settings (for tests)"""
    global _settings
    _settings = None
