# Created: 2026-03-21
# Purpose: 플러그인 프리셋 관리 (JSON 기반 CRUD)

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from audioman.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class PresetData:
    """프리셋 데이터"""
    name: str
    plugin: str
    parameters: dict[str, Any]
    description: str = ""
    created: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "plugin": self.plugin,
            "parameters": {k: v for k, v in self.parameters.items()},
            "description": self.description,
            "created": self.created,
        }


class PresetManager:
    """프리셋 CRUD"""

    def __init__(self, preset_dir: Optional[Path] = None) -> None:
        self._dir = preset_dir or Path(get_settings().preset_dir)

    @staticmethod
    def _safe_component(value: str, field: str) -> str:
        candidate = Path(value)
        if not value or candidate.name != value or value in {".", ".."}:
            raise ValueError(f"{field} must be a single path component")
        return value

    def _plugin_dir(self, plugin: str) -> Path:
        return self._dir / self._safe_component(plugin, "plugin")

    def _preset_path(self, directory: Path, name: str) -> Path:
        safe_name = self._safe_component(name, "preset name")
        return directory / f"{safe_name}.json"

    def save(
        self,
        name: str,
        plugin: str,
        params: dict[str, Any],
        description: str = "",
    ) -> Path:
        """프리셋 저장"""
        plugin_dir = self._plugin_dir(plugin)
        plugin_dir.mkdir(parents=True, exist_ok=True)

        preset = PresetData(
            name=name,
            plugin=plugin,
            parameters=params,
            description=description,
            created=datetime.now().isoformat(),
        )

        path = self._preset_path(plugin_dir, name)
        path.write_text(json.dumps(preset.to_dict(), indent=2, ensure_ascii=False))
        logger.debug(f"프리셋 저장: {path}")
        return path

    def load(self, name: str, plugin: Optional[str] = None) -> PresetData:
        """프리셋 로드"""
        path = self._find_preset(name, plugin)
        if not path:
            raise FileNotFoundError(f"프리셋을 찾을 수 없습니다: '{name}'")

        data = json.loads(path.read_text())
        return PresetData(**data)

    def list(self, plugin: Optional[str] = None) -> list[PresetData]:
        """프리셋 목록"""
        results = []

        if plugin:
            search_dirs = [self._plugin_dir(plugin)]
        else:
            search_dirs = [d for d in self._dir.iterdir() if d.is_dir()] if self._dir.exists() else []

        for d in search_dirs:
            if not d.exists():
                continue
            for f in sorted(d.glob("*.json")):
                try:
                    data = json.loads(f.read_text())
                    results.append(PresetData(**data))
                except Exception:
                    logger.warning(f"프리셋 로드 실패: {f}")

        return results

    def delete(self, name: str, plugin: Optional[str] = None) -> None:
        """프리셋 삭제"""
        path = self._find_preset(name, plugin)
        if not path:
            raise FileNotFoundError(f"프리셋을 찾을 수 없습니다: '{name}'")
        path.unlink()

    def _find_preset(self, name: str, plugin: Optional[str] = None) -> Optional[Path]:
        """프리셋 파일 검색"""
        if plugin:
            path = self._preset_path(self._plugin_dir(plugin), name)
            return path if path.exists() else None

        # 모든 플러그인 디렉토리에서 검색
        if not self._dir.exists():
            return None
        for d in self._dir.iterdir():
            if d.is_dir():
                path = self._preset_path(d, name)
                if path.exists():
                    return path

        return None
