# Created: 2026-03-21
# Purpose: 플러그인 발견, 등록, 캐싱 시스템

import json
import logging
import plistlib
import re
from pathlib import Path
from typing import Optional

from audioman.config.paths import (
    get_au_search_paths,
    get_cache_dir,
    get_vst3_search_paths,
)
from audioman.config.settings import get_settings
from audioman.plugins.parameter import PluginMeta

logger = logging.getLogger(__name__)

# short_name 별칭 매핑
ALIASES = {
    "spectral-de-noise": ["denoise", "spectral-denoise"],
    "voice-de-noise": ["voice-denoise"],
    "guitar-de-noise": ["guitar-denoise"],
    "de-click": ["declick"],
    "de-clip": ["declip"],
    "de-crackle": ["decrackle"],
    "de-ess": ["deess"],
    "de-hum": ["dehum"],
    "de-plosive": ["deplosive"],
    "de-reverb": ["dereverb"],
    "mouth-de-click": ["mouth-declick"],
    "repair-assistant": ["repair"],
}


def _name_to_short_name(name: str) -> str:
    """플러그인 이름에서 short_name 생성
    "RX 10 Spectral De-noise" → "spectral-de-noise"
    """
    # 벤더/버전 접두사 제거: "RX 10 ", "RX 9 " 등
    cleaned = re.sub(r"^RX\s+\d+\s+", "", name)
    # 일반적인 벤더 접두사 제거
    cleaned = re.sub(r"^(iZotope|Waves|FabFilter|Sonnox)\s+", "", cleaned, flags=re.IGNORECASE)
    # 소문자 kebab-case 변환
    short = cleaned.strip().lower()
    short = re.sub(r"\s+", "-", short)
    # 연속 하이픈 정리
    short = re.sub(r"-+", "-", short)
    return short


def _parse_vst3_info(vst3_path: Path) -> Optional[PluginMeta]:
    """VST3 번들에서 Info.plist 파싱하여 PluginMeta 생성"""
    plist_path = vst3_path / "Contents" / "Info.plist"
    if not plist_path.exists():
        return None

    try:
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
    except Exception:
        logger.warning(f"Info.plist 파싱 실패: {plist_path}")
        return None

    name = plist.get("CFBundleName", vst3_path.stem)
    short_name = _name_to_short_name(name)
    aliases = ALIASES.get(short_name, [])

    return PluginMeta(
        name=name,
        short_name=short_name,
        path=str(vst3_path),
        format="vst3",
        vendor=plist.get("CFBundleIdentifier", "").split(".")[1] if "." in plist.get("CFBundleIdentifier", "") else "",
        version=plist.get("CFBundleShortVersionString", ""),
        aliases=aliases,
    )


def _parse_au_info(au_path: Path) -> Optional[PluginMeta]:
    """AU 번들에서 Info.plist 파싱"""
    plist_path = au_path / "Contents" / "Info.plist"
    if not plist_path.exists():
        return None

    try:
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
    except Exception:
        logger.warning(f"Info.plist 파싱 실패: {plist_path}")
        return None

    name = plist.get("CFBundleName", au_path.stem)
    short_name = _name_to_short_name(name)
    aliases = ALIASES.get(short_name, [])

    return PluginMeta(
        name=name,
        short_name=short_name,
        path=str(au_path),
        format="au",
        vendor=plist.get("CFBundleIdentifier", "").split(".")[1] if "." in plist.get("CFBundleIdentifier", "") else "",
        version=plist.get("CFBundleShortVersionString", ""),
        aliases=aliases,
    )


class PluginRegistry:
    """플러그인 발견, 등록, 검색"""

    def __init__(self) -> None:
        self._plugins: dict[str, PluginMeta] = {}  # short_name → meta
        self._alias_map: dict[str, str] = {}  # alias → short_name
        self._cache_path = get_cache_dir() / "plugins.json"

    def scan(
        self,
        extra_paths: Optional[list[str]] = None,
        refresh: bool = False,
    ) -> list[PluginMeta]:
        """시스템에서 VST3/AU 플러그인 검색"""
        if not refresh and self._try_load_cache():
            return list(self._plugins.values())

        self._plugins.clear()
        self._alias_map.clear()

        # VST3 검색
        vst3_paths = get_vst3_search_paths()
        if extra_paths:
            vst3_paths.extend(Path(p) for p in extra_paths)

        settings = get_settings()
        for p in settings.extra_vst3_paths:
            vst3_paths.append(Path(p))

        for search_dir in vst3_paths:
            if not search_dir.exists():
                continue
            for vst3 in sorted(search_dir.glob("**/*.vst3")):
                meta = _parse_vst3_info(vst3)
                if meta:
                    self._register(meta)

        # AU 검색 (macOS만)
        for search_dir in get_au_search_paths():
            if not search_dir.exists():
                continue
            for au in sorted(search_dir.glob("*.component")):
                meta = _parse_au_info(au)
                if meta:
                    # VST3와 중복 시 VST3 우선
                    if meta.short_name not in self._plugins:
                        self._register(meta)

        self._save_cache()
        return list(self._plugins.values())

    def _register(self, meta: PluginMeta) -> None:
        """플러그인 등록 + 별칭 매핑"""
        self._plugins[meta.short_name] = meta
        for alias in meta.aliases:
            self._alias_map[alias] = meta.short_name

    def list(
        self,
        fmt: Optional[str] = None,
        vendor: Optional[str] = None,
    ) -> list[PluginMeta]:
        """등록된 플러그인 목록 (필터 옵션)"""
        if not self._plugins:
            self.scan()

        results = list(self._plugins.values())

        if fmt:
            results = [p for p in results if p.format == fmt]
        if vendor:
            vendor_lower = vendor.lower()
            results = [p for p in results if vendor_lower in p.vendor.lower()]

        return results

    def get(self, name: str) -> Optional[PluginMeta]:
        """이름 또는 별칭으로 플러그인 검색"""
        if not self._plugins:
            self.scan()

        # 정확한 short_name 매칭
        if name in self._plugins:
            return self._plugins[name]

        # 별칭 매칭
        if name in self._alias_map:
            return self._plugins[self._alias_map[name]]

        # 부분 매칭 (short_name에 포함)
        name_lower = name.lower()
        candidates = [
            p for p in self._plugins.values()
            if name_lower in p.short_name or name_lower in p.name.lower()
        ]
        if len(candidates) == 1:
            return candidates[0]

        return None

    def _try_load_cache(self) -> bool:
        """캐시 파일에서 플러그인 목록 로드"""
        if not self._cache_path.exists():
            return False

        try:
            data = json.loads(self._cache_path.read_text())
            for item in data:
                meta = PluginMeta(**item)
                # 캐시된 플러그인 경로가 아직 유효한지 확인
                if Path(meta.path).exists():
                    self._register(meta)
            return bool(self._plugins)
        except Exception:
            logger.warning("플러그인 캐시 로드 실패, 재스캔 필요")
            return False

    def _save_cache(self) -> None:
        """플러그인 목록을 캐시 파일에 저장"""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in self._plugins.values()]
        self._cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# 모듈 레벨 싱글턴
_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry
