# Created: 2026-03-21
# Purpose: Plugin discovery, registration, and caching system

import json
import logging
import plistlib
import re
from pathlib import Path
from typing import Optional

from audioman.config.paths import (
    get_au_search_paths,
    get_vst3_search_paths,
)
from audioman.config.settings import get_settings
from audioman.plugins.parameter import PluginMeta

logger = logging.getLogger(__name__)

# short_name alias mapping
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
    """Build a short_name from a plugin name
    "RX 10 Spectral De-noise" → "spectral-de-noise"
    """
    # strip the vendor/version prefix: "RX 10 ", "RX 9 ", etc.
    cleaned = re.sub(r"^RX\s+\d+\s+", "", name)
    # strip common vendor prefixes
    cleaned = re.sub(r"^(iZotope|Waves|FabFilter|Sonnox)\s+", "", cleaned, flags=re.IGNORECASE)
    # convert to lowercase kebab-case
    short = cleaned.strip().lower()
    short = re.sub(r"\s+", "-", short)
    # collapse repeated hyphens
    short = re.sub(r"-+", "-", short)
    return short


def _parse_vst3_info(vst3_path: Path) -> Optional[PluginMeta]:
    """Parse Info.plist from a VST3 bundle and build a PluginMeta"""
    plist_path = vst3_path / "Contents" / "Info.plist"
    if not plist_path.exists():
        return None

    try:
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
    except Exception:
        logger.warning(f"Failed to parse Info.plist: {plist_path}")
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
    """Parse Info.plist from an AU bundle"""
    plist_path = au_path / "Contents" / "Info.plist"
    if not plist_path.exists():
        return None

    try:
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
    except Exception:
        logger.warning(f"Failed to parse Info.plist: {plist_path}")
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
    """Plugin discovery, registration, and lookup"""

    def __init__(self) -> None:
        self._plugins: dict[str, PluginMeta] = {}  # short_name → meta
        self._alias_map: dict[str, str] = {}  # alias → short_name
        self._cache_path: Optional[Path] = None

    def scan(
        self,
        extra_paths: Optional[list[str]] = None,
        refresh: bool = False,
    ) -> list[PluginMeta]:
        """Scan the system for VST3/AU plugins"""
        settings = get_settings()
        self._cache_path = Path(settings.cache_dir) / "plugins.json"
        if not refresh and self._try_load_cache():
            return list(self._plugins.values())

        self._plugins.clear()
        self._alias_map.clear()

        # VST3 scan
        vst3_paths = get_vst3_search_paths()
        if extra_paths:
            vst3_paths.extend(Path(p) for p in extra_paths)

        for p in settings.extra_vst3_paths:
            vst3_paths.append(Path(p))

        for search_dir in vst3_paths:
            if not search_dir.exists():
                continue
            for vst3 in sorted(search_dir.glob("**/*.vst3")):
                meta = _parse_vst3_info(vst3)
                if meta:
                    self._register(meta)

        # AU scan (macOS only)
        au_paths = get_au_search_paths()
        au_paths.extend(Path(p) for p in settings.extra_au_paths)
        for search_dir in au_paths:
            if not search_dir.exists():
                continue
            for au in sorted(search_dir.glob("*.component")):
                meta = _parse_au_info(au)
                if meta:
                    # prefer VST3 when it duplicates an AU
                    if meta.short_name not in self._plugins:
                        self._register(meta)

        self._save_cache()
        return list(self._plugins.values())

    def _register(self, meta: PluginMeta) -> None:
        """Register a plugin + map its aliases"""
        self._plugins[meta.short_name] = meta
        for alias in meta.aliases:
            self._alias_map[alias] = meta.short_name

    def list(
        self,
        fmt: Optional[str] = None,
        vendor: Optional[str] = None,
    ) -> list[PluginMeta]:
        """List registered plugins (with filter options)"""
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
        """Look up a plugin by name or alias"""
        if not self._plugins:
            self.scan()

        # exact short_name match
        if name in self._plugins:
            return self._plugins[name]

        # alias match
        if name in self._alias_map:
            return self._plugins[self._alias_map[name]]

        # partial match (contained in short_name)
        name_lower = name.lower()
        candidates = [
            p for p in self._plugins.values()
            if name_lower in p.short_name or name_lower in p.name.lower()
        ]
        if len(candidates) == 1:
            return candidates[0]

        return None

    def _try_load_cache(self) -> bool:
        """Load the plugin list from the cache file"""
        if self._cache_path is None:
            self._cache_path = Path(get_settings().cache_dir) / "plugins.json"
        if not self._cache_path.exists():
            return False

        try:
            data = json.loads(self._cache_path.read_text())
            for item in data:
                meta = PluginMeta(**item)
                # check that the cached plugin path is still valid
                if Path(meta.path).exists():
                    self._register(meta)
            return bool(self._plugins)
        except Exception:
            logger.warning("Failed to load the plugin cache; a rescan is needed")
            return False

    def _save_cache(self) -> None:
        """Save the plugin list to the cache file"""
        if self._cache_path is None:
            self._cache_path = Path(get_settings().cache_dir) / "plugins.json"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in self._plugins.values()]
        self._cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# Module-level singleton
_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry
