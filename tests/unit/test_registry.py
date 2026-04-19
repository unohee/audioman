# tests/unit/test_registry.py — 플러그인 레지스트리 단위 테스트

import json
import plistlib

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from audioman.core.registry import (
    _name_to_short_name,
    _parse_vst3_info,
    _parse_au_info,
    PluginRegistry,
    ALIASES,
)
from audioman.plugins.parameter import PluginMeta


class TestNameToShortName:
    """플러그인 이름 → short_name 변환"""

    def test_rx10_prefix(self):
        assert _name_to_short_name("RX 10 Spectral De-noise") == "spectral-de-noise"

    def test_rx9_prefix(self):
        assert _name_to_short_name("RX 9 De-hum") == "de-hum"

    def test_izotope_prefix(self):
        assert _name_to_short_name("iZotope Ozone 11") == "ozone-11"

    def test_waves_prefix(self):
        assert _name_to_short_name("Waves C6") == "c6"

    def test_fabfilter_prefix(self):
        assert _name_to_short_name("FabFilter Pro-Q 3") == "pro-q-3"

    def test_sonnox_prefix(self):
        assert _name_to_short_name("Sonnox Oxford Limiter") == "oxford-limiter"

    def test_simple_name(self):
        assert _name_to_short_name("Limiter") == "limiter"

    def test_spaces_to_hyphens(self):
        assert _name_to_short_name("My Great Plugin") == "my-great-plugin"

    def test_multiple_spaces_collapsed(self):
        assert _name_to_short_name("Some   Plugin") == "some-plugin"

    def test_case_insensitive_vendor(self):
        # "iZotope" 대소문자 무관 처리
        assert _name_to_short_name("IZOTOPE Trash") == "trash"


class TestParseVst3Info:
    """VST3 Info.plist 파싱"""

    def test_valid_plist(self, tmp_path):
        """유효한 VST3 번들에서 PluginMeta 생성"""
        vst3_dir = tmp_path / "TestPlugin.vst3"
        contents_dir = vst3_dir / "Contents"
        contents_dir.mkdir(parents=True)

        plist_data = {
            "CFBundleName": "RX 10 De-click",
            "CFBundleIdentifier": "com.izotope.rx10-declick",
            "CFBundleShortVersionString": "10.4.2",
        }
        plist_path = contents_dir / "Info.plist"
        with open(plist_path, "wb") as f:
            plistlib.dump(plist_data, f)

        meta = _parse_vst3_info(vst3_dir)
        assert meta is not None
        assert meta.name == "RX 10 De-click"
        assert meta.short_name == "de-click"
        assert meta.format == "vst3"
        assert meta.vendor == "izotope"
        assert meta.version == "10.4.2"
        assert "declick" in meta.aliases

    def test_missing_plist(self, tmp_path):
        """Info.plist 없으면 None"""
        vst3_dir = tmp_path / "NoInfo.vst3"
        vst3_dir.mkdir()
        assert _parse_vst3_info(vst3_dir) is None

    def test_invalid_plist(self, tmp_path):
        """손상된 plist → None"""
        vst3_dir = tmp_path / "Bad.vst3"
        contents_dir = vst3_dir / "Contents"
        contents_dir.mkdir(parents=True)
        (contents_dir / "Info.plist").write_text("not a plist")
        assert _parse_vst3_info(vst3_dir) is None

    def test_missing_bundle_name_uses_stem(self, tmp_path):
        """CFBundleName 없으면 폴더명 사용"""
        vst3_dir = tmp_path / "FallbackName.vst3"
        contents_dir = vst3_dir / "Contents"
        contents_dir.mkdir(parents=True)

        plist_data = {"CFBundleIdentifier": "com.test.plugin"}
        with open(contents_dir / "Info.plist", "wb") as f:
            plistlib.dump(plist_data, f)

        meta = _parse_vst3_info(vst3_dir)
        assert meta is not None
        assert meta.name == "FallbackName"


class TestParseAuInfo:
    """AU Info.plist 파싱"""

    def test_valid_plist(self, tmp_path):
        au_dir = tmp_path / "TestAU.component"
        contents_dir = au_dir / "Contents"
        contents_dir.mkdir(parents=True)

        plist_data = {
            "CFBundleName": "Spectral De-noise",
            "CFBundleIdentifier": "com.izotope.spectral-denoise",
            "CFBundleShortVersionString": "1.0",
        }
        with open(contents_dir / "Info.plist", "wb") as f:
            plistlib.dump(plist_data, f)

        meta = _parse_au_info(au_dir)
        assert meta is not None
        assert meta.format == "au"
        assert meta.short_name == "spectral-de-noise"


class TestPluginRegistry:
    """PluginRegistry 클래스"""

    @pytest.fixture
    def registry(self):
        """캐시/설정을 우회한 순수 레지스트리"""
        reg = PluginRegistry.__new__(PluginRegistry)
        reg._plugins = {}
        reg._alias_map = {}
        reg._cache_path = Path("/tmp/audioman_test_cache.json")
        return reg

    def test_register(self, registry):
        meta = PluginMeta(
            name="Test Plugin", short_name="test-plugin",
            path="/tmp/test.vst3", format="vst3",
            aliases=["tp", "test"],
        )
        registry._register(meta)
        assert "test-plugin" in registry._plugins
        assert registry._alias_map["tp"] == "test-plugin"
        assert registry._alias_map["test"] == "test-plugin"

    def test_get_by_short_name(self, registry):
        meta = PluginMeta(
            name="De-noise", short_name="de-noise",
            path="/tmp/denoise.vst3", format="vst3",
        )
        registry._register(meta)
        assert registry.get("de-noise") == meta

    def test_get_by_alias(self, registry):
        meta = PluginMeta(
            name="De-click", short_name="de-click",
            path="/tmp/declick.vst3", format="vst3",
            aliases=["declick"],
        )
        registry._register(meta)
        assert registry.get("declick") == meta

    def test_get_partial_match(self, registry):
        meta = PluginMeta(
            name="Spectral De-noise", short_name="spectral-de-noise",
            path="/tmp/spectral.vst3", format="vst3",
        )
        registry._register(meta)
        # 부분 매칭: "spectral" → unique match
        result = registry.get("spectral")
        assert result == meta

    def test_get_partial_ambiguous(self, registry):
        """부분 매칭 후보가 여러 개면 None"""
        for name in ["spectral-de-noise", "spectral-repair"]:
            registry._register(PluginMeta(
                name=name, short_name=name,
                path=f"/tmp/{name}.vst3", format="vst3",
            ))
        assert registry.get("spectral") is None

    def test_get_not_found(self, registry):
        assert registry.get("nonexistent") is None

    def test_list_all(self, registry):
        for i in range(3):
            registry._register(PluginMeta(
                name=f"P{i}", short_name=f"p{i}",
                path=f"/tmp/p{i}.vst3", format="vst3",
            ))
        assert len(registry.list()) == 3

    def test_list_filter_format(self, registry):
        registry._register(PluginMeta(
            name="VST", short_name="vst-plug",
            path="/tmp/v.vst3", format="vst3",
        ))
        registry._register(PluginMeta(
            name="AU", short_name="au-plug",
            path="/tmp/a.component", format="au",
        ))
        vst_only = registry.list(fmt="vst3")
        assert len(vst_only) == 1
        assert vst_only[0].format == "vst3"

    def test_list_filter_vendor(self, registry):
        registry._register(PluginMeta(
            name="P1", short_name="p1",
            path="/tmp/p1.vst3", format="vst3", vendor="izotope",
        ))
        registry._register(PluginMeta(
            name="P2", short_name="p2",
            path="/tmp/p2.vst3", format="vst3", vendor="waves",
        ))
        results = registry.list(vendor="izotope")
        assert len(results) == 1

    def test_save_and_load_cache(self, tmp_path, registry):
        """캐시 저장 → 로드 라운드트립"""
        registry._cache_path = tmp_path / "plugins.json"

        meta = PluginMeta(
            name="Cached Plugin", short_name="cached",
            path=str(tmp_path / "cached.vst3"), format="vst3",
            vendor="test", aliases=["cache-test"],
        )
        # 캐시 로드 시 경로 존재 확인하므로 더미 파일 생성
        (tmp_path / "cached.vst3").mkdir()

        registry._register(meta)
        registry._save_cache()

        # 새 레지스트리로 캐시 로드
        reg2 = PluginRegistry.__new__(PluginRegistry)
        reg2._plugins = {}
        reg2._alias_map = {}
        reg2._cache_path = registry._cache_path

        assert reg2._try_load_cache() is True
        assert "cached" in reg2._plugins
        assert reg2._alias_map["cache-test"] == "cached"

    def test_cache_skips_missing_paths(self, tmp_path, registry):
        """캐시된 플러그인 경로가 없으면 건너뜀"""
        registry._cache_path = tmp_path / "plugins.json"

        meta = PluginMeta(
            name="Gone", short_name="gone",
            path="/nonexistent/path/gone.vst3", format="vst3",
        )
        registry._register(meta)
        registry._save_cache()

        reg2 = PluginRegistry.__new__(PluginRegistry)
        reg2._plugins = {}
        reg2._alias_map = {}
        reg2._cache_path = registry._cache_path

        # 경로가 없으므로 로드 실패
        assert reg2._try_load_cache() is False
        assert len(reg2._plugins) == 0

    def test_cache_corrupt(self, tmp_path, registry):
        """손상된 캐시 파일 → False"""
        registry._cache_path = tmp_path / "plugins.json"
        registry._cache_path.write_text("not json!!!")

        assert registry._try_load_cache() is False


class TestAliases:
    """ALIASES 매핑 일관성"""

    def test_aliases_are_lowercase(self):
        for key, aliases in ALIASES.items():
            assert key == key.lower(), f"key '{key}' 소문자 아님"
            for alias in aliases:
                assert alias == alias.lower(), f"alias '{alias}' 소문자 아님"

    def test_no_duplicate_aliases(self):
        """서로 다른 키에 같은 별칭이 없어야 함"""
        all_aliases = []
        for aliases in ALIASES.values():
            all_aliases.extend(aliases)
        assert len(all_aliases) == len(set(all_aliases))
