"""OpenSwarm 감사에서 발견된 CLI/config 회귀 방지 테스트."""

import plistlib

import pytest

from audioman.cli import schemas_cmd, stream
from audioman.cli.app import build_parser
from audioman.config import settings as settings_module
from audioman.config.settings import AudiomanSettings
from audioman.core.preset_manager import PresetManager
from audioman.core.registry import PluginRegistry


@pytest.mark.parametrize("argv", [
    ["analyze", "input.wav", "--frame-size", "0"],
    ["analyze", "input.wav", "--hop", "-1"],
    ["analyze", "input.wav", "--spectrum-fft", "0"],
    ["chain", "input.wav", "--steps", "gain", "--output", "out.wav", "--workers", "0"],
    ["process", "input.wav", "--plugin", "x", "--output", "out.wav", "--passes", "0"],
    ["visualize", "input.wav", "--hop", "0"],
])
def test_dimension_cli_options_are_rejected_by_argparse(argv):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(argv)
    assert exc_info.value.code == 2


@pytest.mark.parametrize("raw", [",", "0,128", "64,-1"])
def test_stream_blocks_are_rejected_before_execution(raw):
    with pytest.raises(Exception):
        stream._parse_blocks(raw)


def test_schema_show_rejects_path_traversal():
    args = build_parser().parse_args(["schemas", "show", "../../pyproject"])
    with pytest.raises(SystemExit) as exc_info:
        schemas_cmd._run_show(args)
    assert exc_info.value.code == 2


def test_configured_preset_cache_and_au_paths_are_consumed(tmp_path, monkeypatch):
    preset_dir = tmp_path / "configured-presets"
    cache_dir = tmp_path / "configured-cache"
    au_dir = tmp_path / "configured-au"
    component = au_dir / "Example.component" / "Contents"
    component.mkdir(parents=True)
    with (component / "Info.plist").open("wb") as handle:
        plistlib.dump({"CFBundleName": "Example AU", "CFBundleIdentifier": "com.example.au"}, handle)

    configured = AudiomanSettings(
        preset_dir=str(preset_dir),
        cache_dir=str(cache_dir),
        extra_au_paths=[str(au_dir)],
    )
    monkeypatch.setattr(settings_module, "_settings", configured)

    manager = PresetManager()
    manager.save("configured", "example", {"enabled": True})
    assert (preset_dir / "example" / "configured.json").is_file()

    registry = PluginRegistry()
    plugins = registry.scan(refresh=True)
    assert any(plugin.path.endswith("Example.component") for plugin in plugins)
    assert (cache_dir / "plugins.json").is_file()


def test_toml_settings_source_uses_stdlib_compatible_api(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('default_sample_rate = 48000\nextra_au_paths = ["/tmp/au"]\n')

    class TestSettings(AudiomanSettings):
        config_file = config

    settings = TestSettings()
    assert settings.default_sample_rate == 48000
    assert settings.extra_au_paths == ["/tmp/au"]
