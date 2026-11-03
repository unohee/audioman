# tests/unit/test_preset_manager.py

import pytest
from audioman.core.preset_manager import PresetManager


@pytest.fixture
def manager(tmp_path):
    return PresetManager(preset_dir=tmp_path / "presets")


class TestPresetManager:
    def test_save_and_load(self, manager):
        params = {"threshold": -20.0, "reduction": 12.0}
        manager.save("test_preset", plugin="denoise", params=params, description="test")
        loaded = manager.load("test_preset", plugin="denoise")
        assert loaded.parameters == params
        assert loaded.description == "test"

    def test_list_empty(self, manager):
        presets = manager.list(plugin="denoise")
        assert len(presets) == 0

    def test_list_after_save(self, manager):
        manager.save("p1", plugin="denoise", params={"a": 1})
        manager.save("p2", plugin="denoise", params={"b": 2})
        presets = manager.list(plugin="denoise")
        assert len(presets) == 2

    def test_delete(self, manager):
        manager.save("to_delete", plugin="denoise", params={"x": 1})
        manager.delete("to_delete", plugin="denoise")
        presets = manager.list(plugin="denoise")
        assert len(presets) == 0

    def test_load_nonexistent_raises(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.load("nonexistent", plugin="denoise")

    def test_overwrite(self, manager):
        manager.save("ow", plugin="test", params={"v": 1})
        manager.save("ow", plugin="test", params={"v": 2})
        loaded = manager.load("ow", plugin="test")
        assert loaded.parameters["v"] == 2

    @pytest.mark.parametrize("name,plugin", [
        ("../escape", "denoise"),
        ("preset", "../escape"),
        ("nested/preset", "denoise"),
    ])
    def test_rejects_path_traversal(self, manager, name, plugin):
        with pytest.raises(ValueError, match="single path component"):
            manager.save(name, plugin=plugin, params={})


class TestPresetManagerResidual:
    def test_list_skips_malformed_json(self, manager):
        manager.save("good", plugin="denoise", params={"a": 1})
        bad = manager._plugin_dir("denoise") / "broken.json"
        bad.write_text("{not json")
        presets = manager.list(plugin="denoise")
        assert [p.name for p in presets] == ["good"]

    def test_delete_missing_raises(self, manager):
        with pytest.raises(FileNotFoundError, match="Preset not found"):
            manager.delete("does-not-exist", plugin="denoise")

    def test_find_preset_without_plugin_scans_dirs(self, manager):
        manager.save("scanme", plugin="denoise", params={"a": 1})
        found = manager._find_preset("scanme")
        assert found is not None and found.exists()

    def test_find_preset_without_plugin_missing_dir(self, manager):
        # preset dir never created → early None
        assert manager._find_preset("nothing") is None

    def test_find_preset_without_plugin_not_found(self, manager):
        manager.save("only_one", plugin="denoise", params={})
        assert manager._find_preset("absent") is None

    def test_find_preset_skips_non_directory_entries(self, manager):
        manager._dir.mkdir(parents=True, exist_ok=True)
        (manager._dir / "stray.txt").write_text("x")
        manager.save("real", plugin="denoise", params={})
        assert manager._find_preset("real") is not None
