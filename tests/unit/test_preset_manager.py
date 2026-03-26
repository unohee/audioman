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
