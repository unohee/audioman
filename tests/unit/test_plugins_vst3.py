# tests/unit/test_plugins_vst3.py — VST3PluginWrapper method-level tests.
#
# This host has zero registrable VST3 binaries (AUD-1857), so the wrapper is
# driven against a fake pedalboard plugin object: `pedalboard.load_plugin` is
# replaced through monkeypatch. The pure helper functions
# (find_parameter_info / validate_parameter_value / validate_audio_block) are
# covered in test_vst3_validation.py; this file covers the wrapper methods
# themselves — loading, the fd-redirect window, the parameter cache, and the
# process/reset plumbing.

from __future__ import annotations

import logging
import os

import numpy as np
import pytest

from audioman.plugins import vst3 as vst3_mod
from audioman.plugins.parameter import ParameterInfo
from audioman.plugins.vst3 import VST3PluginWrapper


# --- fakes ------------------------------------------------------------------


class _FakeParam:
    """Stand-in for a pedalboard plugin parameter descriptor."""

    def __init__(self, rng=None, value=None, *, range_error=None, value_error=None):
        self._range = rng
        self._value = value
        self._range_error = range_error
        self._value_error = value_error

    @property
    def range(self):
        if self._range_error is not None:
            raise self._range_error
        return self._range

    @property
    def value(self):
        if self._value_error is not None:
            raise self._value_error
        return self._value


class _FakePlugin:
    """Minimal pedalboard-plugin stand-in.

    ``__setattr__`` rejects names the plugin does not know, which is how a real
    plugin binding reports an unknown parameter (AttributeError). Everything the
    wrapper reads is exposed as a plain attribute or property.
    """

    def __init__(self, parameters=None, attributes=None, transform=None):
        object.__setattr__(self, "_parameters", parameters if parameters is not None else {})
        object.__setattr__(self, "_allowed", set(attributes or ()))
        object.__setattr__(self, "_transform", transform)
        object.__setattr__(self, "processed", [])
        object.__setattr__(self, "reset_calls", 0)
        for key, value in (attributes or {}).items():
            object.__setattr__(self, key, value)

    def __setattr__(self, name, value):
        if name.startswith("_") or name in self._allowed:
            object.__setattr__(self, name, value)
            params = self.__dict__.get("_parameters")
            # A real plugin binding reflects a write back into the descriptor, so a
            # re-read of ``param.value`` must show the new value.
            if isinstance(params, dict) and name in params:
                object.__setattr__(params[name], "_value", value)
            return
        raise AttributeError(f"plugin has no parameter {name!r}")

    @property
    def parameters(self):
        if isinstance(self._parameters, BaseException):
            raise self._parameters
        return self._parameters

    def process(self, audio, sample_rate, reset=True):
        self.processed.append((np.array(audio, copy=True), sample_rate, reset))
        if self._transform is not None:
            return self._transform(audio)
        return audio

    def reset(self):
        object.__setattr__(self, "reset_calls", self.reset_calls + 1)


class _LoaderStub:
    """Records calls to the patched ``pedalboard.load_plugin``."""

    def __init__(self, plugin=None, error=None):
        self.plugin = plugin
        self.error = error
        self.calls: list[str] = []
        self.fd_links: list[dict[int, str]] = []

    def __call__(self, path):
        self.calls.append(path)
        self.fd_links.append(_fd_links())
        if self.error is not None:
            raise self.error
        return self.plugin


_PROCFS_FDS = os.path.isdir("/proc/self/fd")
no_procfs = pytest.mark.skipif(not _PROCFS_FDS, reason="requires /proc/self/fd to inspect open descriptors")


def _fd_links() -> dict[int, str]:
    """Targets of the process's open descriptors, keyed by fd number."""
    links = {}
    for entry in os.listdir("/proc/self/fd"):
        try:
            links[int(entry)] = os.readlink(f"/proc/self/fd/{entry}")
        except OSError:  # the descriptor vanished between listdir and readlink
            continue
    return links


def _fd_stat(fd: int):
    st = os.fstat(fd)
    return (st.st_dev, st.st_ino)


@pytest.fixture
def loader(monkeypatch):
    stub = _LoaderStub()
    monkeypatch.setattr("pedalboard.load_plugin", stub)
    return stub


@pytest.fixture
def wrapper(tmp_path, loader):
    return VST3PluginWrapper(tmp_path / "FakePlugin.vst3")


@pytest.fixture
def loaded_wrapper(wrapper, loader):
    """A wrapper whose fake plugin is already attached (load() short-circuits)."""
    plugin = _FakePlugin()
    loader.plugin = plugin
    wrapper.load()
    return wrapper


# --- load() -----------------------------------------------------------------


class TestLoad:
    def test_name_and_is_loaded_before_load(self, wrapper):
        assert wrapper.name == "FakePlugin"
        assert wrapper.is_loaded is False

    def test_load_marks_wrapper_loaded(self, wrapper, loader):
        plugin = _FakePlugin()
        loader.plugin = plugin
        wrapper.load()
        assert wrapper.is_loaded is True
        assert wrapper._plugin is plugin

    def test_load_passes_stringified_path(self, wrapper, loader):
        loader.plugin = _FakePlugin()
        wrapper.load()
        assert loader.calls == [str(wrapper._path)]
        assert isinstance(loader.calls[0], str)

    def test_load_is_idempotent(self, wrapper, loader):
        plugin = _FakePlugin()
        loader.plugin = plugin
        wrapper.load()
        wrapper.load()
        assert loader.calls == [str(wrapper._path)]
        assert wrapper._plugin is plugin

    @no_procfs
    def test_load_redirects_stdout_to_devnull_during_load(self, wrapper, loader):
        loader.plugin = _FakePlugin()
        wrapper.load()
        assert loader.fd_links[0][1] == os.devnull
        assert loader.fd_links[0][2] == os.devnull

    @no_procfs
    def test_load_restores_stdout_and_stderr(self, wrapper, loader):
        loader.plugin = _FakePlugin()
        before = (_fd_stat(1), _fd_stat(2))
        wrapper.load()
        assert _fd_links()[1] != os.devnull
        assert (_fd_stat(1), _fd_stat(2)) == before

    @no_procfs
    def test_load_does_not_leak_descriptors(self, wrapper, loader):
        loader.plugin = _FakePlugin()
        before = sorted(_fd_links())
        wrapper.load()
        assert sorted(_fd_links()) == before

    @no_procfs
    def test_failed_load_leaves_no_leaked_descriptors(self, wrapper, loader):
        loader.error = ImportError("not a VST3 bundle")
        before = sorted(_fd_links())
        with pytest.raises(ImportError, match="not a VST3 bundle"):
            wrapper.load()
        assert sorted(_fd_links()) == before

    def test_load_failure_raises_and_leaves_wrapper_unloaded(self, wrapper, loader):
        loader.error = ImportError("Unable to load plugin: file not found")
        with pytest.raises(ImportError, match="file not found"):
            wrapper.load()
        assert wrapper._plugin is None
        assert wrapper.is_loaded is False

    @no_procfs
    def test_load_restores_stdout_after_failure(self, wrapper, loader):
        loader.error = ImportError("boom")
        before = (_fd_stat(1), _fd_stat(2))
        with pytest.raises(ImportError):
            wrapper.load()
        assert (_fd_stat(1), _fd_stat(2)) == before

    def test_load_recovers_after_a_failed_attempt(self, wrapper, loader):
        plugin = _FakePlugin()
        loader.error = ImportError("transient")
        with pytest.raises(ImportError):
            wrapper.load()
        loader.error = None
        loader.plugin = plugin
        wrapper.load()
        assert wrapper._plugin is plugin
        assert len(loader.calls) == 2

    def test_load_skips_when_plugin_appeared_while_waiting_for_lock(self, wrapper, loader, monkeypatch):
        """Double-checked locking: a plugin loaded by another thread wins."""
        other = _FakePlugin()

        class _SideEffectLock:
            def __enter__(self):
                wrapper._plugin = other
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(vst3_mod, "_FD_REDIRECT_LOCK", _SideEffectLock())
        wrapper.load()
        assert wrapper._plugin is other
        assert loader.calls == []


# --- get_parameters() -------------------------------------------------------


class TestGetParameters:
    def test_maps_names_ranges_and_types(self, wrapper, loader):
        params = {
            "gain_db": _FakeParam((-18.0, 6.0, 0.5), 3.0),
            "bypass": _FakeParam((0.0, 1.0, None), True),
            "style": _FakeParam(None, "Soft"),
        }
        loader.plugin = _FakePlugin(params)
        infos = {info.name: info for info in wrapper.get_parameters()}

        assert set(infos) == {"gain_db", "bypass", "style"}

        gain = infos["gain_db"]
        assert isinstance(gain, ParameterInfo)
        assert gain.label == "Gain Db"
        assert (gain.min_value, gain.max_value, gain.step_size) == (-18.0, 6.0, 0.5)
        assert gain.default_value is None
        assert gain.current_value == 3.0
        assert gain.type == "float"

        bypass = infos["bypass"]
        assert bypass.type == "bool"
        # ``bool`` is an ``int`` subclass, so the numeric cast in get_parameters
        # applies to it as well; the ``type`` field still reports "bool".
        assert bypass.current_value == 1.0

        style = infos["style"]
        assert style.type == "enum"
        assert style.current_value == "Soft"
        assert (style.min_value, style.max_value, style.step_size) == (None, None, None)

    def test_reads_current_value_from_plugin_attribute(self, wrapper, loader):
        loader.plugin = _FakePlugin(
            {"air_db": _FakeParam((0.0, 12.0, 0.1), value=None)},
            {"air_db": 0.25},
        )
        info = wrapper.get_parameters()[0]
        assert info.current_value == 0.25
        assert isinstance(info.current_value, float)

    def test_underscore_fallback_for_spaced_parameter_name(self, wrapper, loader):
        """A plugin exposing "air db" as an attribute still yields its value."""
        loader.plugin = _FakePlugin(
            {"air db": _FakeParam((0.0, 12.0, 0.1), value=None)},
            {"air_db": 0.5},
        )
        info = wrapper.get_parameters()[0]
        assert info.name == "air db"
        assert info.current_value == 0.5

    def test_tolerates_unreadable_range(self, wrapper, loader):
        loader.plugin = _FakePlugin({"weird": _FakeParam(range_error=RuntimeError("no range"))})
        info = wrapper.get_parameters()[0]
        assert (info.min_value, info.max_value, info.step_size) == (None, None, None)

    def test_tolerates_unreadable_value(self, wrapper, loader):
        loader.plugin = _FakePlugin({"weird": _FakeParam((0.0, 1.0, None), value_error=RuntimeError("no value"))})
        info = wrapper.get_parameters()[0]
        assert info.current_value is None
        assert info.type == "float"

    def test_casts_int_current_value_to_float(self, wrapper, loader):
        loader.plugin = _FakePlugin({"voices": _FakeParam((1.0, 8.0, 1.0), 3)})
        info = wrapper.get_parameters()[0]
        assert info.current_value == 3.0
        assert isinstance(info.current_value, float)

    def test_enum_current_value_kept_as_string(self, wrapper, loader):
        loader.plugin = _FakePlugin({"mode": _FakeParam((0.0, 2.0, 1.0), "Wide")})
        info = wrapper.get_parameters()[0]
        assert info.current_value == "Wide"
        assert info.type == "enum"

    def test_result_is_cached(self, wrapper, loader):
        loader.plugin = _FakePlugin({"gain_db": _FakeParam((0.0, 1.0, None), 0.5)})
        first = wrapper.get_parameters()
        second = wrapper.get_parameters()
        assert second is first
        assert loader.calls == [str(wrapper._path)]

    def test_loads_plugin_on_first_call(self, wrapper, loader):
        loader.plugin = _FakePlugin()
        assert wrapper.is_loaded is False
        wrapper.get_parameters()
        assert wrapper.is_loaded is True

    def test_empty_parameter_table(self, loaded_wrapper):
        assert loaded_wrapper.get_parameters() == []


# --- set_parameters() -------------------------------------------------------


class TestSetParameters:
    def test_empty_mapping_sets_nothing(self, wrapper, loader):
        """An empty mapping still loads (load() precedes the early return) but
        leaves the parameter cache untouched."""
        loader.plugin = _FakePlugin({"gain_db": _FakeParam((0.0, 1.0, None), 0.0)})
        wrapper._parameters = ["sentinel"]
        wrapper.set_parameters({})
        assert loader.calls == [str(wrapper._path)]
        assert wrapper._parameters == ["sentinel"]

    def test_sets_underscore_attribute(self, loaded_wrapper):
        loaded_wrapper._plugin = _FakePlugin(
            {"gain_db": _FakeParam((-18.0, 6.0, 0.5), 0.0)}, {"gain_db": 0.0}
        )
        loaded_wrapper.set_parameters({"gain_db": 3.0})
        assert loaded_wrapper._plugin.gain_db == 3.0

    def test_accepts_space_spelling_for_underscore_parameter(self, loaded_wrapper):
        loaded_wrapper._plugin = _FakePlugin(
            {"gain_db": _FakeParam((-18.0, 6.0, 0.5), 0.0)}, {"gain_db": 0.0}
        )
        loaded_wrapper.set_parameters({"gain db": 1.5})
        assert loaded_wrapper._plugin.gain_db == 1.5

    def test_falls_back_to_space_attribute(self, loaded_wrapper):
        loaded_wrapper._plugin = _FakePlugin(
            {"air db": _FakeParam((0.0, 12.0, 0.1), 0.0)}, {"air db": 0.0}
        )
        loaded_wrapper.set_parameters({"air db": 4.0})
        assert loaded_wrapper._plugin.__dict__["air db"] == 4.0

    def test_unknown_parameter_name_raises_attribute_error(self, loaded_wrapper):
        loaded_wrapper._plugin = _FakePlugin({}, {"known": 0.0})
        with pytest.raises(AttributeError, match="missing param"):
            loaded_wrapper.set_parameters({"missing param": 1.0})
        assert loaded_wrapper._plugin.known == 0.0

    def test_unknown_name_without_metadata_still_sets(self, loaded_wrapper):
        """Names the plugin does not describe have no bounds to check."""
        loaded_wrapper._plugin = _FakePlugin({}, {"extra_gain": 0.0})
        loaded_wrapper.set_parameters({"extra_gain": 99.0})
        assert loaded_wrapper._plugin.extra_gain == 99.0

    def test_unreadable_metadata_logs_warning_and_sets(self, loaded_wrapper, caplog):
        loaded_wrapper._plugin = _FakePlugin(
            NotImplementedError("value strings unavailable"), {"gain_db": 0.0}
        )
        with caplog.at_level(logging.WARNING, logger="audioman.plugins.vst3"):
            loaded_wrapper.set_parameters({"gain_db": 2.5})
        assert loaded_wrapper._plugin.gain_db == 2.5
        assert "skipping bounds validation" in caplog.text

    def test_rejects_nan_before_touching_plugin(self, loaded_wrapper):
        loaded_wrapper._plugin = _FakePlugin(
            {"gain_db": _FakeParam((-18.0, 6.0, 0.5), 0.0)}, {"gain_db": 0.0}
        )
        with pytest.raises(ValueError, match="must be finite"):
            loaded_wrapper.set_parameters({"gain_db": float("nan")})
        assert loaded_wrapper._plugin.gain_db == 0.0

    def test_bool_parameter_value(self, loaded_wrapper):
        loaded_wrapper._plugin = _FakePlugin(
            {"bypass": _FakeParam((0.0, 1.0, None), False)}, {"bypass": False}
        )
        loaded_wrapper.set_parameters({"bypass": True})
        assert loaded_wrapper._plugin.bypass is True

    def test_sets_multiple_parameters(self, loaded_wrapper):
        loaded_wrapper._plugin = _FakePlugin(
            {
                "gain_db": _FakeParam((-18.0, 6.0, 0.5), 0.0),
                "mix": _FakeParam((0.0, 1.0, 0.01), 0.5),
            },
            {"gain_db": 0.0, "mix": 0.5},
        )
        loaded_wrapper.set_parameters({"gain_db": -6.0, "mix": 0.25})
        assert loaded_wrapper._plugin.gain_db == -6.0
        assert loaded_wrapper._plugin.mix == 0.25

    def test_invalidates_parameter_cache(self, loaded_wrapper):
        plugin = _FakePlugin({"gain_db": _FakeParam((-18.0, 6.0, 0.5), 0.0)}, {"gain_db": 0.0})
        loaded_wrapper._plugin = plugin
        before = loaded_wrapper.get_parameters()
        loaded_wrapper.set_parameters({"gain_db": 3.0})
        assert loaded_wrapper._parameters is None
        after = loaded_wrapper.get_parameters()
        assert after is not before
        assert after[0].current_value == 3.0

    def test_loads_plugin_on_demand(self, wrapper, loader):
        loader.plugin = _FakePlugin({"gain_db": _FakeParam((0.0, 1.0, None), 0.0)}, {"gain_db": 0.0})
        wrapper.set_parameters({"gain_db": 0.75})
        assert wrapper.is_loaded is True
        assert wrapper._plugin.gain_db == 0.75


# --- process() --------------------------------------------------------------


class TestProcess:
    def test_promotes_mono_block_to_two_dimensions(self, loaded_wrapper):
        mono = np.linspace(0.0, 1.0, 64, dtype=np.float32)
        loaded_wrapper.process(mono, 48000)
        block, sr, reset = loaded_wrapper._plugin.processed[0]
        assert block.shape == (1, 64)
        assert sr == 48000
        assert reset is True

    def test_casts_int16_block_to_float32(self, loaded_wrapper):
        audio = np.array([[100, -100, 200]], dtype=np.int16)
        out = loaded_wrapper.process(audio, 44100)
        block = loaded_wrapper._plugin.processed[0][0]
        assert block.dtype == np.float32
        assert out.dtype == np.float32
        np.testing.assert_array_equal(block, audio.astype(np.float32))

    def test_casts_float64_block_to_float32(self, loaded_wrapper):
        audio = np.full((2, 32), 0.5, dtype=np.float64)
        out = loaded_wrapper.process(audio, 44100)
        assert loaded_wrapper._plugin.processed[0][0].dtype == np.float32
        assert out.dtype == np.float32

    def test_forwards_sample_rate_and_reset(self, loaded_wrapper):
        audio = np.zeros((2, 16), dtype=np.float32)
        loaded_wrapper.process(audio, 96000, reset=False)
        loaded_wrapper.process(audio, 96000)
        first, second = loaded_wrapper._plugin.processed
        assert (first[1], first[2]) == (96000, False)
        assert second[2] is True

    def test_returns_plugin_output_unchanged(self, loaded_wrapper):
        loaded_wrapper._plugin = _FakePlugin(transform=lambda a: a * 2.0)
        audio = np.full((1, 8), 0.25, dtype=np.float32)
        out = loaded_wrapper.process(audio, 48000)
        np.testing.assert_allclose(out, 0.5)

    def test_loads_plugin_on_demand(self, wrapper, loader):
        loader.plugin = _FakePlugin()
        out = wrapper.process(np.zeros((1, 8), dtype=np.float32), 48000)
        assert wrapper.is_loaded is True
        assert out.shape == (1, 8)

    def test_rejects_empty_block_without_loading(self, wrapper, loader):
        with pytest.raises(ValueError, match="empty"):
            wrapper.process(np.zeros((2, 0), dtype=np.float32), 48000)
        assert wrapper._plugin is None
        assert loader.calls == []

    def test_rejects_nonfinite_block_without_loading(self, wrapper, loader):
        audio = np.zeros((2, 8), dtype=np.float32)
        audio[1, 3] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            wrapper.process(audio, 48000)
        assert wrapper._plugin is None
        assert loader.calls == []


# --- reset() ----------------------------------------------------------------


class TestReset:
    def test_reset_without_plugin_is_a_noop(self, wrapper):
        wrapper.reset()
        assert wrapper._plugin is None

    def test_reset_calls_plugin_reset(self, loaded_wrapper):
        loaded_wrapper.reset()
        loaded_wrapper.reset()
        assert loaded_wrapper._plugin.reset_calls == 2

    def test_reset_does_not_load_the_plugin(self, wrapper, loader):
        wrapper.reset()
        assert loader.calls == []
