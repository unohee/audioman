# tests/unit/test_plugin_wrapper_protocol.py — PluginWrapper contract tests.
#
# Verifies that the Protocol in plugins/base.py matches the real wrapper
# (VST3PluginWrapper) and the streaming layer's 3-arg process(audio, sr, reset)
# contract, via runtime isinstance checks and a real call.

import inspect

import numpy as np
import pytest

from audioman.core.streaming import make_wrapper_process_fn
from audioman.plugins.base import PluginWrapper
from audioman.plugins.vst3 import VST3PluginWrapper


def _param_names(func) -> list[str]:
    return [name for name in inspect.signature(func).parameters if name != "self"]


class ConformingWrapper:
    """Minimal implementation satisfying the Protocol (no real DSP)."""

    def __init__(self) -> None:
        self._loaded = False

    @property
    def name(self) -> str:
        return "conforming"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def get_parameters(self) -> list:
        return []

    def set_parameters(self, params: dict) -> None:
        pass

    def process(self, audio: np.ndarray, sample_rate: int, reset: bool = True) -> np.ndarray:
        return audio

    def reset(self) -> None:
        pass


class NonConformingWrapper:
    """Missing reset() — a Protocol violation."""

    def __init__(self) -> None:
        self._loaded = True

    @property
    def name(self) -> str:
        return "non-conforming"

    def load(self) -> None:
        pass

    def get_parameters(self) -> list:
        return []

    def set_parameters(self, params: dict) -> None:
        pass

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return audio


class TestProtocolConformance:
    def test_conforming_object_passes_isinstance(self):
        assert isinstance(ConformingWrapper(), PluginWrapper)

    def test_nonconforming_object_fails_isinstance(self):
        assert not isinstance(NonConformingWrapper(), PluginWrapper)

    def test_real_vst3_wrapper_conforms(self, tmp_path):
        """The real implementation must satisfy the Protocol (drift guard)."""
        wrapper = VST3PluginWrapper(tmp_path / "Fake.vst3")
        assert isinstance(wrapper, PluginWrapper)

    def test_protocol_process_matches_implementation_signature(self):
        """The Protocol declaration must carry the same parameters as the impl.

        Regression: the Protocol declared process(audio, sample_rate) (2-arg)
        while the implementation and streaming.ProcessFn are 3-arg (with reset).
        """
        assert _param_names(PluginWrapper.process) == _param_names(VST3PluginWrapper.process)
        assert _param_names(PluginWrapper.process) == ["audio", "sample_rate", "reset"]

    def test_reset_defaults_to_true(self):
        """reset must default to True so offline rendering works unchanged."""
        assert inspect.signature(PluginWrapper.process).parameters["reset"].default is True
        assert inspect.signature(VST3PluginWrapper.process).parameters["reset"].default is True


class TestStreamingIntegration:
    def test_streaming_process_fn_calls_three_arg_form(self):
        """The streaming layer must be able to call a Protocol object 3-arg."""
        wrapper = ConformingWrapper()
        fn = make_wrapper_process_fn(wrapper)
        block = np.zeros((1, 32), dtype=np.float32)
        out = fn(block, 48000, False)
        assert out.shape == (1, 32)

    def test_protocol_object_accepts_explicit_reset(self):
        wrapper: PluginWrapper = ConformingWrapper()
        block = np.ones((2, 16), dtype=np.float32)
        assert wrapper.process(block, 44100, reset=False) is block
        assert wrapper.process(block, 44100).shape == (2, 16)


class TestProtocolIsRuntimeCheckable:
    def test_isinstance_rejects_plain_object(self):
        assert not isinstance(object(), PluginWrapper)

    def test_protocol_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            PluginWrapper()
