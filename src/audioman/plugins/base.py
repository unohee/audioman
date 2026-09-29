# Created: 2026-03-21
# Purpose: Plugin wrapper Protocol — the runtime-checkable contract every
# plugin wrapper (VST3, AU, built-in Python plugins) must satisfy.

from typing import Any, Protocol, runtime_checkable

import numpy as np

from audioman.plugins.parameter import ParameterInfo


@runtime_checkable
class PluginWrapper(Protocol):
    """Plugin wrapper interface.

    Implemented by ``audioman.plugins.vst3.VST3PluginWrapper`` and any other
    wrapper that exposes audio through the CLI/engine/streaming layers.

    ``process`` takes a ``reset`` flag because the streaming layer calls
    wrappers block by block and needs to control whether plugin state
    (filter history, lookahead buffers) carries over between blocks. See
    ``audioman.core.streaming.ProcessFn``.
    """

    @property
    def name(self) -> str: ...

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def get_parameters(self) -> list[ParameterInfo]: ...

    def set_parameters(self, params: dict[str, Any]) -> None: ...

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        reset: bool = True,
    ) -> np.ndarray: ...

    def reset(self) -> None: ...
