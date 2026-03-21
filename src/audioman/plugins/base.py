# Created: 2026-03-21
# Purpose: 플러그인 래퍼 Protocol

from typing import Any, Protocol, runtime_checkable

import numpy as np

from audioman.plugins.parameter import ParameterInfo


@runtime_checkable
class PluginWrapper(Protocol):
    """플러그인 래퍼 인터페이스"""

    @property
    def name(self) -> str: ...

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def get_parameters(self) -> list[ParameterInfo]: ...

    def set_parameters(self, params: dict[str, Any]) -> None: ...

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...

    def reset(self) -> None: ...
