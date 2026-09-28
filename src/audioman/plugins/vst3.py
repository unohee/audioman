# Created: 2026-03-21
# Purpose: pedalboard 기반 VST3 플러그인 래퍼

import logging
import math
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from audioman.plugins.parameter import ParameterInfo

logger = logging.getLogger(__name__)
_FD_REDIRECT_LOCK = threading.Lock()


def find_parameter_info(infos: list[ParameterInfo], name: str) -> Optional[ParameterInfo]:
    """Find the ``ParameterInfo`` for a user-supplied parameter name.

    Both spellings the wrapper accepts are matched: the plugin's python
    attribute name (``air_db``) and the space-separated label (``air db``).
    Returns ``None`` when the name is unknown to the plugin.
    """
    wanted = {name, name.replace(" ", "_"), name.replace("_", " ")}
    for info in infos:
        if info.name in wanted:
            return info
    return None


def validate_parameter_value(name: str, value: Any, info: ParameterInfo) -> None:
    """Reject parameter values that are non-finite or outside the plugin's range.

    Bounds come from ``ParameterInfo.min_value`` / ``max_value``, which
    ``VST3PluginWrapper.get_parameters`` fills from the range the plugin itself
    reports. ``None`` bounds mean "unknown" and are skipped. Non-numeric values
    (enum labels, strings with unit suffixes) are left to the pedalboard binding,
    which knows the plugin's valid-value list.

    Raises:
        ValueError: the value is NaN/Inf, or below the minimum / above the maximum.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return  # not a number: enum label or suffixed string, plugin binding validates it
    if not math.isfinite(numeric):
        raise ValueError(f"parameter {name!r} must be finite, got {value!r}")

    low, high = info.min_value, info.max_value
    if low is not None and high is not None and low > high:
        # Garbage metadata from the plugin: both bounds are unusable, so neither
        # is enforced rather than rejecting values that may well be legitimate.
        logger.debug(
            f"parameter {name!r} reports an inverted range ({low} > {high}); "
            "skipping bounds validation"
        )
        return
    if low is not None and numeric < low:
        raise ValueError(f"parameter {name!r} value {value!r} is below the minimum {low}")
    if high is not None and numeric > high:
        raise ValueError(f"parameter {name!r} value {value!r} is above the maximum {high}")


def validate_audio_block(audio: np.ndarray) -> np.ndarray:
    """Normalize an input block to float32 ``(channels, samples)`` and reject junk.

    Empty blocks and non-finite samples are refused here rather than handed to
    the plugin, which would otherwise emit NaN/Inf into the render or, in the
    worst case, crash inside native code.

    Raises:
        ValueError: the block is empty or contains NaN/Inf samples.
    """
    audio = np.asarray(audio)
    if audio.size == 0:
        raise ValueError("audio block is empty; refusing to process it")
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    if audio.dtype != np.float32:
        # A float64 value that overflows float32 becomes Inf here; the finite check
        # below turns it into a ValueError, so numpy's cast warning is redundant.
        with np.errstate(over="ignore"):
            audio = audio.astype(np.float32)

    finite = np.isfinite(audio)
    if not finite.all():
        n_bad = int(finite.size - np.count_nonzero(finite))
        raise ValueError(
            f"audio block contains {n_bad} non-finite sample(s) (NaN/Inf); "
            "refusing to process it"
        )
    return audio


class VST3PluginWrapper:
    """pedalboard load_plugin 기반 VST3 래퍼"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._plugin = None
        self._parameters: Optional[list[ParameterInfo]] = None

    @property
    def name(self) -> str:
        return self._path.stem

    @property
    def is_loaded(self) -> bool:
        return self._plugin is not None

    def load(self) -> None:
        if self._plugin is not None:
            return
        import os
        from pedalboard import load_plugin
        logger.debug(f"VST3 로드: {self._path}")
        # iZotope 플러그인 로드 시 objc 런타임 로그가 stdout에 출력되는 문제 억제
        devnull: Optional[int] = None
        old_stdout: Optional[int] = None
        old_stderr: Optional[int] = None
        with _FD_REDIRECT_LOCK:
            if self._plugin is not None:
                return
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                old_stdout = os.dup(1)
                old_stderr = os.dup(2)
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
                self._plugin = load_plugin(str(self._path))
            finally:
                if old_stdout is not None:
                    try:
                        os.dup2(old_stdout, 1)
                    finally:
                        os.close(old_stdout)
                if old_stderr is not None:
                    try:
                        os.dup2(old_stderr, 2)
                    finally:
                        os.close(old_stderr)
                if devnull is not None:
                    os.close(devnull)

    def get_parameters(self) -> list[ParameterInfo]:
        """플러그인 파라미터 목록 추출"""
        if self._parameters is not None:
            return self._parameters

        self.load()
        params = []

        for attr_name, param in self._plugin.parameters.items():
            # 파라미터 범위 추출
            try:
                rng = param.range
                min_val, max_val, step = rng
            except Exception:
                min_val = max_val = step = None

            # 현재값 읽기
            try:
                current = getattr(param, "value", None)
                if current is None:
                    current = getattr(self._plugin, attr_name, None)
                if current is None:
                    current = getattr(self._plugin, attr_name.replace(" ", "_"), None)
            except Exception:
                current = None

            # 타입 추론
            if isinstance(current, bool):
                param_type = "bool"
            elif isinstance(current, str):
                param_type = "enum"
            else:
                param_type = "float"

            info = ParameterInfo(
                name=attr_name,
                label=attr_name.replace("_", " ").title(),
                min_value=float(min_val) if min_val is not None else None,
                max_value=float(max_val) if max_val is not None else None,
                default_value=None,
                step_size=float(step) if step is not None else None,
                current_value=current if not isinstance(current, (int, float)) else float(current),
                type=param_type,
            )
            params.append(info)

        self._parameters = params
        return params

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set parameters (name -> value mapping)

        Values outside the range the plugin reports (min/max) and NaN/Inf are
        rejected. Clamping out-of-range values silently would hide bugs, so a
        ValueError naming the parameter, its bound, and the value is raised
        instead — the plugin is left untouched.

        Plugins whose metadata cannot be read (some native plugins never return
        value strings, which makes pedalboard raise NotImplementedError) are
        logged and skipped: only the bounds check is lost, setting still works.
        """
        self.load()
        if not params:
            return
        infos: list[ParameterInfo] = []
        try:
            infos = self.get_parameters()
        except Exception as e:
            logger.warning(
                f"Could not read parameter ranges; skipping bounds validation: {e}"
            )
        for name, value in params.items():
            info = find_parameter_info(infos, name)
            if info is not None:
                validate_parameter_value(name, value, info)
            # 언더스코어/공백 양쪽 지원
            attr_name = name.replace(" ", "_")
            try:
                setattr(self._plugin, attr_name, value)
                logger.debug(f"파라미터 설정: {attr_name} = {value}")
            except AttributeError:
                # 공백 포함 이름 시도
                space_name = name.replace("_", " ")
                try:
                    setattr(self._plugin, space_name, value)
                except Exception as e:
                    raise AttributeError(f"파라미터 설정 실패: {name} = {value}") from e
        self._parameters = None

    def process(self, audio: np.ndarray, sample_rate: int, reset: bool = True) -> np.ndarray:
        """Process audio. audio shape: (channels, samples), float32

        Args:
            reset: True(기본)면 처리 전 플러그인 내부 상태를 리셋한다 — 전체 버퍼를
                   한 번에 통과시키는 오프라인 렌더에 맞다. 블록 단위 스트리밍
                   (DAW 재생 재현)에서는 reset=False로 호출해 이전 블록의 상태
                   (필터 히스토리, lookahead 버퍼)를 연속 유지해야 한다.
                   pedalboard 실측: reset=False 연속 호출은 whole-buffer 렌더와
                   비트 단위로 일치(-600dB)하지만, 매 블록 reset=True면 경계마다
                   상태가 끊겨 클릭이 발생(-7.5dB)한다.

        Raises:
            ValueError: the block is empty or carries NaN/Inf samples. Checked
                before the plugin is loaded, so native code never sees them.
        """
        # Validate before loading so a bad block fails fast, without touching the plugin
        audio = validate_audio_block(audio)
        self.load()
        return self._plugin.process(audio, sample_rate, reset=reset)

    def reset(self) -> None:
        """플러그인 상태 리셋"""
        if self._plugin is not None:
            self._plugin.reset()
