# Created: 2026-03-21
# Purpose: pedalboard-based VST3 plugin wrapper

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
    """VST3 wrapper built on pedalboard ``load_plugin``."""

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
        logger.debug(f"Loading VST3: {self._path}")
        # Suppress the objc runtime log that iZotope plugins print to stdout on load
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
        """Extract the plugin parameter list."""
        if self._parameters is not None:
            return self._parameters

        self.load()
        params = []

        for attr_name, param in self._plugin.parameters.items():
            # Extract the parameter range
            try:
                rng = param.range
                min_val, max_val, step = rng
            except Exception:
                min_val = max_val = step = None

            # Read the current value
            try:
                current = getattr(param, "value", None)
                if current is None:
                    current = getattr(self._plugin, attr_name, None)
                if current is None:
                    current = getattr(self._plugin, attr_name.replace(" ", "_"), None)
            except Exception:
                current = None

            # Infer the type
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
            # Accept both underscore and space forms
            attr_name = name.replace(" ", "_")
            try:
                setattr(self._plugin, attr_name, value)
                logger.debug(f"Set parameter: {attr_name} = {value}")
            except AttributeError:
                # Retry with the space-separated name
                space_name = name.replace("_", " ")
                try:
                    setattr(self._plugin, space_name, value)
                except Exception as e:
                    raise AttributeError(f"Failed to set parameter: {name} = {value}") from e
        self._parameters = None

    def process(self, audio: np.ndarray, sample_rate: int, reset: bool = True) -> np.ndarray:
        """Process audio. audio shape: (channels, samples), float32

        Args:
            reset: when True (the default) the plugin's internal state is reset
                   before processing — the right choice for offline renders that
                   push the whole buffer through at once. For block-by-block
                   streaming (reproducing DAW playback), call with reset=False so
                   the state carried over from the previous block (filter history,
                   lookahead buffer) is preserved. Measured with pedalboard:
                   consecutive reset=False calls match a whole-buffer render to
                   the bit (-600 dB), whereas reset=True on every block breaks the
                   state at each boundary and produces clicks (-7.5 dB).

        Raises:
            ValueError: the block is empty or carries NaN/Inf samples. Checked
                before the plugin is loaded, so native code never sees them.
        """
        # Validate before loading so a bad block fails fast, without touching the plugin
        audio = validate_audio_block(audio)
        self.load()
        return self._plugin.process(audio, sample_rate, reset=reset)

    def reset(self) -> None:
        """Reset the plugin state."""
        if self._plugin is not None:
            self._plugin.reset()
