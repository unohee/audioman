# tests/unit/test_vst3_validation.py — VST3 parameter bounds + audio block validation

import numpy as np
import pytest

from audioman.plugins.parameter import ParameterInfo
from audioman.plugins.vst3 import (
    VST3PluginWrapper,
    find_parameter_info,
    validate_audio_block,
    validate_parameter_value,
)


def _info(name="drive", min_value=0.0, max_value=1.0, type="float"):
    return ParameterInfo(
        name=name,
        label=name.replace("_", " ").title(),
        min_value=min_value,
        max_value=max_value,
        default_value=None,
        type=type,
    )


class _FakeParameter:
    def __init__(self, min_value, max_value):
        self.range = (min_value, max_value, None)


class _FakePlugin:
    """Stand-in for a pedalboard plugin: a parameter dict plus process()"""

    def __init__(self, ranges):
        self._ranges = ranges
        self.last_reset = None
        for name in ranges:
            setattr(self, name, None)

    @property
    def parameters(self):
        return {name: _FakeParameter(*bounds) for name, bounds in self._ranges.items()}

    def process(self, audio, sample_rate, reset=True):
        self.last_reset = reset
        return audio


class TestFindParameterInfo:
    """Both spellings (space / underscore) are matched"""

    def test_exact_name(self):
        infos = [_info("air_db"), _info("drive")]
        assert find_parameter_info(infos, "drive").name == "drive"

    def test_space_notation(self):
        infos = [_info("air_db")]
        assert find_parameter_info(infos, "air db").name == "air_db"

    def test_underscore_notation(self):
        infos = [_info("air db")]
        assert find_parameter_info(infos, "air_db").name == "air db"

    def test_unknown_returns_none(self):
        assert find_parameter_info([_info("drive")], "nope") is None

    def test_empty_list(self):
        assert find_parameter_info([], "drive") is None


class TestValidateParameterValue:
    """Out-of-range and non-finite values are rejected"""

    def test_in_range_value_passes(self):
        validate_parameter_value("drive", 0.5, _info())

    def test_boundary_values_pass(self):
        validate_parameter_value("drive", 0.0, _info())
        validate_parameter_value("drive", 1.0, _info())

    def test_below_minimum_raises(self):
        with pytest.raises(ValueError, match="below the minimum"):
            validate_parameter_value("drive", -0.5, _info())

    def test_above_maximum_raises(self):
        with pytest.raises(ValueError, match="above the maximum"):
            validate_parameter_value("drive", 12.0, _info())

    def test_error_names_parameter_and_bound(self):
        with pytest.raises(ValueError) as excinfo:
            validate_parameter_value("bright_db", 100.0, _info("bright_db", -18.0, 6.0))
        message = str(excinfo.value)
        assert "bright_db" in message
        assert "6.0" in message
        assert "100.0" in message

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            validate_parameter_value("drive", float("nan"), _info())

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            validate_parameter_value("drive", float("inf"), _info())

    def test_negative_inf_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            validate_parameter_value("drive", float("-inf"), _info())

    def test_numeric_string_validated(self):
        with pytest.raises(ValueError, match="above the maximum"):
            validate_parameter_value("drive", "5.0", _info())

    def test_unknown_bounds_skip_range_check(self):
        validate_parameter_value("drive", 1e6, _info(min_value=None, max_value=None))

    def test_one_sided_bounds(self):
        validate_parameter_value("drive", 1e6, _info(min_value=0.0, max_value=None))
        with pytest.raises(ValueError, match="below the minimum"):
            validate_parameter_value("drive", -1.0, _info(min_value=0.0, max_value=None))

    def test_enum_label_left_to_plugin(self):
        """Enum labels are not range-checked; the plugin binding validates them"""
        validate_parameter_value("style", "Soft", _info("style", None, None, type="enum"))

    def test_inverted_range_not_enforced(self):
        """Garbage metadata (min > max) must not reject legitimate values"""
        validate_parameter_value("drive", 0.5, _info(min_value=10.0, max_value=-10.0))

    def test_inverted_range_still_rejects_nonfinite(self):
        with pytest.raises(ValueError, match="must be finite"):
            validate_parameter_value("drive", float("nan"), _info(min_value=10.0, max_value=-10.0))

    def test_bool_value(self):
        validate_parameter_value("bypass", True, _info("bypass", False, True, type="bool"))


class TestValidateAudioBlock:
    """Empty blocks and NaN/Inf samples are rejected"""

    def test_valid_block_passes_through(self):
        audio = np.zeros((2, 128), dtype=np.float32)
        out = validate_audio_block(audio)
        assert out.shape == (2, 128)
        assert out.dtype == np.float32

    def test_mono_1d_promoted(self):
        out = validate_audio_block(np.zeros(64, dtype=np.float32))
        assert out.shape == (1, 64)

    def test_int_dtype_cast_to_float32(self):
        out = validate_audio_block(np.ones((1, 8), dtype=np.int16))
        assert out.dtype == np.float32

    def test_empty_1d_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_audio_block(np.zeros(0, dtype=np.float32))

    def test_empty_2d_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_audio_block(np.zeros((2, 0), dtype=np.float32))

    def test_zero_channel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_audio_block(np.zeros((0, 64), dtype=np.float32))

    def test_nan_raises(self):
        audio = np.zeros((2, 128), dtype=np.float32)
        audio[1, 64] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            validate_audio_block(audio)

    def test_inf_raises(self):
        audio = np.zeros((1, 32), dtype=np.float32)
        audio[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            validate_audio_block(audio)

    def test_negative_inf_raises(self):
        audio = np.zeros((1, 32), dtype=np.float32)
        audio[0, 31] = -np.inf
        with pytest.raises(ValueError, match="non-finite"):
            validate_audio_block(audio)

    def test_error_reports_bad_sample_count(self):
        audio = np.zeros((1, 10), dtype=np.float32)
        audio[0, 1] = np.nan
        audio[0, 2] = np.nan
        with pytest.raises(ValueError, match="2 non-finite sample"):
            validate_audio_block(audio)

    def test_empty_int_block_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            validate_audio_block(np.zeros((2, 0), dtype=np.int16))

    def test_float64_overflow_detected_after_cast(self):
        """A value that overflows float32 becomes Inf — must be caught, not passed on"""
        audio = np.zeros((1, 4), dtype=np.float64)
        audio[0, 0] = 1e300
        with pytest.raises(ValueError, match="non-finite"):
            validate_audio_block(audio)

    def test_float64_normal_values_pass(self):
        out = validate_audio_block(np.full((1, 4), 0.5, dtype=np.float64))
        assert out.dtype == np.float32


class TestWrapperWiring:
    """The wrapper really calls the validators — checked without a plugin binary"""

    def _wrapper_with_fake_plugin(self, params):
        wrapper = VST3PluginWrapper("/nonexistent/Plugin.vst3")
        wrapper._plugin = _FakePlugin(params)
        return wrapper

    def test_set_parameters_rejects_out_of_range(self):
        wrapper = self._wrapper_with_fake_plugin({"gain_db": (-18.0, 6.0)})
        with pytest.raises(ValueError, match="above the maximum"):
            wrapper.set_parameters({"gain_db": 99.0})
        assert wrapper._plugin.gain_db is None  # value never reached the plugin

    def test_set_parameters_accepts_valid(self):
        wrapper = self._wrapper_with_fake_plugin({"gain_db": (-18.0, 6.0)})
        wrapper.set_parameters({"gain_db": 3.0})
        assert wrapper._plugin.gain_db == 3.0

    def test_set_parameters_accepts_boundary(self):
        wrapper = self._wrapper_with_fake_plugin({"gain_db": (-18.0, 6.0)})
        wrapper.set_parameters({"gain_db": 6.0})
        assert wrapper._plugin.gain_db == 6.0

    def test_process_rejects_empty(self):
        wrapper = self._wrapper_with_fake_plugin({})
        with pytest.raises(ValueError, match="empty"):
            wrapper.process(np.zeros((0, 0), dtype=np.float32), 48000)

    def test_process_rejects_nan(self):
        wrapper = self._wrapper_with_fake_plugin({})
        audio = np.zeros((2, 32), dtype=np.float32)
        audio[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            wrapper.process(audio, 48000)

    def test_process_passes_valid_block(self):
        wrapper = self._wrapper_with_fake_plugin({})
        audio = np.ones((2, 32), dtype=np.float32) * 0.25
        out = wrapper.process(audio, 48000)
        np.testing.assert_array_equal(out, audio)
        assert wrapper._plugin.last_reset is True

    def test_process_forwards_reset_flag(self):
        wrapper = self._wrapper_with_fake_plugin({})
        wrapper.process(np.zeros((1, 16), dtype=np.float32), 48000, reset=False)
        assert wrapper._plugin.last_reset is False

    def test_process_rejects_bad_block_before_loading(self):
        """A bad block must fail without loading a plugin binary"""
        wrapper = VST3PluginWrapper("/nonexistent/SuchPlugin.vst3")
        with pytest.raises(ValueError, match="empty"):
            wrapper.process(np.zeros((2, 0), dtype=np.float32), 48000)
        assert wrapper._plugin is None  # load() never ran

    def test_process_rejects_nan_before_loading(self):
        wrapper = VST3PluginWrapper("/nonexistent/SuchPlugin.vst3")
        audio = np.zeros((1, 4), dtype=np.float32)
        audio[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            wrapper.process(audio, 48000)
        assert wrapper._plugin is None
