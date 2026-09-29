# tests/unit/test_plugins_parameter.py — PluginMeta / ParameterInfo JSON contract.
#
# ``to_dict`` feeds the ``info`` and ``list``/``scan`` JSON schemas
# (src/audioman/schemas/info.v1.json): every float parameter carries min/max/
# default, enum parameters carry values, bool parameters carry neither.

from __future__ import annotations

import json

from audioman.plugins.parameter import ParameterInfo, PluginMeta


class TestParameterInfoToDict:
    def test_float_parameter_includes_range_fields(self):
        info = ParameterInfo(
            name="gain_db",
            label="Gain Db",
            min_value=-18.0,
            max_value=6.0,
            default_value=0.0,
            step_size=0.5,
            current_value=3.0,
            type="float",
        )
        assert info.to_dict() == {
            "name": "gain_db",
            "label": "Gain Db",
            "type": "float",
            "current_value": 3.0,
            "min": -18.0,
            "max": 6.0,
            "default": 0.0,
        }

    def test_float_parameter_with_unknown_bounds_emits_nulls(self):
        d = ParameterInfo(name="drive", label="Drive", type="float").to_dict()
        assert d["min"] is None
        assert d["max"] is None
        assert d["default"] is None

    def test_step_size_is_not_serialized(self):
        """step_size is a wrapper-side detail; the published schema has no key."""
        info = ParameterInfo(name="mix", label="Mix", step_size=0.01)
        assert "step_size" not in info.to_dict()

    def test_bool_parameter_omits_range_and_values_keys(self):
        info = ParameterInfo(
            name="bypass",
            label="Bypass",
            min_value=0.0,
            max_value=1.0,
            default_value=0.0,
            current_value=True,
            type="bool",
        )
        d = info.to_dict()
        assert d == {
            "name": "bypass",
            "label": "Bypass",
            "type": "bool",
            "current_value": True,
        }
        assert "min" not in d and "max" not in d and "default" not in d and "values" not in d

    def test_bool_parameter_keeps_bool_current_value(self):
        d = ParameterInfo(name="bypass", label="Bypass", current_value=False, type="bool").to_dict()
        assert d["current_value"] is False

    def test_enum_parameter_emits_values_list(self):
        info = ParameterInfo(
            name="style",
            label="Style",
            current_value="Soft",
            type="enum",
            enum_values=["Soft", "Medium", "Hard"],
        )
        d = info.to_dict()
        assert d == {
            "name": "style",
            "label": "Style",
            "type": "enum",
            "current_value": "Soft",
            "values": ["Soft", "Medium", "Hard"],
        }
        assert "min" not in d and "max" not in d and "default" not in d

    def test_enum_parameter_without_values_yields_empty_list(self):
        d = ParameterInfo(name="mode", label="Mode", current_value="A", type="enum").to_dict()
        assert d["values"] == []

    def test_unknown_type_emits_only_base_fields(self):
        d = ParameterInfo(name="x", label="X", type="mystery", min_value=0.0, max_value=1.0).to_dict()
        assert d == {"name": "x", "label": "X", "type": "mystery", "current_value": None}

    def test_current_value_none_is_serialized(self):
        info = ParameterInfo(name="gain_db", label="Gain Db", min_value=-1.0, max_value=1.0)
        assert info.to_dict()["current_value"] is None


class TestPluginMetaToDict:
    def test_all_fields(self):
        meta = PluginMeta(
            name="RX 10 Spectral De-noise",
            short_name="spectral-de-noise",
            path="/Library/Audio/Plug-Ins/VST3/RX.vst3",
            format="vst3",
            vendor="iZotope",
            version="10.0.0",
            aliases=["denoise", "rx-denoise"],
            param_count=12,
        )
        assert meta.to_dict() == {
            "name": "RX 10 Spectral De-noise",
            "short_name": "spectral-de-noise",
            "path": "/Library/Audio/Plug-Ins/VST3/RX.vst3",
            "format": "vst3",
            "vendor": "iZotope",
            "version": "10.0.0",
            "aliases": ["denoise", "rx-denoise"],
            "param_count": 12,
        }

    def test_optional_fields_default(self):
        meta = PluginMeta(name="Reverb", short_name="reverb", path="/p.vst3", format="vst3")
        d = meta.to_dict()
        assert d["vendor"] == ""
        assert d["version"] == ""
        assert d["aliases"] == []
        assert d["param_count"] == 0

    def test_returns_exactly_the_schema_keys(self):
        d = PluginMeta(name="n", short_name="s", path="p", format="au").to_dict()
        assert set(d) == {
            "name",
            "short_name",
            "path",
            "format",
            "vendor",
            "version",
            "aliases",
            "param_count",
        }

    def test_aliases_are_not_shared_between_instances(self):
        first = PluginMeta(name="a", short_name="a", path="p", format="vst3")
        second = PluginMeta(name="b", short_name="b", path="p", format="vst3")
        first.aliases.append("x")
        assert second.aliases == []


class TestJsonSerializability:
    """Both dicts are printed by ``audioman info --json`` — they must round-trip."""

    def test_plugin_meta_round_trips(self):
        meta = PluginMeta(name="n", short_name="s", path="p", format="vst3", aliases=["a"])
        assert json.loads(json.dumps(meta.to_dict())) == meta.to_dict()

    def test_parameter_infos_round_trip(self):
        infos = [
            ParameterInfo(name="gain_db", label="Gain Db", min_value=-1.0, max_value=1.0, current_value=0.5),
            ParameterInfo(name="bypass", label="Bypass", current_value=True, type="bool"),
            ParameterInfo(name="style", label="Style", current_value="Soft", type="enum", enum_values=["Soft"]),
        ]
        payload = {"plugin": PluginMeta(name="n", short_name="s", path="p", format="vst3").to_dict(),
                   "parameters": [i.to_dict() for i in infos]}
        decoded = json.loads(json.dumps(payload))
        assert decoded == payload
        assert [p["type"] for p in decoded["parameters"]] == ["float", "bool", "enum"]
