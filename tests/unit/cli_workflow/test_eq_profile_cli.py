# tests/unit/cli_workflow/test_eq_profile_cli.py
# Purpose: cover `audioman eq-profile` — EQ profiling front-end.
#
# This host has no VST3 plugin (AUD-1857). `core.plugin_analysis` is stubbed with
# deterministic EQResponseResult objects; assertions cover the CLI's own contract
# (plugin resolution, sweep-config parsing, mode dispatch, npy/label artefacts,
# JSON/`--output` emission and error containment).

from __future__ import annotations

import json

import numpy as np
import pytest

from audioman.core.plugin_analysis import EQResponseResult
from audioman.plugins.parameter import PluginMeta


PLUGIN_PATH = "/plugins/EqFake.vst3"

_EQ_KEYS = {"frequencies", "magnitude_db", "phase_deg", "group_delay_ms", "params",
            "sample_rate", "fft_size", "is_minimum_phase", "thd_at_1k"}


def _eq_response(params=None, count=8, thd=0.01, minimum_phase=True, gain_step=0.75):
    freqs = np.linspace(20.0, 20000.0, count).tolist()
    return EQResponseResult(
        frequencies=freqs,
        magnitude_db=[-3.0 + gain_step * i for i in range(count)],
        phase_deg=[-90.0] * count,
        group_delay_ms=[1.5 + 0.1 * i for i in range(count)],
        params=params or {},
        sample_rate=44100,
        fft_size=32768,
        is_minimum_phase=minimum_phase,
        thd_at_1k=thd,
    )


@pytest.fixture
def fake_eq(monkeypatch):
    """Stub the three EQ measurements; returns the recorded call log."""
    import audioman.core.plugin_analysis as real_pa

    calls = []

    def _response(plugin_path, params, bypass_params, **kwargs):
        calls.append(("response", plugin_path, params, bypass_params, kwargs))
        return _eq_response(params=params or {})

    def _sweep(plugin_path, sweep_config, bypass_params, **kwargs):
        calls.append(("sweep", plugin_path, sweep_config, bypass_params, kwargs))
        # One result per swept value, tagged with the swept parameter.
        results = []
        for name, spec in sweep_config.items():
            for value in spec["values"]:
                results.append(_eq_response(
                    params={spec["param"]: value, **spec["fixed"]},
                ))
        return results

    def _nonlinear(plugin_path, params, bypass_params, **kwargs):
        calls.append(("nonlinear", plugin_path, params, bypass_params, kwargs))
        levels = kwargs.get("levels_db") or [-36.0, -24.0, -12.0, -6.0, -3.0, 0.0]
        return [
            _eq_response(
                params={**(params or {}), "_input_level_db": lv},
                thd=0.01 + 0.001 * i,
                # Make the response level-dependent so the deviation logic triggers.
                gain_step=0.75 + 0.05 * i,
            )
            for i, lv in enumerate(levels)
        ]

    monkeypatch.setattr(real_pa, "measure_eq_response", _response)
    monkeypatch.setattr(real_pa, "measure_eq_parameter_sweep", _sweep)
    monkeypatch.setattr(real_pa, "measure_eq_nonlinearity", _nonlinear)
    return calls


@pytest.fixture
def resolved(monkeypatch):
    meta = PluginMeta(name="Fake EQ", short_name="fake-eq", path=PLUGIN_PATH, format="vst3")

    class _Registry:
        def get(self, name):
            return meta if name == "fake-eq" else None

    monkeypatch.setattr("audioman.core.registry.get_registry", lambda: _Registry())


class TestPluginResolution:
    def test_vst3_bundle_path_is_used_as_is(self, tmp_path):
        from audioman.cli.eq_profile import _resolve_plugin

        bundle = tmp_path / "EqFake.vst3"
        bundle.mkdir()
        assert _resolve_plugin(str(bundle)) == str(bundle)

    def test_registry_name_resolves_to_path(self, resolved):
        from audioman.cli.eq_profile import _resolve_plugin

        assert _resolve_plugin("fake-eq") == PLUGIN_PATH

    def test_unknown_plugin_exits_nonzero(self, run_cli, resolved):
        result = run_cli(["eq-profile", "-p", "not-installed", "--mode", "response"])
        assert result.code == 1
        assert "플러그인 없음: 'not-installed'" in result.stderr


class TestParamParsing:
    def test_no_params_returns_none(self):
        from audioman.cli.eq_profile import _parse_params

        assert _parse_params([]) is None

    def test_params_are_parsed_as_floats(self):
        from audioman.cli.eq_profile import _parse_params

        assert _parse_params(["gain=6", "freq=1000"]) == {"gain": 6.0, "freq": 1000.0}

    def test_sweep_config_without_entries_is_empty(self):
        from audioman.cli.eq_profile import _parse_sweep_config

        assert _parse_sweep_config([], []) == {}

    def test_sweep_config_maps_values_and_fixed_params(self):
        from audioman.cli.eq_profile import _parse_sweep_config

        config = _parse_sweep_config(
            ["band1_gain=-12,-6,0,6,12"],
            ["band1_freq=1000", "band1_q=1.0"],
        )
        assert config == {
            "band1_gain_sweep": {
                "param": "band1_gain",
                "values": [-12.0, -6.0, 0.0, 6.0, 12.0],
                "fixed": {"band1_freq": 1000.0, "band1_q": 1.0},
            }
        }

    def test_fixed_param_values_that_are_not_numeric_stay_strings(self):
        from audioman.cli.eq_profile import _parse_sweep_config

        config = _parse_sweep_config(["drive=0,50"], ["style=Soft"])
        assert config["drive_sweep"]["fixed"] == {"style": "Soft"}

    def test_sweep_values_that_are_not_numeric_stay_strings(self):
        from audioman.cli.eq_profile import _parse_sweep_config

        config = _parse_sweep_config(["mode=Soft,Hard"], [])
        assert config["mode_sweep"]["values"] == ["Soft", "Hard"]

    def test_entries_without_equals_are_skipped(self):
        from audioman.cli.eq_profile import _parse_sweep_config

        config = _parse_sweep_config(["malformed", "drive=0,1"], ["alsobad", "q=1"])
        assert list(config) == ["drive_sweep"]
        assert config["drive_sweep"]["fixed"] == {"q": 1.0}


class TestResponseMode:
    def test_json_payload_reports_band_limited_range(self, run_cli, resolved, fake_eq):
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "response"], json_mode=True)
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["plugin"] == PLUGIN_PATH
        assert payload["plugin_type"] == "eq"
        assert payload["mode"] == "response"
        response = payload["response"]
        assert response["is_minimum_phase"] is True
        assert response["thd_at_1k"] == pytest.approx(0.01)
        assert response["fft_size"] == 32768
        assert response["magnitude_range_db"] == [-3.0, 2.25]

    def test_human_output_reports_range_phase_and_thd(self, run_cli, resolved, fake_eq):
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "response"])
        assert result.code == 0, result.stderr
        assert "response 분석 중..." in result.stdout
        assert "주파수 응답: -3.0 ~ 2.2 dB" in result.stdout
        assert "최소위상: 예" in result.stdout
        assert "THD@1kHz: 0.0100%" in result.stdout
        assert "EQ 프로파일링 완료" in result.stderr

    def test_non_minimum_phase_is_reported_as_no(self, run_cli, resolved, monkeypatch):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(real_pa, "measure_eq_response",
                            lambda *a, **k: _eq_response(minimum_phase=False))
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "response"])
        assert result.code == 0, result.stderr
        assert "최소위상: 아니오" in result.stdout

    def test_params_and_bypass_params_are_forwarded(self, run_cli, resolved, fake_eq):
        result = run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "response",
             "--param", "gain=6", "--bypass-param", "bypass=1"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        _name, path, params, bypass, kwargs = fake_eq[0]
        assert path == PLUGIN_PATH
        assert params == {"gain": 6.0}
        assert bypass == {"bypass": 1.0}
        assert kwargs["sample_rate"] == 44100
        assert kwargs["fft_size"] == 32768
        assert kwargs["level_db"] == -12.0
        assert result.payload["response"]["params"] == {"gain": 6.0}

    def test_response_failure_is_contained(self, run_cli, resolved, monkeypatch):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(real_pa, "measure_eq_response",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no sweep")))
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "response"], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["response"] == {"error": "no sweep"}

    def test_response_failure_is_printed_in_human_mode(self, run_cli, resolved, monkeypatch):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(real_pa, "measure_eq_response",
                            lambda *a, **k: (_ for _ in ()).throw(ValueError("bad param")))
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "response"])
        assert result.code == 0, result.stderr
        assert "에러: bad param" in result.stdout


class TestSweepMode:
    def test_sweep_without_params_reports_the_missing_option(self, run_cli, resolved, fake_eq):
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "sweep"], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["sweep"] == {"error": "스윕 파라미터 미지정 (--sweep-param 필요)"}
        assert fake_eq == []  # no measurement was attempted

    def test_sweep_human_message_for_missing_params(self, run_cli, resolved, fake_eq):
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "sweep"])
        assert result.code == 0, result.stderr
        assert "--sweep-param 미지정" in result.stdout

    def test_sweep_measures_once_per_value(self, run_cli, resolved, fake_eq):
        result = run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "sweep",
             "--sweep-param", "band1_gain=-12,-6,0,6,12",
             "--sweep-fixed", "band1_freq=1000"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        sweep = result.payload["sweep"]
        assert sweep["n_settings"] == 5
        assert [m["params"]["band1_gain"] for m in sweep["measurements"]] == [-12.0, -6.0, 0.0, 6.0, 12.0]
        assert sweep["measurements"][0]["params"]["band1_freq"] == 1000.0
        for measurement in sweep["measurements"]:
            assert measurement["is_minimum_phase"] is True
            assert measurement["thd_at_1k"] == pytest.approx(0.01)
            assert measurement["min_db"] <= measurement["peak_db"]

    def test_sweep_human_output_caps_the_list_at_five(self, run_cli, resolved, fake_eq):
        result = run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "sweep",
             "--sweep-param", "band1_gain=-12,-6,0,6,12,18,24"],
        )
        assert result.code == 0, result.stderr
        assert "Measured 7 setting(s)" in result.stderr
        assert "... +2 more" in result.stdout

    def test_sweep_failure_is_contained(self, run_cli, resolved, monkeypatch):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(real_pa, "measure_eq_parameter_sweep",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sweep exploded")))
        result = run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "sweep", "--sweep-param", "gain=0,1"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["sweep"] == {"error": "sweep exploded"}

    def test_sweep_config_reaches_the_engine(self, run_cli, resolved, fake_eq):
        assert run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "sweep",
             "--sweep-param", "gain=0,6", "--sweep-fixed", "q=1"]
        ).code == 0
        _name, _path, config, _bypass, _kwargs = fake_eq[0]
        assert config == {"gain_sweep": {"param": "gain", "values": [0.0, 6.0], "fixed": {"q": 1.0}}}


class TestNonlinearMode:
    def test_level_dependence_is_detected(self, run_cli, resolved, fake_eq):
        result = run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "nonlinear",
             "--levels", "-24", "-12", "0"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        nonlin = result.payload["nonlinear"]
        assert nonlin["n_levels"] == 3
        assert nonlin["levels_db"] == [-24.0, -12.0, 0.0]
        assert len(nonlin["thd_per_level"]) == 3
        assert nonlin["max_response_deviation_db"] > 0.5
        # Must be a JSON boolean, not the string "True" produced by numpy.bool_.
        assert nonlin["is_level_dependent"] is True

    def test_human_output_labels_the_deviation(self, run_cli, resolved, fake_eq):
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "nonlinear"])
        assert result.code == 0, result.stderr
        assert "개 레벨 측정, 최대 편차:" in result.stdout
        assert "비선형 (레벨 의존)" in result.stdout
        assert "dBFS: THD=" in result.stdout

    def test_level_independent_eq_is_labelled_linear(self, run_cli, resolved, monkeypatch):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(
            real_pa, "measure_eq_nonlinearity",
            lambda *a, **k: [
                _eq_response(params={"_input_level_db": lv}, gain_step=0.75)
                for lv in (-24.0, -12.0, 0.0)
            ],
        )
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "nonlinear"])
        assert result.code == 0, result.stderr
        assert "선형 (레벨 무관)" in result.stdout

    def test_a_single_level_is_not_level_dependent(self, run_cli, resolved, monkeypatch):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(
            real_pa, "measure_eq_nonlinearity",
            lambda *a, **k: [_eq_response(params={"_input_level_db": -12.0})],
        )
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "nonlinear"], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["nonlinear"]["n_levels"] == 1
        assert result.payload["nonlinear"]["is_level_dependent"] is False
        assert result.payload["nonlinear"]["max_response_deviation_db"] == 0.0

    def test_requested_levels_are_forwarded(self, run_cli, resolved, fake_eq):
        assert run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "nonlinear", "--levels", "-6", "0"]
        ).code == 0
        _name, _path, _params, _bypass, kwargs = fake_eq[0]
        assert kwargs["levels_db"] == [-6.0, 0.0]

    def test_nonlinear_failure_is_contained(self, run_cli, resolved, monkeypatch):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(real_pa, "measure_eq_nonlinearity",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("levels failed")))
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "nonlinear"], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["nonlinear"] == {"error": "levels failed"}


class TestNpyExport:
    def test_curves_and_labels_are_written(self, run_cli, resolved, fake_eq, tmp_path):
        npy_dir = tmp_path / "curves"
        result = run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "response", "--save-npy", str(npy_dir)],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        for name, expected_shape in [
            ("frequency_response_curves.npy", (1, 8)),
            ("phase_response_curves.npy", (1, 8)),
            ("group_delay_curves.npy", (1, 8)),
            ("frequency_axis.npy", (8,)),
        ]:
            path = npy_dir / name
            assert path.exists(), name
            assert np.load(str(path)).shape == expected_shape

        labels = json.loads((npy_dir / "settings_labels.json").read_text(encoding="utf-8"))
        assert labels == [{}]

    def test_multiple_settings_stack_into_rows(self, run_cli, resolved, fake_eq, tmp_path):
        npy_dir = tmp_path / "stack"
        result = run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "sweep",
             "--sweep-param", "gain=0,6,12", "--save-npy", str(npy_dir)],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert np.load(str(npy_dir / "frequency_response_curves.npy")).shape == (3, 8)
        labels = json.loads((npy_dir / "settings_labels.json").read_text(encoding="utf-8"))
        assert [lab["gain"] for lab in labels] == [0.0, 6.0, 12.0]

    def test_human_mode_prints_the_saved_shape(self, run_cli, resolved, fake_eq, tmp_path):
        npy_dir = tmp_path / "curves"
        result = run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "response", "--save-npy", str(npy_dir)]
        )
        assert result.code == 0, result.stderr
        assert "곡선 저장:" in result.stderr
        assert "1 settings × 8 bins" in result.stderr

    def test_save_npy_is_skipped_when_no_measurement_succeeded(
        self, run_cli, resolved, monkeypatch, tmp_path
    ):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(real_pa, "measure_eq_response",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
        npy_dir = tmp_path / "empty"
        result = run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "response", "--save-npy", str(npy_dir)],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert not npy_dir.exists()
        assert result.payload["response"] == {"error": "nope"}


class TestAllModeAndOutput:
    def test_all_mode_runs_the_three_analyses(self, run_cli, resolved, fake_eq):
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "all"], json_mode=True)
        assert result.code == 0, result.stderr
        payload = result.payload
        assert "response" in payload
        # sweep needs --sweep-param; in `all` mode it reports the missing option.
        assert payload["sweep"] == {"error": "스윕 파라미터 미지정 (--sweep-param 필요)"}
        assert "nonlinear" in payload
        assert [name for name, *_ in fake_eq] == ["response", "nonlinear"]

    def test_all_mode_with_sweep_params_runs_everything(self, run_cli, resolved, fake_eq):
        result = run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "all", "--sweep-param", "gain=0,6"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert [name for name, *_ in fake_eq] == ["response", "sweep", "nonlinear"]
        assert result.payload["sweep"]["n_settings"] == 2

    def test_output_writes_the_envelope_to_disk(self, run_cli, resolved, fake_eq, tmp_path):
        out = tmp_path / "eq.json"
        result = run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "response", "-o", str(out)]
        )
        assert result.code == 0, result.stderr
        assert out.exists()
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["command"] == "eq-profile"
        assert saved["plugin"] == PLUGIN_PATH
        assert "response" in saved
        assert "결과 저장" in result.stderr

    def test_json_without_output_prints_the_envelope(self, run_cli, resolved, fake_eq):
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "response"], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["$schema"] == "audioman://schema/eq-profile.v1.json"
        assert result.payload["command"] == "eq-profile"

    def test_output_and_json_together_write_both(self, run_cli, resolved, fake_eq, tmp_path):
        out = tmp_path / "both.json"
        result = run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "response", "-o", str(out)], json_mode=True
        )
        assert result.code == 0, result.stderr
        assert result.payload["command"] == "eq-profile"
        assert json.loads(out.read_text(encoding="utf-8"))["command"] == "eq-profile"

    def test_human_completion_message_without_json_or_output(self, run_cli, resolved, fake_eq):
        result = run_cli(["eq-profile", "-p", "fake-eq", "--mode", "response"])
        assert result.code == 0, result.stderr
        assert "EQ 프로파일링 완료" in result.stderr

    def test_common_options_reach_the_engine(self, run_cli, resolved, fake_eq):
        assert run_cli(
            ["eq-profile", "-p", "fake-eq", "--mode", "response",
             "--sample-rate", "48000", "--fft-size", "16384", "--level", "-6",
             "--sweep-duration", "3"]
        ).code == 0
        _name, _path, _params, _bypass, kwargs = fake_eq[0]
        assert kwargs == {
            "sample_rate": 48000,
            "fft_size": 16384,
            "sweep_duration": 3.0,
            "level_db": -6.0,
        }
