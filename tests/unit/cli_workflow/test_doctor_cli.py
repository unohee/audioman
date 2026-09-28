# tests/unit/cli_workflow/test_doctor_cli.py
# Purpose: cover `audioman doctor` — plugin analysis front-end.
#
# This host has no VST3 plugin (AUD-1857), so `core.plugin_analysis` is stubbed
# with deterministic result objects. What is asserted here is the CLI's own
# contract: plugin resolution, mode dispatch, payload shape per mode, the
# per-mode error containment, CLAP output files, comparison mode and --output.

from __future__ import annotations

import json

import numpy as np
import pytest

from audioman.core.plugin_analysis import (
    DynamicsResult,
    HarmonicResult,
    LinearResult,
    OscilloscopeResult,
    PerformanceResult,
    SweepResult,
    WaveshaperV2Result,
)
from audioman.plugins.parameter import PluginMeta


PLUGIN_PATH = "/plugins/Fake.vst3"


# ---------------------------------------------------------------------------
# Deterministic stand-ins for the analysis engine
# ---------------------------------------------------------------------------


def _linear(fs=6):
    freqs = np.linspace(20.0, 20000.0, fs).tolist()
    mags = np.linspace(-1.5, 2.5, fs).tolist()
    return LinearResult(
        frequencies=freqs,
        magnitude_db=mags,
        phase_deg=[0.0] * fs,
        sample_rate=44100,
        fft_size=16384,
        method="impulse",
    )


def _imd():
    result = _thd()
    result.imd_percent = 0.4321
    result.method = "imd"
    return result


def _thd():
    return HarmonicResult(
        thd_percent=0.1234,
        thd_plus_n_percent=0.2345,
        fundamental_freq=1000.0,
        fundamental_db=-6.0,
        harmonics=[{"order": i, "freq": 1000.0 * i, "db": -60.0 - i} for i in range(1, 8)],
    )


def _sweep(points=4):
    return SweepResult(
        frequencies=[100.0 * i for i in range(1, points + 1)],
        thd_per_freq=[0.1 * i for i in range(points)],
        gain_per_freq=[-1.0 * i for i in range(points)],
    )


def _dynamics():
    return DynamicsResult(
        input_levels_db=[-80.0, -40.0, -20.0, 0.0],
        output_levels_db=[-80.0, -44.0, -28.0, -12.0],
        gain_reduction_db=[0.0, -4.0, -8.0, -12.0],
    )


def _oscilloscope():
    return OscilloscopeResult(
        input_signal=np.zeros(8, dtype=np.float32),
        output_signal=np.zeros(8, dtype=np.float32),
        waveshaper_input=[-1.0, -0.5, 0.0, 0.5, 1.0],
        waveshaper_output=[-0.9, -0.45, 0.0, 0.45, 0.9],
    )


def _waveshaper_v2(points=8):
    inputs = np.linspace(-1.0, 1.0, points)
    return WaveshaperV2Result(
        input_values=inputs,
        output_values=inputs * 0.8,
        n_points=points,
        levels_db=[-24.0, -12.0, -6.0],
        input_coverage=0.75,
        is_symmetric=True,
    )


def _performance():
    return PerformanceResult(
        buffer_sizes=[64, 256, 1024],
        process_times_ms=[0.1, 0.4, 1.6],
        samples_per_second=[640000.0, 640000.0, 640000.0],
        realtime_ratio=[100.0, 95.0, 90.0],
    )


_DEFAULT_RESULTS = {
    "measure_linear": _linear(),
    "measure_thd": _thd(),
    "measure_imd": _imd(),
    "measure_sweep": _sweep(),
    "measure_dynamics_ramp": _dynamics(),
    "measure_dynamics_ar": _dynamics(),
    "measure_waveshaper": _oscilloscope(),
    "measure_waveshaper_v2": _waveshaper_v2(),
    "measure_performance": _performance(),
    "measure_clap_profile": {
        "n_settings": 5,
        "embedding_dim": 512,
        "labels": [f"drive={v}" for v in (0, 25, 50, 75, 100)],
        "embeddings_npy": np.zeros((5, 512), dtype=np.float32),
        "params": {"drive": [0, 25, 50, 75, 100]},
    },
    "compare_linear": {
        "diff_magnitude_db": [0.5, -13.5, 2.0],
    },
}


@pytest.fixture
def fake_pa(monkeypatch):
    """Replace every core.plugin_analysis measurement with a fixed stub."""
    import audioman.core.plugin_analysis as real_pa

    calls = []
    for name, result in _DEFAULT_RESULTS.items():
        def _stub(*args, _name=name, _result=result, **kwargs):
            calls.append((_name, args, kwargs))
            return _result
        monkeypatch.setattr(real_pa, name, _stub)
    return calls


def _resolve_fake(monkeypatch):
    """Minimal registry so `--plugin name` resolves without a real bundle."""
    meta = PluginMeta(name="Fake EQ", short_name="fake-eq", path=PLUGIN_PATH, format="vst3")

    class _Registry:
        def get(self, name):
            return meta if name == "fake-eq" else None

    monkeypatch.setattr("audioman.core.registry.get_registry", lambda: _Registry())


class TestPluginResolution:
    def test_existing_vst3_path_is_used_as_is(self, tmp_path):
        from audioman.cli.doctor import _resolve_plugin

        bundle = tmp_path / "Fake.vst3"
        bundle.mkdir()
        assert _resolve_plugin(str(bundle)) == str(bundle)

    def test_component_bundle_is_used_as_is(self, tmp_path):
        from audioman.cli.doctor import _resolve_plugin

        bundle = tmp_path / "Fake.component"
        bundle.mkdir()
        assert _resolve_plugin(str(bundle)) == str(bundle)

    def test_registry_name_resolves_to_its_path(self, monkeypatch):
        from audioman.cli.doctor import _resolve_plugin

        _resolve_fake(monkeypatch)
        assert _resolve_plugin("fake-eq") == PLUGIN_PATH

    def test_unknown_name_exits_nonzero(self, run_cli, monkeypatch):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "not-installed", "--mode", "linear"])
        assert result.code == 1
        assert "플러그인 없음: 'not-installed'" in result.stderr

    def test_directory_with_other_suffix_is_not_treated_as_a_bundle(self, run_cli, tmp_path, monkeypatch):
        _resolve_fake(monkeypatch)
        other = tmp_path / "plugin.dylib"
        other.write_bytes(b"x")
        result = run_cli(["doctor", "-p", str(other), "--mode", "linear"])
        assert result.code == 1
        assert "플러그인 없음" in result.stderr


class TestModes:
    """One assertion block per analysis mode: dispatch + payload shape."""

    def test_linear_mode_payload(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "linear", "--sample-rate", "48000",
             "--fft-size", "8192", "--level", "-3"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "doctor"
        assert payload["plugin"] == PLUGIN_PATH
        assert payload["mode"] == "linear"
        assert payload["linear"] == {
            "method": "impulse",
            "fft_size": 16384,
            "freq_count": 6,
            "magnitude_range_db": [-1.5, 2.5],
        }

    def test_linear_mode_human_output(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "linear"])
        assert result.code == 0, result.stderr
        assert "linear 분석 중..." in result.stdout
        assert "주파수 응답: 6 bins" in result.stdout
        assert "분석 완료" in result.stderr

    def test_thd_mode_payload_and_harmonics(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "thd"], json_mode=True)
        assert result.code == 0, result.stderr
        thd = result.payload["thd"]
        assert thd["thd_percent"] == pytest.approx(0.1234)
        assert thd["thd_plus_n_percent"] == pytest.approx(0.2345)
        assert thd["fundamental_freq"] == 1000.0
        assert len(thd["harmonics"]) == 7

    def test_thd_mode_human_output_prints_five_harmonics(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "thd"])
        assert result.code == 0, result.stderr
        assert "THD: 0.1234%" in result.stdout
        assert "Fundamental: 1000.0Hz @ -6.0dB" in result.stdout
        assert result.stdout.count("차:") == 5

    def test_imd_mode_payload(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "imd"], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["imd"]["imd_percent"] == pytest.approx(0.4321)
        assert len(result.payload["imd"]["harmonics"]) <= 10

    def test_imd_mode_human_output(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "imd"])
        assert result.code == 0, result.stderr
        assert "IMD: 0.4321%" in result.stdout

    def test_sweep_mode_payload_ranges(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "sweep"], json_mode=True)
        assert result.code == 0, result.stderr
        sweep = result.payload["sweep"]
        assert sweep["freq_count"] == 4
        assert sweep["thd_range"] == [0.0, 0.3]
        assert sweep["gain_range"] == [-3.0, 0.0]

    def test_sweep_mode_human_output(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "sweep"])
        assert result.code == 0, result.stderr
        assert "스윕: 4 points" in result.stdout
        assert "THD range: 0.0000% ~ 0.3000%" in result.stdout

    def test_sweep_mode_with_empty_curves_reports_zero_ranges(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        empty = SweepResult(frequencies=[100.0], thd_per_freq=[], gain_per_freq=[])
        monkeypatch.setattr(real_pa, "measure_sweep", lambda *a, **k: empty)
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "sweep"], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["sweep"]["thd_range"] == [0, 0]
        assert result.payload["sweep"]["gain_range"] == [0, 0]

    def test_dynamics_mode_payload(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "dynamics"], json_mode=True)
        assert result.code == 0, result.stderr
        dyn = result.payload["dynamics"]
        assert dyn["input_range"] == [-80.0, 0.0]
        assert dyn["output_range"] == [-80.0, -12.0]
        assert dyn["max_gain_reduction"] == -12.0
        assert dyn["io_curve"][0] == [-80.0, -80.0]

    def test_dynamics_mode_human_output(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "dynamics"])
        assert result.code == 0, result.stderr
        assert "I/O: -80.0~0.0 dB" in result.stdout
        assert "Max gain reduction: -12.0 dB" in result.stdout

    def test_attack_release_mode_payload(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "attack-release"], json_mode=True)
        assert result.code == 0, result.stderr
        ar = result.payload["attack_release"]
        assert ar["input_levels"] == [-80.0, -40.0, -20.0, 0.0]
        assert ar["envelope_points"] == 4

    def test_attack_release_mode_human_output(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "attack-release"])
        assert result.code == 0, result.stderr
        assert "Envelope: 4 points" in result.stdout

    def test_waveshaper_v2_payload_and_human_output(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        json_result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "waveshaper", "--ws-points", "8"],
            json_mode=True,
        )
        assert json_result.code == 0, json_result.stderr
        ws = json_result.payload["waveshaper"]
        assert ws["version"] == "v2"
        assert ws["points"] == 8
        assert ws["levels_db"] == [-24.0, -12.0, -6.0]
        assert ws["input_coverage"] == 0.75
        assert ws["is_symmetric"] is True
        # A perfect correlation would be 1.0; the stub is scaled by 0.8.
        assert ws["linearity"] == pytest.approx(1.0, abs=1e-6)
        assert ws["is_linear"] is True
        assert len(ws["input_values"]) == 8

        human = run_cli(["doctor", "-p", "fake-eq", "--mode", "waveshaper"])
        assert human.code == 0, human.stderr
        assert "Waveshaper v2: 8 points, 3 levels, coverage=75.0%" in human.stdout
        assert "대칭: 예 (홀수 하모닉)" in human.stdout
        assert "선형 플러그인" in human.stdout

    def test_waveshaper_nonlinear_result_is_flagged(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        inputs = np.linspace(-1.0, 1.0, 8)
        nonlinear = WaveshaperV2Result(
            input_values=inputs,
            output_values=inputs ** 3,  # not affine: correlation stays high but not 0.999+
            n_points=8,
            levels_db=[-12.0],
            input_coverage=0.4,
            is_symmetric=True,
        )
        monkeypatch.setattr(real_pa, "measure_waveshaper_v2", lambda *a, **k: nonlinear)
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "waveshaper"])
        assert result.code == 0, result.stderr
        assert "coverage=40.0%" in result.stdout

    def test_waveshaper_legacy_payload(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "waveshaper", "--legacy-waveshaper"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        ws = result.payload["waveshaper"]
        assert ws["version"] == "v1"
        assert ws["points"] == 5
        assert ws["linearity"] == pytest.approx(1.0, abs=1e-6)
        assert ws["is_linear"] is True

    def test_waveshaper_v2_with_few_points_skips_the_correlation(self, run_cli, monkeypatch, fake_pa):
        """n_points <= 2 has no meaningful correlation: linearity must be 1.0."""
        import audioman.core.plugin_analysis as real_pa

        tiny = WaveshaperV2Result(
            input_values=np.array([-1.0, 1.0]),
            output_values=np.array([-1.0, 1.0]),
            n_points=2,
            levels_db=[-6.0],
            input_coverage=1.0,
            is_symmetric=True,
        )
        monkeypatch.setattr(real_pa, "measure_waveshaper_v2", lambda *a, **k: tiny)
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "waveshaper"], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["waveshaper"]["points"] == 2
        assert result.payload["waveshaper"]["linearity"] == 1.0

    def test_legacy_waveshaper_with_two_points_is_treated_as_linear(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        tiny = OscilloscopeResult(
            input_signal=np.zeros(2, dtype=np.float32),
            output_signal=np.zeros(2, dtype=np.float32),
            waveshaper_input=[-1.0, 1.0],
            waveshaper_output=[-1.0, 1.0],
        )
        monkeypatch.setattr(real_pa, "measure_waveshaper", lambda *a, **k: tiny)
        _resolve_fake(monkeypatch)
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "waveshaper", "--legacy-waveshaper"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["waveshaper"]["linearity"] == 1.0
        assert result.payload["waveshaper"]["is_linear"] is True

    def test_legacy_waveshaper_nonlinear_is_flagged_in_human_output(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        bent = OscilloscopeResult(
            input_signal=np.zeros(5, dtype=np.float32),
            output_signal=np.zeros(5, dtype=np.float32),
            waveshaper_input=[-1.0, -0.5, 0.0, 0.5, 1.0],
            waveshaper_output=[-0.9, -0.2, 0.0, 0.5, 1.0],
        )
        monkeypatch.setattr(real_pa, "measure_waveshaper", lambda *a, **k: bent)
        _resolve_fake(monkeypatch)
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "waveshaper", "--legacy-waveshaper"]
        )
        assert result.code == 0, result.stderr
        assert "비선형 (" in result.stdout
        assert "선형 플러그인" not in result.stdout

    def test_legacy_waveshaper_human_output(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "waveshaper", "--legacy-waveshaper"]
        )
        assert result.code == 0, result.stderr
        assert "Waveshaper (legacy): 5 points" in result.stdout
        assert "선형 플러그인" in result.stdout

    def test_performance_mode_payload_and_table(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        json_result = run_cli(["doctor", "-p", "fake-eq", "--mode", "performance"], json_mode=True)
        assert json_result.code == 0, json_result.stderr
        perf = json_result.payload["performance"]
        assert perf["buffer_sizes"] == [64, 256, 1024]
        assert perf["process_times_ms"] == [0.1, 0.4, 1.6]
        assert perf["realtime_ratio"] == [100.0, 95.0, 90.0]

        human = run_cli(["doctor", "-p", "fake-eq", "--mode", "performance"])
        assert human.code == 0, human.stderr
        for token in ("64", "0.100", "100.0x"):
            assert token in human.stdout

    def test_all_mode_runs_every_mode_and_omits_clap(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "all"], json_mode=True)
        assert result.code == 0, result.stderr
        payload = result.payload
        for key in ("linear", "thd", "imd", "sweep", "dynamics", "attack_release",
                    "waveshaper", "performance"):
            assert key in payload, key
            assert "error" not in payload[key], payload[key]
        assert "clap" not in payload

    def test_parameters_are_forwarded_to_the_engine(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        captured = {}

        def fake_thd(plugin_path, params, *a, **k):
            captured["path"] = plugin_path
            captured["params"] = params
            return _thd()

        monkeypatch.setattr(real_pa, "measure_thd", fake_thd)
        _resolve_fake(monkeypatch)
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "thd", "--param", "drive=75", "--param", "mode=A"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert captured["path"] == PLUGIN_PATH
        assert captured["params"] == {"drive": 75.0, "mode": "A"}

    def test_default_mode_is_all(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq"], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["mode"] == "all"
        assert "performance" in result.payload


class TestErrorContainment:
    def test_one_failing_mode_does_not_abort_the_rest(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        def boom(*a, **k):
            raise RuntimeError("plugin crashed")

        monkeypatch.setattr(real_pa, "measure_sweep", boom)
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "all"], json_mode=True)
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["sweep"] == {"error": "plugin crashed"}
        assert "error" not in payload["linear"]
        assert "error" not in payload["performance"]

    def test_failing_mode_is_printed_in_human_mode(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(real_pa, "measure_thd",
                            lambda *a, **k: (_ for _ in ()).throw(ValueError("bad params")))
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "all"])
        assert result.code == 0, result.stderr
        assert "에러: bad params" in result.stdout

    def test_single_mode_failure_still_emits_the_envelope(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(real_pa, "measure_linear",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ir")))
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "linear"], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["linear"] == {"error": "no ir"}
        assert result.payload["$schema"].startswith("audioman://schema/")


class TestClap:
    def test_default_drive_sweep_is_used_when_no_sweep_is_given(self, run_cli, monkeypatch):
        import audioman.core.plugin_analysis as real_pa

        captured = {}

        def fake_profile(plugin_path, sweeps, base_params, **kwargs):
            captured["sweeps"] = sweeps
            captured["base"] = base_params
            return _DEFAULT_RESULTS["measure_clap_profile"]

        monkeypatch.setattr(real_pa, "measure_clap_profile", fake_profile)
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "linear", "--clap"], json_mode=True)
        assert result.code == 0, result.stderr
        assert captured["sweeps"] == {"drive": [0, 25, 50, 75, 100]}
        assert result.payload["clap"]["n_settings"] == 5
        assert result.payload["clap"]["embedding_dim"] == 512
        assert result.payload["clap"]["labels"][0] == "drive=0"

    def test_explicit_sweep_parses_numbers_and_enum_labels(self, run_cli, monkeypatch):
        import audioman.core.plugin_analysis as real_pa

        captured = {}

        def fake_profile(plugin_path, sweeps, base_params, **kwargs):
            captured["sweeps"] = sweeps
            return _DEFAULT_RESULTS["measure_clap_profile"]

        monkeypatch.setattr(real_pa, "measure_clap_profile", fake_profile)
        _resolve_fake(monkeypatch)
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "linear",
             "--clap-sweep", "drive=0,50,100", "--clap-sweep", "style=Soft,Hard",
             "--clap-sweep", "malformed"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert captured["sweeps"] == {
            "drive": [0.0, 50.0, 100.0],
            "style": ["Soft", "Hard"],
        }

    def test_clap_human_output_truncates_after_five_labels(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        many = dict(_DEFAULT_RESULTS["measure_clap_profile"])
        many["labels"] = [f"drive={v}" for v in range(8)]
        many["n_settings"] = len(many["labels"])
        monkeypatch.setattr(real_pa, "measure_clap_profile", lambda *a, **k: many)
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "linear", "--clap"])
        assert result.code == 0, result.stderr
        assert "8개 설정 × 512dim 임베딩" in result.stdout
        assert "... +3 more" in result.stdout

    def test_clap_output_writes_npy_and_labels(self, run_cli, monkeypatch, fake_pa, tmp_path):
        _resolve_fake(monkeypatch)
        npy = tmp_path / "clap.npy"
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "linear", "--clap", "--clap-output", str(npy)],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert npy.exists()
        loaded = np.load(str(npy))
        assert loaded.shape == (5, 512)
        label_path = tmp_path / "clap_labels.json"
        assert label_path.exists()
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        assert labels["labels"][0] == "drive=0"
        assert labels["params"] == {"drive": [0, 25, 50, 75, 100]}

    def test_clap_output_reports_the_shape_in_human_mode(self, run_cli, monkeypatch, fake_pa, tmp_path):
        _resolve_fake(monkeypatch)
        npy = tmp_path / "clap.npy"
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "linear", "--clap", "--clap-output", str(npy)]
        )
        assert result.code == 0, result.stderr
        assert "CLAP 임베딩 저장" in result.stderr
        assert "(5, 512)" in result.stderr

    def test_missing_laion_clap_is_reported_as_an_install_hint(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(
            real_pa, "measure_clap_profile",
            lambda *a, **k: (_ for _ in ()).throw(ImportError("no laion_clap")),
        )
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "linear", "--clap"], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["clap"] == {"error": "laion-clap 미설치: pip install laion-clap"}

    def test_clap_human_mode_prints_the_install_hint(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(
            real_pa, "measure_clap_profile",
            lambda *a, **k: (_ for _ in ()).throw(ImportError("no laion_clap")),
        )
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "linear", "--clap"])
        assert result.code == 0, result.stderr
        assert "laion-clap 미설치" in result.stdout

    def test_other_clap_failure_is_contained(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(
            real_pa, "measure_clap_profile",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("clap exploded")),
        )
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "linear", "--clap"], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["clap"] == {"error": "clap exploded"}

    def test_clap_failure_is_printed_in_human_mode(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(
            real_pa, "measure_clap_profile",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("clap exploded")),
        )
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "linear", "--clap"])
        assert result.code == 0, result.stderr
        assert "에러: clap exploded" in result.stdout

    def test_clap_sweep_alone_triggers_profiling(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "linear", "--clap-sweep", "drive=0,1"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert "clap" in result.payload


class TestCompare:
    def test_compare_mode_reports_the_max_difference(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "linear", "--compare", "fake-eq"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["compare"] == {"max_diff_db": 13.5}

    def test_compare_human_output(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(["doctor", "-p", "fake-eq", "--mode", "linear", "--compare", "fake-eq"])
        assert result.code == 0, result.stderr
        assert "비교: fake-eq vs fake-eq" in result.stdout
        assert "최대 차이: 13.50 dB" in result.stdout

    def test_compare_failure_is_contained(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        monkeypatch.setattr(real_pa, "compare_linear",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mismatch")))
        _resolve_fake(monkeypatch)
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "linear", "--compare", "fake-eq"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["compare"] == {"error": "mismatch"}

    def test_compare_second_plugin_must_resolve(self, run_cli, monkeypatch, fake_pa):
        _resolve_fake(monkeypatch)
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "linear", "--compare", "nope"],
        )
        assert result.code == 1
        assert "플러그인 없음: 'nope'" in result.stderr

    def test_compare_parameters_are_forwarded(self, run_cli, monkeypatch, fake_pa):
        import audioman.core.plugin_analysis as real_pa

        captured = {}

        def fake_compare(path1, path2, params1, params2, sr):
            captured.update(path1=path1, path2=path2, p1=params1, p2=params2, sr=sr)
            return {"diff_magnitude_db": [0.0]}

        monkeypatch.setattr(real_pa, "compare_linear", fake_compare)
        _resolve_fake(monkeypatch)
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "linear", "--compare", "fake-eq",
             "--param", "a=1", "--compare-param", "b=2"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert captured["p1"] == {"a": 1.0}
        assert captured["p2"] == {"b": 2.0}
        assert captured["path2"] == PLUGIN_PATH


class TestOutputFile:
    def test_output_writes_the_envelope_to_disk(self, run_cli, monkeypatch, fake_pa, tmp_path):
        _resolve_fake(monkeypatch)
        out = tmp_path / "doctor.json"
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "linear", "-o", str(out)]
        )
        assert result.code == 0, result.stderr
        assert out.exists()
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["command"] == "doctor"
        assert saved["plugin"] == PLUGIN_PATH
        assert "linear" in saved
        assert "결과 저장" in result.stderr

    def test_output_with_json_prints_nothing_extra(self, run_cli, monkeypatch, fake_pa, tmp_path):
        _resolve_fake(monkeypatch)
        out = tmp_path / "doctor.json"
        result = run_cli(
            ["doctor", "-p", "fake-eq", "--mode", "linear", "-o", str(out)], json_mode=True
        )
        assert result.code == 0, result.stderr
        assert result.payload["command"] == "doctor"
        assert json.loads(out.read_text(encoding="utf-8"))["command"] == "doctor"
