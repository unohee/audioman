"""uat compare / smoke / pdc and the dropout-block gate: positive controls PASS, negative controls FAIL."""

from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pedalboard import Distortion, Gain, Pedalboard

from audioman.core.discontinuity import dropout_block_findings
from audioman.core.pdc import estimate_lag, judge_pdc
from audioman.core.uat_gates import TakeSet, compare_takes, dropout_blocks, smoke_check
from audioman.plugins.parameter import PluginMeta
from test_json_contract import _resolve_ref, validate_against

SR = 48000


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------


def _source(seconds: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(1)
    t = np.arange(int(SR * seconds)) / SR
    mono = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(len(t))
    return np.stack([mono, mono]).astype(np.float32)


def _plugin(audio: np.ndarray) -> np.ndarray:
    return Pedalboard([Distortion(drive_db=12), Gain(gain_db=-6)])(audio, SR)


def _delay(audio: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([np.zeros((audio.shape[0], n), audio.dtype), audio[:, :-n]], axis=1)


@pytest.fixture
def takes():
    src = _source()
    off = _plugin(src)
    return src, off


def _gates(report) -> dict[str, dict]:
    return {g["name"]: g for g in report.to_dict()["gates"]}


def _failing(report) -> set[str]:
    return {name for name, g in _gates(report).items() if g["status"] == "FAIL"}


# ---------------------------------------------------------------------------
# compare: positive and negative controls
# ---------------------------------------------------------------------------


def test_positive_control_passes_every_gate(takes):
    src, off = takes
    report = compare_takes(TakeSet(src, off, off.copy(), SR, bypassed=src.copy(), render_seconds=0.5))
    data = report.to_dict()
    assert data["verdict"] == "PASS", _failing(report)
    assert set(_gates(report)) == {
        "realtime_present", "offline_present", "rt_matches_offline", "no_dropout_blocks",
        "no_unprocessed_blocks", "delay_compensated", "processed", "bypass_equals_source",
        "faster_than_real_time",
    }
    assert data["missing_gates"] == []


def test_zeroed_block_fails(takes):
    src, off = takes
    rt = off.copy()
    rt[:, SR + 100: SR + 100 + 512] = 0.0  # one lost 512-sample host buffer, off the analysis grid
    failing = _failing(compare_takes(TakeSet(src, off, rt, SR)))
    assert {"rt_matches_offline", "no_dropout_blocks"} <= failing


def test_one_sample_lag_in_the_render_fails_delay_compensation(takes):
    src, off = takes
    late = _delay(off, 1)
    report = compare_takes(TakeSet(src, late, late.copy(), SR))
    assert _failing(report) == {"delay_compensated"}
    assert "+1 samples" in _gates(report)["delay_compensated"]["reason"]


def test_one_sample_lag_in_the_realtime_take_only_fails_rt_match(takes):
    src, off = takes
    report = compare_takes(TakeSet(src, off, _delay(off, 1), SR))
    assert "rt_matches_offline" in _failing(report)
    assert "offset by 1 samples" in _gates(report)["rt_matches_offline"]["reason"]


def test_bypass_with_delay_fails(takes):
    src, off = takes
    report = compare_takes(TakeSet(src, off, off.copy(), SR, bypassed=_delay(src, 64)))
    assert _failing(report) == {"bypass_equals_source"}
    assert "64 samples" in _gates(report)["bypass_equals_source"]["reason"]


def test_bypass_that_alters_the_signal_fails(takes):
    src, off = takes
    report = compare_takes(TakeSet(src, off, off.copy(), SR, bypassed=src * 0.5))
    assert _failing(report) == {"bypass_equals_source"}


def test_silent_output_fails(takes):
    src, _ = takes
    silent = np.zeros_like(src)
    failing = _failing(compare_takes(TakeSet(src, silent, silent.copy(), SR)))
    assert {"realtime_present", "offline_present", "delay_compensated"} <= failing


def test_non_finite_output_fails(takes):
    src, off = takes
    bad = off.copy()
    bad[0, 10] = np.nan
    gates = _gates(compare_takes(TakeSet(src, off, bad, SR)))
    assert gates["realtime_present"]["status"] == "FAIL" and gates["realtime_present"]["reason"] == "non-finite samples"
    assert gates["rt_matches_offline"]["status"] == "FAIL"  # a NaN residual is never a match


def test_passthrough_fails_processed_and_unprocessed_blocks_are_caught(takes):
    src, off = takes
    assert _failing(compare_takes(TakeSet(src, src.copy(), src.copy(), SR))) == {"processed"}
    # a transparent plugin may declare it; the gate is then not required
    report = compare_takes(TakeSet(src, src.copy(), src.copy(), SR), expect_processing=False)
    assert report.to_dict()["verdict"] == "PASS" and "processed" not in _gates(report)
    # a real-time take that let one block through unprocessed
    rt = off.copy()
    rt[:, SR: SR + 1024] = src[:, SR: SR + 1024]
    assert "no_unprocessed_blocks" in _failing(compare_takes(TakeSet(src, off, rt, SR)))


def test_slow_render_fails(takes):
    src, off = takes
    report = compare_takes(TakeSet(src, off, off.copy(), SR, render_seconds=10.0))
    assert _failing(report) == {"faster_than_real_time"}


def test_dropout_ignores_blocks_that_are_quiet_in_the_reference():
    ref = _source(1.0)
    ref[:, :SR // 2] = 0.0  # the plugin is meant to be silent here
    out = ref.copy()
    assert dropout_blocks(out, ref) == []
    out[:, SR // 2 + 1000: SR // 2 + 2000] = 0.0
    assert dropout_blocks(out, ref)


# ---------------------------------------------------------------------------
# PDC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lag", [0, 1, 96, -5, 4000])
def test_estimate_lag_is_exact_both_ways(lag):
    src = _source()
    processed = _delay(src, lag) if lag > 0 else (src if lag == 0 else np.concatenate(
        [src[:, -lag:], np.zeros((2, -lag), np.float32)], axis=1))
    est = estimate_lag(src, processed, SR)
    assert est.lag_samples == lag and est.ncc > 0.99 and not est.at_edge
    assert judge_pdc(est, lag, SR).status == "PASS"
    assert est.to_dict()["lag_samples"] == lag


def test_steady_state_estimate_ignores_a_warm_up_transient():
    src = _source()
    processed = _delay(src, 300)
    processed[:, :2400] = 0.9  # a loud start-up burst that would fool an onset estimate
    est = estimate_lag(src, processed, SR)
    assert est.lag_samples == 300 and est.settle_samples > 2400


def test_pdc_residual_beyond_tolerance_fails():
    src = _source()
    est = estimate_lag(src, _delay(src, 480), SR)  # 10 ms late
    verdict = judge_pdc(est, 0, SR)
    assert verdict.status == "FAIL" and verdict.residual_samples == 480 and "ms" in verdict.reason
    assert judge_pdc(est, 470, SR).status == "PASS"  # within 2 ms


def test_pdc_fails_when_latency_is_not_established():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, SR)).astype(np.float32)
    b = rng.standard_normal((2, SR)).astype(np.float32)  # unrelated signal
    verdict = judge_pdc(estimate_lag(a, b, SR), 0, SR)
    assert verdict.status == "FAIL" and "not established" in verdict.reason


def test_pdc_edge_of_window_is_not_trusted():
    # Brown noise has a broad correlation peak: with the true lag outside the
    # window, the best in-window lag slides onto the edge instead of a random spot.
    brown = np.cumsum(np.random.default_rng(9).standard_normal(SR))
    brown = (brown - brown.mean()) / np.max(np.abs(brown))
    src = np.stack([brown, brown]).astype(np.float32)
    est = estimate_lag(src, _delay(src, 2000), SR, max_lag_ms=10.0, settle_sec=0.0)
    assert est.at_edge
    assert "search edge" in judge_pdc(est, est.lag_samples, SR).reason


def test_pdc_lag_outside_the_window_on_white_noise_is_not_established():
    src = np.random.default_rng(9).standard_normal((2, SR)).astype(np.float32)
    est = estimate_lag(src, _delay(src, 2000), SR, max_lag_ms=10.0)
    assert "not established" in judge_pdc(est, est.lag_samples, SR).reason


def test_residual_of_empty_signals_is_minus_infinity():
    from audioman.core.uat_gates import residual_peak_dbfs

    assert residual_peak_dbfs(np.zeros(0), np.zeros(0)) == float("-inf")


def test_estimate_lag_on_silence_has_zero_correlation():
    z = np.zeros((2, SR), np.float32)
    assert estimate_lag(z, z, SR).ncc == 0.0


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def test_smoke_check_controls():
    src = _source(1.0)
    ok = smoke_check(src, _plugin(src), SR, n_parameters=3)
    assert all(g.status == "PASS" for g in ok) and len(ok) == 4
    bad = {g.name: g.status for g in smoke_check(src, src[:, :-10], SR)}
    assert bad["shape_matches"] == "FAIL"
    nan = src.copy()
    nan[0, 0] = np.inf
    assert {g.name: g.status for g in smoke_check(src, nan, SR)}["finite"] == "FAIL"
    tail = src.copy()
    tail[:, SR // 2:] = 0
    assert {g.name: g.status for g in smoke_check(src, tail, SR)}["second_half_audible"] == "FAIL"
    assert {g.name: g.status for g in smoke_check(src, src, SR, n_parameters=0, label="x")}["x.parameters_exposed"] == "FAIL"


# ---------------------------------------------------------------------------
# stream triage dropout gate
# ---------------------------------------------------------------------------


def test_dropout_block_findings():
    ref = _source(1.0)
    assert dropout_block_findings(ref, ref, SR) == []
    lost = ref.copy()
    lost[:, 10000:10512] = 0.0
    found = dropout_block_findings(lost, ref, SR, file="x.wav")
    assert len(found) == 1
    f = found[0].to_dict()
    assert f["code"] == "SAMPLE_DROPOUT" and f["severity"] == "critical"
    assert f["measurement"]["dropout_blocks"] >= 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class _Result:
    code: int
    stdout: str
    stderr: str


def _run(argv: list[str]) -> _Result:
    from audioman.cli import app

    plain_before = os.environ.get("AUDIOMAN_PLAIN")
    out, err = io.StringIO(), io.StringIO()
    real = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        app.main(argv)
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdout, sys.stderr = real
        if plain_before is None:
            os.environ.pop("AUDIOMAN_PLAIN", None)
        else:
            os.environ["AUDIOMAN_PLAIN"] = plain_before
    return _Result(code, out.getvalue(), err.getvalue())


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from audioman.config import settings as settings_module

    monkeypatch.setenv("AUDIOMAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_module.reset_settings()
    yield
    settings_module.reset_settings()


def _write(path: Path, audio: np.ndarray, sr: int = SR) -> str:
    sf.write(path, audio.T, sr, subtype="FLOAT")
    return str(path)


def _validated(result: _Result) -> dict:
    payload = json.loads(result.stdout)
    validate_against(payload, _resolve_ref("audioman://schema/uat.v1.json"))
    assert payload["digest"].startswith(payload["command"])
    return payload


def test_cli_compare_positive_and_negative(tmp_path, takes):
    src, off = takes
    files = {
        "source": _write(tmp_path / "src.wav", src),
        "offline": _write(tmp_path / "off.wav", off),
        "realtime": _write(tmp_path / "rt.wav", off),
        "bypassed": _write(tmp_path / "byp.wav", src),
    }
    argv = ["--json", "uat", "compare", *sum(([f"--{k}", v] for k, v in files.items()), []), "--render-seconds", "0.4"]
    ok = _run(argv)
    assert ok.code == 0, ok.stderr
    assert _validated(ok)["report"]["verdict"] == "PASS"

    zeroed = off.copy()
    zeroed[:, SR: SR + 4096] = 0.0
    files["realtime"] = _write(tmp_path / "rt_bad.wav", zeroed)
    bad = _run(["--json", "uat", "compare", *sum(([f"--{k}", v] for k, v in files.items()), [])])
    assert bad.code == 1
    payload = _validated(bad)
    assert payload["report"]["verdict"] == "FAIL" and "no_dropout_blocks" in payload["digest"]

    human = _run(["uat", "compare", "--source", files["source"], "--offline", files["offline"],
                  "--realtime", files["offline"], "--transparent"])
    assert human.code == 0 and "rt_matches_offline" in human.stdout


def test_cli_compare_rejects_mismatched_rates_and_bad_files(tmp_path, takes):
    src, off = takes
    s = _write(tmp_path / "src.wav", src)
    other = _write(tmp_path / "off44.wav", off, sr=44100)
    assert _run(["uat", "compare", "--source", s, "--offline", other, "--realtime", other]).code == 1
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"nope")
    assert _run(["uat", "compare", "--source", str(junk), "--offline", s, "--realtime", s]).code == 1


def test_cli_smoke_builtin_and_check_output(tmp_path):
    ok = _run(["--json", "uat", "smoke", "--plugin", "builtin:distortion,gain", "--rates", "44100,48000",
               "--duration", "0.5"])
    assert ok.code == 0, ok.stderr
    gates = {g["name"] for g in _validated(ok)["report"]["gates"]}
    assert {"44100.finite", "48000.second_half_audible"} <= gates

    src = _source(1.0)
    s = _write(tmp_path / "probe.wav", src)
    good = _write(tmp_path / "host_out.wav", _plugin(src))
    assert _run(["--json", "uat", "smoke", "--source", s, "--check-output", good]).code == 0
    short = _write(tmp_path / "host_short.wav", _plugin(src)[:, :-100], sr=44100)
    res = _run(["--json", "uat", "smoke", "--source", s, "--check-output", short])
    assert res.code == 1
    failing = {g["name"] for g in _validated(res)["report"]["gates"] if g["status"] == "FAIL"}
    assert {"shape_matches", "sample_rate_matches"} <= failing


@pytest.mark.parametrize(
    "argv",
    [
        ["uat", "smoke"],
        ["uat", "smoke", "--check-output", "x.wav"],
        ["uat", "smoke", "--plugin", "builtin:wobble"],
        ["uat", "pdc"],
        ["uat"],
    ],
)
def test_cli_usage_errors(argv):
    assert _run(argv).code == 1


class _FakeWrapper:
    instances: list = []
    fail = False

    def __init__(self, path):
        self.path = path
        self.name = "fake"
        self._plugin = type("P", (), {"latency_samples": 64})()
        self.params = None
        _FakeWrapper.instances.append(self)

    def load(self):
        pass

    def set_parameters(self, params):
        self.params = params

    def get_parameters(self):
        return [object(), object()]

    def process(self, audio, sample_rate, reset=True):
        if _FakeWrapper.fail:
            raise RuntimeError("plugin crashed")
        return _delay(np.asarray(audio, dtype=np.float32) * 0.5, 64)


@pytest.fixture
def fake_vst3(monkeypatch, tmp_path):
    meta = PluginMeta(name="Fake", short_name="fake", path=str(tmp_path / "Fake.vst3"), format="vst3")
    monkeypatch.setattr("audioman.core.registry.get_registry",
                        lambda: type("R", (), {"get": staticmethod(lambda n: meta if n == "fake" else None)})())
    monkeypatch.setattr("audioman.plugins.vst3.VST3PluginWrapper", _FakeWrapper)
    _FakeWrapper.instances = []
    _FakeWrapper.fail = False
    return _FakeWrapper


def test_cli_smoke_and_pdc_with_an_external_plugin(fake_vst3):
    ok = _run(["--json", "uat", "smoke", "--plugin", "fake", "--param", "drive=3", "--rates", "48000",
               "--duration", "0.5"])
    assert ok.code == 0, ok.stderr
    assert "48000.parameters_exposed" in {g["name"] for g in _validated(ok)["report"]["gates"]}
    assert fake_vst3.instances[0].params == {"drive": 3.0} or fake_vst3.instances[0].params

    pdc = _run(["--json", "uat", "pdc", "--plugin", "fake"])
    assert pdc.code == 0, pdc.stderr
    report = _validated(pdc)["report"]
    measured = {m["name"]: m["value"] for m in report["measurements"]}
    assert measured["measured_latency_samples"] == 64 == measured["reported_latency_samples"]

    lying = _run(["--json", "uat", "pdc", "--plugin", "fake", "--reported", "0", "--tolerance-ms", "0.5"])
    assert lying.code == 1

    fake_vst3.fail = True
    crashed = _run(["--json", "uat", "smoke", "--plugin", "fake", "--rates", "48000", "--duration", "0.5"])
    assert crashed.code == 1
    assert "48000.finite" in _validated(crashed)["report"]["missing_gates"]


def test_cli_smoke_unknown_plugin_fails(fake_vst3):
    assert _run(["uat", "smoke", "--plugin", "nope"]).code == 1


def test_cli_pdc_files_mode(tmp_path):
    src = _source()
    ref = _write(tmp_path / "ref.wav", src)
    proc = _write(tmp_path / "proc.wav", _delay(src, 96))
    good = _run(["--json", "uat", "pdc", "--reference", ref, "--processed", proc, "--reported", "96"])
    assert good.code == 0 and _validated(good)["report"]["verdict"] == "PASS"
    bad = _run(["uat", "pdc", "--reference", ref, "--processed", proc, "--reported", "0", "--tolerance-ms", "1"])
    assert bad.code == 1
    other = _write(tmp_path / "proc44.wav", src, sr=44100)
    assert _run(["uat", "pdc", "--reference", ref, "--processed", other, "--reported", "0"]).code == 1
    assert _run(["--json", "uat", "pdc", "--plugin", "builtin:gain"]).code == 0
