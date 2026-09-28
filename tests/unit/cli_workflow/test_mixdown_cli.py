# tests/unit/cli_workflow/test_mixdown_cli.py
# Purpose: cover `audioman mixdown` — track config wiring, dry-run plan,
#          session-file mode, automix integration and the failure paths.
#
# This host has no VST3 plugin (AUD-1857), so plugin-dependent paths use a fake
# registry + fake wrapper. Everything else (mixing, automix, JSON contract,
# clipping warning) runs for real.

from __future__ import annotations

import json

import pytest

from audioman.plugins.parameter import PluginMeta

from .conftest import write_wav


@pytest.fixture
def tracks(tmp_path):
    return [
        write_wav(tmp_path / "a.wav", sample_rate=8000, duration=0.2, frequency=440.0),
        write_wav(tmp_path / "b.wav", sample_rate=8000, duration=0.2, frequency=660.0),
    ]


class _ZeroLatency:
    """LatencyMeasurement stand-in: a plugin that reports no delay."""

    plugin_name = "fake"
    reported_latency = 0
    measured_latency = 0
    confidence = 0.0
    used_latency = 0


class TestParsing:
    def test_empty_gain_list_is_empty(self):
        from audioman.cli.mixdown import _parse_float_list

        assert _parse_float_list("") == []
        assert _parse_float_list("   ") == []

    def test_gain_list_is_split_and_trimmed(self):
        from audioman.cli.mixdown import _parse_float_list

        assert _parse_float_list(" -3, 0 ,+2.5") == [-3.0, 0.0, 2.5]

    def test_malformed_gain_list_raises_value_error(self):
        from audioman.cli.mixdown import _parse_float_list

        with pytest.raises(ValueError):
            _parse_float_list("loud")


class TestDryRun:
    def test_plan_lists_tracks_with_gain_and_pan(self, run_cli, tracks, tmp_path):
        out = tmp_path / "mix.wav"
        result = run_cli(
            ["mixdown", str(tracks[0]), str(tracks[1]), "-o", str(out),
             "--gain=-3,0", "--pan=-1,1", "--dry-run"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "mixdown"
        assert payload["dry_run"] is True
        assert payload["track_count"] == 2
        assert payload["output"] == str(out)
        assert payload["tracks"][0]["gain_db"] == -3.0
        assert payload["tracks"][0]["pan"] == -1.0
        assert payload["tracks"][1]["pan"] == 1.0
        assert not out.exists()

    def test_human_plan_prints_track_and_master_lines(self, run_cli, tracks, tmp_path):
        result = run_cli(
            ["mixdown", str(tracks[0]), "-o", str(tmp_path / "mix.wav"),
             "--master", "limiter:threshold=-1", "--dry-run"],
        )
        assert result.code == 0, result.stderr
        assert "Mixdown Plan" in result.stdout
        assert "gain=+0.0dB" in result.stdout
        # Note: the rich console consumes "[limiter]" as a markup token, so only
        # the label prefix is observable here; the chain content is asserted via
        # the JSON payload in TestExecution/TestSessionMode.
        assert "Master:" in result.stdout
        assert "Automix" not in result.stdout
        assert not (tmp_path / "mix.wav").exists()

    def test_missing_gain_entries_default_to_unity(self, run_cli, tracks, tmp_path):
        result = run_cli(
            ["mixdown", str(tracks[0]), str(tracks[1]), "-o", str(tmp_path / "m.wav"),
             "--gain=-6", "--dry-run"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert [t["gain_db"] for t in result.payload["tracks"]] == [-6.0, 0.0]

    def test_no_input_files_is_a_usage_error(self, run_cli, tmp_path):
        result = run_cli(["mixdown", "-o", str(tmp_path / "m.wav")])
        assert result.code == 1
        assert "입력 파일을 지정하세요" in result.stderr
        assert "--session" in result.stderr


class TestExecution:
    def test_mixdown_writes_the_output_and_reports_it(self, run_cli, tracks, tmp_path):
        import soundfile as sf

        out = tmp_path / "mix.wav"
        result = run_cli(
            ["mixdown", str(tracks[0]), str(tracks[1]), "-o", str(out), "--gain=-12"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["output_path"] == str(out)
        assert payload["track_count"] == 2
        assert payload["sample_rate"] == 8000
        assert payload["master_chain"] is None
        assert payload["master_latency_samples"] == 0
        assert payload["clipping_detected"] is False
        assert payload["duration_seconds"] >= 0
        assert out.exists()
        audio, sr = sf.read(str(out), always_2d=True)
        assert sr == 8000
        assert audio.shape[1] == 2
        assert audio.shape[0] == pytest.approx(1600, abs=8)

    def test_loud_tracks_report_clipping(self, run_cli, tmp_path):
        loud = write_wav(tmp_path / "loud.wav", sample_rate=8000, duration=0.1, amplitude=0.95)
        result = run_cli(["mixdown", str(loud), str(loud), "-o", str(tmp_path / "m.wav")])
        assert result.code == 0, result.stderr
        assert "클리핑 감지" in result.stderr
        assert "리미터 추가" in result.stderr

    def test_human_output_reports_mixdown_complete(self, run_cli, tracks, tmp_path):
        out = tmp_path / "mix.wav"
        result = run_cli(["mixdown", str(tracks[0]), "-o", str(out)])
        assert result.code == 0, result.stderr
        assert "Mixdown complete" in result.stderr
        assert "Tracks: 1" in result.stdout
        assert str(out) in result.stdout
        assert "SR:     8000 Hz" in result.stdout

    def test_unreadable_input_surfaces_as_a_mixdown_failure(self, run_cli, tmp_path):
        bogus = tmp_path / "bogus.wav"
        bogus.write_bytes(b"not audio")
        result = run_cli(["mixdown", str(bogus), "-o", str(tmp_path / "m.wav")])
        assert result.code == 1
        assert "믹스다운 실패" in result.stderr

    def test_master_chain_plugin_lookup_failure_is_reported(self, run_cli, tracks, tmp_path):
        result = run_cli(
            ["mixdown", str(tracks[0]), "-o", str(tmp_path / "m.wav"),
             "--master", "definitely-not-installed"],
        )
        assert result.code == 1
        assert "믹스다운 실패" in result.stderr
        assert "definitely-not-installed" in result.stderr

    def test_master_chain_runs_through_the_registry(self, run_cli, tracks, tmp_path, monkeypatch):
        """Master chain wiring: registry lookup -> wrapper load -> process."""
        calls = {"processed": 0}

        class _FakeWrapper:
            def __init__(self, path):
                calls["path"] = path

            def load(self):
                calls["loaded"] = True

            def set_parameters(self, params):
                calls["params"] = params

            def process(self, audio, sr, reset=False):
                calls["processed"] += 1
                return audio

        meta = PluginMeta(name="Fake Limit", short_name="limiter",
                          path="/plugins/FakeLimiter.vst3", format="vst3")

        class _FakeRegistry:
            def get(self, name):
                assert name == "limiter"
                return meta

        monkeypatch.setattr("audioman.core.mixer.get_registry", lambda: _FakeRegistry())
        monkeypatch.setattr("audioman.core.mixer.VST3PluginWrapper", _FakeWrapper)
        # Delay compensation measures the chain through core.latency's own imports.
        monkeypatch.setattr("audioman.core.latency.get_registry", lambda: _FakeRegistry())
        monkeypatch.setattr("audioman.core.latency.VST3PluginWrapper", _FakeWrapper)
        monkeypatch.setattr("audioman.core.latency.measure_plugin_latency",
                            lambda wrapper, sr, **kw: _ZeroLatency())

        out = tmp_path / "mastered.wav"
        result = run_cli(
            ["mixdown", str(tracks[0]), "-o", str(out), "--master", "limiter:threshold=-1"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert calls["path"] == "/plugins/FakeLimiter.vst3"
        assert calls["loaded"] is True
        assert calls["params"] == {"threshold": -1.0}
        assert calls["processed"] == 1
        payload = result.payload
        assert payload["master_chain"] == [{"plugin": "limiter", "params": {"threshold": -1.0}}]
        assert payload["master_latency_samples"] == 0
        assert out.exists()

    def test_no_compensation_skips_latency_measurement(self, run_cli, tracks, tmp_path, monkeypatch):
        class _FakeWrapper:
            def __init__(self, path):
                pass

            def load(self):
                pass

            def set_parameters(self, params):
                pass

            def process(self, audio, sr, reset=False):
                return audio

        meta = PluginMeta(name="Fake", short_name="fake", path="/p/Fake.vst3", format="vst3")
        monkeypatch.setattr("audioman.core.mixer.get_registry",
                            lambda: type("R", (), {"get": lambda self, n: meta})())
        monkeypatch.setattr("audioman.core.mixer.VST3PluginWrapper", _FakeWrapper)
        monkeypatch.setattr(
            "audioman.core.latency.measure_chain_latency",
            lambda *a, **k: pytest.fail("latency must not be measured with --no-compensation"),
        )

        out = tmp_path / "noc.wav"
        result = run_cli(
            ["mixdown", str(tracks[0]), "-o", str(out), "--master", "fake", "--no-compensation"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["master_latency_samples"] == 0

    def test_per_track_chain_runs_before_mixing(self, run_cli, tracks, tmp_path, monkeypatch):
        seen = []

        class _FakeWrapper:
            def __init__(self, path):
                pass

            def load(self):
                pass

            def set_parameters(self, params):
                pass

            def process(self, audio, sr, reset=False):
                seen.append(audio.shape[1])
                return audio

        meta = PluginMeta(name="Fake", short_name="fake", path="/p/Fake.vst3", format="vst3")
        monkeypatch.setattr("audioman.core.mixer.get_registry",
                            lambda: type("R", (), {"get": lambda self, n: meta})())
        monkeypatch.setattr("audioman.core.mixer.VST3PluginWrapper", _FakeWrapper)

        out = tmp_path / "chained.wav"
        result = run_cli(
            ["mixdown", str(tracks[0]), str(tracks[1]), "-o", str(out),
             "--chain", "fake|fake"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert seen == [1600, 1600]
        assert result.payload["tracks"][0]["chain"] == [{"plugin": "fake", "params": {}}]


class TestAutomix:
    def test_dry_run_plan_includes_automix_gains(self, run_cli, tracks, tmp_path):
        result = run_cli(
            ["mixdown", str(tracks[0]), str(tracks[1]), "-o", str(tmp_path / "m.wav"),
             "--automix", "--dry-run"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert "automix" in payload
        assert payload["automix"]["target_profile"]["type"] == "pink_noise"
        assert len(payload["automix"]["gains_db"]) == 2
        # Automix gains are applied on top of the CLI gains.
        assert payload["tracks"][0]["gain_db"] == pytest.approx(payload["automix"]["gains_db"][0])

    def test_human_plan_prints_automix_block(self, run_cli, tracks, tmp_path):
        result = run_cli(
            ["mixdown", str(tracks[0]), "-o", str(tmp_path / "m.wav"), "--automix", "--dry-run"],
        )
        assert result.code == 0, result.stderr
        assert "Automix (target: pink_noise)" in result.stdout
        assert "Residual:" in result.stdout

    def test_executed_automix_reports_applied_gains(self, run_cli, tracks, tmp_path):
        out = tmp_path / "auto.wav"
        result = run_cli(
            ["mixdown", str(tracks[0]), str(tracks[1]), "-o", str(out), "--automix"],
        )
        assert result.code == 0, result.stderr
        assert "Automix Applied" in result.stdout
        assert "other:" in result.stdout
        assert out.exists()

    def test_explicit_target_overrides_the_default(self, run_cli, tracks, tmp_path):
        result = run_cli(
            ["mixdown", str(tracks[0]), "-o", str(tmp_path / "m.wav"),
             "--automix", "--target", "rock", "--dry-run"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["automix"]["target_profile"]["type"] == "genre"
        assert result.payload["automix"]["target_profile"]["genre"] == "rock"

    def test_reference_target_without_a_path_falls_back_to_pink_noise(self, run_cli, tracks, tmp_path):
        """`reference` needs --reference; without it automix keeps the default profile."""
        result = run_cli(
            ["mixdown", str(tracks[0]), "-o", str(tmp_path / "m.wav"),
             "--automix", "--target", "reference", "--dry-run"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["automix"]["target_profile"]["type"] == "pink_noise"

    def test_automix_with_a_reference_file_uses_its_spectrum(self, run_cli, tracks, tmp_path):
        reference = write_wav(tmp_path / "ref.wav", sample_rate=8000, duration=0.3)
        result = run_cli(
            ["mixdown", str(tracks[0]), "-o", str(tmp_path / "m.wav"),
             "--automix", "--reference", str(reference), "--dry-run"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["automix"]["target_profile"]["type"] == "reference"
        assert result.payload["automix"]["target_profile"]["path"] == str(reference)

    def test_automix_failure_on_a_missing_track_is_reported(self, run_cli, tmp_path):
        result = run_cli(
            ["mixdown", str(tmp_path / "missing.wav"), "-o", str(tmp_path / "m.wav"),
             "--automix"],
        )
        assert result.code == 1
        assert "Automix 분석 실패" in result.stderr


class TestSessionMode:
    def _session(self, tmp_path, tracks, *, master=None, extra=None):
        data = {
            "output": str(tmp_path / "session_out.wav"),
            "format": "PCM_24",
            "tracks": [
                {"path": str(tracks[0]), "gain_db": -3.0, "pan": -0.5},
                {"path": str(tracks[1]), "gain_db": -6.0, "pan": 0.5},
            ],
        }
        if master:
            data["master"] = {"chain": master}
        if extra:
            data.update(extra)
        path = tmp_path / "session.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_session_file_drives_the_plan(self, run_cli, tracks, tmp_path):
        session = self._session(tmp_path, tracks)
        result = run_cli(
            ["mixdown", "-o", str(tmp_path / "cli_out.wav"), "--session", str(session),
             "--dry-run"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["track_count"] == 2
        assert payload["tracks"][0]["gain_db"] == -3.0
        assert payload["tracks"][1]["pan"] == 0.5
        # --output wins over the session's own output field.
        assert payload["output"] == str(tmp_path / "cli_out.wav")

    def test_session_file_master_chain_is_parsed(self, run_cli, tracks, tmp_path):
        session = self._session(tmp_path, tracks, master="limiter:threshold=-1")
        result = run_cli(
            ["mixdown", "-o", str(tmp_path / "o.wav"), "--session", str(session), "--dry-run"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["master_chain"] == [
            {"plugin": "limiter", "params": {"threshold": -1.0}}
        ]

    def test_session_file_executes_a_real_mixdown(self, run_cli, tracks, tmp_path):
        session = self._session(tmp_path, tracks)
        out = tmp_path / "rendered.wav"
        result = run_cli(["mixdown", "-o", str(out), "--session", str(session)])
        assert result.code == 0, result.stderr
        assert out.exists()
        assert "Mixdown complete" in result.stderr

    def test_unloadable_session_file_is_reported(self, run_cli, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        result = run_cli(["mixdown", "-o", str(tmp_path / "o.wav"), "--session", str(bad)])
        assert result.code == 1
        assert "세션 파일 로드 실패" in result.stderr

    def test_missing_session_file_is_reported(self, run_cli, tmp_path):
        result = run_cli(
            ["mixdown", "-o", str(tmp_path / "o.wav"), "--session", str(tmp_path / "nope.json")]
        )
        assert result.code == 1
        assert "세션 파일 로드 실패" in result.stderr


class TestEarlyReturnsAfterErrors:
    """A printed error must stop the command instead of cascading into a crash.

    print_error normally exits; these tests neutralise that exit to prove the
    explicit `return` guards are what stop the run (otherwise the next statement
    would raise NameError/UnboundLocalError on the unset locals).
    """

    def test_session_load_failure_returns_before_using_the_session(self, run_cli, tmp_path, monkeypatch):
        from audioman.cli import mixdown as mixdown_cli

        seen = []
        monkeypatch.setattr(mixdown_cli, "print_error", lambda msg: seen.append(msg))

        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        result = run_cli(["mixdown", "-o", str(tmp_path / "o.wav"), "--session", str(bad)])

        assert result.code == 0
        assert seen and "세션 파일 로드 실패" in seen[0]
        assert not (tmp_path / "o.wav").exists()

    def test_missing_inputs_return_before_building_tracks(self, run_cli, tmp_path, monkeypatch):
        from audioman.cli import mixdown as mixdown_cli

        seen = []
        monkeypatch.setattr(mixdown_cli, "print_error", lambda msg: seen.append(msg))

        result = run_cli(["mixdown", "-o", str(tmp_path / "o.wav")])

        assert result.code == 0
        assert seen and "입력 파일을 지정하세요" in seen[0]
        assert not (tmp_path / "o.wav").exists()

    def test_automix_failure_returns_before_applying_gains(self, run_cli, tmp_path, monkeypatch):
        from audioman.cli import mixdown as mixdown_cli

        seen = []
        monkeypatch.setattr(mixdown_cli, "print_error", lambda msg: seen.append(msg))

        result = run_cli(
            ["mixdown", str(tmp_path / "missing.wav"), "-o", str(tmp_path / "o.wav"),
             "--automix"],
        )

        assert result.code == 0
        assert seen and "Automix 분석 실패" in seen[0]
        assert not (tmp_path / "o.wav").exists()

    def test_mixdown_failure_returns_before_reporting(self, run_cli, tmp_path, monkeypatch):
        from audioman.cli import mixdown as mixdown_cli

        seen = []
        monkeypatch.setattr(mixdown_cli, "print_error", lambda msg: seen.append(msg))

        bogus = tmp_path / "bogus.wav"
        bogus.write_bytes(b"not audio")
        result = run_cli(["mixdown", str(bogus), "-o", str(tmp_path / "o.wav")])

        assert result.code == 0
        assert seen and "믹스다운 실패" in seen[0]
        assert "Mixdown complete" not in result.stderr


class TestUngroupedAutomixRendering:
    """When automix returns no instrument groups the per-track branch renders."""

    def _ungrouped_result(self, gains, paths):
        from audioman.core.automix import AutomixResult

        return AutomixResult(
            gains_db=list(gains),
            band_analysis=[
                {"path": str(p), "bands": {"sub": -30.0, "mid": -20.0}, "rms_db": -24.0}
                for p in paths
            ],
            target_profile={"type": "pink_noise", "bands": {"sub": -20.0}},
            residual_error_db=-12.5,
            groups=None,
        )

    def test_dry_run_renders_per_track_gains_and_bands(self, run_cli, tracks, tmp_path, monkeypatch):
        result_obj = self._ungrouped_result([3.0, -4.0], tracks)
        monkeypatch.setattr("audioman.core.automix.automix", lambda **kw: result_obj)

        result = run_cli(
            ["mixdown", str(tracks[0]), str(tracks[1]), "-o", str(tmp_path / "m.wav"),
             "--automix", "--dry-run"],
        )
        assert result.code == 0, result.stderr
        assert "Track 1: +3.0dB" in result.stdout
        assert "sub=-30  mid=-20" in result.stdout
        assert "Residual: -12.5dB" in result.stdout

    def test_executed_mixdown_renders_per_track_gains(self, run_cli, tracks, tmp_path, monkeypatch):
        result_obj = self._ungrouped_result([3.0, -4.0], tracks)
        monkeypatch.setattr("audioman.core.automix.automix", lambda **kw: result_obj)

        out = tmp_path / "auto.wav"
        result = run_cli(["mixdown", str(tracks[0]), str(tracks[1]), "-o", str(out), "--automix"])
        assert result.code == 0, result.stderr
        assert "Automix Applied" in result.stdout
        assert "Track 1: +3.0dB" in result.stdout
        assert "Track 2: -4.0dB" in result.stdout
        assert out.exists()

    def test_human_plan_prints_grouped_automix(self, run_cli, tracks, tmp_path):
        """The real automix groups tracks by filename; both index lists are rendered."""
        result = run_cli(
            ["mixdown", str(tracks[0]), str(tracks[1]), "-o", str(tmp_path / "m.wav"),
             "--automix", "--dry-run"],
        )
        assert result.code == 0, result.stderr
        assert "Automix (target: pink_noise)" in result.stdout
        assert "a.wav" in result.stdout and "b.wav" in result.stdout


class TestMasterChainHumanOutput:
    def test_master_chain_steps_and_latency_are_reported(self, run_cli, tracks, tmp_path, monkeypatch):
        class _FakeWrapper:
            def __init__(self, path):
                pass

            def load(self):
                pass

            def set_parameters(self, params):
                pass

            def process(self, audio, sr, reset=False):
                return audio

        meta = PluginMeta(name="Fake Limit", short_name="limiter",
                          path="/plugins/Fake.vst3", format="vst3")

        class _FakeRegistry:
            def get(self, name):
                return meta

        class _Latency:
            plugin_name = "limiter"
            reported_latency = 64
            measured_latency = 64
            confidence = 1.0
            used_latency = 64

        monkeypatch.setattr("audioman.core.mixer.get_registry", lambda: _FakeRegistry())
        monkeypatch.setattr("audioman.core.mixer.VST3PluginWrapper", _FakeWrapper)
        monkeypatch.setattr("audioman.core.latency.get_registry", lambda: _FakeRegistry())
        monkeypatch.setattr("audioman.core.latency.VST3PluginWrapper", _FakeWrapper)
        monkeypatch.setattr("audioman.core.latency.measure_plugin_latency",
                            lambda wrapper, sr, **kw: _Latency())

        out = tmp_path / "lat.wav"
        result = run_cli(
            ["mixdown", str(tracks[0]), "-o", str(out), "--master", "limiter:threshold=-1"],
        )
        assert result.code == 0, result.stderr
        assert "Master: 1 steps" in result.stdout
        assert "Master latency compensation: 64 samples" in result.stdout
        assert out.exists()
