# tests/unit/cli_extra2/test_commit_cmd.py
# Purpose: cover `audioman commit` — dry-run latency measurement, the real
#          commit with/without delay compensation and tail trim, and the error
#          paths.
#
# This host has no registrable VST3 plugin (AUD-1857), so registry lookups and
# the plugin wrapper are stubbed with a deterministic delay line. The delay
# compensation and tail-trim assertions therefore land on exact sample counts.

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from .conftest import install_registry, install_wrapper, make_meta, write_wav

SR = 8000


@pytest.fixture
def source(tmp_path):
    """Stereo sine with a click-free body, long enough for delay assertions."""
    return write_wav(tmp_path / "in.wav", sample_rate=SR, duration=0.5, amplitude=0.3)


class TestChainParsing:
    def test_empty_chain_is_rejected_before_any_work(self, run_cli, source, tmp_path, silent_error):
        """`--chain ,` parses to zero steps; the guard must report and return."""
        seen = silent_error("audioman.cli.commit_cmd")

        result = run_cli(
            ["commit", str(source), "-o", str(tmp_path / "out.wav"), "--chain", ","],
        )

        assert result.code == 0, result.stderr
        assert seen and "처리 단계가 비어있습니다" in seen[0]
        assert not (tmp_path / "out.wav").exists()

    def test_empty_chain_exits_nonzero_through_the_real_error_path(self, run_cli, source, tmp_path):
        result = run_cli(
            ["commit", str(source), "-o", str(tmp_path / "out.wav"), "--chain", ","],
        )
        assert result.code == 1
        assert "처리 단계가 비어있습니다" in result.stderr


class TestDryRun:
    def test_json_payload_measures_the_chain(self, run_cli, source, tmp_path, monkeypatch):
        meta = make_meta(short_name="fake-denoiser", name="Fake De-noise")
        stub = install_registry(monkeypatch, ("audioman.core.latency", "audioman.core.commit"), [meta])
        install_wrapper(monkeypatch, ("audioman.core.latency",), delay=8, reported=8)

        result = run_cli(
            ["commit", str(source), "-o", str(tmp_path / "out.wav"),
             "--chain", "fake-denoiser:threshold=-20", "--dry-run"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "commit"
        assert payload["dry_run"] is True
        assert payload["chain"] == [{"plugin": "fake-denoiser", "params": {"threshold": -20.0}}]
        measurement = payload["latency"][0]
        assert measurement["plugin_name"] == "fake-denoiser"
        assert measurement["measured_latency"] == 8
        assert measurement["reported_latency"] == 8
        assert measurement["used_latency"] == 8
        assert payload["total_latency_samples"] == 8
        assert stub.get_calls == ["fake-denoiser"]
        # Dry-run must not create the output.
        assert not (tmp_path / "out.wav").exists()

    def test_human_report_prints_each_plugin_and_the_total(self, run_cli, source, tmp_path, monkeypatch):
        meta = make_meta(short_name="fake-denoiser", name="Fake De-noise")
        install_registry(monkeypatch, ("audioman.core.latency",), [meta])
        install_wrapper(monkeypatch, ("audioman.core.latency",), delay=8, reported=8)

        result = run_cli(
            ["commit", str(source), "-o", str(tmp_path / "out.wav"),
             "--chain", "fake-denoiser", "--dry-run"],
        )

        assert result.code == 0, result.stderr
        assert "Latency Measurement (dry-run)" in result.stdout
        assert "fake-denoiser: 8 samples (measured=8, reported=8 ✓)" in result.stdout
        assert "confidence=100%" in result.stdout
        assert "Total: 8 samples (0.2ms @ 48000Hz)" in result.stdout

    def test_measured_reported_mismatch_is_flagged(self, run_cli, source, tmp_path, monkeypatch):
        meta = make_meta(short_name="fake-denoiser")
        install_registry(monkeypatch, ("audioman.core.latency",), [meta])
        install_wrapper(monkeypatch, ("audioman.core.latency",), delay=8, reported=3)

        result = run_cli(
            ["commit", str(source), "-o", str(tmp_path / "o.wav"),
             "--chain", "fake-denoiser", "--dry-run"],
        )

        assert result.code == 0, result.stderr
        # The "≠" marker is what tells the user the plugin's own report is wrong.
        assert "≠" in result.stdout
        assert "measured=8, reported=3" in result.stdout

    def test_zero_latency_chain_reports_zero_total(self, run_cli, source, tmp_path, monkeypatch):
        meta = make_meta(short_name="fake-passthrough")
        install_registry(monkeypatch, ("audioman.core.latency",), [meta])
        install_wrapper(monkeypatch, ("audioman.core.latency",), delay=0)

        result = run_cli(
            ["commit", str(source), "-o", str(tmp_path / "o.wav"),
             "--chain", "fake-passthrough", "--dry-run"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        assert result.payload["total_latency_samples"] == 0

    def test_measurement_failure_is_reported(self, run_cli, source, tmp_path, monkeypatch, silent_error):
        from audioman.core import latency as latency_module

        def _boom(*args, **kwargs):
            raise RuntimeError("measurement exploded")

        monkeypatch.setattr(latency_module, "measure_chain_latency", _boom)
        seen = silent_error("audioman.cli.commit_cmd")

        result = run_cli(
            ["commit", str(source), "-o", str(tmp_path / "o.wav"),
             "--chain", "fake-denoiser", "--dry-run"],
        )

        assert result.code == 0, result.stderr
        assert seen and "레이턴시 측정 실패" in seen[0]

    def test_unknown_plugin_in_dry_run_is_reported(self, run_cli, source, tmp_path):
        result = run_cli(
            ["commit", str(source), "-o", str(tmp_path / "o.wav"),
             "--chain", "definitely-not-installed", "--dry-run"],
        )
        assert result.code == 1
        assert "레이턴시 측정 실패" in result.stderr
        assert "definitely-not-installed" in result.stderr


class TestCommit:
    """Shared stub install: `commit_file` measures through `core.latency` and
    processes through its own wrapper, so both modules must be patched."""

    def _install(self, monkeypatch, *, names=("fake-denoiser",), delay=8, tail=0,
                 reported=8, measure_delay=None):
        metas = [make_meta(short_name=name) for name in names]
        stub = install_registry(
            monkeypatch, ("audioman.core.latency", "audioman.core.commit"), metas,
        )
        measure_wrappers = install_wrapper(
            monkeypatch, ("audioman.core.latency",),
            delay=delay if measure_delay is None else measure_delay,
            reported=reported,
        )
        process_wrappers = install_wrapper(
            monkeypatch, ("audioman.core.commit",), delay=delay, tail=tail,
        )
        return stub, measure_wrappers, process_wrappers

    def test_json_payload_describes_the_committed_file(self, run_cli, source, tmp_path, monkeypatch):
        stub, _, processors = self._install(monkeypatch)

        out = tmp_path / "out.wav"
        result = run_cli(
            ["commit", str(source), "-o", str(out), "--chain", "fake-denoiser:threshold=-6"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["input_path"] == str(source)
        assert payload["output_path"] == str(out)
        assert payload["steps"] == [{"plugin": "fake-denoiser", "params": {"threshold": -6.0}}]
        assert payload["total_latency_samples"] == 8
        assert payload["latency_compensation"][0]["plugin_name"] == "fake-denoiser"
        assert payload["duration_seconds"] >= 0
        assert stub.get_calls == ["fake-denoiser", "fake-denoiser"]

        # The processing wrapper got the parsed parameters.
        assert processors[0].applied == [{"threshold": -6.0}]
        assert out.exists()
        audio, sr = sf.read(str(out), always_2d=True)
        assert sr == SR
        assert audio.shape[0] == 4000  # tail_trim keeps the original length

    def test_human_report_prints_the_compensation_block(self, run_cli, source, tmp_path, monkeypatch):
        self._install(monkeypatch)

        result = run_cli(["commit", str(source), "-o", str(tmp_path / "out.wav"),
                          "--chain", "fake-denoiser"])

        assert result.code == 0, result.stderr
        assert "Commit complete" in result.stderr
        assert f"Input:  {source}" in result.stdout
        assert "Chain:  1 steps" in result.stdout
        assert "1. fake-denoiser" in result.stdout
        assert "Delay Compensation" in result.stdout
        assert "fake-denoiser: 8 samples (confidence=100%)" in result.stdout
        assert "Total: 8 samples" in result.stdout
        assert "Time:" in result.stdout

    def test_no_compensation_skips_measurement_and_the_block(self, run_cli, source, tmp_path, monkeypatch):
        self._install(monkeypatch)
        monkeypatch.setattr(
            "audioman.core.commit.measure_chain_latency",
            lambda *a, **k: pytest.fail("latency must not be measured with --no-compensation"),
        )

        result = run_cli(
            ["commit", str(source), "-o", str(tmp_path / "out.wav"),
             "--chain", "fake-denoiser", "--no-compensation"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        assert result.payload["total_latency_samples"] == 0
        assert result.payload["latency_compensation"] == []

    def test_zero_latency_human_report_says_no_compensation_needed(
        self, run_cli, source, tmp_path, monkeypatch,
    ):
        self._install(monkeypatch, names=("fake-passthrough",), delay=0, reported=0)

        result = run_cli(["commit", str(source), "-o", str(tmp_path / "o.wav"),
                          "--chain", "fake-passthrough"])

        assert result.code == 0, result.stderr
        assert "Latency: 0 samples" in result.stdout
        assert "Delay Compensation" not in result.stdout

    def test_no_tail_trim_keeps_the_plugin_tail(self, run_cli, source, tmp_path, monkeypatch):
        self._install(monkeypatch, delay=0, tail=500, reported=0)

        out = tmp_path / "tail.wav"
        result = run_cli(
            ["commit", str(source), "-o", str(out), "--chain", "fake-denoiser", "--no-tail-trim"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        audio, _ = sf.read(str(out), always_2d=True)
        assert audio.shape[0] == 4500  # 4000 original + 500 tail

    def test_tail_trim_is_on_by_default(self, run_cli, source, tmp_path, monkeypatch):
        self._install(monkeypatch, delay=0, tail=500, reported=0)

        out = tmp_path / "trimmed.wav"
        result = run_cli(
            ["commit", str(source), "-o", str(out), "--chain", "fake-denoiser"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        audio, _ = sf.read(str(out), always_2d=True)
        assert audio.shape[0] == 4000

    def test_delay_compensation_removes_the_leading_silence(self, run_cli, tmp_path, monkeypatch):
        """The compensation shifts the signal back by the measured latency."""
        impulse_path = tmp_path / "impulse.wav"
        data = np.zeros((4000, 2), dtype=np.float32)
        data[0, :] = 1.0
        sf.write(str(impulse_path), data, SR, subtype="FLOAT")

        self._install(monkeypatch, names=("fake-delay",))

        compensated = tmp_path / "compensated.wav"
        uncompensated = tmp_path / "uncompensated.wav"
        assert run_cli(
            ["commit", str(impulse_path), "-o", str(compensated), "--chain", "fake-delay"],
        ).code == 0
        assert run_cli(
            ["commit", str(impulse_path), "-o", str(uncompensated),
             "--chain", "fake-delay", "--no-compensation"],
        ).code == 0

        comp_audio, _ = sf.read(str(compensated), always_2d=True)
        uncomp_audio, _ = sf.read(str(uncompensated), always_2d=True)
        assert int(np.argmax(np.abs(comp_audio[:, 0]))) == 0
        assert int(np.argmax(np.abs(uncomp_audio[:, 0]))) == 8

    def test_commit_failure_is_reported(self, run_cli, source, tmp_path, monkeypatch, silent_error):
        seen = silent_error("audioman.cli.commit_cmd")
        bogus = tmp_path / "bogus.wav"
        bogus.write_bytes(b"not audio")

        result = run_cli(["commit", str(bogus), "-o", str(tmp_path / "o.wav"),
                          "--chain", "fake-denoiser"])

        assert result.code == 0, result.stderr
        assert seen and "커밋 실패" in seen[0]
        assert "Commit complete" not in result.stderr

    def test_unknown_plugin_fails_the_commit(self, run_cli, source, tmp_path):
        result = run_cli(
            ["commit", str(source), "-o", str(tmp_path / "o.wav"),
             "--chain", "definitely-not-installed"],
        )
        assert result.code == 1
        assert "커밋 실패" in result.stderr
        assert "definitely-not-installed" in result.stderr

    def test_multi_step_chain_is_applied_in_order(self, run_cli, source, tmp_path, monkeypatch):
        stub, _, _ = self._install(monkeypatch, names=("step-one", "step-two"), delay=0, reported=0)

        result = run_cli(
            ["commit", str(source), "-o", str(tmp_path / "o.wav"),
             "--chain", "step-one,step-two:gain=-3"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        assert result.payload["steps"] == [
            {"plugin": "step-one", "params": {}},
            {"plugin": "step-two", "params": {"gain": -3.0}},
        ]
        # Each step is resolved twice: once for the latency measurement pass,
        # once for the processing pass.
        assert stub.get_calls == ["step-one", "step-two", "step-one", "step-two"]

    def test_human_report_lists_every_step(self, run_cli, source, tmp_path, monkeypatch):
        self._install(monkeypatch, names=("step-one", "step-two"), delay=0, reported=0)

        result = run_cli(
            ["commit", str(source), "-o", str(tmp_path / "o.wav"),
             "--chain", "step-one,step-two"],
        )

        assert result.code == 0, result.stderr
        assert "Chain:  2 steps" in result.stdout
        assert "1. step-one" in result.stdout
        assert "2. step-two" in result.stdout
