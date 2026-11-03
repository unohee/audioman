# tests/unit/cli_workflow/test_stream_cli.py
# Purpose: cover `audioman stream` (bench / triage / compare / play).
#
# `builtin:<effect>` resolves through pedalboard, so these tests need no VST3
# plugin: this host has none (AUD-1857). Plugin-dependent wiring is exercised
# with a fake registry + fake wrapper instead of a real plugin.

from __future__ import annotations

import argparse
import sys

import numpy as np
import pytest

from audioman.cli import stream
from audioman.plugins.parameter import PluginMeta

from .conftest import write_wav


def _clicky_wav(path, *, sample_rate=8000, block_size=512, edge_index=3):
    """Sine with one sample-aligned spike at an exact block boundary."""
    import soundfile as sf

    duration = 0.25
    n = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)
    mono = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    idx = block_size * edge_index
    mono[idx] = 0.95
    mono[idx + 1] = -0.95
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack([mono, mono], axis=1), sample_rate, subtype="PCM_16")
    return path


class TestBlockParsing:
    """`--blocks` accepted forms and the rejection messages."""

    def test_absent_blocks_use_the_daw_defaults(self):
        assert stream._parse_blocks(None) == list(stream.DEFAULT_BLOCK_SIZES)

    def test_string_is_split_into_ints(self):
        assert stream._parse_blocks("64, 128,512") == [64, 128, 512]

    def test_already_parsed_list_passes_through(self):
        blocks = [32, 64]
        assert stream._parse_blocks(blocks) is blocks

    def test_non_integer_token_is_reported_as_usage_error(self):
        with pytest.raises(argparse.ArgumentTypeError, match="comma-separated integers"):
            stream._parse_blocks("64,not-a-number")

    def test_non_positive_block_is_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError, match="positive integers"):
            stream._parse_blocks("64,0")

    def test_empty_string_falls_back_to_defaults(self):
        assert stream._parse_blocks("") == list(stream.DEFAULT_BLOCK_SIZES)


class TestInputResolution:
    """synthetic input keywords vs. file paths."""

    @pytest.mark.parametrize("spec", ["sine", "impulse", "two-tone"])
    def test_synthetic_keywords_return_audio_and_rate(self, spec):
        args = argparse.Namespace(input=spec, sample_rate=8000, duration=0.05)
        audio, sr = stream._load_input(args)
        assert sr == 8000
        assert audio.shape[0] == 2
        assert audio.shape[1] == pytest.approx(400, abs=2)
        assert np.isfinite(audio).all()

    def test_file_input_is_read_with_its_own_sample_rate(self, tmp_path):
        path = write_wav(tmp_path / "tone.wav", sample_rate=11025, duration=0.1)
        args = argparse.Namespace(input=str(path), sample_rate=48000, duration=0.0)
        audio, sr = stream._load_input(args)
        assert sr == 11025
        assert audio.shape[1] == pytest.approx(1102, abs=2)

    def test_missing_file_exits_nonzero_with_message(self, tmp_path, capsys):
        args = argparse.Namespace(input=str(tmp_path / "nope.wav"), sample_rate=8000, duration=0.1)
        with pytest.raises(SystemExit) as exc:
            stream._load_input(args)
        assert exc.value.code == 1
        assert "input not found" in capsys.readouterr().err


class TestProcessFnBuilding:
    def test_builtin_effect_builds_a_callable(self):
        args = argparse.Namespace(plugin="builtin:reverb", param=[])
        fn = stream._build_process_fn(args, 8000)
        block = np.zeros((2, 256), dtype=np.float32)
        out = fn(block, 8000, True)
        assert np.asarray(out).shape == block.shape

    def test_multiple_builtin_effects_chain(self):
        args = argparse.Namespace(plugin="builtin:reverb,delay", param=[])
        fn = stream._build_process_fn(args, 8000)
        block = np.zeros((2, 128), dtype=np.float32)
        assert np.asarray(fn(block, 8000, True)).shape == block.shape

    def test_unknown_builtin_effect_names_the_alternatives(self, capsys):
        args = argparse.Namespace(plugin="builtin:nope", param=[])
        with pytest.raises(SystemExit) as exc:
            stream._build_process_fn(args, 8000)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "unknown builtin effect: nope" in err
        assert "reverb" in err

    def test_unknown_plugin_points_at_the_builtin_escape_hatch(self, capsys):
        args = argparse.Namespace(plugin="definitely-not-installed", param=[])
        with pytest.raises(SystemExit) as exc:
            stream._build_process_fn(args, 8000)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "plugin not found: 'definitely-not-installed'" in err
        assert "builtin:reverb" in err

    def test_registry_hit_loads_wrapper_and_applies_params(self, monkeypatch):
        """Registry-resolved plugin path: wrapper load + set_parameters wiring."""
        calls = {}

        class _FakeWrapper:
            def __init__(self, path):
                calls["path"] = path

            def load(self):
                calls["loaded"] = True

            def set_parameters(self, params):
                calls["params"] = params

            def process(self, block, sr, reset=False):
                return block

        meta = PluginMeta(
            name="Fake EQ", short_name="fake-eq", path="/plugins/Fake.vst3", format="vst3"
        )

        class _FakeRegistry:
            def get(self, name):
                assert name == "fake-eq"
                return meta

        monkeypatch.setattr("audioman.core.registry.get_registry", lambda: _FakeRegistry())
        monkeypatch.setattr("audioman.plugins.vst3.VST3PluginWrapper", _FakeWrapper)

        args = argparse.Namespace(plugin="fake-eq", param=["drive=75"])
        fn = stream._build_process_fn(args, 8000)

        assert calls["path"] == "/plugins/Fake.vst3"
        assert calls["loaded"] is True
        assert calls["params"] == {"drive": 75.0}
        block = np.ones((2, 64), dtype=np.float32)
        assert fn(block, 8000, False).shape == block.shape

    def test_registry_hit_without_params_skips_set_parameters(self, monkeypatch):
        calls = {}

        class _FakeWrapper:
            def __init__(self, path):
                calls["path"] = path

            def load(self):
                calls["loaded"] = True

            def set_parameters(self, params):  # pragma: no cover - must not run
                calls["params"] = params

        meta = PluginMeta(name="Fake", short_name="fake", path="/p/Fake.vst3", format="vst3")

        class _FakeRegistry:
            def get(self, name):
                return meta

        monkeypatch.setattr("audioman.core.registry.get_registry", lambda: _FakeRegistry())
        monkeypatch.setattr("audioman.plugins.vst3.VST3PluginWrapper", _FakeWrapper)

        stream._build_process_fn(argparse.Namespace(plugin="fake", param=[]), 8000)
        assert "params" not in calls


class TestNoSubcommand:
    def test_bare_stream_exits_with_usage_directed_message(self, run_cli):
        result = run_cli(["stream"])
        assert result.code == 1
        assert "stream requires a subcommand: bench | triage | compare | play" in result.stderr


class TestBench:
    def test_json_payload_reports_every_block_size(self, run_cli, tmp_path):
        result = run_cli(
            ["stream", "bench", "sine", "-p", "builtin:gain", "--duration", "0.2",
             "--sample-rate", "8000", "--blocks", "64,512"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "stream bench"
        assert payload["plugin"] == "builtin:gain"
        assert payload["sample_rate"] == 8000
        assert [r["block_size"] for r in payload["reports"]] == [64, 512]
        for report in payload["reports"]:
            assert report["blocks"] > 0
            assert report["rt_factor_p50"] > 0
            assert report["est_max_tracks"] >= 0

    def test_human_report_lists_blocks_and_prints_success(self, run_cli):
        result = run_cli(
            ["stream", "bench", "sine", "-p", "builtin:gain", "--duration", "0.1",
             "--sample-rate", "8000", "--blocks", "64,128"],
        )
        assert result.code == 0, result.stderr
        assert "# RT bench — builtin:gain @ 8000Hz" in result.stdout
        assert "block\tdeadline_ms\tproc_ms_avg" in result.stdout
        assert "64\t" in result.stdout and "128\t" in result.stdout
        assert "no xruns; smallest safe est_tracks" in result.stderr

    def test_missing_blocks_option_uses_default_sweep(self, run_cli):
        result = run_cli(
            ["stream", "bench", "sine", "-p", "builtin:gain", "--duration", "0.05",
             "--sample-rate", "8000"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert [r["block_size"] for r in result.payload["reports"]] == stream.DEFAULT_BLOCK_SIZES

    def test_slow_plugin_reports_xruns(self, run_cli, monkeypatch):
        """A process fn that blows the block deadline must be reported as an xrun."""
        import time as _time

        def slow_fn(block, sr, reset):
            _time.sleep(0.02)
            return np.zeros_like(block)

        monkeypatch.setattr(stream, "_build_process_fn", lambda args, sr: slow_fn)
        result = run_cli(
            ["stream", "bench", "sine", "-p", "builtin:gain", "--duration", "0.1",
             "--sample-rate", "8000", "--blocks", "64"],
        )
        assert result.code == 0, result.stderr
        assert "xruns at block=64" in result.stderr
        assert "blocks missed deadline" in result.stderr


class TestTriage:
    def test_json_payload_carries_stream_block_and_summaries(self, run_cli, tmp_path):
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.3)
        result = run_cli(
            ["stream", "triage", str(path), "-p", "builtin:gain", "-b", "512"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "stream triage"
        assert payload["$schema"] == "audioman://schema/stream.v1.json"
        assert payload["stream"]["block_size"] == 512
        assert payload["stream"]["reset_first"] is True
        assert payload["stream"]["offline_summary"]["mode"] == "offline"
        assert payload["stream"]["streamed_summary"]["mode"] == "streamed"
        assert payload["stream"]["reset_per_block"] is False
        assert "summary" in payload

    def test_block_aligned_click_is_reported_with_table_and_hint(self, run_cli, tmp_path):
        path = _clicky_wav(tmp_path / "clicky.wav", block_size=512)
        result = run_cli(["stream", "triage", str(path), "-p", "builtin:gain", "-b", "512"])
        assert result.code == 0, result.stderr
        assert "CLICK_DENSITY" in result.stdout
        assert "block_aligned" in result.stdout
        assert "critical finding(s)" in result.stderr

    def test_clean_input_reports_no_findings(self, run_cli, tmp_path):
        path = write_wav(tmp_path / "clean.wav", sample_rate=8000, duration=0.2)
        result = run_cli(["stream", "triage", str(path), "-p", "builtin:gain", "-b", "512"])
        assert result.code == 0, result.stderr
        assert "clean — no clicks/discontinuities at block_size=512" in result.stderr

    def test_reset_flags_are_propagated_to_the_payload(self, run_cli, tmp_path):
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.2)
        result = run_cli(
            ["stream", "triage", str(path), "-p", "builtin:gain", "-b", "256",
             "--reset-per-block", "--no-reset-first"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["stream"]["reset_per_block"] is True
        assert result.payload["stream"]["reset_first"] is False

    def test_output_flag_writes_the_streamed_audio(self, run_cli, tmp_path):
        import soundfile as sf

        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.2)
        out = tmp_path / "streamed.wav"
        result = run_cli(
            ["stream", "triage", str(path), "-p", "builtin:gain", "-b", "256",
             "-o", str(out)],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert out.exists()
        audio, sr = sf.read(str(out), always_2d=True)
        assert sr == 8000
        # Block streaming is lossless: exactly the 1600 source frames, no padding.
        assert audio.shape == (1600, 2)
        # The payload must agree with the artefact on disk.
        assert result.payload["stream"]["streamed_summary"]["audio_seconds"] == 0.2

    def test_null_test_finding_renders_the_max_diff_column(self, run_cli, tmp_path, monkeypatch):
        """A streamed-vs-offline mismatch renders the null-test row (max_diff_db column).

        The same plugin instance processes offline first and then every streamed
        block, so the streamed output drifts away from the offline reference and
        the null test must report SAMPLE_DROPOUT with a level_db figure.
        """
        state = {"calls": 0}

        def drifting_fn(block, sr, reset):
            state["calls"] += 1
            out = np.array(block, dtype=np.float32)
            if state["calls"] > 1:
                out = out + 0.5
            return out

        monkeypatch.setattr(stream, "_build_process_fn", lambda args, sr: drifting_fn)

        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.2)
        result = run_cli(["stream", "triage", str(path), "-p", "builtin:gain", "-b", "512"])
        assert result.code == 0, result.stderr
        assert "SAMPLE_DROPOUT" in result.stdout
        assert "max_diff_db" not in result.stdout  # that key never leaks into the table
        assert "level_db" in result.stdout


class TestCompare:
    def test_json_payload_compares_each_block_size_with_offline(self, run_cli, tmp_path):
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.2)
        result = run_cli(
            ["stream", "compare", str(path), "-p", "builtin:gain", "--blocks", "64,128,256"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "stream compare"
        assert set(payload["vs_offline_db"]) == {"64", "128", "256"}
        assert set(payload["cross_block_db"]) == {"64_vs_128", "128_vs_256"}

    def test_deterministic_plugin_matches_offline(self, run_cli, tmp_path):
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.2)
        result = run_cli(
            ["stream", "compare", str(path), "-p", "builtin:gain", "--blocks", "64,256"],
        )
        assert result.code == 0, result.stderr
        assert "# streamed vs offline — builtin:gain" in result.stdout
        assert "# cross block-size diff" in result.stdout
        assert "all block sizes match offline render" in result.stderr

    def test_block_size_sensitive_plugin_is_flagged(self, run_cli, tmp_path, monkeypatch):
        """A plugin whose output depends on the block length must trip the warning."""
        def block_sensitive_fn(block, sr, reset):
            # Gain depends on how many frames the host handed over.
            return np.array(block, dtype=np.float32) * (1.0 + 0.5 * (block.shape[1] > 64))

        monkeypatch.setattr(stream, "_build_process_fn", lambda args, sr: block_sensitive_fn)
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.2)
        result = run_cli(
            ["stream", "compare", str(path), "-p", "builtin:gain", "--blocks", "64,256"],
        )
        assert result.code == 0, result.stderr
        assert "block streaming diverges from offline" in result.stderr
        assert "block-size sensitive" in result.stderr


class _CallbackStop(Exception):
    pass


class _FakeOutputStream:
    """Minimal sounddevice.OutputStream stand-in that drives the callback.

    Plays the callback until it raises CallbackStop (the command's own signal
    that the source ran out), with a hard iteration cap so a bug can never hang
    the suite.
    """

    def __init__(self, *, samplerate, channels, dtype, blocksize, callback,
                 finished_callback, status="", raise_interrupt=False,
                 max_blocks=64):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.callback = callback
        self.finished_callback = finished_callback
        self.status = status
        self.raise_interrupt = raise_interrupt
        self.max_blocks = max_blocks
        self.calls = 0
        self.captured = []

    def __enter__(self):
        if self.raise_interrupt:
            raise KeyboardInterrupt
        for _ in range(self.max_blocks):
            outdata = np.zeros((self.blocksize, self.channels), dtype=np.float32)
            try:
                self.callback(outdata, self.blocksize, None, self.status)
            except _CallbackStop:
                break
            self.captured.append(outdata.copy())
            self.calls += 1
        self.finished_callback()
        return self

    def __exit__(self, *exc):
        return False


class _FakeSoundDevice:
    CallbackStop = _CallbackStop

    def __init__(self, **defaults):
        self.defaults = defaults

    def OutputStream(self, **kwargs):
        merged = dict(self.defaults)
        sink = merged.pop("sink")
        merged.update(kwargs)
        stream = _FakeOutputStream(**merged)
        sink.append(stream)
        return stream


class TestPlay:
    """`stream play` drives an audio device; sounddevice is faked, no device opens."""

    def _install_fake_sd(self, monkeypatch, **kwargs):
        sink = []
        kwargs["sink"] = sink
        monkeypatch.setitem(sys.modules, "sounddevice", _FakeSoundDevice(**kwargs))
        return sink

    def test_playback_consumes_the_whole_file_without_underflows(self, run_cli, tmp_path, monkeypatch):
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.1)
        sink = self._install_fake_sd(monkeypatch)
        result = run_cli(["stream", "play", str(path), "-p", "builtin:gain", "-b", "256"])
        assert result.code == 0, result.stderr
        assert "playback complete, no underflows" in result.stderr
        stream_obj = sink[0]
        assert stream_obj.samplerate == 8000
        assert stream_obj.blocksize == 256
        # 800 frames / 256-frame blocks: callback ran until the tail block.
        assert stream_obj.calls == 4
        # Last block is a partial one: the remainder must be zero-filled.
        last = stream_obj.captured[-1]
        assert last.shape == (256, 2)
        assert np.all(last[32:] == 0.0)

    def test_mono_process_output_is_duplicated_to_both_channels(self, run_cli, tmp_path, monkeypatch):
        def mono_fn(block, sr, reset):
            return block.mean(axis=0, keepdims=True)

        monkeypatch.setattr(stream, "_build_process_fn", lambda args, sr: mono_fn)
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.05)
        sink = self._install_fake_sd(monkeypatch)
        result = run_cli(["stream", "play", str(path), "-p", "builtin:gain", "-b", "128"])
        assert result.code == 0, result.stderr
        block = sink[0].captured[0]
        assert np.array_equal(block[:, 0], block[:, 1])

    def test_portaudio_underflow_status_is_counted(self, run_cli, tmp_path, monkeypatch):
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.05)
        self._install_fake_sd(monkeypatch, status="input overflow")
        result = run_cli(["stream", "play", str(path), "-p", "builtin:gain", "-b", "128"])
        assert result.code == 0, result.stderr
        assert "PortAudio underflow(s)" in result.stderr
        assert "real xruns at block=128" in result.stderr

    def test_reset_per_block_flag_resets_on_every_block(self, run_cli, tmp_path, monkeypatch):
        seen = []

        def recording_fn(block, sr, reset):
            seen.append(reset)
            return block

        monkeypatch.setattr(stream, "_build_process_fn", lambda args, sr: recording_fn)
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.1)
        self._install_fake_sd(monkeypatch)
        result = run_cli(
            ["stream", "play", str(path), "-p", "builtin:gain", "-b", "128", "--reset-per-block"],
        )
        assert result.code == 0, result.stderr
        assert seen == [True] * 7  # 800 frames / 128 = 7 blocks

    def test_default_resets_only_the_first_block(self, run_cli, tmp_path, monkeypatch):
        """A real DAW resets once at playback start, then keeps plugin state."""
        seen = []

        def recording_fn(block, sr, reset):
            seen.append(reset)
            return block

        monkeypatch.setattr(stream, "_build_process_fn", lambda args, sr: recording_fn)
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.1)
        self._install_fake_sd(monkeypatch)
        result = run_cli(["stream", "play", str(path), "-p", "builtin:gain", "-b", "128"])
        assert result.code == 0, result.stderr
        assert seen == [True] + [False] * 6

    def test_processed_audio_reaches_the_device_buffer(self, run_cli, tmp_path, monkeypatch):
        """The callback must write the plugin output, not silence."""
        import soundfile as sf

        def half_gain(block, sr, reset):
            return np.array(block, dtype=np.float32) * 0.5

        monkeypatch.setattr(stream, "_build_process_fn", lambda args, sr: half_gain)
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.1, amplitude=0.8)
        sink = self._install_fake_sd(monkeypatch)
        assert run_cli(["stream", "play", str(path), "-p", "builtin:gain", "-b", "256"]).code == 0

        source = sf.read(str(path), dtype="float32", always_2d=True)[0].T
        first = sink[0].captured[0]
        assert np.allclose(first[:, 0], source[0, :256] * 0.5, atol=1e-6)
        assert np.allclose(first[:, 1], source[1, :256] * 0.5, atol=1e-6)

    def test_keyboard_interrupt_stops_playback_quietly(self, run_cli, tmp_path, monkeypatch):
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.05)
        self._install_fake_sd(monkeypatch, raise_interrupt=True)
        result = run_cli(["stream", "play", str(path), "-p", "builtin:gain", "-b", "128"])
        assert result.code == 0, result.stderr
        assert "Traceback" not in result.stderr
        assert "playback complete" in result.stderr

    def test_play_emits_no_json_payload_even_with_the_global_flag(self, run_cli, tmp_path, monkeypatch):
        """`play` is audible output, not a document: no payload is ever printed."""
        path = write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.05)
        self._install_fake_sd(monkeypatch)
        result = run_cli(
            ["stream", "play", str(path), "-p", "builtin:gain", "-b", "128"], json_mode=True
        )
        assert result.code == 0, result.stderr
        assert result.stdout.strip() == ""
        assert "playback complete" in result.stderr
