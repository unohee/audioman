# Created: 2026-09-28
# Purpose: cli/fx.py coverage — the 11 effects + batch/directory mode (AUD-1851).
#
# fx is pure built-in DSP with no plugins, so it runs as-is on this host. For each
# effect the test asserts on (a) the JSON payload and (b) the sample count/amplitude
# of the output file actually written to disk. It checks that the effect really took
# hold, not just the values.

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from audioman.cli import fx
from harness import (
    read_wav,
    run_command,
    write_pattern_wav,
    write_silence_wav,
    write_wav,
)


def _json(result) -> dict:
    return json.loads(result.out)


def _write_silence_tone_silence(path, *, sample_rate=44100, seg_sec=0.2):
    """A silence-tone-silence 3-segment WAV. Exercises both pad paths of trim-silence."""
    n = int(sample_rate * seg_sec)
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(n, dtype=np.float32) / sample_rate)
    mono = np.concatenate([np.zeros(n, dtype=np.float32), tone,
                           np.zeros(n, dtype=np.float32)]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack([mono, mono], axis=1), sample_rate, subtype="PCM_16")
    return path


def _run_effect(tmp_path, effect, *extra, source="tone.wav", sr=44100, **write_kwargs):
    """Run one fx invocation on a fresh tone; return (result, output path)."""
    src = write_wav(tmp_path / source, sample_rate=sr, duration=1.0, **write_kwargs)
    out = tmp_path / "out" / f"{effect}.wav"
    result = run_command(["--json", "fx", str(src), effect, "-o", str(out), *extra])
    assert result.code == 0, result.err
    return result, out


class TestFadeIn:
    def test_samples_path_ramps_up_and_payload_matches(self, tmp_path):
        result, out = _run_effect(tmp_path, "fade-in", "--samples", "4410")
        payload = _json(result)
        assert payload["command"] == "fx"
        assert payload["effect"] == "fade-in"
        assert payload["output"] == str(out)
        assert payload["output_stats"]["frames"] == 44100
        assert payload["time_seconds"] >= 0

        audio, sr = read_wav(out)
        assert sr == 44100
        assert np.abs(audio[0, 0]) == pytest.approx(0.0, abs=1e-6)
        # A 440Hz sine crosses zero sample by sample, so compare a window envelope.
        faded = float(np.abs(audio[0, :100]).max())
        mid = float(np.abs(audio[0, 2000:3000]).max())
        after = float(np.abs(audio[0, 5000:6000]).max())
        assert faded < 0.1
        assert faded < mid < after
        assert after == pytest.approx(0.5, abs=0.01)  # amplitude back to the original

    def test_duration_path_converts_seconds_to_samples(self, tmp_path):
        result, out = _run_effect(tmp_path, "fade-in", "--duration", "0.1")
        assert _json(result)["effect"] == "fade-in"
        audio, _sr = read_wav(out)
        assert abs(float(audio[0, 0])) < abs(float(audio[0, 4409]))  # 0.1s = 4410 samples

    def test_default_length_is_a_tenth_of_a_second(self, tmp_path):
        src = write_wav(tmp_path / "t.wav")
        out = tmp_path / "d.wav"
        result = run_command(["--json", "fx", str(src), "fade-in", "-o", str(out)])
        assert result.code == 0
        audio, _ = read_wav(out)
        # Default length is sr//10 = 4410 samples: attenuated before, original after.
        assert float(np.abs(audio[0, :100]).max()) < 0.1
        assert float(np.abs(audio[0, 4410:]).max()) == pytest.approx(0.5, abs=0.01)

    def test_curve_choice_changes_the_ramp_shape(self, tmp_path):
        linear_result, linear_out = _run_effect(tmp_path, "fade-in", "--samples", "4410")
        assert _json(linear_result)["effect"] == "fade-in"
        cosine_src = write_wav(tmp_path / "c.wav")
        cosine_out = tmp_path / "cosine.wav"
        result = run_command([
            "--json", "fx", str(cosine_src), "fade-in",
            "--samples", "4410", "--curve", "equal_power", "-o", str(cosine_out),
        ])
        assert result.code == 0
        linear, _ = read_wav(linear_out)
        equal_power, _ = read_wav(cosine_out)
        assert not np.allclose(linear[0, :4410], equal_power[0, :4410])

    def test_rejects_unknown_curve(self, tmp_path):
        src = write_wav(tmp_path / "t.wav")
        result = run_command(["--json", "fx", str(src), "fade-in", "--curve", "nope",
                                  "-o", str(tmp_path / "x.wav")])
        assert result.code == 2       # argparse rejects the choices violation
        assert "invalid choice" in result.err
        assert not (tmp_path / "x.wav").exists()


class TestFadeOut:
    def test_samples_path_ramps_down(self, tmp_path):
        result, out = _run_effect(tmp_path, "fade-out", "--samples", "4410")
        assert _json(result)["effect"] == "fade-out"
        audio, _ = read_wav(out)
        assert abs(float(audio[0, -1])) < 1e-6
        assert np.abs(audio[0, 30000]) > 0.4

    def test_duration_path_converts_seconds(self, tmp_path):
        _result, out = _run_effect(tmp_path, "fade-out", "--duration", "0.2")
        audio, _ = read_wav(out)
        assert abs(float(audio[0, -1])) < 1e-6

    def test_default_length_applies(self, tmp_path):
        src = write_wav(tmp_path / "t.wav")
        out = tmp_path / "d.wav"
        assert run_command(["--json", "fx", str(src), "fade-out", "-o", str(out)]).code == 0
        audio, _ = read_wav(out)
        assert abs(float(audio[0, -1])) < 1e-6


class TestPad:
    def test_head_and_tail_milliseconds_extend_length(self, tmp_path):
        result, out = _run_effect(
            tmp_path, "pad", "--head-ms", "100", "--tail-ms", "50"
        )
        payload = _json(result)
        assert payload["effect"] == "pad"
        assert payload["input_stats"]["frames"] == 44100
        # 4410 head + 2205 tail + 44100 orig
        assert payload["output_stats"]["frames"] == 44100 + 4410 + 2205
        audio, _ = read_wav(out)
        assert np.allclose(audio[:, :4410], 0.0)
        assert np.allclose(audio[:, -2205:], 0.0)

    def test_seconds_override_milliseconds(self, tmp_path):
        result, out = _run_effect(
            tmp_path, "pad", "--head-ms", "999", "--head-sec", "0.1",
            "--tail-ms", "999", "--tail-sec", "0.05",
        )
        payload = _json(result)
        assert payload["output_stats"]["frames"] == 44100 + 4410 + 2205

    def test_defaults_pad_nothing(self, tmp_path):
        result, _out = _run_effect(tmp_path, "pad")
        payload = _json(result)
        assert payload["output_stats"]["frames"] == payload["input_stats"]["frames"]


class TestRemoveDc:
    def test_removes_constant_offset(self, tmp_path):
        result, out = _run_effect(tmp_path, "remove-dc", offset=0.3)
        assert _json(result)["effect"] == "remove-dc"
        audio, _ = read_wav(out)
        assert abs(float(audio.mean())) < 1e-3
        # The input carries DC, so its RMS drops.
        assert _json(result)["output_stats"]["rms"] < _json(result)["input_stats"]["rms"]


class TestTrim:
    def test_start_and_end_samples(self, tmp_path):
        result, out = _run_effect(tmp_path, "trim", "--start", "1000", "--end", "5000")
        payload = _json(result)
        assert payload["effect"] == "trim"
        assert payload["output_stats"]["frames"] == 4000
        audio, _ = read_wav(out)
        assert audio.shape[1] == 4000

    def test_start_sec_and_end_sec_override_samples(self, tmp_path):
        result, _out = _run_effect(
            tmp_path, "trim", "--start", "99999", "--end", "99999",
            "--start-sec", "0.1", "--end-sec", "0.2",
        )
        assert _json(result)["output_stats"]["frames"] == 4410

    def test_defaults_keep_whole_file(self, tmp_path):
        result, _out = _run_effect(tmp_path, "trim")
        payload = _json(result)
        assert payload["output_stats"]["frames"] == payload["input_stats"]["frames"]


class TestCutRegion:
    def test_sample_arguments_remove_the_middle(self, tmp_path):
        result, out = _run_effect(tmp_path, "cut-region", "--start", "1000", "--end", "5000")
        payload = _json(result)
        assert payload["effect"] == "cut-region"
        assert payload["output_stats"]["frames"] == 44100 - 4000
        audio, _ = read_wav(out)
        assert audio.shape[1] == 44100 - 4000

    def test_seconds_arguments_and_crossfade_ms(self, tmp_path):
        result, _out = _run_effect(
            tmp_path, "cut-region", "--start-sec", "0.1", "--end-sec", "0.2",
            "--crossfade-ms", "5",
        )
        # The crossfade consumes an extra left/right tail (cf=220 samples).
        assert _json(result)["output_stats"]["frames"] == 44100 - 4410 - 220

    def test_crossfade_samples_argument(self, tmp_path):
        result, _out = _run_effect(
            tmp_path, "cut-region", "--start", "100", "--end", "200", "--crossfade", "50",
        )
        assert _json(result)["output_stats"]["frames"] == 44100 - 100 - 50


class TestSplice:
    def test_insert_mode_lengthens_output_by_clip_length(self, tmp_path):
        src = write_wav(tmp_path / "base.wav", duration=1.0)
        clip = write_wav(tmp_path / "clip.wav", duration=0.25, freq=880.0)
        out = tmp_path / "spliced.wav"
        result = run_command([
            "--json", "fx", str(src), "splice", "--clip", str(clip),
            "--position", "1000", "-o", str(out),
        ])
        assert result.code == 0
        payload = _json(result)
        assert payload["effect"] == "splice"
        assert payload["output_stats"]["frames"] == 44100 + 11025

    def test_position_sec_and_crossfade_ms(self, tmp_path):
        src = write_wav(tmp_path / "base.wav")
        clip = write_wav(tmp_path / "clip.wav", duration=0.25)
        out = tmp_path / "spliced.wav"
        result = run_command([
            "--json", "fx", str(src), "splice", "--clip", str(clip),
            "--position-sec", "0.5", "--crossfade-ms", "5", "-o", str(out),
        ])
        assert result.code == 0
        # insert mode: base + clip, and the crossfade overlaps on both the left and right.
        assert _json(result)["output_stats"]["frames"] == 44100 + 11025 - 2 * 220

    @pytest.mark.parametrize("mode", ["overwrite", "mix"])
    def test_length_preserving_modes(self, tmp_path, mode):
        src = write_wav(tmp_path / "base.wav")
        clip = write_wav(tmp_path / "clip.wav", duration=0.25)
        out = tmp_path / f"{mode}.wav"
        result = run_command([
            "--json", "fx", str(src), "splice", "--clip", str(clip),
            "--position", "0", "--mode", mode, "-o", str(out),
        ])
        assert result.code == 0
        assert _json(result)["output_stats"]["frames"] == 44100

    def test_mix_mode_changes_samples_in_the_window(self, tmp_path):
        src = write_wav(tmp_path / "base.wav")
        clip = write_wav(tmp_path / "clip.wav", duration=0.25, amp=0.25)
        out = tmp_path / "mix.wav"
        assert run_command([
            "--json", "fx", str(src), "splice", "--clip", str(clip),
            "--position", "0", "--mode", "mix", "-o", str(out),
        ]).code == 0
        base, _ = read_wav(src)
        mixed, _ = read_wav(out)
        assert not np.allclose(base[:, :11025], mixed[:, :11025])
        assert np.allclose(base[:, 11025:], mixed[:, 11025:])

    def test_mono_clip_is_broadcast_to_stereo_input(self, tmp_path):
        src = write_wav(tmp_path / "base.wav", channels=2)
        clip = write_wav(tmp_path / "clip.wav", duration=0.1, channels=1)
        out = tmp_path / "bcast.wav"
        result = run_command([
            "--json", "fx", str(src), "splice", "--clip", str(clip),
            "--position", "0", "-o", str(out),
        ])
        assert result.code == 0
        audio, _ = read_wav(out)
        assert audio.shape[0] == 2
        assert np.allclose(audio[0], audio[1])

    def test_stereo_clip_against_mono_input_is_refused_by_dsp(self, tmp_path):
        """A mono input with a stereo clip is downmixed by the CLI but refused by dsp.

        `read_audio` reads even a mono file as a (1, n) 2-D array, while the downmix
        result is 1-D, so the shapes disagree. Pin this crash as the contract (an
        explicit ValueError, not a silent misbehaviour).
        """
        src = write_wav(tmp_path / "base.wav", channels=1)
        clip = write_wav(tmp_path / "clip.wav", duration=0.1, channels=2)
        with pytest.raises(ValueError):
            run_command([
                "--json", "fx", str(src), "splice", "--clip", str(clip),
                "--position", "0", "-o", str(tmp_path / "down.wav"),
            ])

    def test_unsupported_channel_conversion_raises(self, tmp_path):
        src = write_wav(tmp_path / "base.wav", channels=2)
        clip = write_wav(tmp_path / "clip.wav", duration=0.1, channels=3)
        with pytest.raises(ValueError, match="Cannot convert channels"):
            run_command([
                "--json", "fx", str(src), "splice", "--clip", str(clip),
                "--position", "0", "-o", str(tmp_path / "x.wav"),
            ])

    def test_sample_rate_mismatch_raises(self, tmp_path):
        src = write_wav(tmp_path / "base.wav", sample_rate=44100)
        clip = write_wav(tmp_path / "clip.wav", sample_rate=22050, duration=0.1)
        with pytest.raises(ValueError, match="Sample rate mismatch"):
            run_command([
                "--json", "fx", str(src), "splice", "--clip", str(clip),
                "--position", "0", "-o", str(tmp_path / "x.wav"),
            ])

    def test_directory_input_is_refused_with_exit_1(self, tmp_path):
        write_wav(tmp_path / "in" / "a.wav")
        result = run_command([
            "--json", "fx", str(tmp_path / "in"), "splice",
            "--clip", str(tmp_path / "in" / "a.wav"), "-o", str(tmp_path / "out"),
        ])
        assert result.code == 1
        assert "splice can only be applied to a single file" in result.err


class TestTrimSilence:
    def test_removes_leading_and_trailing_silence(self, tmp_path):
        src = write_pattern_wav(tmp_path / "pattern.wav", silence_segments=1)
        out = tmp_path / "trimmed.wav"
        result = run_command([
            "--json", "fx", str(src), "trim-silence", "--threshold", "-40", "-o", str(out),
        ])
        assert result.code == 0
        payload = _json(result)
        assert payload["effect"] == "trim-silence"
        assert payload["output_stats"]["frames"] < payload["input_stats"]["frames"]

    def test_pad_samples_keep_boundary_margin(self, tmp_path):
        # Both ends must be silent for pad to be added on each side.
        src = _write_silence_tone_silence(tmp_path / "bracketed.wav")
        unpadded = tmp_path / "unpadded.wav"
        padded = tmp_path / "padded.wav"
        first = run_command(["--json", "fx", str(src), "trim-silence", "-o", str(unpadded)])
        second = run_command([
            "--json", "fx", str(src), "trim-silence", "--pad", "500", "-o", str(padded),
        ])
        assert first.code == 0 and second.code == 0
        # pad leaves pad_samples at each of the head and tail boundaries.
        assert _json(second)["output_stats"]["frames"] == _json(first)["output_stats"]["frames"] + 1000

    def test_pad_is_clamped_at_the_file_boundary(self, tmp_path):
        # With only leading silence, pad is clipped at the start of the file (0).
        src = write_pattern_wav(tmp_path / "lead-only.wav", silence_segments=1)
        unpadded = tmp_path / "u.wav"
        padded = tmp_path / "p.wav"
        first = run_command(["--json", "fx", str(src), "trim-silence", "-o", str(unpadded)])
        second = run_command([
            "--json", "fx", str(src), "trim-silence", "--pad", "500", "-o", str(padded),
        ])
        assert _json(second)["output_stats"]["frames"] - _json(first)["output_stats"]["frames"] == 500

    def test_all_silent_input_is_returned_unchanged(self, tmp_path):
        src = write_silence_wav(tmp_path / "silent.wav")
        out = tmp_path / "still-silent.wav"
        result = run_command(["--json", "fx", str(src), "trim-silence", "-o", str(out)])
        assert result.code == 0
        payload = _json(result)
        assert payload["output_stats"]["frames"] == payload["input_stats"]["frames"]


class TestNormalize:
    def test_default_peak_targets_minus_one_db(self, tmp_path):
        src = write_wav(tmp_path / "quiet.wav", amp=0.1)
        out = tmp_path / "normalized.wav"
        result = run_command(["--json", "fx", str(src), "normalize", "-o", str(out)])
        assert result.code == 0
        payload = _json(result)
        assert payload["effect"] == "normalize"
        expected = 10 ** (-1.0 / 20.0)
        assert payload["output_stats"]["peak"] == pytest.approx(expected, abs=1e-4)

    def test_explicit_peak_db(self, tmp_path):
        src = write_wav(tmp_path / "quiet.wav", amp=0.1)
        out = tmp_path / "peak.wav"
        result = run_command(["--json", "fx", str(src), "normalize", "--peak", "-6", "-o", str(out)])
        assert result.code == 0
        assert _json(result)["output_stats"]["peak"] == pytest.approx(10 ** (-6 / 20), abs=1e-4)

    def test_target_rms_db(self, tmp_path):
        src = write_wav(tmp_path / "quiet.wav", amp=0.1)
        out = tmp_path / "rms.wav"
        result = run_command([
            "--json", "fx", str(src), "normalize", "--target-rms", "-20", "-o", str(out),
        ])
        assert result.code == 0
        assert _json(result)["output_stats"]["rms"] == pytest.approx(10 ** (-20 / 20), abs=1e-3)


class TestGate:
    def test_low_level_passage_is_muted_and_tone_preserved(self, tmp_path):
        """The gate must attenuate only the below-threshold part (not kill the whole file)."""
        quiet_len, tone_len = 22050, 22050
        src = tmp_path / "gate-fixture.wav"
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(tone_len, dtype=np.float32) / 44100)
        quiet = np.full(quiet_len, 0.0005, dtype=np.float32)  # -66 dB, below the threshold
        mono = np.concatenate([quiet, tone]).astype(np.float32)
        sf.write(str(src), np.stack([mono, mono], axis=1), 44100, subtype="PCM_16")

        out = tmp_path / "gated.wav"
        result = run_command([
            "--json", "fx", str(src), "gate", "--threshold", "-40",
            "--attack", "0.01", "--release", "0.05", "-o", str(out),
        ])
        assert result.code == 0
        assert _json(result)["effect"] == "gate"

        gated, _sr = read_wav(out)
        # The low-level part is attenuated and the tone part survives unchanged.
        assert float(np.abs(gated[:, :20000]).max()) < 0.001
        assert float(np.abs(gated[:, 25000:]).max()) == pytest.approx(0.5, abs=0.01)
        assert gated.shape[1] == quiet_len + tone_len


class TestGain:
    def test_positive_db_raises_peak(self, tmp_path):
        src = write_wav(tmp_path / "t.wav", amp=0.5)
        out = tmp_path / "loud.wav"
        result = run_command(["--json", "fx", str(src), "gain", "--db", "6", "-o", str(out)])
        assert result.code == 0
        payload = _json(result)
        assert payload["effect"] == "gain"
        assert payload["output_stats"]["peak"] == pytest.approx(0.5 * 10 ** (6 / 20), abs=1e-3)

    def test_negative_db_lowers_peak(self, tmp_path):
        src = write_wav(tmp_path / "t.wav", amp=0.5)
        out = tmp_path / "quiet.wav"
        result = run_command(["--json", "fx", str(src), "gain", "--db", "-6", "-o", str(out)])
        assert result.code == 0
        assert _json(result)["output_stats"]["peak"] == pytest.approx(0.5 * 10 ** (-6 / 20), abs=1e-3)

    def test_db_is_required(self, tmp_path):
        src = write_wav(tmp_path / "t.wav")
        result = run_command(["--json", "fx", str(src), "gain", "-o", str(tmp_path / "x.wav")])
        assert result.code == 2
        assert "required" in result.err


class TestHumanOutput:
    def test_plain_single_reports_stats_and_writes_file(self, tmp_path):
        src = write_wav(tmp_path / "t.wav")
        out = tmp_path / "gained.wav"
        result = run_command(["--plain", "fx", str(src), "gain", "--db", "3", "-o", str(out)])
        assert result.code == 0
        assert out.exists()
        assert "gain complete" in result.err
        # print_success goes to the stderr console in both plain and rich modes.
        assert "gain complete" not in result.out
        assert "RMS:" in result.out

    def test_rich_single_reports_stats(self, tmp_path):
        src = write_wav(tmp_path / "t.wav")
        out = tmp_path / "gained.wav"
        result = run_command(["fx", str(src), "gain", "--db", "3", "-o", str(out)])
        assert result.code == 0
        assert "gain complete" in result.err
        assert "RMS:" in result.out
        assert "Peak:" in result.out
        assert "Time:" in result.out

    def test_missing_input_exits_1(self, tmp_path):
        result = run_command([
            "--json", "fx", str(tmp_path / "ghost.wav"), "gain", "--db", "3",
            "-o", str(tmp_path / "out.wav"),
        ])
        assert result.code == 1
        assert "File not found" in result.err
        assert not (tmp_path / "out.wav").exists()


class TestNoEffect:
    def test_missing_effect_name_is_rejected_by_the_parser(self, tmp_path):
        src = write_wav(tmp_path / "t.wav")
        result = run_command(["--json", "fx", str(src), "-o", str(tmp_path / "o.wav")])
        assert result.code == 2
        assert "fade-in" in result.err and "gain" in result.err

    def test_unknown_effect_in_namespace_raises_from_apply_effect(self, tmp_path):
        """_apply_effect refuses an effect name that bypassed argparse choices.

        The parser enforces subcommand names, but a path that calls run(args) directly
        can pass an arbitrary string. Check that it fails explicitly instead of silently
        returning the input unchanged.
        """
        import argparse

        src = write_wav(tmp_path / "t.wav")
        audio, sr = read_wav(src)
        args = argparse.Namespace(effect="nonexistent-effect")
        with pytest.raises(ValueError, match="Unknown effect: nonexistent-effect"):
            fx._apply_effect(audio, sr, args)

    def test_run_without_effect_reports_usage_and_exits_1(self, tmp_path, capsys):
        """Calling run() directly with an effect-less Namespace prints usage and exits.

        argparse requires a subcommand, so this branch is only reachable through a
        hand-built Namespace. (the contract for other entry points calling run(args))
        """
        import argparse

        src = write_wav(tmp_path / "t.wav")
        args = argparse.Namespace(
            input=str(src), effect=None, output=str(tmp_path / "o.wav"), json=True,
        )
        with pytest.raises(SystemExit) as excinfo:
            fx.run(args)
        assert excinfo.value.code == 1
        assert "Specify an effect" in capsys.readouterr().err


class TestBatch:
    def test_plain_directory_run_processes_every_file(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        write_wav(in_dir / "b.wav", freq=880.0)
        out_dir = tmp_path / "out"

        result = run_command(["--plain", "fx", str(in_dir), "gain", "--db", "-3",
                                  "-o", str(out_dir)])
        assert result.code == 0
        assert (out_dir / "a.wav").exists()
        assert (out_dir / "b.wav").exists()
        assert "Batch complete: 2 succeeded, 0 failed / 2 total" in result.err

    def test_json_directory_run_emits_one_payload_per_file(self, tmp_path):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        write_wav(in_dir / "b.wav")
        out_dir = tmp_path / "out"

        result = run_command(["--json", "fx", str(in_dir), "gain", "--db", "-3",
                                  "-o", str(out_dir)])
        assert result.code == 0
        payloads = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert len(payloads) == 2
        assert {p["effect"] for p in payloads} == {"gain"}
        assert {p["output"] for p in payloads} == {str(out_dir / "a.wav"), str(out_dir / "b.wav")}
        assert all("output_rms" in p and "output_peak" in p for p in payloads)

    def test_recursive_and_suffix(self, tmp_path):
        in_dir = tmp_path / "in"
        (in_dir / "nested").mkdir(parents=True)
        write_wav(in_dir / "top.wav")
        write_wav(in_dir / "nested" / "deep.wav")
        out_dir = tmp_path / "out"

        result = run_command([
            "--plain", "fx", str(in_dir), "gain", "--db", "-3",
            "-o", str(out_dir), "--recursive", "--suffix", "_fx",
        ])
        assert result.code == 0
        assert (out_dir / "top_fx.wav").exists()
        assert (out_dir / "nested" / "deep_fx.wav").exists()

    def test_non_recursive_skips_subdirectories(self, tmp_path):
        in_dir = tmp_path / "in"
        (in_dir / "nested").mkdir(parents=True)
        write_wav(in_dir / "top.wav")
        write_wav(in_dir / "nested" / "deep.wav")
        out_dir = tmp_path / "out"

        result = run_command(["--plain", "fx", str(in_dir), "gain", "--db", "-3",
                                  "-o", str(out_dir)])
        assert result.code == 0
        assert (out_dir / "top.wav").exists()
        assert not (out_dir / "nested" / "deep.wav").exists()

    def test_empty_directory_exits_1(self, tmp_path):
        in_dir = tmp_path / "empty"
        in_dir.mkdir()
        result = run_command(["--plain", "fx", str(in_dir), "gain", "--db", "-3",
                                  "-o", str(tmp_path / "out")])
        assert result.code == 1
        assert "No audio files found" in result.err

    def test_batch_file_level_error_payload(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        out_dir = tmp_path / "out"

        # Make _apply_effect blow up only on the second call to observe the per-file failure.
        calls = {"n": 0}
        real_apply = fx._apply_effect

        def flaky(audio, sr, args):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return real_apply(audio, sr, args)

        monkeypatch.setattr(fx, "_apply_effect", flaky)

        result = run_command(["--json", "fx", str(in_dir), "gain", "--db", "-3",
                                  "-o", str(out_dir)])
        assert result.code == 0  # batch must not raise the exit code (contract)
        payloads = [json.loads(line) for line in result.out.splitlines() if line.strip()]
        assert len(payloads) == 1
        assert payloads[0]["error"] == "boom"
        assert "input" in payloads[0]

    def test_batch_human_path_reports_warning(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")

        def boom(audio, sr, args):
            raise RuntimeError("bad input")

        monkeypatch.setattr(fx, "_apply_effect", boom)
        result = run_command(["--plain", "fx", str(in_dir), "gain", "--db", "-3",
                                  "-o", str(tmp_path / "out")])
        assert result.code == 0
        assert "warning:" in result.err
        assert "bad input" in result.err
        assert "Batch complete: 0 succeeded, 1 failed / 1 total" in result.err
