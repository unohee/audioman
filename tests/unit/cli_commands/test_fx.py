# Created: 2026-09-28
# Purpose: cli/fx.py 커버리지 — 11개 이펙트 + 배치/디렉토리 모드 (AUD-1851).
#
# fx는 플러그인을 쓰지 않는 순수 내장 DSP라 이 호스트에서 그대로 돈다. 각
# 이펙트는 (a) JSON 페이로드와 (b) 실제로 디스크에 쓰인 출력 파일의 샘플
# 수/진폭을 단정한다. 값이 아니라 "효과가 실제로 적용됐는지"를 본다.

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
    """무음-톤-무음 3구간 WAV. trim-silence의 앞뒤 pad 경로를 모두 태운다."""
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
        # 440Hz 사인은 개별 샘플이 영점을 지나므로 윈도우 envelope으로 비교한다.
        faded = float(np.abs(audio[0, :100]).max())
        mid = float(np.abs(audio[0, 2000:3000]).max())
        after = float(np.abs(audio[0, 5000:6000]).max())
        assert faded < 0.1
        assert faded < mid < after
        assert after == pytest.approx(0.5, abs=0.01)  # 원음 진폭 회복

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
        # 기본 길이는 sr//10 = 4410 샘플: 그 앞은 눌리고 뒤는 원음이 살아 있다.
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
        assert result.code == 2       # argparse가 choices 위반을 거부
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
        # 입력은 DC가 있으므로 RMS가 줄어든다.
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
        # crossfade는 좌/우 꼬리 길이만큼을 추가로 소비한다 (cf=220 samples).
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
        # insert mode: base + clip, 그리고 crossfade가 좌/우 양쪽에서 겹친다.
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
        """모노 입력 + 스테레오 클립은 CLI에서 다운믹스되지만 dsp가 거부한다.

        `read_audio`가 모노 파일도 (1, n) 2-D로 읽는 반면, 다운믹스 결과는
        1-D가 되어 shape이 어긋난다. 크래시를 계약으로 고정한다 (조용한
        오작동이 아니라 명시적 ValueError).
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
        with pytest.raises(ValueError, match="채널 변환 불가"):
            run_command([
                "--json", "fx", str(src), "splice", "--clip", str(clip),
                "--position", "0", "-o", str(tmp_path / "x.wav"),
            ])

    def test_sample_rate_mismatch_raises(self, tmp_path):
        src = write_wav(tmp_path / "base.wav", sample_rate=44100)
        clip = write_wav(tmp_path / "clip.wav", sample_rate=22050, duration=0.1)
        with pytest.raises(ValueError, match="Sample rate 불일치"):
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
        assert "splice는 단일 파일에만 적용 가능" in result.err


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
        # 앞뒤 모두 무음이 있어야 pad가 양쪽에서 더해진다.
        src = _write_silence_tone_silence(tmp_path / "bracketed.wav")
        unpadded = tmp_path / "unpadded.wav"
        padded = tmp_path / "padded.wav"
        first = run_command(["--json", "fx", str(src), "trim-silence", "-o", str(unpadded)])
        second = run_command([
            "--json", "fx", str(src), "trim-silence", "--pad", "500", "-o", str(padded),
        ])
        assert first.code == 0 and second.code == 0
        # pad는 앞뒤 경계에 각각 pad_samples를 남긴다.
        assert _json(second)["output_stats"]["frames"] == _json(first)["output_stats"]["frames"] + 1000

    def test_pad_is_clamped_at_the_file_boundary(self, tmp_path):
        # 앞쪽 무음만 있는 파일에서는 pad가 파일 시작(0)에서 잘린다.
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
        """게이트는 임계값 아래 구간만 눌러야 한다 (파일 전체를 죽이면 안 된다)."""
        quiet_len, tone_len = 22050, 22050
        src = tmp_path / "gate-fixture.wav"
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(tone_len, dtype=np.float32) / 44100)
        quiet = np.full(quiet_len, 0.0005, dtype=np.float32)  # -66 dB, 임계값 아래
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
        # 저레벨 구간은 눌리고, 톤 구간은 그대로 남는다.
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
        # print_success는 plain/rich 모두 stderr 콘솔로 나간다.
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
        assert "파일 없음" in result.err
        assert not (tmp_path / "out.wav").exists()


class TestNoEffect:
    def test_missing_effect_name_is_rejected_by_the_parser(self, tmp_path):
        src = write_wav(tmp_path / "t.wav")
        result = run_command(["--json", "fx", str(src), "-o", str(tmp_path / "o.wav")])
        assert result.code == 2
        assert "fade-in" in result.err and "gain" in result.err

    def test_unknown_effect_in_namespace_raises_from_apply_effect(self, tmp_path):
        """argparse의 choices를 우회한 effect 이름은 _apply_effect가 거부한다.

        서브커맨드 이름은 parser가 강제하지만, run(args)를 직접 부르는 경로에서는
        임의 문자열이 들어올 수 있다. 이때 조용히 원본을 통과시키지 않고
        명시적으로 실패하는지 확인한다.
        """
        import argparse

        src = write_wav(tmp_path / "t.wav")
        audio, sr = read_wav(src)
        args = argparse.Namespace(effect="nonexistent-effect")
        with pytest.raises(ValueError, match="알 수 없는 이펙트: nonexistent-effect"):
            fx._apply_effect(audio, sr, args)

    def test_run_without_effect_reports_usage_and_exits_1(self, tmp_path, capsys):
        """`effect`가 없는 Namespace로 run()을 직접 부르면 안내 후 종료한다.

        argparse는 서브커맨드를 요구하므로 이 분기는 손으로 만든 Namespace로만
        도달한다. (다른 진입점이 run(args)를 직접 호출할 때의 계약)
        """
        import argparse

        src = write_wav(tmp_path / "t.wav")
        args = argparse.Namespace(
            input=str(src), effect=None, output=str(tmp_path / "o.wav"), json=True,
        )
        with pytest.raises(SystemExit) as excinfo:
            fx.run(args)
        assert excinfo.value.code == 1
        assert "이펙트를 지정해주세요" in capsys.readouterr().err


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
        assert "배치 완료: 2 성공, 0 실패 / 2 전체" in result.err

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
        assert "오디오 파일이 없습니다" in result.err

    def test_batch_file_level_error_payload(self, tmp_path, monkeypatch):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        write_wav(in_dir / "a.wav")
        out_dir = tmp_path / "out"

        # 두 번째 호출에서만 _apply_effect가 터지도록 만들어 per-file 실패를 관찰한다.
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
        assert result.code == 0  # 배치는 종료코드를 올리지 않는다 (계약)
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
        assert "배치 완료: 0 성공, 1 실패 / 1 전체" in result.err
