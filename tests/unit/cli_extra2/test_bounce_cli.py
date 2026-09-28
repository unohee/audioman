# tests/unit/cli_extra2/test_bounce_cli.py
# Purpose: cover `audioman bounce` end to end — gain/pan lists, per-track
#          chains, session-file mode, dry-run, and the human success report.
#
# Plugin-dependent steps go through the stub registry + recording wrapper
# (AUD-1857: this host has no registrable VST3). The mixing itself, the file
# written to disk and the JSON contract run for real.

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from .conftest import install_registry, install_wrapper, make_meta, write_wav


@pytest.fixture
def tracks(tmp_path):
    return [
        write_wav(tmp_path / "kick.wav", sample_rate=8000, frequency=220.0),
        write_wav(tmp_path / "bass.wav", sample_rate=8000, frequency=330.0),
    ]


class TestFloatListParsing:
    def test_blank_list_is_empty(self):
        from audioman.cli.bounce import _parse_float_list

        assert _parse_float_list("") == []
        assert _parse_float_list("  ") == []

    def test_values_are_split_and_trimmed(self):
        from audioman.cli.bounce import _parse_float_list

        assert _parse_float_list(" -3, 0 ,+2.5") == [-3.0, 0.0, 2.5]

    def test_garbage_raises_value_error(self):
        from audioman.cli.bounce import _parse_float_list

        with pytest.raises(ValueError):
            _parse_float_list("loud")


class TestDryRun:
    def test_json_plan_reports_gain_pan_and_output(self, run_cli, tracks, tmp_path):
        out = tmp_path / "mix.wav"
        result = run_cli(
            ["bounce", str(tracks[0]), str(tracks[1]), "-o", str(out),
             "--gain=-3,0", "--pan=-1,1", "--dry-run"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "bounce"
        assert payload["dry_run"] is True
        assert payload["track_count"] == 2
        assert payload["output"] == str(out)
        assert [t["gain_db"] for t in payload["tracks"]] == [-3.0, 0.0]
        assert [t["pan"] for t in payload["tracks"]] == [-1.0, 1.0]
        # Dry-run must not touch the filesystem.
        assert not out.exists()

    def test_missing_gain_entries_default_to_unity_and_center(self, run_cli, tracks, tmp_path):
        result = run_cli(
            ["bounce", str(tracks[0]), str(tracks[1]), "-o", str(tmp_path / "m.wav"),
             "--gain=-6", "--dry-run"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert [t["gain_db"] for t in result.payload["tracks"]] == [-6.0, 0.0]
        assert [t["pan"] for t in result.payload["tracks"]] == [0.0, 0.0]

    def test_human_plan_lists_each_track(self, run_cli, tracks, tmp_path):
        result = run_cli(
            ["bounce", str(tracks[0]), str(tracks[1]), "-o", str(tmp_path / "m.wav"),
             "--gain=-3,0", "--dry-run"],
        )
        assert result.code == 0, result.stderr
        assert "Bounce Plan" in result.stdout
        assert "2 tracks" in result.stdout
        assert "kick.wav" in result.stdout
        assert "gain=-3.0dB" in result.stdout
        assert "pan=+0.0" in result.stdout

    def test_chain_column_is_rendered_for_chained_tracks(self, run_cli, tracks, tmp_path):
        result = run_cli(
            ["bounce", str(tracks[0]), str(tracks[1]), "-o", str(tmp_path / "m.wav"),
             "--chain=fake-denoiser|", "--dry-run"],
        )
        assert result.code == 0, result.stderr
        # `[fake-denoiser]` in the plan is a rich markup token, so the whole
        # bracketed segment is swallowed by the console in both modes. Only the
        # arrow that precedes it is observable here; the chain content itself is
        # asserted through the JSON payload below.
        assert "→" in result.stdout
        assert "fake-denoiser" not in result.stdout

    def test_dry_run_chain_payload_matches_the_dash_placeholder(self, run_cli, tracks, tmp_path):
        """The '|' separator keeps track/chain alignment: an empty slot means "no chain"."""
        result = run_cli(
            ["bounce", str(tracks[0]), str(tracks[1]), "-o", str(tmp_path / "m.wav"),
             "--chain=fake-denoiser|", "--dry-run"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        tracks_payload = result.payload["tracks"]
        assert tracks_payload[0]["chain"] == [{"plugin": "fake-denoiser", "params": {}}]
        assert "chain" not in tracks_payload[1]


class TestUsageErrors:
    def test_no_inputs_stops_with_usage_error(self, run_cli, tmp_path):
        result = run_cli(["bounce", "-o", str(tmp_path / "m.wav")])
        assert result.code == 1
        assert "입력 파일을 지정하세요" in result.stderr
        assert "--session" in result.stderr

    def test_no_inputs_returns_before_building_tracks(self, run_cli, tmp_path, silent_error):
        """The explicit `return` (not `print_error`'s exit) must stop the command.

        Otherwise the code would fall through to the track loop with `gains`
        unbound and die on a NameError.
        """
        seen = silent_error("audioman.cli.bounce")
        result = run_cli(["bounce", "-o", str(tmp_path / "m.wav")])

        assert result.code == 0, result.stderr
        assert seen and "입력 파일을 지정하세요" in seen[0]
        assert not (tmp_path / "m.wav").exists()

    def test_malformed_gain_list_keeps_the_process_alive(self, run_cli, tracks, tmp_path, silent_error):
        """A bad --gain value must report and stop, not fall through into bounce()."""
        seen = silent_error("audioman.cli.bounce")
        result = run_cli(
            ["bounce", str(tracks[0]), "-o", str(tmp_path / "m.wav"), "--gain=not-a-number"],
        )
        assert result.code == 0
        assert seen and "--gain" in seen[0]
        assert not (tmp_path / "m.wav").exists()

    def test_unloadable_session_is_reported_and_nothing_is_written(
        self, run_cli, tmp_path, silent_error,
    ):
        seen = silent_error("audioman.cli.bounce")
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")

        result = run_cli(["bounce", "-o", str(tmp_path / "m.wav"), "--session", str(bad)])

        assert result.code == 0
        assert seen and "세션 파일 로드 실패" in seen[0]
        assert not (tmp_path / "m.wav").exists()

    def test_bounce_failure_is_reported(self, run_cli, tmp_path, silent_error):
        seen = silent_error("audioman.cli.bounce")
        bogus = tmp_path / "bogus.wav"
        bogus.write_bytes(b"not audio")

        result = run_cli(["bounce", str(bogus), "-o", str(tmp_path / "m.wav")])

        assert result.code == 0
        assert seen and "바운스 실패" in seen[0]
        assert "Bounce complete" not in result.stderr


class TestExecution:
    def test_json_payload_describes_the_written_file(self, run_cli, tracks, tmp_path):
        out = tmp_path / "mix.wav"
        result = run_cli(
            ["bounce", str(tracks[0]), str(tracks[1]), "-o", str(out), "--gain=-12"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["output_path"] == str(out)
        assert payload["track_count"] == 2
        assert payload["sample_rate"] == 8000
        assert payload["clipping_detected"] is False
        assert payload["duration_seconds"] >= 0

        audio, sr = sf.read(str(out), always_2d=True)
        assert sr == 8000
        assert audio.shape[1] == 2
        assert audio.shape[0] == 1600

    def test_human_success_report_lists_tracks_and_output(self, run_cli, tracks, tmp_path):
        out = tmp_path / "mix.wav"
        result = run_cli(["bounce", str(tracks[0]), str(tracks[1]), "-o", str(out)])

        assert result.code == 0, result.stderr
        assert "Bounce complete" in result.stderr
        assert "Tracks: 2" in result.stdout
        assert str(out) in result.stdout
        assert "SR:     8000 Hz" in result.stdout
        assert "Time:" in result.stdout

    def test_loud_sum_reports_clipping(self, run_cli, tmp_path):
        loud = write_wav(tmp_path / "loud.wav", sample_rate=8000, amplitude=0.95)
        result = run_cli(["bounce", str(loud), str(loud), "-o", str(tmp_path / "m.wav")])

        assert result.code == 0, result.stderr
        assert "클리핑 감지" in result.stderr
        assert "Bounce complete" in result.stderr

    def test_gain_is_actually_applied_to_the_output(self, run_cli, tracks, tmp_path):
        out_full = tmp_path / "full.wav"
        out_quiet = tmp_path / "quiet.wav"
        assert run_cli(["bounce", str(tracks[0]), "-o", str(out_full)]).code == 0
        assert run_cli(
            ["bounce", str(tracks[0]), "-o", str(out_quiet), "--gain=-6"],
        ).code == 0

        full, _ = sf.read(str(out_full), always_2d=True)
        quiet, _ = sf.read(str(out_quiet), always_2d=True)
        ratio = float(np.max(np.abs(quiet))) / float(np.max(np.abs(full)))
        assert ratio == pytest.approx(10 ** (-6 / 20.0), rel=0.05)

    def test_per_track_chain_runs_through_the_registry(self, run_cli, tracks, tmp_path, monkeypatch):
        meta = make_meta(short_name="fake-denoiser", name="Fake De-noise")
        stub = install_registry(
            monkeypatch, ("audioman.core.mixer",), [meta],
        )
        wrappers = install_wrapper(monkeypatch, ("audioman.core.mixer",), gain=0.5)

        out = tmp_path / "chained.wav"
        result = run_cli(
            ["bounce", str(tracks[0]), "-o", str(out), "--chain=fake-denoiser:threshold=-20"],
            json_mode=True,
        )

        assert result.code == 0, result.stderr
        assert stub.get_calls == ["fake-denoiser"]
        assert len(wrappers) == 1
        assert wrappers[0].loaded is True
        assert wrappers[0].applied == [{"threshold": -20.0}]
        assert wrappers[0].processed == 1
        assert result.payload["tracks"][0]["chain"] == [
            {"plugin": "fake-denoiser", "params": {"threshold": -20.0}}
        ]

        chained, _ = sf.read(str(out), always_2d=True)
        bare_out = tmp_path / "bare.wav"
        assert run_cli(["bounce", str(tracks[0]), "-o", str(bare_out)]).code == 0
        bare, _ = sf.read(str(bare_out), always_2d=True)
        assert float(np.max(np.abs(chained))) == pytest.approx(
            0.5 * float(np.max(np.abs(bare))), rel=0.08
        )

    def test_unknown_plugin_in_chain_fails_the_bounce(self, run_cli, tracks, tmp_path):
        result = run_cli(
            ["bounce", str(tracks[0]), "-o", str(tmp_path / "m.wav"),
             "--chain=definitely-not-installed"],
        )
        assert result.code == 1
        assert "바운스 실패" in result.stderr
        assert "definitely-not-installed" in result.stderr


class TestSessionMode:
    def _session(self, tmp_path, tracks, *, subtype="PCM_16", sample_rate=8000):
        payload = {
            "output": str(tmp_path / "session_out.wav"),
            "subtype": subtype,
            "sample_rate": sample_rate,
            "tracks": [
                {"path": tracks[0].name, "gain_db": -3.0, "pan": -0.5},
                {"path": tracks[1].name, "gain_db": -6.0, "pan": 0.5},
            ],
        }
        path = tmp_path / "session.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_session_drives_the_plan_and_supplies_the_output_default(
        self, run_cli, tracks, tmp_path, silent_error,
    ):
        """`--output` is required, so a session run supplies it explicitly.

        The session file must be the source of tracks/gain/pan/sample_rate.
        """
        session = self._session(tmp_path, tracks)
        out = tmp_path / "cli_out.wav"
        result = run_cli(
            ["bounce", "-o", str(out), "--session", str(session), "--dry-run"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["track_count"] == 2
        assert payload["output"] == str(out)
        assert [t["gain_db"] for t in payload["tracks"]] == [-3.0, -6.0]
        assert [t["pan"] for t in payload["tracks"]] == [-0.5, 0.5]
        assert payload["tracks"][0]["path"] == str((tmp_path / tracks[0].name).resolve())

    def test_session_output_path_is_used_when_no_override_is_given(self, run_cli, tracks, tmp_path):
        """The session file supplies the output path when `--output` is empty.

        The parser marks `--output` required, so this drives `bounce.run` with a
        namespace whose `output` is empty — the branch
        `args.output if args.output else session.output` exists to serve.
        """
        import argparse

        from audioman.cli import bounce as bounce_cli
        from .conftest import run_command

        session = self._session(tmp_path, tracks)
        args = argparse.Namespace(
            inputs=[], output="", gain="", pan="", chain="", session=str(session),
            dry_run=True, json=True,
        )

        result = run_command(bounce_cli.run, args)

        assert result.code == 0, result.stderr
        assert result.payload["output"] == str(tmp_path / "session_out.wav")
        assert not (tmp_path / "session_out.wav").exists()  # dry-run

    def test_session_sample_rate_and_subtype_reach_the_written_file(
        self, run_cli, tracks, tmp_path,
    ):
        session = self._session(tmp_path, tracks, subtype="PCM_16")
        out = tmp_path / "out.wav"
        result = run_cli(["bounce", "-o", str(out), "--session", str(session)], json_mode=True)

        assert result.code == 0, result.stderr
        assert result.payload["sample_rate"] == 8000
        info = sf.info(str(out))
        assert info.subtype == "PCM_16"
        assert info.samplerate == 8000

    def test_missing_session_file_is_reported(self, run_cli, tmp_path):
        result = run_cli(
            ["bounce", "-o", str(tmp_path / "o.wav"), "--session", str(tmp_path / "nope.json")],
        )
        assert result.code == 1
        assert "세션 파일 로드 실패" in result.stderr
