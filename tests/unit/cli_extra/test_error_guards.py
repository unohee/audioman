# tests/unit/cli_extra/test_error_guards.py
# Purpose: prove the explicit `return` / `continue` guards that follow a
#          `print_error(...)` are what stop these commands (AUD-1859/AUD-1860).
#
# `print_error` normally exits, so those guards are dead in production for the
# happy reason: the failure is already fatal. They are still load-bearing —
# several of them were missing before, and without them the command cascaded
# into a traceback (`ZeroDivisionError`, `TypeError`) whenever the exit was
# bypassed. These tests neutralise the exit to exercise the guards, matching the
# convention in tests/unit/cli_workflow/test_mixdown_cli.py.
#
# The remaining tests in this module cover the small pure helpers whose branches
# the CLI cannot reach from a well-formed input (a malformed CHANGELOG, a
# missing schema directory, a symlinked schema, ...).

from __future__ import annotations

import json

import pytest

from .conftest import (
    write_corrupt_wav,
    write_ground_truth,
    write_stems,
    write_tone,
)


def record_errors(monkeypatch, module, seen: list):
    """Replace a CLI module's `print_error` with a recording no-op.

    Returns the list of messages; the caller asserts on them and on whatever
    ran (or did not run) afterwards.
    """
    def _fake(message):
        seen.append(str(message))

    monkeypatch.setattr(module, "print_error", _fake)


# ---------------------------------------------------------------------------
# cli/fader_compare.py
# ---------------------------------------------------------------------------


class TestFaderCompareGuards:
    @pytest.fixture
    def stems_dir(self, tmp_path):
        """The directory holding the stems (write_stems returns the file list)."""
        directory = tmp_path / "stems"
        write_stems(directory)
        return directory

    def test_missing_file_returns_before_reading(self, run_cli, tmp_path, monkeypatch):
        from audioman.cli import fader_compare as fc

        seen: list = []
        record_errors(monkeypatch, fc, seen)

        result = run_cli(["fader-compare", str(tmp_path / "ghost.json")])
        assert result.code == 0
        assert seen and "File not found" in seen[0]
        assert "Traceback" not in result.stderr

    def test_malformed_json_returns_before_touching_the_fields(self, run_cli, tmp_path,
                                                              monkeypatch):
        from audioman.cli import fader_compare as fc

        seen: list = []
        record_errors(monkeypatch, fc, seen)

        bad = tmp_path / "broken.json"
        bad.write_text("{not json", encoding="utf-8")

        result = run_cli(["fader-compare", str(bad)])
        assert result.code == 0
        assert seen and "Cannot read ground truth JSON" in seen[0]
        assert "Traceback" not in result.stderr

    def test_non_object_json_returns_before_field_access(self, run_cli, tmp_path, monkeypatch):
        from audioman.cli import fader_compare as fc

        seen: list = []
        record_errors(monkeypatch, fc, seen)

        bad = tmp_path / "list.json"
        bad.write_text("[1, 2]", encoding="utf-8")

        result = run_cli(["fader-compare", str(bad)])
        assert result.code == 0
        assert seen and "Top level of the ground truth JSON is not an object" in seen[0]

    def test_invalid_source_dir_returns_before_glob(self, run_cli, tmp_path, monkeypatch):
        from audioman.cli import fader_compare as fc

        seen: list = []
        record_errors(monkeypatch, fc, seen)

        bad = tmp_path / "typed.json"
        bad.write_text(json.dumps({"source_dir": 5, "gains": {"a": -1}}), encoding="utf-8")

        result = run_cli(["fader-compare", str(bad)])
        assert result.code == 0
        assert seen and "ground truth source_dir is not valid" in seen[0]
        # The guard stops the run before Path(5).glob() can raise TypeError.
        assert "Traceback" not in result.stderr

    def test_missing_gains_returns_before_automix(self, run_cli, tmp_path, stems_dir,
                                                 monkeypatch):
        from audioman.cli import fader_compare as fc

        seen: list = []
        record_errors(monkeypatch, fc, seen)

        bad = tmp_path / "nogains.json"
        bad.write_text(json.dumps({"source_dir": str(stems_dir), "gains": {}}),
                       encoding="utf-8")

        result = run_cli(["fader-compare", str(bad)])
        assert result.code == 0
        assert seen and "has no 'gains' field" in seen[0]
        assert "Fader-test vs Automix" not in result.stdout

    def test_source_dir_without_wavs_returns_before_the_zip(self, run_cli, tmp_path,
                                                            monkeypatch):
        from audioman.cli import fader_compare as fc

        seen: list = []
        record_errors(monkeypatch, fc, seen)

        empty = tmp_path / "empty"
        empty.mkdir()
        gt = write_ground_truth(tmp_path / "g.json", empty, {"a": -1.0})

        result = run_cli(["fader-compare", str(gt)])
        assert result.code == 0
        assert seen and "No wav files in source_dir" in seen[0]
        assert "Traceback" not in result.stderr

    def test_automix_failure_returns_before_the_summary(self, run_cli, tmp_path,
                                                        stems_dir, monkeypatch):
        from audioman.cli import fader_compare as fc

        seen: list = []
        record_errors(monkeypatch, fc, seen)

        (stems_dir / "broken.wav").write_bytes(b"not audio")
        gt = write_ground_truth(tmp_path / "g.json", stems_dir, {"kick": -1.0})

        result = run_cli(["fader-compare", str(gt)])
        assert result.code == 0
        assert seen and "automix failed" in seen[0]
        assert "Fader-test vs Automix" not in result.stdout

    def test_gain_cardinality_mismatch_returns_before_indexing(self, run_cli, tmp_path,
                                                               stems_dir, monkeypatch):
        """A truncated gain vector must be rejected, not silently zipped."""
        from audioman.cli import fader_compare as fc
        from audioman.core import automix as automix_module
        from audioman.core.automix import AutomixResult

        seen: list = []
        record_errors(monkeypatch, fc, seen)

        monkeypatch.setattr(automix_module, "automix", lambda **kwargs: AutomixResult(
            gains_db=[0.0], band_analysis=[], target_profile={}, residual_error_db=0.0,
        ))

        gt = write_ground_truth(tmp_path / "g.json", stems_dir,
                               {"kick": -1.0, "snare": -2.0, "bass": -3.0})
        result = run_cli(["fader-compare", str(gt)])

        assert result.code == 0
        assert seen and "different number of gains than input tracks" in seen[0]
        assert "tracks=3" in seen[0] and "gains=1" in seen[0]
        # The guard stops the run before the statistics divide by a bad count.
        assert "Fader-test vs Automix" not in result.stdout

    def test_non_numeric_gain_returns_before_the_arithmetic(self, run_cli, tmp_path,
                                                            stems_dir, monkeypatch):
        from audioman.cli import fader_compare as fc

        seen: list = []
        record_errors(monkeypatch, fc, seen)

        gt = write_ground_truth(tmp_path / "g.json", stems_dir, {"kick": "loud"})
        result = run_cli(["fader-compare", str(gt)])

        assert result.code == 0
        assert seen and "gain is not a number" in seen[0]
        assert "Fader-test vs Automix" not in result.stdout

    def test_no_matching_tracks_returns_before_dividing(self, run_cli, tmp_path,
                                                        stems_dir, monkeypatch):
        """Covered explicitly: this was the ZeroDivisionError site."""
        from audioman.cli import fader_compare as fc

        seen: list = []
        record_errors(monkeypatch, fc, seen)

        gt = write_ground_truth(tmp_path / "g.json", stems_dir, {"unrelated": -1.0})
        result = run_cli(["fader-compare", str(gt)])

        assert result.code == 0
        assert seen and "No tracks matched" in seen[0]
        assert "Traceback" not in result.stderr
        assert "Fader-test vs Automix" not in result.stdout


# ---------------------------------------------------------------------------
# cli/observe.py
# ---------------------------------------------------------------------------


class TestObserveGuards:
    def test_single_file_read_failure_returns(self, run_cli, tmp_path, monkeypatch):
        from audioman.cli import observe as observe_cli

        seen: list = []
        record_errors(monkeypatch, observe_cli, seen)

        bad = write_corrupt_wav(tmp_path / "corrupt.wav")
        result = run_cli(["observe", str(bad)])

        assert result.code == 0
        assert seen and "Format not recognised" in seen[0]
        # The guard stops the run before the human renderer touches a payload
        # that was never built.
        assert "Findings:" not in result.stdout

    def test_empty_directory_guard_returns_before_iterating(self, run_cli, tmp_path,
                                                            monkeypatch):
        from audioman.cli import observe as observe_cli

        seen: list = []
        record_errors(monkeypatch, observe_cli, seen)

        empty = tmp_path / "empty"
        empty.mkdir()
        result = run_cli(["observe", str(empty)])

        assert result.code == 0
        assert seen and "No audio files in" in seen[0]
        # The guard returns before the batch loop, so nothing else is reported.
        assert "Batch complete" not in result.stderr
        assert result.stdout == ""


# ---------------------------------------------------------------------------
# cli/obs.py
# ---------------------------------------------------------------------------


class TestObsGuards:
    def test_probe_missing_input_returns(self, run_cli, tmp_path, monkeypatch):
        from audioman.cli import obs as obs_cli

        seen: list = []
        record_errors(monkeypatch, obs_cli, seen)

        result = run_cli(["obs", "probe", str(tmp_path / "ghost.mp4")])
        assert result.code == 0
        assert seen and "File/folder not found" in seen[0]
        assert "OBS topology" not in result.stdout

    def test_probe_failure_continues_with_the_next_video(self, run_cli, tmp_path, monkeypatch):
        """A failing probe must not stop the remaining files in the directory."""
        from audioman.cli import obs as obs_cli

        seen: list = []
        record_errors(monkeypatch, obs_cli, seen)

        videos = tmp_path / "videos"
        videos.mkdir()
        (videos / "a_bad.mp4").write_bytes(b"stub")
        (videos / "b_good.mp4").write_bytes(b"stub")

        probed: list[str] = []

        def fake_probe(video, *, probe_seconds=None):
            probed.append(video.name)
            if video.name.startswith("a_"):
                raise RuntimeError("ffprobe exploded")

            class _Report:
                def to_dict(self):
                    return {
                        "topology": "single",
                        "n_streams": 1,
                        "active_indices": [0],
                        "unique_signal_groups": [[0]],
                        "duration_sec": 1.0,
                    }

            return _Report()

        monkeypatch.setattr(obs_cli.obs_core, "probe_topology", fake_probe)

        result = run_cli(["--json", "obs", "probe", str(videos)])
        assert result.code == 0
        assert seen and "ffprobe exploded" in seen[0]
        # Both were attempted; only the healthy one landed in the payload.
        assert probed == ["a_bad.mp4", "b_good.mp4"]
        payload = result.payload
        assert payload["count"] == 1
        assert payload["files"][0]["video"].endswith("b_good.mp4")

    def test_dry_run_missing_input_returns(self, run_cli, tmp_path, monkeypatch):
        from audioman.cli import obs as obs_cli

        seen: list = []
        record_errors(monkeypatch, obs_cli, seen)

        result = run_cli(["obs", "dry-run", str(tmp_path / "ghost.mp4")])
        assert result.code == 0
        assert seen and "File/folder not found" in seen[0]

    def test_dry_run_failure_continues_with_the_next_video(self, run_cli, tmp_path,
                                                           monkeypatch):
        from audioman.cli import obs as obs_cli

        seen: list = []
        record_errors(monkeypatch, obs_cli, seen)

        videos = tmp_path / "videos"
        videos.mkdir()
        (videos / "a_bad.mp4").write_bytes(b"stub")
        (videos / "b_good.mp4").write_bytes(b"stub")

        attempted: list[str] = []

        def fake_dry_run(video, *, analysis_seconds, analysis_start_sec):
            attempted.append(video.name)
            if video.name.startswith("a_"):
                raise RuntimeError("ffprobe exploded")

            class _Topology:
                def to_dict(self):
                    return {"topology": "silent", "n_streams": 0, "active_indices": [],
                            "unique_signal_groups": [], "duration_sec": 0.0, "tracks": []}

            class _Report:
                def to_dict(self):
                    return {"video": str(video), "topology": _Topology().to_dict(),
                            "tracks": [], "treatments": [], "notes": ["stub"]}

            return _Report()

        monkeypatch.setattr(obs_cli.obs_core, "dry_run_video", fake_dry_run)

        result = run_cli(["--json", "obs", "dry-run", str(videos)])
        assert result.code == 0
        assert seen and "ffprobe exploded" in seen[0]
        assert attempted == ["a_bad.mp4", "b_good.mp4"]
        assert result.payload["count"] == 1


# ---------------------------------------------------------------------------
# cli/changelog_cmd.py
# ---------------------------------------------------------------------------


class TestChangelogGuards:
    def test_changelog_lookup_returns_none_when_nothing_matches(self, monkeypatch, tmp_path):
        """`_find_changelog` must report "not found" instead of raising."""
        from audioman.cli import changelog_cmd

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            changelog_cmd, "__file__", str(tmp_path / "nowhere" / "cli" / "changelog_cmd.py"),
        )
        assert changelog_cmd._find_changelog() is None

    def test_since_filter_skips_entries_without_a_version(self):
        """A hand-edited entry must be dropped, not raise KeyError."""
        from audioman.cli.changelog_cmd import filter_since

        entries = [
            {"version": "0.3.0", "date": None, "sections": {}},
            {"date": "2026-01-01", "sections": {}},  # missing "version"
        ]
        kept = filter_since(entries, "0.2.0")
        assert [e["version"] for e in kept] == ["0.3.0"]



# ---------------------------------------------------------------------------
# cli/schemas_cmd.py
# ---------------------------------------------------------------------------


class TestSchemasGuards:
    def test_missing_schema_directory_lists_nothing(self, monkeypatch, tmp_path):
        from audioman.cli import schemas_cmd

        monkeypatch.setattr(schemas_cmd, "_schemas_dir", lambda: tmp_path / "absent")
        assert schemas_cmd._list_schemas() == []

    def test_undecodable_schema_file_is_skipped(self, monkeypatch, tmp_path):
        from audioman.cli import schemas_cmd

        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        (schema_dir / "good.v1.json").write_text(
            json.dumps({"$id": "audioman://schema/good.v1.json", "title": "Good"}),
            encoding="utf-8",
        )
        (schema_dir / "broken.v1.json").write_text("{not json", encoding="utf-8")

        monkeypatch.setattr(schemas_cmd, "_schemas_dir", lambda: schema_dir)
        listed = schemas_cmd._list_schemas()
        assert [s["name"] for s in listed] == ["good.v1"]

    def test_symlinked_schema_cannot_escape_the_directory(self, run_cli, tmp_path,
                                                          monkeypatch):
        """The containment check rejects a resolved path outside the schema dir."""
        from audioman.cli import schemas_cmd

        secret = tmp_path / "secret.json"
        secret.write_text('{"$id": "leaked"}', encoding="utf-8")
        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        (schema_dir / "escape.json").symlink_to(secret)

        monkeypatch.setattr(schemas_cmd, "_schemas_dir", lambda: schema_dir)
        result = run_cli(["schemas", "show", "escape"])
        assert result.code == 2
        assert "invalid schema name" in result.stderr
        assert "leaked" not in result.stdout

    def test_schema_without_a_trailing_newline_gets_one(self, run_cli, tmp_path,
                                                        monkeypatch):
        from audioman.cli import schemas_cmd

        schema_dir = tmp_path / "schemas"
        schema_dir.mkdir()
        (schema_dir / "tight.v1.json").write_text('{"$id": "audioman://schema/tight.v1.json"}',
                                                  encoding="utf-8")

        monkeypatch.setattr(schemas_cmd, "_schemas_dir", lambda: schema_dir)
        result = run_cli(["schemas", "show", "tight.v1"])
        assert result.code == 0, result.stderr
        assert result.stdout.endswith("}\n")
        assert not result.stdout.endswith("\n\n")


# ---------------------------------------------------------------------------
# cli/screen.py
# ---------------------------------------------------------------------------


class TestScreenGuards:
    def test_detector_failure_reports_instead_of_returning_none(self, run_cli, tmp_path,
                                                                monkeypatch):
        """Every exception arm funnels into print_error, which exits."""
        from audioman.cli import screen as screen_cli

        seen: list = []
        record_errors(monkeypatch, screen_cli, seen)

        def boom(path, *, issues, backend):
            raise RuntimeError("detector exploded")

        monkeypatch.setattr(screen_cli, "screen_file", boom)

        src = write_tone(tmp_path / "tone.wav")
        # The neutralised exit reaches the trailing raise; production never does
        # because print_error exits first.
        with pytest.raises(AssertionError, match="unreachable"):
            run_cli(["screen", str(src)])
        assert seen == ["detector exploded"]

    def test_event_note_renders_salience(self):
        """`salience` only appears on the essentia hum events (optional extra)."""
        from audioman.cli.screen import _event_note

        assert _event_note({
            "frequency_hz": 60.0, "salience": 0.83, "snr_db": None,
        }) == "60.0 Hz, salience 0.83"

    def test_event_note_omits_a_null_snr(self):
        from audioman.cli.screen import _event_note

        assert _event_note({"frequency_hz": 120.0, "snr_db": None}) == "120.0 Hz"

    def test_event_note_joins_every_detail_it_knows(self):
        from audioman.cli.screen import _event_note

        note = _event_note({
            "frequency_hz": 60.0,
            "snr_db": 18.5,
            "salience": 0.5,
            "ratio_db": 12.25,
            "rms_db": -30.5,
            "tones": [{"frequency_hz": 8000.0}, {"frequency_hz": 12000.0}],
        })
        assert note == ("60.0 Hz, SNR 18.5 dB, salience 0.5, ratio 12.25 dB, "
                        "RMS -30.5 dB, tones 8000.0Hz, 12000.0Hz")

    def test_event_note_is_empty_for_a_bare_event(self):
        from audioman.cli.screen import _event_note

        assert _event_note({"type": "click"}) == ""

    def test_event_note_caps_the_rendered_tone_list(self):
        from audioman.cli.screen import _event_note

        tones = [{"frequency_hz": float(1000 * i)} for i in range(1, 8)]
        note = _event_note({"tones": tones})
        assert note.count("Hz") == 4
        assert "7000.0" not in note
