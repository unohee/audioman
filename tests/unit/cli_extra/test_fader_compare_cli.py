# tests/unit/cli_extra/test_fader_compare_cli.py
# Purpose: cover `audioman fader-compare` end to end (AUD-1851) plus the
#          ground-truth validation added for AUD-1860.
#
# The command consumes the JSON that `fader-test` exports, so the fixtures
# reproduce that shape (conftest.write_ground_truth) next to real stem WAVs and
# automix runs for real on them — no stubbed gain vector, so the comparison
# arithmetic is exercised against genuine recommendations.

from __future__ import annotations

import json

import pytest

from .conftest import write_ground_truth, write_stems


@pytest.fixture
def stems_dir(tmp_path):
    return tmp_path / "stems"


@pytest.fixture
def stems(stems_dir):
    return write_stems(stems_dir)


def ground_truth_for(tmp_path, stems_dir, gains=None):
    """A fader-test export whose gains roughly match what automix produces."""
    if gains is None:
        gains = {"kick": -15.0, "snare": -21.0, "bass": -21.0}
    return write_ground_truth(tmp_path / "gains.json", stems_dir, gains)


class TestJsonPayload:
    def test_envelope_and_summary_shape(self, run_cli, tmp_path, stems_dir, stems):
        gt = ground_truth_for(tmp_path, stems_dir)

        result = run_cli(["--json", "fader-compare", str(gt)])
        assert result.code == 0, result.stderr

        payload = result.payload
        assert payload["$schema"] == "audioman://schema/fader-compare.v1.json"
        assert payload["command"] == "fader-compare"
        assert payload["ground_truth"] == str(gt)
        assert payload["automix_target"] == "archive_techno_standard"
        assert payload["n_tracks_matched"] == 3

        summary = payload["summary"]
        assert set(summary) == {
            "mean_abs_error_db", "max_abs_error_db", "within_3dB_pct", "within_6dB_pct",
        }
        assert 0.0 <= summary["within_3dB_pct"] <= 100.0
        assert 0.0 <= summary["within_6dB_pct"] <= 100.0

    def test_per_track_diff_is_automix_minus_ground_truth(self, run_cli, tmp_path,
                                                          stems_dir, stems):
        gt = ground_truth_for(tmp_path, stems_dir)

        payload = run_cli(["--json", "fader-compare", str(gt)]).payload
        assert len(payload["tracks"]) == 3

        for row in payload["tracks"]:
            assert set(row) == {"track", "ground_truth_db", "automix_db", "diff_db"}
            expected = round(row["automix_db"] - row["ground_truth_db"], 2)
            assert row["diff_db"] == pytest.approx(expected, abs=0.011)

    def test_summary_statistics_follow_the_track_rows(self, run_cli, tmp_path,
                                                      stems_dir, stems):
        gt = ground_truth_for(tmp_path, stems_dir)

        payload = run_cli(["--json", "fader-compare", str(gt)]).payload
        diffs = [abs(row["diff_db"]) for row in payload["tracks"]]
        summary = payload["summary"]

        assert summary["mean_abs_error_db"] == pytest.approx(
            round(sum(diffs) / len(diffs), 2), abs=0.02)
        assert summary["max_abs_error_db"] == pytest.approx(round(max(diffs), 2), abs=0.02)

    def test_exact_ground_truth_yields_zero_error(self, run_cli, tmp_path,
                                                 stems_dir, stems):
        """Feed automix's own numbers back: every track is a perfect match."""
        first = run_cli(["--json", "fader-compare",
                         str(ground_truth_for(tmp_path, stems_dir))]).payload
        echoes = {row["track"]: row["automix_db"] for row in first["tracks"]}
        gt = write_ground_truth(tmp_path / "echo.json", stems_dir, echoes)

        payload = run_cli(["--json", "fader-compare", str(gt)]).payload
        assert payload["summary"]["mean_abs_error_db"] == 0.0
        assert payload["summary"]["within_3dB_pct"] == 100.0

    def test_target_override_is_reported(self, run_cli, tmp_path, stems_dir, stems):
        gt = ground_truth_for(tmp_path, stems_dir)
        result = run_cli([
            "--json", "fader-compare", str(gt), "--target", "electronica",
        ])
        assert result.code == 0, result.stderr
        assert result.payload["automix_target"] == "electronica"

    def test_tracks_missing_from_the_ground_truth_are_skipped(self, run_cli, tmp_path,
                                                              stems_dir, stems):
        gt = ground_truth_for(tmp_path, stems_dir, gains={"kick": -15.0})

        payload = run_cli(["--json", "fader-compare", str(gt)]).payload
        assert payload["n_tracks_matched"] == 1
        assert [row["track"] for row in payload["tracks"]] == ["kick"]


class TestHumanOutput:
    def test_headline_statistics_are_printed(self, run_cli, tmp_path, stems_dir, stems):
        gt = ground_truth_for(tmp_path, stems_dir)
        result = run_cli(["fader-compare", str(gt)])
        assert result.code == 0, result.stderr

        out = result.stdout
        assert "Fader-test vs Automix (archive_techno_standard)" in out
        assert "matched tracks: 3" in out
        assert "mean |error|:" in out
        assert "max |error|:" in out
        assert "within ±3 dB:" in out
        assert "within ±6 dB:" in out

    def test_disagreement_table_sorts_worst_first(self, run_cli, tmp_path,
                                                  stems_dir, stems):
        # kick is off by 20 dB, the others line up: kick must be the first row.
        gt = ground_truth_for(tmp_path, stems_dir,
                              gains={"kick": 5.0, "snare": -21.0, "bass": -21.0})
        result = run_cli(["fader-compare", str(gt)])

        assert "Track\tGround truth (you)\tAutomix\tDiff\t" in result.stdout
        rows = [line for line in result.stdout.splitlines()
                if line.split("\t")[0] in ("kick", "snare", "bass")]
        assert rows[0].startswith("kick\t")
        assert "↓" in rows[0]  # diff below -3 dB is marked as too quiet

    def test_upward_and_neutral_markers_are_rendered(self, run_cli, tmp_path,
                                                     stems_dir, stems):
        gt = ground_truth_for(tmp_path, stems_dir,
                              gains={"kick": -30.0, "snare": -21.0, "bass": -21.0})
        result = run_cli(["fader-compare", str(gt)])

        row = next(line for line in result.stdout.splitlines() if line.startswith("kick\t"))
        assert "↑" in row
        assert row.split("\t")[3].startswith("+")  # positive diff carries a sign

    def test_closest_matches_block_is_printed(self, run_cli, tmp_path,
                                              stems_dir, stems):
        gt = ground_truth_for(tmp_path, stems_dir)
        result = run_cli(["fader-compare", str(gt)])

        assert "Closest matches" in result.stdout
        assert "gt=" in result.stdout and "auto=" in result.stdout and "diff=" in result.stdout

    def test_long_track_names_are_truncated_in_the_table(self, run_cli, tmp_path):
        stems_dir = tmp_path / "long_stems"
        long_name = "a_very_long_kick_track_name_that_exceeds_the_column.wav"
        write_stems(stems_dir, names=(long_name,))
        gt = write_ground_truth(tmp_path / "g.json", stems_dir, {long_name[:-4]: -15.0})

        result = run_cli(["fader-compare", str(gt)])
        assert result.code == 0, result.stderr
        row = next(line for line in result.stdout.splitlines()
                   if line.split("\t")[0].startswith("a_very_long"))
        assert len(row.split("\t")[0]) == 25


class TestGroundTruthValidation:
    """AUD-1860: every malformed ground truth reports and exits, never tracebacks."""

    def test_missing_file_exits_nonzero(self, run_cli, tmp_path):
        result = run_cli(["fader-compare", str(tmp_path / "ghost.json")])

        assert result.code == 1
        assert "파일 없음" in result.stderr
        assert "Traceback" not in result.stderr

    def test_malformed_json_exits_nonzero(self, run_cli, tmp_path):
        bad = tmp_path / "broken.json"
        bad.write_text("{not json at all", encoding="utf-8")

        result = run_cli(["fader-compare", str(bad)])
        assert result.code == 1
        assert "ground truth JSON을 읽을 수 없습니다" in result.stderr
        assert "Traceback" not in result.stderr

    def test_non_object_json_exits_nonzero(self, run_cli, tmp_path):
        bad = tmp_path / "list.json"
        bad.write_text("[1, 2, 3]", encoding="utf-8")

        result = run_cli(["fader-compare", str(bad)])
        assert result.code == 1
        assert "최상위가 객체가 아닙니다" in result.stderr
        assert "Traceback" not in result.stderr

    def test_non_string_source_dir_exits_nonzero(self, run_cli, tmp_path):
        """Was `TypeError: argument should be a str or an os.PathLike`."""
        bad = tmp_path / "typed.json"
        bad.write_text(json.dumps({"source_dir": 5, "gains": {"a": -1}}), encoding="utf-8")

        result = run_cli(["fader-compare", str(bad)])
        assert result.code == 1
        assert "source_dir가 유효하지 않습니다" in result.stderr
        assert "Traceback" not in result.stderr

    def test_nonexistent_source_dir_exits_nonzero(self, run_cli, tmp_path):
        bad = tmp_path / "ghostdir.json"
        bad.write_text(json.dumps({
            "source_dir": str(tmp_path / "nope"), "gains": {"a": -1},
        }), encoding="utf-8")

        result = run_cli(["fader-compare", str(bad)])
        assert result.code == 1
        assert "source_dir가 유효하지 않습니다" in result.stderr

    def test_missing_gains_field_exits_nonzero(self, run_cli, tmp_path, stems_dir, stems):
        bad = tmp_path / "nogains.json"
        bad.write_text(json.dumps({"source_dir": str(stems_dir)}), encoding="utf-8")

        result = run_cli(["fader-compare", str(bad)])
        assert result.code == 1
        assert "'gains' 필드가 없습니다" in result.stderr

    def test_non_mapping_gains_exits_nonzero(self, run_cli, tmp_path, stems_dir, stems):
        bad = tmp_path / "listgains.json"
        bad.write_text(json.dumps({
            "source_dir": str(stems_dir), "gains": [1, 2, 3],
        }), encoding="utf-8")

        result = run_cli(["fader-compare", str(bad)])
        assert result.code == 1
        assert "'gains' 필드가 없습니다" in result.stderr

    def test_source_dir_without_wavs_exits_nonzero(self, run_cli, tmp_path):
        """Was `ZeroDivisionError` after print_error returned."""
        empty = tmp_path / "empty_stems"
        empty.mkdir()
        gt = write_ground_truth(tmp_path / "g.json", empty, {"a": -1.0})

        result = run_cli(["fader-compare", str(gt)])
        assert result.code == 1
        assert "source_dir에 wav 없음" in result.stderr
        assert "Traceback" not in result.stderr

    def test_no_matching_track_names_exits_nonzero(self, run_cli, tmp_path,
                                                   stems_dir, stems):
        """Was `ZeroDivisionError` dividing by an empty row list."""
        gt = ground_truth_for(tmp_path, stems_dir, gains={"unrelated": -1.0})

        result = run_cli(["fader-compare", str(gt)])
        assert result.code == 1
        assert "매칭된 트랙이 없습니다" in result.stderr
        assert "Traceback" not in result.stderr

    def test_non_numeric_gain_exits_nonzero(self, run_cli, tmp_path, stems_dir, stems):
        gt = ground_truth_for(tmp_path, stems_dir, gains={"kick": "loud"})

        result = run_cli(["fader-compare", str(gt)])
        assert result.code == 1
        assert "gain 값이 숫자가 아닙니다" in result.stderr
        assert "Traceback" not in result.stderr

    def test_undecodable_stem_reports_the_automix_failure(self, run_cli, tmp_path,
                                                          stems_dir, stems):
        (stems_dir / "broken.wav").write_bytes(b"not audio")
        gt = ground_truth_for(tmp_path, stems_dir)

        result = run_cli(["fader-compare", str(gt)])
        assert result.code == 1
        assert "automix 실패" in result.stderr
        assert "Traceback" not in result.stderr


class TestNameMatching:
    def test_whitespace_in_stem_falls_back_to_the_raw_name(self, run_cli, tmp_path):
        """`pad .wav` strips to `pad`; the raw-name lookup catches `pad `."""
        stems_dir = tmp_path / "spaced"
        write_stems(stems_dir, names=("pad .wav",))
        gt = write_ground_truth(tmp_path / "g.json", stems_dir, {"pad ": -15.0})

        payload = run_cli(["--json", "fader-compare", str(gt)]).payload
        assert payload["n_tracks_matched"] == 1
        assert payload["tracks"][0]["track"] == "pad"


class TestArgumentValidation:
    def test_ground_truth_positional_is_required(self, run_cli):
        result = run_cli(["fader-compare"])
        assert result.code == 2
        assert "ground_truth" in result.stderr

    def test_default_target_is_the_archive_profile(self, run_cli, tmp_path,
                                                   stems_dir, stems):
        gt = ground_truth_for(tmp_path, stems_dir)
        payload = run_cli(["--json", "fader-compare", str(gt)]).payload
        assert payload["automix_target"] == "archive_techno_standard"


class TestReferenceTarget:
    """`--reference` is forwarded to automix when `--target reference` is used."""

    def test_reference_wav_is_accepted(self, run_cli, tmp_path, stems_dir, stems):
        from .conftest import write_tone

        ref = write_tone(tmp_path / "reference.wav", frequency=220.0)
        gt = ground_truth_for(tmp_path, stems_dir)

        result = run_cli([
            "--json", "fader-compare", str(gt),
            "--target", "reference", "--reference", str(ref),
        ])
        assert result.code == 0, result.stderr

        payload = result.payload
        assert payload["automix_target"] == "reference"
        assert payload["n_tracks_matched"] == 3

    def test_reference_gains_differ_from_the_genre_profile(self, run_cli, tmp_path,
                                                           stems_dir, stems):
        """A quiet reference must not produce the same recommendations."""
        from .conftest import write_tone

        ref = write_tone(tmp_path / "reference.wav", frequency=220.0, amplitude=0.05)
        gt = ground_truth_for(tmp_path, stems_dir)

        by_genre = run_cli(["--json", "fader-compare", str(gt)]).payload
        by_reference = run_cli([
            "--json", "fader-compare", str(gt),
            "--target", "reference", "--reference", str(ref),
        ]).payload

        genre_gains = {r["track"]: r["automix_db"] for r in by_genre["tracks"]}
        ref_gains = {r["track"]: r["automix_db"] for r in by_reference["tracks"]}
        assert genre_gains != ref_gains
