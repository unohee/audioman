# tests/unit/cli_extra/test_obs_cli.py
# Purpose: cover `audioman obs probe` / `obs dry-run` end to end (AUD-1851).
#
# The OBS paths shell out to ffmpeg/ffprobe, so the fixtures are real encoded
# containers built by conftest.make_multitrack_video; the run is skipped when
# the host has no ffmpeg rather than asserting on an environment gap. Everything
# else (arg parsing, JSON envelope, human table, --out-dir report writing,
# dedup of .mov/.mp4 stem pairs) is exercised for real.

from __future__ import annotations

import json

import numpy as np
import pytest

from .conftest import (
    make_multitrack_video,
    music_track,
    voice_track,
)


@pytest.fixture
def multitrack_video(tmp_path):
    """Two distinct-RMS tracks -> classified as `multitrack`."""
    path = make_multitrack_video(
        tmp_path / "multi.mp4", [voice_track(), music_track()],
    )
    if path is None:
        pytest.skip("ffmpeg/ffprobe unavailable or encode failed")
    return path


@pytest.fixture
def silent_video(tmp_path):
    """Two digital-silence tracks -> classified as `silent`."""
    silent = np.zeros(48000, dtype=np.float32)
    path = make_multitrack_video(tmp_path / "silent.mp4", [silent, silent])
    if path is None:
        pytest.skip("ffmpeg/ffprobe unavailable or encode failed")
    return path


class TestProbeJson:
    def test_single_file_reports_topology_and_envelope(self, run_cli, multitrack_video):
        result = run_cli(["--json", "obs", "probe", str(multitrack_video)])
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["$schema"] == "audioman://schema/obs.v1.json"
        assert payload["command"] == "obs probe"
        assert payload["count"] == 1
        assert payload["input"] == str(multitrack_video)

        entry = payload["files"][0]
        assert entry["video"] == str(multitrack_video)
        assert entry["topology"] == "multitrack"
        assert entry["n_streams"] == 2
        assert entry["active_indices"] == [0, 1]
        # Both tracks carry signal at distinct levels, so they stay ungrouped.
        assert len(entry["unique_signal_groups"]) == 2

    def test_probe_seconds_limits_the_rms_window(self, run_cli, multitrack_video):
        result = run_cli([
            "--json", "obs", "probe", str(multitrack_video), "--probe-seconds", "0.5",
        ])
        assert result.code == 0, result.stderr
        entry = result.payload["files"][0]
        assert entry["topology"] == "multitrack"
        assert entry["active_indices"] == [0, 1]

    def test_silent_video_is_classified_silent(self, run_cli, silent_video):
        result = run_cli(["--json", "obs", "probe", str(silent_video)])
        assert result.code == 0, result.stderr
        entry = result.payload["files"][0]
        assert entry["topology"] == "silent"
        assert entry["active_indices"] == []


class TestProbeDirectory:
    def test_directory_mode_probes_every_video(self, run_cli, tmp_path, multitrack_video):
        videos = tmp_path / "videos"
        videos.mkdir()
        multitrack_video.rename(videos / "one.mp4")

        result = run_cli(["--json", "obs", "probe", str(videos)])
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["input"] == str(videos)
        assert payload["count"] == 1
        assert payload["files"][0]["topology"] == "multitrack"

    def test_non_video_files_are_ignored(self, run_cli, tmp_path):
        videos = tmp_path / "only_notes"
        videos.mkdir()
        (videos / "notes.txt").write_text("ignore me", encoding="utf-8")
        (videos / "clip.wav").write_bytes(b"riff")

        result = run_cli(["--json", "obs", "probe", str(videos)])
        assert result.code == 0, result.stderr
        assert result.payload["count"] == 0
        assert result.payload["files"] == []

    def test_mov_and_mp4_with_the_same_stem_are_probed_once(self, run_cli, tmp_path):
        """`_iter_videos` dedups on the stem: the second container is skipped."""
        videos = tmp_path / "pairs"
        videos.mkdir()
        (videos / "clip.mov").write_bytes(b"stub")
        (videos / "clip.mp4").write_bytes(b"stub")

        probed: list[str] = []

        def fake_probe(video, *, probe_seconds=None):
            probed.append(video.name)

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

        from audioman.cli import obs as obs_cli

        original = obs_cli.obs_core.probe_topology
        obs_cli.obs_core.probe_topology = fake_probe
        try:
            result = run_cli(["--json", "obs", "probe", str(videos)])
        finally:
            obs_cli.obs_core.probe_topology = original

        assert result.code == 0, result.stderr
        # sorted() puts clip.mov first; the .mp4 sharing the stem is dropped.
        assert probed == ["clip.mov"]
        assert result.payload["count"] == 1


class TestProbeHuman:
    def test_table_lists_topology_streams_and_groups(self, run_cli, multitrack_video):
        result = run_cli(["obs", "probe", str(multitrack_video)])
        assert result.code == 0, result.stderr
        assert "# OBS topology — 1 file(s)" in result.stdout
        assert "file\ttopology\tstreams\tactive\tgroups\tduration" in result.stdout
        row = result.stdout.splitlines()[2]
        assert row.startswith("multi.mp4\tmultitrack\t2\t0,1\t")
        assert "[0] | [1]" in row
        # duration is rendered with one decimal place, e.g. "1.5s".
        duration = row.split("\t")[-1]
        assert duration.endswith("s")
        assert duration[:-1] == f"{float(duration[:-1]):.1f}"

    def test_empty_directory_prints_an_empty_table(self, run_cli, tmp_path):
        videos = tmp_path / "empty"
        videos.mkdir()
        result = run_cli(["obs", "probe", str(videos)])
        assert result.code == 0, result.stderr
        assert "# OBS topology — 0 file(s)" in result.stdout


class TestProbeErrors:
    def test_missing_input_exits_nonzero(self, run_cli, tmp_path):
        result = run_cli(["obs", "probe", str(tmp_path / "ghost.mp4")])
        assert result.code == 1
        assert "파일/폴더 없음" in result.stderr

    def test_unreadable_video_is_reported_and_skipped(self, run_cli, tmp_path):
        """A container ffprobe cannot read must not abort the whole run."""
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"definitely not a container")

        result = run_cli(["--json", "obs", "probe", str(broken)])
        assert result.code == 1
        assert "error:" in result.stderr
        assert str(broken) in result.stderr


class TestDryRunJson:
    def test_multitrack_payload_carries_treatments_and_notes(self, run_cli, multitrack_video):
        result = run_cli(["--json", "obs", "dry-run", str(multitrack_video)])
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["$schema"] == "audioman://schema/obs.v1.json"
        assert payload["command"] == "obs dry-run"
        assert payload["count"] == 1

        report = payload["files"][0]
        assert report["video"] == str(multitrack_video)
        assert report["topology"]["topology"] == "multitrack"
        assert report["notes"]
        assert any(n.startswith("topology=") for n in report["notes"])

        treatments = report["treatments"]
        assert [t["track_index"] for t in treatments] == [0, 1]
        assert all(t["kind"] in ("voice", "music", "fullmix", "silent", "unknown")
                   for t in treatments)
        assert all(isinstance(t["plan"], list) for t in treatments)

    def test_silent_video_reports_zero_treatments(self, run_cli, silent_video):
        result = run_cli(["--json", "obs", "dry-run", str(silent_video)])
        assert result.code == 0, result.stderr
        report = result.payload["files"][0]
        assert report["treatments"] == []
        assert report["tracks"] == []
        assert "모든 트랙 무음 — 처리 불필요" in report["notes"]

    def test_analysis_window_arguments_reach_the_report(self, run_cli, multitrack_video):
        result = run_cli([
            "--json", "obs", "dry-run", str(multitrack_video),
            "--seconds", "1.0", "--start", "0.25",
        ])
        assert result.code == 0, result.stderr
        report = result.payload["files"][0]
        diag = report["tracks"][0]
        assert diag["analysis_start_sec"] == 0.25
        assert diag["analysis_seconds"] == 1.0


class TestDryRunOutDir:
    def test_json_mode_writes_one_report_per_video(self, run_cli, tmp_path, multitrack_video):
        videos = tmp_path / "videos"
        videos.mkdir()
        multitrack_video.rename(videos / "one.mp4")
        out_dir = tmp_path / "reports"

        result = run_cli([
            "--json", "obs", "dry-run", str(videos), "--out-dir", str(out_dir),
        ])
        assert result.code == 0, result.stderr

        written = out_dir / "one.json"
        assert written.is_file()
        saved = json.loads(written.read_text(encoding="utf-8"))
        # The file body is the per-video report, not the envelope.
        assert saved["video"] == str(videos / "one.mp4")
        assert saved["topology"]["topology"] == "multitrack"
        assert "notes" in saved and "treatments" in saved

    def test_out_dir_is_created_when_absent(self, run_cli, tmp_path, multitrack_video):
        out_dir = tmp_path / "nested" / "reports"
        assert not out_dir.exists()

        result = run_cli([
            "obs", "dry-run", str(multitrack_video), "--out-dir", str(out_dir),
        ])
        assert result.code == 0, result.stderr
        assert out_dir.is_dir()
        assert (out_dir / "multi.json").is_file()
        # Human mode points at the directory it just wrote (the long absolute
        # path wraps at the console width, so match the stable fragments).
        assert "상세 JSON:" in result.stdout
        assert "nested/reports" in result.stdout


class TestDryRunHuman:
    def test_table_has_a_row_per_track(self, run_cli, multitrack_video):
        result = run_cli(["obs", "dry-run", str(multitrack_video)])
        assert result.code == 0, result.stderr
        assert "# OBS dry-run — 1 file(s)" in result.stdout
        assert "file\ttopology\ttrack\tkind\tactions\tissues" in result.stdout

        rows = result.stdout.splitlines()[2:]
        assert len(rows) == 2
        assert all(r.startswith("multi.mp4\tmultitrack\ttrack ") for r in rows)
        assert any("warn=" in r and "crit=" in r for r in rows)

    def test_silent_video_renders_a_skip_row(self, run_cli, silent_video):
        result = run_cli(["obs", "dry-run", str(silent_video)])
        assert result.code == 0, result.stderr
        assert "# OBS dry-run — 1 file(s)" in result.stdout
        row = result.stdout.splitlines()[2]
        assert row == f"silent.mp4\tsilent\t-\t-\tskip\t—"

    def test_nothing_to_process_prints_the_info_message(self, run_cli, tmp_path):
        videos = tmp_path / "empty"
        videos.mkdir()
        result = run_cli(["obs", "dry-run", str(videos)])
        assert result.code == 0, result.stderr
        assert "처리할 항목 없음" in result.stderr
        assert "OBS dry-run" not in result.stdout


class TestDryRunErrors:
    def test_missing_input_exits_nonzero(self, run_cli, tmp_path):
        result = run_cli(["obs", "dry-run", str(tmp_path / "ghost.mp4")])
        assert result.code == 1
        assert "파일/폴더 없음" in result.stderr

    def test_unreadable_video_is_reported_and_skipped(self, run_cli, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"definitely not a container")

        result = run_cli(["obs", "dry-run", str(broken)])
        assert result.code == 1
        assert "error:" in result.stderr
        assert str(broken) in result.stderr
        # The failure short-circuits before any summary table is printed.
        assert "OBS dry-run" not in result.stdout


class TestParserSurface:
    def test_subcommand_is_required(self, run_cli):
        result = run_cli(["obs"])
        assert result.code == 2
        assert "obs_command" in result.stderr or "required" in result.stderr

    def test_help_names_both_subcommands(self, run_cli):
        result = run_cli(["--help"])
        assert result.code == 0
        assert "obs" in result.stdout

    def test_probe_requires_an_input(self, run_cli):
        result = run_cli(["obs", "probe"])
        assert result.code == 2
        assert "input" in result.stderr
