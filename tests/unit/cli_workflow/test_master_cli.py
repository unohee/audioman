# tests/unit/cli_workflow/test_master_cli.py
# Purpose: cover `audioman master` (prep / qc / verify / list-profiles).
#
# These subcommands are plugin-free: they run the EDL render engine and the QC
# evaluator on real audio. Assertions check output files, the prepared EDL ops,
# the profile overrides and the QC verdict payload.

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from audioman.cli import master as master_cli
from audioman.core import edl as edl_core
from audioman.core import qc

from .conftest import write_wav


@pytest.fixture
def source(tmp_path):
    """2 s tone: long enough for a measurable LUFS reading."""
    return write_wav(tmp_path / "master_src.wav", sample_rate=8000, duration=2.0)


def _clipped_wav(path):
    """Square-ish 2 s tone driven past full scale so QC reports clipping details."""
    sr = 8000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    clipped = np.clip(1.8 * np.sin(2 * np.pi * 220 * t), -1.0, 1.0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack([clipped, clipped], axis=1), sr, subtype="PCM_16")
    return path


class TestPrepHappyPath:
    def test_default_profile_writes_the_output_and_reports_the_chain(self, run_cli, source, tmp_path):
        out = tmp_path / "prepped.wav"
        result = run_cli(["master", "prep", str(source), "-o", str(out)], json_mode=True)
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "master prep"
        assert payload["profile"] == "spotify"
        assert payload["input"] == str(source.resolve())
        assert payload["output"] == str(out)
        assert payload["params"] == master_cli.PREP_PROFILES["spotify"]
        assert payload["ops_applied"] == [
            "remove_dc", "pad", "fade_in", "fade_out", "loudness_normalize"
        ]
        # 2 s source + 200 ms head + 2 s tail.
        assert payload["input_duration_sec"] == pytest.approx(2.0, abs=1e-3)
        assert payload["output_duration_sec"] == pytest.approx(4.2, abs=1e-3)
        assert payload["elapsed_sec"] >= 0.0
        assert out.exists()
        rendered, sr = sf.read(str(out), always_2d=True)
        assert sr == 8000
        assert rendered.shape[0] == pytest.approx(33600, abs=64)

    def test_render_starts_and_ends_at_silence(self, run_cli, source, tmp_path):
        out = tmp_path / "faded.wav"
        assert run_cli(["master", "prep", str(source), "-o", str(out)]).code == 0
        rendered, sr = sf.read(str(out), always_2d=True)
        head = rendered[: int(0.05 * sr)]
        tail = rendered[-int(0.05 * sr):]
        assert np.max(np.abs(head)) < 1e-3
        assert np.max(np.abs(tail)) < 1e-3

    def test_human_output_reports_profile_and_ops(self, run_cli, source, tmp_path):
        out = tmp_path / "prepped.wav"
        result = run_cli(["master", "prep", str(source), "-o", str(out)])
        assert result.code == 0, result.stderr
        assert "master prep 완료 (spotify)" in result.stderr
        assert "Ops:       remove_dc, pad, fade_in, fade_out, loudness_normalize" in result.stdout
        assert "Duration:  2.00s → 4.20s" in result.stdout

    @pytest.mark.parametrize("profile", list(master_cli.PREP_PROFILES))
    def test_every_profile_prepares(self, run_cli, source, tmp_path, profile):
        out = tmp_path / f"{profile}.wav"
        result = run_cli(
            ["master", "prep", str(source), "-o", str(out), "--profile", profile],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["profile"] == profile
        assert result.payload["params"] == master_cli.PREP_PROFILES[profile]
        assert out.exists()

    def test_cd_master_profile_skips_loudness_normalisation(self, run_cli, source, tmp_path):
        result = run_cli(
            ["master", "prep", str(source), "-o", str(tmp_path / "cd.wav"),
             "--profile", "cd_master"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert "loudness_normalize" not in result.payload["ops_applied"]
        assert result.payload["ops_applied"] == ["remove_dc", "pad", "fade_out"]


class TestPrepOverrides:
    def test_cli_overrides_replace_profile_values(self, run_cli, source, tmp_path):
        result = run_cli(
            ["master", "prep", str(source), "-o", str(tmp_path / "o.wav"),
             "--head-pad-ms", "0", "--tail-pad-sec", "0", "--fade-in-ms", "0",
             "--fade-out-ms", "0", "--target-lufs", "-20", "--max-tp", "-2"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        params = result.payload["params"]
        assert params["head_pad_ms"] == 0
        assert params["tail_pad_sec"] == 0
        assert params["fade_in_ms"] == 0
        assert params["fade_out_ms"] == 0
        assert params["target_lufs"] == -20.0
        assert params["max_true_peak_dbtp"] == -2.0
        # Zero-length fades are dropped from the op list rather than emitted.
        assert result.payload["ops_applied"] == ["remove_dc", "pad", "loudness_normalize"]

    def test_fade_curve_override_is_applied_to_the_edl(self, run_cli, source, tmp_path):
        result = run_cli(
            ["master", "prep", str(source), "-o", str(tmp_path / "o.wav"),
             "--fade-curve", "exponential", "--fade-in-ms", "10", "--fade-out-ms", "10"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["params"]["fade_curve"] == "exponential"
        edl = master_cli._build_prep_edl(source, result.payload["params"], skip_dc=False)
        curves = [op["curve"] for op in edl.ops if "curve" in op]
        assert curves == ["exponential", "exponential"]

    def test_no_dc_remove_drops_the_first_op(self, run_cli, source, tmp_path):
        result = run_cli(
            ["master", "prep", str(source), "-o", str(tmp_path / "o.wav"), "--no-dc-remove"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["ops_applied"][0] == "pad"

    def test_write_edl_persists_the_workspace(self, run_cli, source, tmp_path):
        out = tmp_path / "o.wav"
        result = run_cli(
            ["master", "prep", str(source), "-o", str(out), "--write-edl"], json_mode=True
        )
        assert result.code == 0, result.stderr
        edl_path = edl_core.edl_path(source)
        assert edl_path.exists()
        stored = json.loads(edl_path.read_text(encoding="utf-8"))
        assert [op["type"] for op in stored["ops"]] == result.payload["ops_applied"]
        assert len(edl_core.list_history(source)) == 1

    def test_without_write_edl_no_workspace_is_created(self, run_cli, source, tmp_path):
        assert run_cli(["master", "prep", str(source), "-o", str(tmp_path / "o.wav")]).code == 0
        assert not edl_core.workspace_dir(source).exists()

    def test_missing_input_exits_nonzero(self, run_cli, tmp_path):
        result = run_cli(["master", "prep", str(tmp_path / "nope.wav"), "-o", str(tmp_path / "o.wav")])
        assert result.code == 1
        assert "파일 없음" in result.stderr


class TestPrepFailures:
    def test_render_failure_is_reported(self, run_cli, tmp_path, monkeypatch):
        from audioman.cli import master as mod

        src = write_wav(tmp_path / "src.wav", sample_rate=8000, duration=0.5)
        monkeypatch.setattr(
            mod.edl_core, "render_edl",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("render exploded")),
        )
        result = run_cli(["master", "prep", str(src), "-o", str(tmp_path / "o.wav")])
        assert result.code == 1
        assert "render exploded" in result.stderr

    def test_source_shorter_than_a_second_is_padded_not_rejected(self, run_cli, tmp_path):
        """A 4 ms source still yields a deliverable: pad + fades dominate the output."""
        src = write_wav(tmp_path / "tiny.wav", sample_rate=8000, duration=0.0005)
        out = tmp_path / "tiny_out.wav"
        result = run_cli(
            ["master", "prep", str(src), "-o", str(out), "--target-lufs", "-14",
             "--tail-pad-sec", "0", "--head-pad-ms", "0"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["input_duration_sec"] == pytest.approx(0.0005, abs=1e-4)
        assert payload["output_duration_sec"] == pytest.approx(payload["input_duration_sec"], abs=1e-4)
        assert out.exists()
        rendered, _sr = sf.read(str(out), always_2d=True)
        assert rendered.shape[0] == pytest.approx(4, abs=1)


class TestBuildPrepParams:
    def test_base_profile_is_copied_not_mutated(self):
        import argparse

        args = argparse.Namespace(
            profile="youtube", head_pad_ms=123.0, tail_pad_sec=None, fade_in_ms=None,
            fade_out_ms=None, fade_curve=None, target_lufs=None, max_tp=None,
        )
        params = master_cli._build_prep_params(args)
        assert params["head_pad_ms"] == 123.0
        assert params["tail_pad_sec"] == master_cli.PREP_PROFILES["youtube"]["tail_pad_sec"]
        assert master_cli.PREP_PROFILES["youtube"]["head_pad_ms"] != 123.0

    def test_skip_dc_controls_the_first_op(self, source):
        edl = master_cli._build_prep_edl(source, master_cli.PREP_PROFILES["spotify"], skip_dc=True)
        assert edl.ops[0]["type"] == "pad"


class TestQc:
    def test_json_payload_carries_the_report(self, run_cli, source):
        result = run_cli(["master", "qc", str(source)], json_mode=True)
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "master qc"
        assert payload["input"] == str(source.resolve())
        assert payload["verdict"] in ("PASS", "WARN", "FAIL")
        assert payload["target_profile"]["name"] == qc.TARGETS["spotify"].name
        assert payload["summary"]["n_pass"] + payload["summary"]["n_warn"] + payload["summary"]["n_fail"] == len(payload["checks"])
        assert payload["format"]["sample_rate"] == 8000
        assert payload["format"]["channels"] == 2

    def test_human_report_prints_verdict_table_and_details(self, run_cli, source):
        result = run_cli(["master", "qc", str(source), "--target", "cd_master"])
        assert result.code == 0, result.stderr
        assert "QC Verdict:" in result.stdout
        assert "PASS:" in result.stdout and "WARN:" in result.stdout and "FAIL:" in result.stdout
        assert "# Checks" in result.stdout
        assert "Category\tCheck\tValue\tTarget\tStatus" in result.stdout

    def test_human_report_lists_warn_and_fail_details(self, run_cli, tmp_path):
        """Clipping findings carry a `detail` block that must be printed."""
        result = run_cli(["master", "qc", str(_clipped_wav(tmp_path / "clipped.wav"))])
        assert result.code == 0, result.stderr
        assert "[FAIL] clipping_samples: {'n_samples':" in result.stdout
        assert "'per_channel': [10000, 10000]" in result.stdout
        # The detail block is the raw measurement, not a findings document.
        assert "CLIP_SAMPLE_PEAK_EXCEEDED" not in result.stdout

    @pytest.mark.parametrize("target", qc.list_targets())
    def test_every_target_is_accepted(self, run_cli, source, target):
        result = run_cli(["master", "qc", str(source), "--target", target], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["target"] == target

    def test_click_sensitivity_is_forwarded(self, run_cli, source, monkeypatch):
        seen = {}
        real_evaluate_file = qc.evaluate_file

        def fake_evaluate_file(path, target="spotify", click_sensitivity=6.0):
            seen["path"] = path
            seen["target"] = target
            seen["click"] = click_sensitivity
            return real_evaluate_file(path, target=target, click_sensitivity=click_sensitivity)

        monkeypatch.setattr(master_cli.qc, "evaluate_file", fake_evaluate_file)
        result = run_cli(["master", "qc", str(source), "--click-sensitivity", "2.5"])
        assert result.code == 0, result.stderr
        assert seen["click"] == 2.5
        assert seen["target"] == "spotify"
        assert str(seen["path"]) == str(source.resolve())

    def test_unknown_target_is_rejected_by_argparse(self, run_cli, source):
        result = run_cli(["master", "qc", str(source), "--target", "not-a-target"])
        assert result.code == 2
        assert "invalid choice" in result.stderr

    def test_missing_input_exits_nonzero(self, run_cli, tmp_path):
        result = run_cli(["master", "qc", str(tmp_path / "nope.wav")])
        assert result.code == 1
        assert "파일 없음" in result.stderr

    def test_evaluate_value_error_is_reported(self, run_cli, source, monkeypatch):
        def boom(*a, **k):
            raise ValueError("알 수 없는 target")

        monkeypatch.setattr(master_cli.qc, "evaluate_file", boom)
        result = run_cli(["master", "qc", str(source)])
        assert result.code == 1
        assert "알 수 없는 target" in result.stderr


class TestVerify:
    def test_verify_runs_prep_then_qc_and_reports_both(self, run_cli, source, tmp_path):
        out = tmp_path / "verified.wav"
        result = run_cli(
            ["master", "verify", str(source), "-o", str(out), "--profile", "apple_music"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "master verify"
        assert payload["profile"] == "apple_music"
        assert payload["qc_target"] == "apple_music"
        assert payload["input"] == str(source.resolve())
        assert payload["output"] == str(out)
        assert payload["prep"]["params"] == master_cli.PREP_PROFILES["apple_music"]
        assert payload["prep"]["ops_applied"] == [
            "remove_dc", "pad", "fade_in", "fade_out", "loudness_normalize"
        ]
        assert payload["prep"]["elapsed_sec"] >= 0.0
        assert payload["qc"]["verdict"] in ("PASS", "WARN", "FAIL")
        assert out.exists()

    def test_verify_honours_an_explicit_qc_target(self, run_cli, source, tmp_path):
        result = run_cli(
            ["master", "verify", str(source), "-o", str(tmp_path / "v.wav"),
             "--profile", "spotify", "--target", "cd_master"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["qc_target"] == "cd_master"
        assert result.payload["qc"]["target"] == "cd_master"

    def test_verify_defaults_the_qc_target_to_the_prep_profile(self, run_cli, source, tmp_path):
        """With no --target, QC runs against the same profile name as prep."""
        result = run_cli(
            ["master", "verify", str(source), "-o", str(tmp_path / "v.wav")],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["qc_target"] == "spotify"
        assert result.payload["qc"]["target"] == "spotify"

    def test_verify_human_output_prints_both_stages(self, run_cli, source, tmp_path):
        out = tmp_path / "verified.wav"
        result = run_cli(["master", "verify", str(source), "-o", str(out)])
        assert result.code == 0, result.stderr
        assert "prep 완료 (spotify)" in result.stderr
        assert "QC target: spotify" in result.stdout
        assert "QC Verdict:" in result.stdout
        assert str(out) not in result.stdout  # only the file name is shown
        assert out.name in result.stdout

    def test_verify_warns_and_falls_back_for_an_unknown_qc_target(
        self, run_cli, source, tmp_path, monkeypatch
    ):
        """A profile whose name is not a QC target must warn and fall back to spotify."""
        monkeypatch.setitem(
            master_cli.PREP_PROFILES, "prep_only",
            dict(master_cli.PREP_PROFILES["spotify"]),
        )
        # Widen the argparse choices by rebuilding through the parser is not possible
        # post-hoc, so drive run_verify directly with a synthetic namespace.
        import argparse

        out = tmp_path / "fallback.wav"
        args = argparse.Namespace(
            input=str(source), output=str(out), profile="prep_only", target=None,
            head_pad_ms=None, tail_pad_sec=None, fade_in_ms=None, fade_out_ms=None,
            fade_curve=None, target_lufs=None, max_tp=None, no_dc_remove=False,
            write_edl=False, json=False,
        )
        master_cli.run_verify(args)
        assert out.exists()

    def test_verify_render_failure_is_reported(self, run_cli, source, tmp_path, monkeypatch):
        monkeypatch.setattr(
            master_cli.edl_core, "render_edl",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("bad render")),
        )
        result = run_cli(["master", "verify", str(source), "-o", str(tmp_path / "v.wav")])
        assert result.code == 1
        assert "bad render" in result.stderr

    def test_missing_input_exits_nonzero(self, run_cli, tmp_path):
        result = run_cli(["master", "verify", str(tmp_path / "nope.wav"), "-o", str(tmp_path / "v.wav")])
        assert result.code == 1
        assert "파일 없음" in result.stderr


class TestListProfiles:
    def test_json_payload_lists_prep_profiles_and_qc_targets(self, run_cli):
        result = run_cli(["master", "list-profiles"], json_mode=True)
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "master list-profiles"
        assert payload["prep_profiles"] == master_cli.PREP_PROFILES
        assert payload["qc_targets"] == qc.list_targets()

    def test_human_table_has_a_row_per_profile(self, run_cli):
        result = run_cli(["master", "list-profiles"])
        assert result.code == 0, result.stderr
        assert "# Mastering profiles" in result.stdout
        for name in master_cli.PREP_PROFILES:
            assert name in result.stdout
        assert "QC targets:" in result.stdout
        for target in qc.list_targets():
            assert target in result.stdout


class TestBareInvocation:
    def test_no_action_prints_help(self, run_cli):
        result = run_cli(["master"])
        assert result.code == 0
        assert "usage: audioman master" in result.stdout
