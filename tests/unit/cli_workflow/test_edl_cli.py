# tests/unit/cli_workflow/test_edl_cli.py
# Purpose: cover `audioman edl` — the non-destructive edit workflow
#          (init / add / list / undo / redo / render / status / clear).
#
# Every subcommand runs for real against a tmp_path workspace; assertions check
# the on-disk EDL, the history/redo directories and the undo/redo semantics,
# not just that the function returned.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audioman.core import edl as edl_core

from .conftest import write_wav


def _edl_json(source: Path) -> dict:
    return json.loads(edl_core.edl_path(source).read_text(encoding="utf-8"))


def _op_types(source: Path) -> list[str]:
    return [op["type"] for op in _edl_json(source)["ops"]]


@pytest.fixture
def source(tmp_path) -> Path:
    return write_wav(tmp_path / "src.wav", sample_rate=8000, duration=0.5)


@pytest.fixture
def initialized(run_cli, source):
    """A source file with an initialized EDL workspace."""
    result = run_cli(["edl", "init", str(source)])
    assert result.code == 0, result.stderr
    return source


class TestParamParsing:
    """`--param key=value` type inference."""

    def test_json_prefix_decodes_arrays_and_objects(self):
        from audioman.cli.edl import _parse_value

        assert _parse_value('json:[1, 2, 3]') == [1, 2, 3]
        assert _parse_value('json:{"a": 1}') == {"a": 1}

    @pytest.mark.parametrize("raw,expected", [
        ("true", True),
        ("False", False),
        ("null", None),
        ("none", None),
        ("12", 12),
        ("-3.5", -3.5),
        ("1e3", 1000.0),
        ("cosine", "cosine"),
        ("dehum", "dehum"),
    ])
    def test_scalar_inference(self, raw, expected):
        from audioman.cli.edl import _parse_value

        value = _parse_value(raw)
        assert value == expected
        assert isinstance(value, type(expected))

    def test_params_are_stripped_and_typed(self):
        from audioman.cli.edl import _parse_params

        assert _parse_params([" db = -3.0 ", "curve=cosine"]) == {"db": -3.0, "curve": "cosine"}

    def test_item_without_equals_is_a_usage_error(self):
        from audioman.cli.edl import _parse_params

        with pytest.raises(SystemExit) as exc:
            _parse_params(["threshold"])
        assert exc.value.code == 1

    def test_value_containing_equals_keeps_the_rest(self):
        from audioman.cli.edl import _parse_params

        assert _parse_params(["expr=a=b"]) == {"expr": "a=b"}


class TestInit:
    def test_init_writes_edl_and_snapshot_history(self, initialized):
        src = initialized
        data = _edl_json(src)
        assert data["source"] == str(src.resolve())
        assert data["sample_rate"] == 8000
        assert data["channels"] == 2
        assert data["duration_sec"] == pytest.approx(0.5, abs=1e-3)
        assert data["ops"] == []
        assert edl_core.workspace_dir(src).is_dir()
        assert len(edl_core.list_history(src)) == 1

    def test_json_payload_reports_workspace_and_hash(self, run_cli, source):
        result = run_cli(["edl", "init", str(source)], json_mode=True)
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "edl init"
        assert payload["source"] == str(source.resolve())
        assert payload["edl_path"] == str(edl_core.edl_path(source))
        assert payload["workspace"] == str(edl_core.workspace_dir(source))
        assert payload["duration_sec"] == pytest.approx(0.5, abs=1e-3)
        assert payload["sample_rate"] == 8000
        assert payload["channels"] == 2
        assert payload["source_sha256"] == edl_core.file_sha256(source)

    def test_reinitializing_warns_but_succeeds(self, run_cli, initialized):
        result = run_cli(["edl", "init", str(initialized)])
        assert result.code == 0, result.stderr
        assert "warning: 이미 초기화된 EDL이 있습니다" in result.stderr
        # The workspace is re-created from the source, so the op list is reset.
        assert _edl_json(initialized)["ops"] == []

    def test_missing_input_exits_nonzero(self, run_cli, tmp_path):
        result = run_cli(["edl", "init", str(tmp_path / "missing.wav")])
        assert result.code == 1
        assert "파일 없음" in result.stderr


class TestAdd:
    @pytest.mark.parametrize("op_type,params,expect_keys", [
        ("cut_region", ["start_sec=0.1", "end_sec=0.2"], {"start_sec": 0.1, "end_sec": 0.2}),
        ("trim", ["start_sec=0.05", "end_sec=0.4"], {"start_sec": 0.05, "end_sec": 0.4}),
        ("trim_silence", ["threshold_db=-40.0"], {"threshold_db": -40.0}),
        ("fade_in", ["duration_sec=0.01", "curve=cosine"], {"duration_sec": 0.01, "curve": "cosine"}),
        ("fade_out", ["duration_sec=0.02"], {"duration_sec": 0.02}),
        ("normalize", ["peak_db=-1.0"], {"peak_db": -1.0}),
        ("gain", ["db=-3.0"], {"db": -3.0}),
        ("gate", ["threshold_db=-45.0", "attack_sec=0.005"], {"threshold_db": -45.0, "attack_sec": 0.005}),
        ("remove_dc", [], {}),
        ("loudness_normalize", ["target_lufs=-14.0"], {"target_lufs": -14.0}),
    ])
    def test_op_is_stored_with_parsed_params(self, run_cli, initialized, op_type, params, expect_keys):
        argv = ["edl", "add", "-s", str(initialized), op_type]
        for p in params:
            argv += ["--param", p]
        result = run_cli(argv)
        assert result.code == 0, result.stderr
        ops = _edl_json(initialized)["ops"]
        assert len(ops) == 1
        assert ops[0]["type"] == op_type
        for key, value in expect_keys.items():
            assert ops[0][key] == value

    def test_json_payload_reports_op_and_count(self, run_cli, initialized):
        result = run_cli(
            ["edl", "add", "-s", str(initialized), "gain", "--param", "db=-6"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "edl add"
        assert payload["op"] == {"type": "gain", "db": -6.0}
        assert payload["n_ops"] == 1

    def test_op_types_accumulate_in_order(self, run_cli, initialized):
        for op_type, params in [
            ("remove_dc", []),
            ("fade_in", ["duration_sec=0.01"]),
            ("gain", ["db=-2"]),
        ]:
            argv = ["edl", "add", "-s", str(initialized), op_type]
            for p in params:
                argv += ["--param", p]
            assert run_cli(argv).code == 0
        assert _op_types(initialized) == ["remove_dc", "fade_in", "gain"]

    def test_human_output_lists_each_parameter(self, run_cli, initialized):
        result = run_cli(
            ["edl", "add", "-s", str(initialized), "gain", "--param", "db=-3"],
        )
        assert result.code == 0, result.stderr
        assert "op 추가: gain (총 1개)" in result.stderr
        assert "db: -3" in result.stdout

    def test_splice_requires_a_clip_path(self, run_cli, initialized, tmp_path):
        clip = write_wav(tmp_path / "clip.wav", sample_rate=8000, duration=0.05)
        result = run_cli(
            ["edl", "add", "-s", str(initialized), "splice",
             "--param", f"clip={clip}", "--param", "position_sec=0.1", "--param", "mode=insert"],
            json_mode=True,
        )
        assert result.code == 0, result.stderr
        assert result.payload["op"]["clip"] == str(clip)

    def test_unknown_op_type_is_rejected_before_saving(self, run_cli, initialized):
        result = run_cli(["edl", "add", "-s", str(initialized), "not-an-op"])
        assert result.code == 1
        assert "알 수 없는 op type" in result.stderr
        assert _edl_json(initialized)["ops"] == []

    def test_missing_required_param_is_rejected(self, run_cli, initialized):
        result = run_cli(["edl", "add", "-s", str(initialized), "gain"])
        assert result.code == 1
        assert "필수 키 누락" in result.stderr
        assert _edl_json(initialized)["ops"] == []

    def test_add_without_init_exits_nonzero(self, run_cli, tmp_path):
        path = write_wav(tmp_path / "fresh.wav")
        result = run_cli(["edl", "add", "-s", str(path), "remove_dc"])
        assert result.code == 1
        assert "EDL이 초기화되지 않았습니다" in result.stderr


class TestList:
    def test_empty_edl_says_so(self, run_cli, initialized):
        result = run_cli(["edl", "list", "-s", str(initialized)])
        assert result.code == 0, result.stderr
        assert "ops 없음" in result.stdout

    def test_table_lists_every_op_with_one_based_index(self, run_cli, initialized):
        run_cli(["edl", "add", "-s", str(initialized), "remove_dc"])
        run_cli(["edl", "add", "-s", str(initialized), "gain", "--param", "db=-1"])
        result = run_cli(["edl", "list", "-s", str(initialized)])
        assert result.code == 0, result.stderr
        assert "# EDL ops (2)" in result.stdout
        assert "#\ttype\tparams" in result.stdout
        assert "1\tremove_dc\t" in result.stdout
        assert "2\tgain\tdb=-1" in result.stdout

    def test_json_payload_round_trips_ops(self, run_cli, initialized):
        run_cli(["edl", "add", "-s", str(initialized), "normalize", "--param", "peak_db=-2"])
        result = run_cli(["edl", "list", "-s", str(initialized)], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["n_ops"] == 1
        assert result.payload["ops"] == [{"type": "normalize", "peak_db": -2.0}]

    def test_list_without_init_exits_nonzero(self, run_cli, tmp_path):
        path = write_wav(tmp_path / "fresh.wav")
        result = run_cli(["edl", "list", "-s", str(path)])
        assert result.code == 1
        assert "EDL이 초기화되지 않았습니다" in result.stderr


class TestUndoRedo:
    def test_undo_removes_the_last_op_and_moves_the_snapshot_to_redo(self, run_cli, initialized):
        run_cli(["edl", "add", "-s", str(initialized), "remove_dc"])
        run_cli(["edl", "add", "-s", str(initialized), "gain", "--param", "db=-3"])
        assert _op_types(initialized) == ["remove_dc", "gain"]

        result = run_cli(["edl", "undo", "-s", str(initialized)])
        assert result.code == 0, result.stderr
        assert "undo 완료. 현재 op 수: 1" in result.stderr
        assert _op_types(initialized) == ["remove_dc"]
        assert len(edl_core.list_redo(initialized)) == 1

    def test_undo_without_history_warns(self, run_cli, initialized):
        result = run_cli(["edl", "undo", "-s", str(initialized)])
        assert result.code == 0, result.stderr
        assert "되돌릴 op이 없습니다" in result.stderr

    def test_undo_json_payload_reports_both_outcomes(self, run_cli, initialized):
        empty = run_cli(["edl", "undo", "-s", str(initialized)], json_mode=True)
        assert empty.code == 0
        assert empty.payload["undone"] is False
        assert empty.payload["reason"] == "history empty"

        run_cli(["edl", "add", "-s", str(initialized), "remove_dc"])
        done = run_cli(["edl", "undo", "-s", str(initialized)], json_mode=True)
        assert done.code == 0
        assert done.payload["undone"] is True
        assert done.payload["n_ops"] == 0

    def test_redo_restores_the_undone_op(self, run_cli, initialized):
        run_cli(["edl", "add", "-s", str(initialized), "remove_dc"])
        run_cli(["edl", "add", "-s", str(initialized), "gain", "--param", "db=-3"])
        run_cli(["edl", "undo", "-s", str(initialized)])

        result = run_cli(["edl", "redo", "-s", str(initialized)])
        assert result.code == 0, result.stderr
        assert "redo 완료. 현재 op 수: 2" in result.stderr
        assert _op_types(initialized) == ["remove_dc", "gain"]
        assert edl_core.list_redo(initialized) == []
        # The redo snapshot was moved back into history, so the state is undoable again.
        assert len(edl_core.list_history(initialized)) == 3
        assert run_cli(["edl", "undo", "-s", str(initialized)]).code == 0
        assert _op_types(initialized) == ["remove_dc"]

    def test_redo_without_redo_state_warns(self, run_cli, initialized):
        result = run_cli(["edl", "redo", "-s", str(initialized)])
        assert result.code == 0, result.stderr
        assert "redo할 op이 없습니다" in result.stderr

    def test_redo_json_payload_reports_both_outcomes(self, run_cli, initialized):
        empty = run_cli(["edl", "redo", "-s", str(initialized)], json_mode=True)
        assert empty.code == 0
        assert empty.payload["redone"] is False

        run_cli(["edl", "add", "-s", str(initialized), "remove_dc"])
        run_cli(["edl", "undo", "-s", str(initialized)])
        done = run_cli(["edl", "redo", "-s", str(initialized)], json_mode=True)
        assert done.code == 0
        assert done.payload["redone"] is True
        assert done.payload["n_ops"] == 1

    def test_new_op_after_undo_clears_the_redo_queue(self, run_cli, initialized):
        run_cli(["edl", "add", "-s", str(initialized), "remove_dc"])
        run_cli(["edl", "add", "-s", str(initialized), "gain", "--param", "db=-3"])
        run_cli(["edl", "undo", "-s", str(initialized)])
        assert len(edl_core.list_redo(initialized)) == 1

        run_cli(["edl", "add", "-s", str(initialized), "fade_in", "--param", "duration_sec=0.01"])
        assert _op_types(initialized) == ["remove_dc", "fade_in"]
        assert edl_core.list_redo(initialized) == []
        assert run_cli(["edl", "redo", "-s", str(initialized)]).code == 0
        assert _op_types(initialized) == ["remove_dc", "fade_in"]

    def test_undo_without_init_exits_nonzero(self, run_cli, tmp_path):
        path = write_wav(tmp_path / "fresh.wav")
        result = run_cli(["edl", "undo", "-s", str(path)])
        assert result.code == 1
        assert "EDL이 초기화되지 않았습니다" in result.stderr

    def test_redo_without_init_warns_but_does_not_write_a_workspace(self, run_cli, tmp_path):
        """`redo` skips the initialized check that `undo` performs; it warns instead."""
        path = write_wav(tmp_path / "fresh.wav")
        result = run_cli(["edl", "redo", "-s", str(path)])
        assert result.code == 0, result.stderr
        assert "redo할 op이 없습니다" in result.stderr
        assert not edl_core.edl_path(path).exists()

    def test_undo_undo_walks_back_two_ops(self, run_cli, initialized):
        for op_type, params in [
            ("remove_dc", []),
            ("gain", ["db=-1"]),
            ("gain", ["db=-2"]),
        ]:
            argv = ["edl", "add", "-s", str(initialized), op_type]
            for p in params:
                argv += ["--param", p]
            run_cli(argv)
        assert _op_types(initialized) == ["remove_dc", "gain", "gain"]

        run_cli(["edl", "undo", "-s", str(initialized)])
        run_cli(["edl", "undo", "-s", str(initialized)])
        assert _op_types(initialized) == ["remove_dc"]


class TestRender:
    def test_render_applies_every_op_and_writes_output(self, run_cli, initialized, tmp_path):
        run_cli(["edl", "add", "-s", str(initialized), "trim",
                 "--param", "start_sec=0.1", "--param", "end_sec=0.4"])
        run_cli(["edl", "add", "-s", str(initialized), "gain", "--param", "db=-6"])
        out = tmp_path / "rendered.wav"

        result = run_cli(["edl", "render", "-s", str(initialized), "-o", str(out)], json_mode=True)
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["command"] == "edl render"
        assert payload["output_path"] == str(out)
        assert payload["n_ops"] == 2
        assert payload["input_duration_sec"] == pytest.approx(0.5, abs=1e-3)
        assert payload["output_duration_sec"] == pytest.approx(0.3, abs=1e-3)
        assert payload["sample_rate"] == 8000
        assert payload["channels"] == 2
        assert payload["edl_path"] == str(edl_core.edl_path(initialized))
        assert out.exists()

    def test_rendered_audio_is_quieter_after_gain(self, run_cli, initialized, tmp_path):
        import soundfile as sf

        run_cli(["edl", "add", "-s", str(initialized), "gain", "--param", "db=-6"])
        out = tmp_path / "gain.wav"
        assert run_cli(["edl", "render", "-s", str(initialized), "-o", str(out)]).code == 0
        audio, _ = sf.read(str(out), always_2d=True)
        assert np.max(np.abs(audio)) == pytest.approx(0.3 * 10 ** (-6 / 20), rel=0.05)

    @pytest.mark.parametrize("op_type,params", [
        ("cut_region", ["start_sec=0.1", "end_sec=0.2"]),
        ("trim", ["start_sec=0.05", "end_sec=0.45"]),
        ("splice", ["position_sec=0.2", "mode=insert"]),
        ("fade_in", ["duration_sec=0.05"]),
        ("fade_out", ["duration_sec=0.05"]),
        ("remove_dc", []),
        ("normalize", ["peak_db=-1.0"]),
        ("gain", ["db=-3.0"]),
        ("gate", ["threshold_db=-45.0"]),
        ("loudness_normalize", ["target_lufs=-14.0", "max_true_peak_dbtp=-1.0"]),
    ])
    def test_every_op_type_renders(self, run_cli, initialized, tmp_path, op_type, params):
        if op_type == "splice":
            clip = write_wav(tmp_path / "clip.wav", sample_rate=8000, duration=0.05)
            params = [f"clip={clip}", *params]
        argv = ["edl", "add", "-s", str(initialized), op_type]
        for p in params:
            argv += ["--param", p]
        assert run_cli(argv).code == 0, op_type

        out = tmp_path / f"{op_type}.wav"
        result = run_cli(["edl", "render", "-s", str(initialized), "-o", str(out)], json_mode=True)
        assert result.code == 0, f"{op_type}: {result.stderr}"
        assert out.exists()
        assert result.payload["n_ops"] == 1

    def test_no_dc_remove_flag_lifts_dc_offset(self, run_cli, tmp_path):
        import soundfile as sf

        src = write_wav(tmp_path / "dc.wav", sample_rate=8000, duration=0.3, dc_offset=0.05)
        assert run_cli(["edl", "init", str(src)]).code == 0
        run_cli(["edl", "add", "-s", str(src), "remove_dc"])
        out = tmp_path / "dc_removed.wav"
        assert run_cli(["edl", "render", "-s", str(src), "-o", str(out)]).code == 0
        audio, _ = sf.read(str(out), always_2d=True)
        assert abs(float(np.mean(audio))) < 0.01

    def test_source_mutation_is_detected_by_sha_verification(self, run_cli, initialized):
        run_cli(["edl", "add", "-s", str(initialized), "gain", "--param", "db=-1"])
        # Rewrite the source so its hash no longer matches the EDL snapshot.
        write_wav(initialized, sample_rate=8000, duration=0.5, frequency=880.0)
        result = run_cli(["edl", "render", "-s", str(initialized), "-o",
                          str(initialized.parent / "out.wav")])
        assert result.code == 1
        assert "source 파일이 변경됨" in result.stderr

    def test_no_verify_skips_the_hash_check(self, run_cli, initialized, tmp_path):
        run_cli(["edl", "add", "-s", str(initialized), "gain", "--param", "db=-1"])
        write_wav(initialized, sample_rate=8000, duration=0.5, frequency=880.0)
        out = tmp_path / "forced.wav"
        result = run_cli(["edl", "render", "-s", str(initialized), "-o", str(out), "--no-verify"])
        assert result.code == 0, result.stderr
        assert out.exists()

    def test_failing_op_is_reported_with_its_index(self, run_cli, initialized, tmp_path):
        clip = write_wav(tmp_path / "clip.wav", sample_rate=16000, duration=0.05)
        result = run_cli(
            ["edl", "add", "-s", str(initialized), "splice",
             "--param", f"clip={clip}", "--param", "position_sec=0.1", "--param", "mode=insert"],
        )
        assert result.code == 0, result.stderr
        render = run_cli(["edl", "render", "-s", str(initialized), "-o", str(tmp_path / "x.wav")])
        assert render.code == 1
        assert "op #1 (splice) 실패" in render.stderr
        assert "sample rate 불일치" in render.stderr

    def test_render_without_init_exits_nonzero(self, run_cli, tmp_path):
        path = write_wav(tmp_path / "fresh.wav")
        result = run_cli(["edl", "render", "-s", str(path), "-o", str(tmp_path / "o.wav")])
        assert result.code == 1
        assert "EDL이 초기화되지 않았습니다" in result.stderr


class TestStatus:
    def test_uninitialized_workspace_reports_it(self, run_cli, tmp_path):
        path = write_wav(tmp_path / "fresh.wav")
        result = run_cli(["edl", "status", "-s", str(path)])
        assert result.code == 0, result.stderr
        assert "초기화되지 않음" in result.stdout

    def test_uninitialized_json_payload_is_machine_readable(self, run_cli, tmp_path):
        path = write_wav(tmp_path / "fresh.wav")
        result = run_cli(["edl", "status", "-s", str(path)], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["command"] == "edl status"
        assert result.payload["initialized"] is False
        assert "n_ops" not in result.payload

    def test_status_counts_ops_history_and_redo(self, run_cli, initialized):
        run_cli(["edl", "add", "-s", str(initialized), "remove_dc"])
        run_cli(["edl", "add", "-s", str(initialized), "gain", "--param", "db=-3"])
        run_cli(["edl", "undo", "-s", str(initialized)])
        result = run_cli(["edl", "status", "-s", str(initialized)], json_mode=True)
        assert result.code == 0, result.stderr
        payload = result.payload
        assert payload["initialized"] is True
        assert payload["n_ops"] == 1
        assert payload["history_depth"] == 2
        assert payload["redo_depth"] == 1
        assert payload["edl_path"] == str(edl_core.edl_path(initialized))
        assert payload["workspace"] == str(edl_core.workspace_dir(initialized))
        assert payload["duration_sec"] == pytest.approx(0.5, abs=1e-3)

    def test_human_status_shows_each_counter(self, run_cli, initialized):
        run_cli(["edl", "add", "-s", str(initialized), "remove_dc"])
        result = run_cli(["edl", "status", "-s", str(initialized)])
        assert result.code == 0, result.stderr
        assert "Ops:           1" in result.stdout
        assert "History depth: 2" in result.stdout
        assert "Redo depth:    0" in result.stdout
        assert str(edl_core.workspace_dir(initialized)) in result.stdout


class TestClear:
    def test_clear_drops_ops_but_keeps_history(self, run_cli, initialized):
        run_cli(["edl", "add", "-s", str(initialized), "remove_dc"])
        run_cli(["edl", "add", "-s", str(initialized), "gain", "--param", "db=-3"])
        history_before = len(edl_core.list_history(initialized))

        result = run_cli(["edl", "clear", "-s", str(initialized)])
        assert result.code == 0, result.stderr
        assert "모든 op 삭제 (history는 유지)" in result.stderr
        assert _edl_json(initialized)["ops"] == []
        assert len(edl_core.list_history(initialized)) == history_before + 1

    def test_clear_is_undoable_back_to_the_previous_op_list(self, run_cli, initialized):
        run_cli(["edl", "add", "-s", str(initialized), "remove_dc"])
        run_cli(["edl", "clear", "-s", str(initialized)])
        assert _op_types(initialized) == []
        assert run_cli(["edl", "undo", "-s", str(initialized)]).code == 0
        assert _op_types(initialized) == ["remove_dc"]

    def test_clear_json_payload_reports_zero_ops(self, run_cli, initialized):
        run_cli(["edl", "add", "-s", str(initialized), "remove_dc"])
        result = run_cli(["edl", "clear", "-s", str(initialized)], json_mode=True)
        assert result.code == 0, result.stderr
        assert result.payload["command"] == "edl clear"
        assert result.payload["n_ops"] == 0
        assert result.payload["source"] == str(initialized.resolve())

    def test_clear_without_init_exits_nonzero(self, run_cli, tmp_path):
        path = write_wav(tmp_path / "fresh.wav")
        result = run_cli(["edl", "clear", "-s", str(path)])
        assert result.code == 1
        assert "EDL이 초기화되지 않았습니다" in result.stderr


class TestBareInvocation:
    def test_no_action_prints_help(self, run_cli):
        result = run_cli(["edl"])
        assert result.code == 0
        assert "usage: audioman edl" in result.stdout
