# tests/unit/test_edl.py — 비파괴 EDL 워크플로우

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audioman.core import edl as edl_core


@pytest.fixture
def long_wav(tmp_path, sample_rate):
    """3초짜리 신호: 1초 사인 + 1초 무음 + 1초 노이즈."""
    sr = sample_rate
    t = np.arange(sr) / sr
    sine = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    silence = np.zeros(sr, dtype=np.float32)
    rng = np.random.RandomState(0)
    noise = (rng.randn(sr) * 0.2).astype(np.float32)
    audio = np.concatenate([sine, silence, noise])
    path = tmp_path / "long.wav"
    sf.write(str(path), audio, sr, subtype="PCM_24")
    return path


class TestInitAndLoad:
    def test_init_creates_edl_with_metadata(self, long_wav, sample_rate):
        edl = edl_core.init_edl(long_wav)
        assert edl.source == str(long_wav.resolve())
        assert edl.sample_rate == sample_rate
        assert edl.channels == 1
        assert abs(edl.duration_sec - 3.0) < 0.01
        assert edl.ops == []
        assert len(edl.source_sha256) == 64

    def test_save_load_roundtrip(self, long_wav, tmp_path):
        edl = edl_core.init_edl(long_wav)
        edl_core.add_op(edl, {"type": "fade_in", "duration_sec": 0.1})
        edl_core.add_op(edl, {"type": "gain", "db": -3.0})

        out = tmp_path / "edit.json"
        edl_core.save_edl(edl, out)
        loaded = edl_core.load_edl(out)
        assert loaded.ops == edl.ops
        assert loaded.source_sha256 == edl.source_sha256

    def test_init_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            edl_core.init_edl(tmp_path / "nope.wav")


class TestValidation:
    def test_unknown_op_type(self):
        with pytest.raises(ValueError, match="알 수 없는 op type"):
            edl_core.validate_op({"type": "explode"})

    def test_missing_required_keys(self):
        with pytest.raises(ValueError, match="필수 키 누락"):
            edl_core.validate_op({"type": "cut_region", "start_sec": 0})

    def test_valid_op_passes(self):
        edl_core.validate_op({"type": "cut_region", "start_sec": 0, "end_sec": 1.0})
        edl_core.validate_op({"type": "fade_in", "duration_sec": 0.1})
        edl_core.validate_op({"type": "trim_silence"})
        edl_core.validate_op({"type": "normalize"})


class TestRender:
    def test_render_no_ops_matches_source(self, long_wav, tmp_path):
        edl = edl_core.init_edl(long_wav)
        out = tmp_path / "out.wav"
        result = edl_core.render_edl(edl, out)
        assert result.n_ops == 0
        assert abs(result.input_duration_sec - result.output_duration_sec) < 1e-6

        original, _ = sf.read(str(long_wav), always_2d=True)
        rendered, _ = sf.read(str(out), always_2d=True)
        assert original.shape == rendered.shape
        # PCM_24 양자화 오차 허용
        assert np.allclose(original, rendered, atol=1e-4)

    def test_render_cut_region(self, long_wav, tmp_path, sample_rate):
        edl = edl_core.init_edl(long_wav)
        edl_core.add_op(edl, {"type": "cut_region", "start_sec": 1.0, "end_sec": 2.0})
        out = tmp_path / "cut.wav"
        result = edl_core.render_edl(edl, out)
        # 1초 삭제 → 2초 남음
        assert abs(result.output_duration_sec - 2.0) < 0.01

    def test_render_chain_of_ops(self, long_wav, tmp_path):
        edl = edl_core.init_edl(long_wav)
        # 가운데 무음 자르고 → fade_in → gain
        edl_core.add_op(edl, {"type": "cut_region", "start_sec": 1.0, "end_sec": 2.0})
        edl_core.add_op(edl, {"type": "fade_in", "duration_sec": 0.1})
        edl_core.add_op(edl, {"type": "gain", "db": -6.0})
        out = tmp_path / "chain.wav"
        result = edl_core.render_edl(edl, out)
        assert result.n_ops == 3
        assert abs(result.output_duration_sec - 2.0) < 0.01

        # gain -6dB로 줄였으니 원본보다 작아야 함
        original, _ = sf.read(str(long_wav), always_2d=True)
        rendered, _ = sf.read(str(out), always_2d=True)
        original_peak = float(np.max(np.abs(original)))
        rendered_peak = float(np.max(np.abs(rendered)))
        # -6dB = 약 0.5x. 허용 오차로 0.55x 이내
        assert rendered_peak < original_peak * 0.55

    def test_render_invalid_op_raises(self, long_wav, tmp_path):
        edl = edl_core.init_edl(long_wav)
        # validate_op을 우회해 직접 ops에 주입 → render에서 _apply_op이 잡아냄
        edl.ops.append({"type": "explode"})
        with pytest.raises(RuntimeError, match="op #1"):
            edl_core.render_edl(edl, tmp_path / "x.wav")

    def test_source_modification_detected(self, long_wav, tmp_path, sample_rate):
        edl = edl_core.init_edl(long_wav)
        # source 파일 변조
        sr = sample_rate
        new_audio = np.ones(sr, dtype=np.float32) * 0.1
        sf.write(str(long_wav), new_audio, sr, subtype="PCM_24")

        with pytest.raises(ValueError, match="source 파일이 변경됨"):
            edl_core.render_edl(edl, tmp_path / "x.wav")

    def test_no_verify_skips_check(self, long_wav, tmp_path, sample_rate):
        edl = edl_core.init_edl(long_wav)
        sf.write(str(long_wav), np.ones(sample_rate, dtype=np.float32) * 0.1, sample_rate, subtype="PCM_24")
        # no_verify=True면 통과
        result = edl_core.render_edl(edl, tmp_path / "x.wav", verify_source=False)
        assert result.n_ops == 0


class TestWorkspaceUndoRedo:
    def test_workspace_paths_under_source_dir(self, long_wav):
        ws = edl_core.workspace_dir(long_wav)
        assert ws.parent.name == ".audioman"
        assert ws.parent.parent == long_wav.parent

    def test_undo_redo_roundtrip(self, long_wav):
        # init → 두 op 추가 → undo → redo
        edl = edl_core.init_edl(long_wav)
        edl_path = edl_core.edl_path(long_wav)
        edl_core.workspace_dir(long_wav).mkdir(parents=True, exist_ok=True)
        edl_core.save_edl(edl, edl_path)
        edl_core.snapshot_history(edl, long_wav)

        edl_core.add_op(edl, {"type": "fade_in", "duration_sec": 0.1})
        edl_core.save_edl(edl, edl_path)
        edl_core.snapshot_history(edl, long_wav)

        edl_core.add_op(edl, {"type": "gain", "db": -6.0})
        edl_core.save_edl(edl, edl_path)
        edl_core.snapshot_history(edl, long_wav)

        assert len(edl_core.list_history(long_wav)) == 3
        assert len(edl_core.list_redo(long_wav)) == 0

        # undo 1번 → ops 1개로
        rolled = edl_core.undo(long_wav)
        assert rolled is not None
        assert len(rolled.ops) == 1
        assert rolled.ops[0]["type"] == "fade_in"
        assert len(edl_core.list_redo(long_wav)) == 1

        # redo → ops 2개로 복원
        forward = edl_core.redo(long_wav)
        assert forward is not None
        assert len(forward.ops) == 2
        assert forward.ops[1]["type"] == "gain"
        assert len(edl_core.list_redo(long_wav)) == 0

    def test_undo_with_no_history(self, long_wav):
        edl = edl_core.init_edl(long_wav)
        edl_path = edl_core.edl_path(long_wav)
        edl_core.workspace_dir(long_wav).mkdir(parents=True, exist_ok=True)
        edl_core.save_edl(edl, edl_path)
        edl_core.snapshot_history(edl, long_wav)
        # history가 1개뿐 → undo 불가
        result = edl_core.undo(long_wav)
        assert result is None

    def test_new_op_after_undo_clears_redo(self, long_wav):
        edl = edl_core.init_edl(long_wav)
        edl_path = edl_core.edl_path(long_wav)
        edl_core.workspace_dir(long_wav).mkdir(parents=True, exist_ok=True)
        edl_core.save_edl(edl, edl_path)
        edl_core.snapshot_history(edl, long_wav)

        edl_core.add_op(edl, {"type": "fade_in", "duration_sec": 0.1})
        edl_core.save_edl(edl, edl_path)
        edl_core.snapshot_history(edl, long_wav)

        edl_core.add_op(edl, {"type": "gain", "db": -6.0})
        edl_core.save_edl(edl, edl_path)
        edl_core.snapshot_history(edl, long_wav)

        edl_core.undo(long_wav)
        assert len(edl_core.list_redo(long_wav)) == 1

        # 새 op 추가하면 redo는 비워져야 함 (다른 길로 갔으므로)
        edl = edl_core.load_edl(edl_path)
        edl_core.add_op(edl, {"type": "normalize"})
        edl_core.save_edl(edl, edl_path)
        edl_core.snapshot_history(edl, long_wav)  # default clear_redo=True
        assert len(edl_core.list_redo(long_wav)) == 0


class TestParamSerialization:
    def test_int_float_bool_roundtrip(self, long_wav, tmp_path):
        edl = edl_core.init_edl(long_wav)
        edl_core.add_op(edl, {
            "type": "process",
            "plugin": "denoise",
            "params": {"reduction_db": 12.5, "adaptive": True, "passes": 2},
        })
        path = tmp_path / "edit.json"
        edl_core.save_edl(edl, path)
        loaded = edl_core.load_edl(path)
        params = loaded.ops[0]["params"]
        assert params["reduction_db"] == 12.5
        assert params["adaptive"] is True
        assert params["passes"] == 2
