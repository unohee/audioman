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


# ---------------------------------------------------------------------------
# Additional op coverage: every op type through render, plus validation edges.
# ---------------------------------------------------------------------------


class TestEDLDataclassEdges:
    def test_version_too_new_rejected(self):
        with pytest.raises(ValueError, match="지원하지 않는 EDL version"):
            edl_core.EDL.from_dict({
                "version": edl_core.EDL_VERSION + 1,
                "source": "/a.wav", "source_sha256": "x",
                "sample_rate": 48000, "channels": 2, "duration_sec": 1.0,
            })

    def test_from_dict_defaults(self):
        edl = edl_core.EDL.from_dict({
            "source": "/a.wav", "source_sha256": "abc",
            "sample_rate": "48000", "channels": "2", "duration_sec": "1.5",
        })
        assert edl.version == 1
        assert edl.sample_rate == 48000
        assert edl.channels == 2
        assert edl.ops == []
        assert edl.created_at == ""

    def test_to_dict_roundtrip(self):
        edl = edl_core.EDL(
            source="/a.wav", source_sha256="h", sample_rate=44100, channels=1,
            duration_sec=2.0, ops=[{"type": "normalize"}],
            created_at="2026-01-01T00:00:00+00:00", modified_at="2026-01-02T00:00:00+00:00",
        )
        d = edl.to_dict()
        assert d["version"] == edl_core.EDL_VERSION
        assert d["ops"] == [{"type": "normalize"}]
        assert edl_core.EDL.from_dict(d).ops == [{"type": "normalize"}]


class TestValidationEdges:
    def test_non_dict_op_rejected(self):
        with pytest.raises(ValueError, match="op은 dict여야 합니다"):
            edl_core.validate_op(["not", "a", "dict"])

    def test_missing_type_key_rejected(self):
        with pytest.raises(ValueError, match="알 수 없는 op type"):
            edl_core.validate_op({"start_sec": 0, "end_sec": 1})

    def test_add_op_stores_copy_and_updates_modified(self, long_wav):
        edl = edl_core.init_edl(long_wav)
        before = edl.modified_at
        op = {"type": "gain", "db": -1.0}
        edl_core.add_op(edl, op)
        op["db"] = -99.0  # mutating the caller's dict must not affect the EDL
        assert edl.ops[0]["db"] == -1.0
        assert edl.modified_at >= before

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="EDL 파일 없음"):
            edl_core.load_edl(tmp_path / "nope.json")


class TestRenderAllOpTypes:
    """Each op type routed through _apply_op / render_edl."""

    @pytest.fixture
    def src(self, tmp_path):
        sr = 48000
        t = np.arange(sr * 2) / sr
        mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.stack([mono, mono])
        path = tmp_path / "src.wav"
        sf.write(str(path), stereo.T, sr, subtype="FLOAT")
        return path

    def _render(self, src, tmp_path, ops, name="out.wav"):
        edl = edl_core.init_edl(src)
        for op in ops:
            edl_core.add_op(edl, op)
        out = tmp_path / name
        return edl_core.render_edl(edl, out), out

    def test_trim_op(self, src, tmp_path):
        result, out = self._render(src, tmp_path, [{"type": "trim", "start_sec": 0.5, "end_sec": 1.5}])
        assert result.output_duration_sec == pytest.approx(1.0, abs=0.01)

    def test_trim_silence_op(self, src, tmp_path):
        # source is continuous tone → trim_silence returns nearly the whole file
        result, _ = self._render(src, tmp_path, [{"type": "trim_silence"}])
        assert result.output_duration_sec == pytest.approx(2.0, abs=0.05)

    def test_fade_in_and_out_ops(self, src, tmp_path):
        result, out = self._render(src, tmp_path, [
            {"type": "fade_in", "duration_sec": 0.2},
            {"type": "fade_out", "duration_sec": 0.2, "curve": "cosine"},
        ])
        rendered, _ = sf.read(str(out), always_2d=True)
        assert abs(rendered[0, 0]) < 1e-4
        assert abs(rendered[-1, 0]) < 1e-4

    def test_pad_ms_and_sec_ops(self, src, tmp_path):
        result, out = self._render(src, tmp_path, [
            {"type": "pad", "head_ms": 250.0, "tail_sec": 0.5},
        ])
        assert result.output_duration_sec == pytest.approx(2.75, abs=0.02)
        rendered, _ = sf.read(str(out), always_2d=True)
        assert np.max(np.abs(rendered[:12000, 0])) == 0.0

    def test_remove_dc_op(self, tmp_path, src):
        # add DC to the source then remove it
        data, sr = sf.read(str(src), always_2d=True)
        data = data + 0.05
        sf.write(str(src), data, sr, subtype="FLOAT")
        result, out = self._render(src, tmp_path, [{"type": "remove_dc"}])
        rendered, _ = sf.read(str(out), always_2d=True)
        assert abs(float(np.mean(rendered[:, 0]))) < 1e-4

    def test_loudness_normalize_op(self, src, tmp_path):
        from audioman.core.loudness import integrated_lufs
        result, out = self._render(src, tmp_path, [
            {"type": "loudness_normalize", "target_lufs": -23.0, "max_true_peak_dbtp": -1.0},
        ])
        rendered, sr = sf.read(str(out), always_2d=True)
        lufs = integrated_lufs(rendered.T.astype(np.float32), sr)
        assert lufs == pytest.approx(-23.0, abs=1.0)

    def test_normalize_op_defaults_to_minus_1dbfs(self, src, tmp_path):
        _, out = self._render(src, tmp_path, [{"type": "normalize"}])
        rendered, _ = sf.read(str(out), always_2d=True)
        assert float(np.max(np.abs(rendered))) == pytest.approx(10 ** (-1.0 / 20.0), abs=0.01)

    def test_normalize_op_rms(self, src, tmp_path):
        _, out = self._render(src, tmp_path, [{"type": "normalize", "target_rms_db": -20.0}])
        rendered, _ = sf.read(str(out), always_2d=True)
        rms = float(np.sqrt(np.mean(rendered ** 2)))
        assert rms == pytest.approx(10 ** (-20.0 / 20.0), abs=0.01)

    def test_gate_op(self, tmp_path):
        sr = 48000
        audio = np.zeros((2, sr), dtype=np.float32)
        t = np.arange(sr // 2) / sr
        audio[:, sr // 2:] = 0.4 * np.sin(2 * np.pi * 440 * t)
        path = tmp_path / "gated.wav"
        sf.write(str(path), audio.T, sr, subtype="FLOAT")
        result, out = self._render(path, tmp_path, [{"type": "gate", "threshold_db": -30.0}])
        rendered, _ = sf.read(str(out), always_2d=True)
        assert float(np.sqrt(np.mean(rendered[:sr // 4, 0] ** 2))) < 0.01

    def test_splice_op_insert(self, src, tmp_path):
        sr = 48000
        clip = np.full((2, sr // 2), 0.2, dtype=np.float32)
        clip_path = tmp_path / "clip.wav"
        sf.write(str(clip_path), clip.T, sr, subtype="FLOAT")
        result, _ = self._render(src, tmp_path, [
            {"type": "splice", "clip": str(clip_path), "position_sec": 1.0, "mode": "insert"},
        ])
        assert result.output_duration_sec == pytest.approx(2.5, abs=0.02)

    def test_splice_op_overwrite(self, src, tmp_path):
        sr = 48000
        clip = np.full((2, sr // 2), 0.2, dtype=np.float32)
        clip_path = tmp_path / "clip.wav"
        sf.write(str(clip_path), clip.T, sr, subtype="FLOAT")
        result, _ = self._render(src, tmp_path, [
            {"type": "splice", "clip": str(clip_path), "position_sec": 0.5, "mode": "overwrite"},
        ])
        assert result.output_duration_sec == pytest.approx(2.0, abs=0.02)

    def test_splice_sample_rate_mismatch_raises(self, src, tmp_path):
        clip_path = tmp_path / "clip44.wav"
        sf.write(str(clip_path), np.zeros((100, 2), dtype=np.float32), 44100, subtype="FLOAT")
        edl = edl_core.init_edl(src)
        edl_core.add_op(edl, {
            "type": "splice", "clip": str(clip_path), "position_sec": 0.1, "mode": "insert",
        })
        with pytest.raises(RuntimeError, match="sample rate 불일치"):
            edl_core.render_edl(edl, tmp_path / "out.wav")

    def test_splice_mono_clip_expanded_to_stereo(self, src, tmp_path):
        clip_path = tmp_path / "mono.wav"
        sf.write(str(clip_path), np.full(24000, 0.2, dtype=np.float32), 48000, subtype="FLOAT")
        result, out = self._render(src, tmp_path, [
            {"type": "splice", "clip": str(clip_path), "position_sec": 0.5, "mode": "insert"},
        ])
        rendered, _ = sf.read(str(out), always_2d=True)
        assert rendered.shape[1] == 2
        assert result.output_duration_sec == pytest.approx(2.5, abs=0.02)

    def test_splice_stereo_clip_into_mono_source(self, tmp_path):
        sr = 48000
        src = tmp_path / "mono_src.wav"
        sf.write(str(src), np.zeros(sr, dtype=np.float32), sr, subtype="FLOAT")
        clip_path = tmp_path / "stereo.wav"
        sf.write(str(clip_path), np.full((12000, 2), 0.3, dtype=np.float32), sr, subtype="FLOAT")
        result, _ = self._render(src, tmp_path, [
            {"type": "splice", "clip": str(clip_path), "position_sec": 0.25, "mode": "insert"},
        ])
        assert result.output_duration_sec == pytest.approx(1.25, abs=0.02)

    def test_splice_3ch_conversion_rejected(self, tmp_path):
        sr = 48000
        src = tmp_path / "three.wav"
        sf.write(str(src), np.zeros((1000, 3), dtype=np.float32), sr, subtype="FLOAT")
        clip_path = tmp_path / "clip.wav"
        sf.write(str(clip_path), np.zeros(1000, dtype=np.float32), sr, subtype="FLOAT")
        edl = edl_core.init_edl(src)
        edl_core.add_op(edl, {
            "type": "splice", "clip": str(clip_path), "position_sec": 0.0, "mode": "insert",
        })
        with pytest.raises(RuntimeError, match="채널 변환 불가"):
            edl_core.render_edl(edl, tmp_path / "out.wav")

    def test_cut_region_with_crossfade_ms(self, src, tmp_path):
        result, _ = self._render(src, tmp_path, [
            {"type": "cut_region", "start_sec": 0.5, "end_sec": 1.0, "crossfade_ms": 10.0},
        ])
        assert result.output_duration_sec == pytest.approx(1.5, abs=0.02)

    def test_render_result_to_dict(self, src, tmp_path):
        result, _ = self._render(src, tmp_path, [{"type": "normalize"}], name="dict.wav")
        d = result.to_dict()
        assert d["n_ops"] == 1
        assert d["sample_rate"] == 48000
        assert d["channels"] == 2
        assert d["edl_path"] is None


class TestRenderErrors:
    def test_missing_source_raises(self, tmp_path):
        edl = edl_core.EDL(
            source=str(tmp_path / "gone.wav"), source_sha256="x",
            sample_rate=48000, channels=2, duration_sec=1.0,
        )
        with pytest.raises(FileNotFoundError, match="EDL source 파일 없음"):
            edl_core.render_edl(edl, tmp_path / "o.wav")

    def test_sample_rate_mismatch_raises(self, tmp_path):
        sr = 48000
        src = tmp_path / "sr.wav"
        sf.write(str(src), np.zeros(sr, dtype=np.float32), sr, subtype="FLOAT")
        edl = edl_core.init_edl(src)
        edl.sample_rate = 44100  # tamper
        with pytest.raises(ValueError, match="sample rate 불일치"):
            edl_core.render_edl(edl, tmp_path / "o.wav")

    def test_edl_path_recorded(self, tmp_path):
        src = tmp_path / "p.wav"
        sf.write(str(src), np.zeros(1000, dtype=np.float32), 48000, subtype="FLOAT")
        edl = edl_core.init_edl(src)
        result = edl_core.render_edl(edl, tmp_path / "o.wav", edl_path=tmp_path / "e.json")
        assert result.edl_path == str(tmp_path / "e.json")


class TestHistoryHelpers:
    def test_next_index_skips_non_numeric_stems(self, long_wav):
        hist = edl_core.history_dir(long_wav)
        hist.mkdir(parents=True, exist_ok=True)
        (hist / "0003.json").write_text("{}")
        (hist / "README.json").write_text("{}")
        assert edl_core._next_index(hist) == 4

    def test_list_sorted_returns_empty_when_missing(self, long_wav):
        assert edl_core.list_history(long_wav) == []
        assert edl_core.list_redo(long_wav) == []

    def test_redo_without_snapshots_is_none(self, long_wav):
        assert edl_core.redo(long_wav) is None

    def test_snapshot_clear_redo_false_keeps_redo(self, long_wav):
        edl = edl_core.init_edl(long_wav)
        edl_core.snapshot_history(edl, long_wav)
        rd = edl_core.redo_dir(long_wav)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "0001.json").write_text("{}")
        edl_core.snapshot_history(edl, long_wav, clear_redo=False)
        assert len(edl_core.list_redo(long_wav)) == 1


class TestPadVariants:
    @pytest.fixture
    def src(self, tmp_path):
        sr = 48000
        path = tmp_path / "pad_src.wav"
        sf.write(str(path), np.full((sr, 2), 0.1, dtype=np.float32), sr, subtype="FLOAT")
        return path

    def _render(self, src, tmp_path, ops):
        edl = edl_core.init_edl(src)
        for op in ops:
            edl_core.add_op(edl, op)
        out = tmp_path / "pad_out.wav"
        return edl_core.render_edl(edl, out), out

    def test_pad_head_sec_and_tail_ms(self, src, tmp_path):
        # exercises both the head_sec branch and the tail_ms fallback branch
        result, out = self._render(src, tmp_path, [
            {"type": "pad", "head_sec": 0.25, "tail_ms": 500.0},
        ])
        assert result.output_duration_sec == pytest.approx(1.75, abs=0.02)
        rendered, _ = sf.read(str(out), always_2d=True)
        assert np.max(np.abs(rendered[:12000, 0])) == 0.0
        assert np.max(np.abs(rendered[-24000:, 0])) == 0.0


class TestPluginOpsInEDL:
    """`process` and `chain` ops — registry/wrapper faked (no VST3 host)."""

    @pytest.fixture
    def src(self, tmp_path):
        sr = 48000
        path = tmp_path / "plugin_src.wav"
        sf.write(str(path), np.full((sr, 2), 0.4, dtype=np.float32), sr, subtype="FLOAT")
        return path

    def _install_fakes(self, monkeypatch, missing=None):
        from types import SimpleNamespace
        from audioman.core import registry as registry_mod
        from audioman.plugins import vst3 as vst3_mod

        metas = {
            "gain": SimpleNamespace(short_name="gain", path="/g.vst3"),
            "pad0": SimpleNamespace(short_name="pad0", path="/p.vst3"),
        }
        made = []

        class _W:
            def __init__(self, path):
                self.path = path
                self.params = None
                made.append(self)

            def load(self):
                pass

            def set_parameters(self, p):
                self.params = p

            def process(self, audio, sr):
                return (audio * 0.5).astype(audio.dtype)

        def _get(name):
            if name == missing:
                return None
            return metas.get(name)

        monkeypatch.setattr(registry_mod, "get_registry",
                            lambda: SimpleNamespace(get=_get))
        monkeypatch.setattr(vst3_mod, "VST3PluginWrapper", _W)
        return made

    def test_process_op_with_params_and_passes(self, monkeypatch, tmp_path, src):
        made = self._install_fakes(monkeypatch)
        edl = edl_core.init_edl(src)
        edl_core.add_op(edl, {
            "type": "process", "plugin": "gain", "params": {"g": 2.0}, "passes": 3,
        })
        out = tmp_path / "process_out.wav"
        result = edl_core.render_edl(edl, out)
        assert result.n_ops == 1
        assert made[0].params == {"g": 2.0}
        rendered, _ = sf.read(str(out), always_2d=True)
        # 3 passes of x0.5 applied to the same running buffer
        np.testing.assert_allclose(rendered, 0.4 * 0.125, atol=1e-5)

    def test_process_op_without_params(self, monkeypatch, tmp_path, src):
        made = self._install_fakes(monkeypatch)
        edl = edl_core.init_edl(src)
        edl_core.add_op(edl, {"type": "process", "plugin": "gain"})
        edl_core.render_edl(edl, tmp_path / "o.wav")
        assert made[0].params is None

    def test_process_op_unknown_plugin_raises(self, monkeypatch, tmp_path, src):
        self._install_fakes(monkeypatch, missing="nope")
        edl = edl_core.init_edl(src)
        edl_core.add_op(edl, {"type": "process", "plugin": "nope"})
        with pytest.raises(RuntimeError, match="플러그인을 찾을 수 없음"):
            edl_core.render_edl(edl, tmp_path / "o.wav")

    def test_chain_op_applies_each_step(self, monkeypatch, tmp_path, src):
        made = self._install_fakes(monkeypatch)
        edl = edl_core.init_edl(src)
        edl_core.add_op(edl, {
            "type": "chain",
            "steps": [
                {"plugin": "gain", "params": {"g": 1.0}},
                {"plugin": "pad0"},
            ],
        })
        out = tmp_path / "chain_out.wav"
        edl_core.render_edl(edl, out)
        assert len(made) == 2
        assert made[0].params == {"g": 1.0}
        assert made[1].params is None
        rendered, _ = sf.read(str(out), always_2d=True)
        np.testing.assert_allclose(rendered, 0.4 * 0.25, atol=1e-5)

    def test_chain_op_unknown_plugin_raises(self, monkeypatch, tmp_path, src):
        self._install_fakes(monkeypatch, missing="ghost")
        edl = edl_core.init_edl(src)
        edl_core.add_op(edl, {"type": "chain", "steps": [{"plugin": "ghost"}]})
        with pytest.raises(RuntimeError, match="chain 플러그인 없음"):
            edl_core.render_edl(edl, tmp_path / "o.wav")


class TestNextIndexFreshDirectory:
    def test_missing_dir_returns_one(self, tmp_path):
        assert edl_core._next_index(tmp_path / "does_not_exist") == 1
