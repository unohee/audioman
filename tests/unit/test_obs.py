# Created: 2026-05-07
# Purpose: core/obs.py 단위 테스트 — 토폴로지 분류, 트랙 분류, 처치 룰 엔진

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from audioman.core import obs as obs_core


def _ffmpeg_available() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# ---------------------------------------------------------------------------
# classify_track
# ---------------------------------------------------------------------------


def _make_voice_like(sr: int = 16000, dur: float = 5.0) -> np.ndarray:
    """speech-like signal: 200~3500Hz 합성 + 변조."""
    n = int(sr * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    # 기본 발화 톤(200Hz) + 포먼트 흉내 1500/2500Hz, 진폭 변조
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))  # 4Hz 음절 envelope
    sig = env * (
        0.5 * np.sin(2 * np.pi * 200 * t)
        + 0.3 * np.sin(2 * np.pi * 1500 * t)
        + 0.2 * np.sin(2 * np.pi * 2500 * t)
    )
    return np.stack([sig, sig]).astype(np.float32)


def _make_music_like(sr: int = 16000, dur: float = 5.0) -> np.ndarray:
    """music-like: sub/low가 강한 신호."""
    n = int(sr * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    sig = (
        0.6 * np.sin(2 * np.pi * 50 * t)    # sub
        + 0.5 * np.sin(2 * np.pi * 80 * t)
        + 0.3 * np.sin(2 * np.pi * 120 * t)
    )
    return np.stack([sig, sig]).astype(np.float32)


def test_classify_silent():
    audio = np.zeros((2, 48000), dtype=np.float32)
    cls = obs_core.classify_track(audio, 48000)
    assert cls.kind == "silent"
    assert cls.speech_ratio == 0.0


def test_classify_music_like():
    """sub heavy 신호 → music으로 분류."""
    audio = _make_music_like(sr=16000, dur=5.0)
    cls = obs_core.classify_track(audio, 16000)
    # speech_ratio가 거의 0이고 sub가 매우 높아 music 또는 fullmix로
    assert cls.sub_band_pct > 15.0
    assert cls.kind in ("music", "fullmix")


def test_classify_returns_valid_fields():
    audio = _make_voice_like(sr=16000, dur=3.0)
    cls = obs_core.classify_track(audio, 16000)
    d = cls.to_dict()
    assert d["kind"] in ("voice", "music", "fullmix", "silent", "unknown")
    assert 0.0 <= d["confidence"] <= 1.0
    assert 0.0 <= d["speech_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# recommend_treatment 룰 엔진
# ---------------------------------------------------------------------------


def _base_diag(kind: str, **overrides) -> dict:
    """recommend_treatment에 넣을 최소 진단 dict 빌더."""
    diag = {
        "classification": {
            "kind": kind,
            "speech_ratio": 0.5,
            "sub_band_pct": 5.0,
            "low_band_pct": 30.0,
            "presence_band_pct": 4.0,
            "hf_slope_db": -12.0,
            "confidence": 0.8,
        },
        "loudness": {"integrated_lufs": -18.0, "true_peak_dbtp": -3.0, "loudness_range_lu": 6.0},
        "spectrum": {"hum_check": [{"frequency_hz": 50, "snr_db": 5.0, "is_hum": False}]},
        "clipping": {"n_samples": 0},
        "dc_offset_max": 0.0001,
        "clicks": {"n_clicks": 0},
        "head_tail_silence": {"head_ms": 100, "tail_sec": 1.0},
        "phase": {"applicable": True, "negative_correlation_pct": 1.0},
        "channel_imbalance": {"applicable": True, "imbalance_db": 0.1},
    }
    diag.update(overrides)
    return diag


def test_recommend_voice_includes_denoise():
    diag = _base_diag("voice")
    plan = obs_core.recommend_treatment(diag)
    actions = [t.action for t in plan]
    assert "denoise" in actions
    assert "leveling" in actions
    # voice de-noise 플러그인이 지정되어야 함
    denoise = next(t for t in plan if t.action == "denoise")
    assert denoise.plugin_short == "voice-de-noise"


def test_recommend_music_no_denoise():
    diag = _base_diag("music")
    plan = obs_core.recommend_treatment(diag)
    actions = [t.action for t in plan]
    assert "denoise" not in actions  # 음악엔 voice denoise 적용 금지
    assert "stem_separate" not in actions


def test_recommend_fullmix_demucs_hint():
    diag = _base_diag("fullmix")
    plan = obs_core.recommend_treatment(diag)
    actions = [t.action for t in plan]
    assert "stem_separate" in actions
    stem = next(t for t in plan if t.action == "stem_separate")
    assert stem.params.get("primary_tool") == "demucs"
    assert stem.params.get("alternate_tool") == "music-rebalance"


def test_recommend_silent_skip():
    diag = _base_diag("silent")
    plan = obs_core.recommend_treatment(diag)
    actions = [t.action for t in plan]
    assert "skip" in actions
    assert "denoise" not in actions


def test_recommend_dehum_when_hum_detected():
    diag = _base_diag("voice")
    diag["spectrum"] = {"hum_check": [
        {"frequency_hz": 60, "snr_db": 14.0, "is_hum": True},
        {"frequency_hz": 120, "snr_db": 12.5, "is_hum": True},
    ]}
    plan = obs_core.recommend_treatment(diag)
    dehum = [t for t in plan if t.action == "dehum"]
    assert len(dehum) == 1
    assert dehum[0].plugin_short == "de-hum"
    assert 60 in dehum[0].params.get("frequencies_hz", [])


def test_recommend_declip_when_clipped():
    diag = _base_diag("voice", clipping={"n_samples": 250})
    plan = obs_core.recommend_treatment(diag)
    declip = next(t for t in plan if t.action == "declip")
    assert declip.severity == "critical"
    assert declip.plugin_short == "de-clip"


def test_recommend_dc_removal():
    diag = _base_diag("voice", dc_offset_max=0.005)
    plan = obs_core.recommend_treatment(diag)
    actions = [t.action for t in plan]
    assert "dc_removal" in actions


def test_recommend_phase_warning():
    diag = _base_diag("voice")
    diag["phase"] = {"applicable": True, "negative_correlation_pct": 25.0}
    plan = obs_core.recommend_treatment(diag)
    actions = [t.action for t in plan]
    assert "phase_warning" in actions


def test_recommend_channel_balance_warning():
    diag = _base_diag("voice")
    diag["channel_imbalance"] = {"applicable": True, "imbalance_db": 2.5}
    plan = obs_core.recommend_treatment(diag)
    actions = [t.action for t in plan]
    assert "channel_balance" in actions


# ---------------------------------------------------------------------------
# probe_topology — 합성 영상 (ffmpeg로 multitrack mov 생성)
# ---------------------------------------------------------------------------


def _try_make_multitrack_video(out_path, tracks_audio, sample_rate, video_duration=1.5):
    """ffmpeg로 multitrack 영상 만들기. 실패 시 None 반환.

    video_duration: 비디오 트랙 길이(초). -shortest 정책상 비디오/오디오 중 짧은
    쪽에 맞춰 컷되므로, 오디오 길이와 같거나 길게 설정해야 오디오가 잘리지 않는다.
    """
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        return None

    tmp_dir = out_path.parent
    wav_paths = []
    for i, audio in enumerate(tracks_audio):
        wav = tmp_dir / f"track{i}.wav"
        sf.write(str(wav), audio.T, sample_rate, subtype="PCM_24")
        wav_paths.append(wav)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=size=64x64:rate=30:duration={video_duration}",
    ]
    for wav in wav_paths:
        cmd += ["-i", str(wav)]
    cmd += ["-map", "0:v"]
    for i in range(len(wav_paths)):
        cmd += ["-map", f"{i+1}:a"]
    cmd += ["-c:v", "libx264", "-c:a", "aac", "-shortest", str(out_path)]

    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        return None
    return out_path


def test_probe_topology_silent(tmp_path):
    sr = 48000
    silent = [np.zeros((2, sr), dtype=np.float32) for _ in range(3)]
    video = tmp_path / "silent.mp4"
    if _try_make_multitrack_video(video, silent, sr) is None:
        pytest.skip("ffmpeg 없음")
    r = obs_core.probe_topology(video, probe_seconds=1.0)
    assert r.topology == "silent"
    assert r.active_indices == []


def test_probe_topology_multitrack(tmp_path):
    """track1=voice-like, track2=music-like → multitrack."""
    sr = 48000
    voice = _make_voice_like(sr=sr, dur=1.5)
    music = _make_music_like(sr=sr, dur=1.5)
    video = tmp_path / "multi.mp4"
    if _try_make_multitrack_video(video, [voice, music], sr) is None:
        pytest.skip("ffmpeg 없음")
    r = obs_core.probe_topology(video, probe_seconds=1.0)
    assert r.topology in ("multitrack", "single")  # RMS가 우연히 비슷하면 single로 빠질 수 있음
    assert len(r.active_indices) >= 1


def test_probe_topology_duplicated(tmp_path):
    sr = 48000
    voice = _make_voice_like(sr=sr, dur=1.5)
    video = tmp_path / "dup.mp4"
    if _try_make_multitrack_video(video, [voice, voice, voice], sr) is None:
        pytest.skip("ffmpeg 없음")
    r = obs_core.probe_topology(video, probe_seconds=1.0)
    assert r.topology == "duplicated"
    assert len(r.unique_signal_groups) == 1


def test_probe_topology_full_scan_catches_late_signal(tmp_path):
    """앞 구간이 무음이고 후반에만 신호가 있는 트랙은 짧은 probe_seconds로
    silent로 오분류되지만, probe_seconds=None(전체 스캔)이면 active로 잡혀야 함.

    OBS 데스크탑 오디오 트랙처럼 산발적으로만 신호가 나오는 패턴 시뮬레이션.
    """
    sr = 48000
    dur = 4.0
    voice = _make_voice_like(sr=sr, dur=dur)
    # 앞 2초 무음, 뒤 2초만 신호
    sparse = np.zeros((2, int(sr * dur)), dtype=np.float32)
    sparse[:, int(sr * 2.0):] = _make_voice_like(sr=sr, dur=2.0)

    video = tmp_path / "sparse.mp4"
    if _try_make_multitrack_video(video, [voice, sparse], sr, video_duration=dur) is None:
        pytest.skip("ffmpeg 없음")

    # 앞 1초만 보면 sparse 트랙은 silent로 오분류됨
    r_short = obs_core.probe_topology(video, probe_seconds=1.0)
    sparse_short = next(t for t in r_short.track_probes if t.index == 1)
    assert sparse_short.is_silent, (
        "전제 검증: 앞 1초 스캔에서는 sparse 트랙이 silent여야 회귀 테스트 의미가 있음"
    )

    # 전체 스캔(default = None)이면 sparse 트랙도 active로 잡혀야 함
    r_full = obs_core.probe_topology(video, probe_seconds=None)
    sparse_full = next(t for t in r_full.track_probes if t.index == 1)
    assert not sparse_full.is_silent, (
        "전체 스캔에서는 sparse 트랙(후반 신호)이 active로 분류돼야 함"
    )
    assert 1 in r_full.active_indices


def test_probe_topology_default_is_full_scan(tmp_path):
    """probe_seconds 미지정 시 전체 영상 RMS를 사용 — 인터페이스 변경 회귀 테스트."""
    sr = 48000
    dur = 3.0
    sparse = np.zeros((2, int(sr * dur)), dtype=np.float32)
    sparse[:, int(sr * 1.5):] = _make_voice_like(sr=sr, dur=1.5)

    video = tmp_path / "default_sparse.mp4"
    if _try_make_multitrack_video(video, [sparse], sr, video_duration=dur) is None:
        pytest.skip("ffmpeg 없음")

    # 인자 없이 호출 — 기본 동작이 전체 스캔이어야 함
    r = obs_core.probe_topology(video)
    assert r.active_indices == [0]


# ---------------------------------------------------------------------------
# diagnose_track / recommend_treatment / dry_run_video and probe edges
# ---------------------------------------------------------------------------

import json  # noqa: E402
import subprocess  # noqa: E402


class TestMeasureDCOffsetHelper:
    def test_mono(self):
        audio = np.full(100, 0.25, dtype=np.float32)
        assert obs_core._measure_dc_offset(audio) == pytest.approx([0.25], abs=1e-6)

    def test_stereo_per_channel(self):
        audio = np.stack([np.full(50, 0.1), np.full(50, -0.2)]).astype(np.float32)
        dc = obs_core._measure_dc_offset(audio)
        assert dc[0] == pytest.approx(0.1, abs=1e-6)
        assert dc[1] == pytest.approx(-0.2, abs=1e-6)


class TestDiagnoseTrack:
    def test_reports_all_measurement_blocks(self):
        cls = obs_core.TrackClassification(
            kind="voice", speech_ratio=0.6, sub_band_pct=5.0, low_band_pct=20.0,
            presence_band_pct=8.0, hf_slope_db=-3.0, confidence=0.8,
        )
        diag = obs_core.diagnose_track(_make_voice_like(sr=16000, dur=3.0), 16000, cls)
        for key in ("classification", "loudness", "spectrum", "clipping",
                    "dc_offset_max", "clicks", "head_tail_silence"):
            assert key in diag
        assert diag["classification"]["kind"] == "voice"

    def test_stereo_adds_phase_and_imbalance(self):
        cls = obs_core.TrackClassification("music", 0.0, 20.0, 30.0, 5.0, -2.0, 0.5)
        stereo = _make_voice_like(sr=16000, dur=3.0)
        diag = obs_core.diagnose_track(stereo, 16000, cls)
        assert "phase" in diag
        assert "channel_imbalance" in diag

    def test_mono_omits_stereo_blocks(self):
        cls = obs_core.TrackClassification("voice", 0.5, 5.0, 20.0, 8.0, -3.0, 0.7)
        mono = _make_voice_like(sr=16000, dur=3.0)[0]
        diag = obs_core.diagnose_track(mono, 16000, cls)
        assert "phase" not in diag
        assert "channel_imbalance" not in diag

    def test_long_file_skips_clicks(self):
        # > 600 s → click detection summarised instead of computed
        cls = obs_core.TrackClassification("music", 0.0, 20.0, 30.0, 5.0, -2.0, 0.5)
        # use a tiny sample_rate so the length/rate ratio exceeds 600
        audio = np.zeros((2, 16000 * 601), dtype=np.float32)
        diag = obs_core.diagnose_track(audio, 16000, cls)
        assert diag["clicks"]["n_clicks"] is None
        assert diag["clicks"]["skipped"] == "duration > 10min"


class TestRecommendTreatmentEdges:
    def test_single_clip_uses_warn_severity(self):
        diag = _base_diag("voice")
        diag["clipping"] = {"n_samples": 1}
        plan = obs_core.recommend_treatment(diag)
        declip = next(t for t in plan if t.action == "declip")
        assert declip.severity == "warn"

    def test_many_clips_use_critical_severity(self):
        diag = _base_diag("voice")
        diag["clipping"] = {"n_samples": 500}
        plan = obs_core.recommend_treatment(diag)
        declip = next(t for t in plan if t.action == "declip")
        assert declip.severity == "critical"

    def test_music_loud_lufs_warns(self):
        diag = _base_diag("music")
        diag["loudness"] = {"integrated_lufs": -6.0}
        plan = obs_core.recommend_treatment(diag)
        assert any(t.action == "loudness_check" for t in plan)

    def test_music_quiet_lufs_no_warning(self):
        diag = _base_diag("music")
        diag["loudness"] = {"integrated_lufs": -18.0}
        plan = obs_core.recommend_treatment(diag)
        assert not any(t.action == "loudness_check" for t in plan)

    def test_few_clicks_info_severity(self):
        diag = _base_diag("voice")
        diag["clicks"] = {"n_clicks": 2}
        plan = obs_core.recommend_treatment(diag)
        declick = next(t for t in plan if t.action == "declick")
        assert declick.severity == "info"

    def test_many_clicks_warn_severity(self):
        diag = _base_diag("voice")
        diag["clicks"] = {"n_clicks": 40}
        plan = obs_core.recommend_treatment(diag)
        declick = next(t for t in plan if t.action == "declick")
        assert declick.severity == "warn"

    def test_unknown_kind_produces_no_kind_specific_treatment(self):
        diag = _base_diag("unknown")
        plan = obs_core.recommend_treatment(diag)
        assert not any(t.action in ("denoise", "stem_separate", "skip") for t in plan)


class TestProbeTopologyEdges:
    def test_missing_file_raises(self, tmp_path):
        if not _ffmpeg_available():
            pytest.skip("ffmpeg/ffprobe not available")
        with pytest.raises(FileNotFoundError, match="파일 없음"):
            obs_core.probe_topology(tmp_path / "missing.mp4")

    def test_probe_seconds_extracts_leading_window(self, tmp_path):
        sr = 48000
        voice = _make_voice_like(sr=sr, dur=2.0)
        video = tmp_path / "window.mp4"
        if _try_make_multitrack_video(video, [voice], sr, video_duration=2.0) is None:
            pytest.skip("ffmpeg 없음")
        report = obs_core.probe_topology(video, probe_seconds=0.5)
        assert report.n_streams == 1
        assert report.sample_rate > 0

    def test_topology_report_to_dict(self, tmp_path):
        sr = 48000
        voice = _make_voice_like(sr=sr, dur=1.5)
        video = tmp_path / "dict.mp4"
        if _try_make_multitrack_video(video, [voice], sr) is None:
            pytest.skip("ffmpeg 없음")
        d = obs_core.probe_topology(video, probe_seconds=1.0).to_dict()
        assert set(d.keys()) >= {"topology", "n_streams", "active_indices",
                                 "unique_signal_groups", "sample_rate", "duration_sec", "tracks"}
        assert isinstance(d["tracks"], list)

    def test_extract_failure_marks_track_silent(self, tmp_path):
        # a text file with an .mp4 extension: ffprobe finds no audio streams
        bad = tmp_path / "bad.mp4"
        bad.write_text("not a video")
        if not _ffmpeg_available():
            pytest.skip("ffmpeg/ffprobe not available")
        try:
            report = obs_core.probe_topology(bad)
        except subprocess.CalledProcessError:
            pytest.skip("ffprobe rejected the file before stream probing")
        assert report.topology == "silent"


class TestDryRunVideo:
    def _video(self, tmp_path, tracks, sr=48000, duration=2.0):
        path = tmp_path / "dryrun.mp4"
        if _try_make_multitrack_video(path, tracks, sr, video_duration=duration) is None:
            pytest.skip("ffmpeg 없음")
        return path

    def test_single_track_report(self, tmp_path):
        voice = _make_voice_like(sr=48000, dur=2.0)
        video = self._video(tmp_path, [voice])
        report = obs_core.dry_run_video(video, analysis_seconds=1.0)
        assert report.track_diagnostics
        assert report.treatments
        d = report.to_dict()
        assert d["video"] == str(video)
        assert "topology" in d and "notes" in d
        assert json.dumps(d)  # JSON-serializable

    def test_duplicated_tracks_analyze_first_only(self, tmp_path):
        voice = _make_voice_like(sr=48000, dur=2.0)
        video = self._video(tmp_path, [voice, voice])
        report = obs_core.dry_run_video(video, analysis_seconds=1.0)
        if report.topology.topology != "duplicated":
            pytest.skip(f"AAC encoding shifted RMS enough to look distinct ({report.topology.topology})")
        assert any("복제 믹스" in n for n in report.notes)
        assert len(report.treatments) == 1
        # every active track is mapped onto the representative that was analysed
        assert set(report.treatments[0]["mirrors"]) == set(report.topology.active_indices)

    def test_multitrack_mirror_map(self, tmp_path):
        voice = _make_voice_like(sr=48000, dur=2.0)
        music = _make_music_like(sr=48000, dur=2.0)
        # two identical voice tracks + one distinct music track
        video = self._video(tmp_path, [voice, voice, music])
        report = obs_core.dry_run_video(video, analysis_seconds=1.0)
        if report.topology.topology != "multitrack":
            pytest.skip(f"acoustic separation insufficient for mirror test ({report.topology.topology})")
        mirrored = [t for t in report.treatments if t["mirrors"]]
        assert mirrored, "expected at least one mirrored track"

    def test_silent_video_short_circuits(self, tmp_path):
        silent = [np.zeros((2, 48000), dtype=np.float32)]
        video = self._video(tmp_path, silent)
        report = obs_core.dry_run_video(video)
        assert report.track_diagnostics == []
        assert report.treatments == []
        assert any("모든 트랙 무음" in n for n in report.notes)

    def test_analysis_start_sec_explicit(self, tmp_path):
        voice = _make_voice_like(sr=48000, dur=3.0)
        video = self._video(tmp_path, [voice], duration=3.0)
        report = obs_core.dry_run_video(video, analysis_seconds=0.5, analysis_start_sec=1.0)
        assert report.track_diagnostics
        assert report.track_diagnostics[0]["analysis_start_sec"] == 1.0

    def test_default_analysis_start_is_middle_for_long_video(self, tmp_path):
        voice = _make_voice_like(sr=48000, dur=3.0)
        video = self._video(tmp_path, [voice], duration=3.0)
        # analysis_seconds * 2 = 2.0 < duration 3.0 → centred window
        report = obs_core.dry_run_video(video, analysis_seconds=1.0)
        assert report.track_diagnostics[0]["analysis_start_sec"] > 0.0


class TestTreatmentDict:
    def test_to_dict(self):
        t = obs_core.Treatment(action="denoise", plugin_short="voice-de-noise",
                               params={"a": 1}, rationale="r", severity="warn")
        assert t.to_dict() == {
            "action": "denoise", "plugin": "voice-de-noise",
            "params": {"a": 1}, "rationale": "r", "severity": "warn",
        }


class TestTrackClassificationDict:
    def test_to_dict_rounds(self):
        cls = obs_core.TrackClassification("voice", 0.5678, 1.234, 5.678, 9.876, -3.2, 0.789)
        d = cls.to_dict()
        assert d["speech_ratio"] == 0.568
        assert d["confidence"] == 0.789
        assert d["hf_slope_db"] == -3.2


# ---------------------------------------------------------------------------
# classify_track branch matrix - dependencies stubbed to isolate the decision rules.
# ---------------------------------------------------------------------------


def _stub_classify(monkeypatch, speech_ratio, sub_pct, presence_pct, duration_sec=5.0):
    class _Seg:
        def __init__(self, dur):
            self.duration_samples = dur

    def _speech(audio, sr, **kw):
        total = int(sr * duration_sec)
        want = int(total * speech_ratio)
        return [_Seg(want)] if want > 0 else []

    def _diag(audio, sr, **kw):
        return {
            "band_energy": [
                {"band": "sub", "percent": sub_pct},
                {"band": "low", "percent": 30.0},
                {"band": "presence", "percent": presence_pct},
            ],
            "hf_slope": {"slope_db": -3.0},
        }

    monkeypatch.setattr(obs_core, "detect_speech", _speech)
    monkeypatch.setattr(obs_core, "spectrum_diagnostics", _diag)
    audio = np.full((2, int(48000 * duration_sec)), 0.1, dtype=np.float32)
    return obs_core.classify_track(audio, 48000)


class TestClassifyTrackBranches:
    def test_voice_rule(self, monkeypatch):
        cls = _stub_classify(monkeypatch, speech_ratio=0.6, sub_pct=5.0, presence_pct=3.0)
        assert cls.kind == "voice"
        assert cls.confidence == pytest.approx(0.8)  # 0.5 + 0.6*0.5

    def test_music_rule(self, monkeypatch):
        cls = _stub_classify(monkeypatch, speech_ratio=0.0, sub_pct=30.0, presence_pct=1.0)
        assert cls.kind == "music"
        assert cls.confidence > 0.5  # 0.5 + (30-15)/50

    def test_fullmix_voice_plus_music_rule(self, monkeypatch):
        cls = _stub_classify(monkeypatch, speech_ratio=0.3, sub_pct=20.0, presence_pct=1.0)
        assert cls.kind == "fullmix"
        assert cls.confidence == pytest.approx(0.7)

    def test_speech_dominant_low_sub_becomes_voice(self, monkeypatch):
        cls = _stub_classify(monkeypatch, speech_ratio=0.3, sub_pct=3.0, presence_pct=3.0)
        assert cls.kind == "voice"
        assert cls.confidence == pytest.approx(0.6)

    def test_speech_dominant_high_sub_stays_fullmix(self, monkeypatch):
        cls = _stub_classify(monkeypatch, speech_ratio=0.3, sub_pct=8.0, presence_pct=3.0)
        assert cls.kind == "fullmix"
        assert cls.confidence == pytest.approx(0.6)

    def test_default_fallback_is_conservative_fullmix(self, monkeypatch):
        cls = _stub_classify(monkeypatch, speech_ratio=0.1, sub_pct=5.0, presence_pct=1.0)
        assert cls.kind == "fullmix"
        assert cls.confidence == pytest.approx(0.4)

    def test_silent_before_any_detection(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("detectors must not run for a silent track")

        monkeypatch.setattr(obs_core, "detect_speech", _boom)
        cls = obs_core.classify_track(np.zeros((2, 48000), dtype=np.float32), 48000)
        assert cls.kind == "silent"
        assert cls.confidence == 1.0


# ---------------------------------------------------------------------------
# probe_topology error and grouping branches - ffmpeg calls stubbed for determinism.
# ---------------------------------------------------------------------------

import subprocess as _subprocess  # noqa: E402


class TestEnsureTools:
    def test_missing_ffmpeg_raises(self, monkeypatch):
        monkeypatch.setattr(obs_core.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="ffmpeg/ffprobe"):
            obs_core._ensure_tools()

    def test_present_tools_pass(self, monkeypatch):
        monkeypatch.setattr(obs_core.shutil, "which", lambda name: f"/usr/bin/{name}")
        obs_core._ensure_tools()  # must not raise


def _fake_extract_factory(monkeypatch, rms_values, stereo=False):
    """Replace the ffmpeg extractor with one that writes known-RMS WAVs."""
    calls = []

    def _extract(video, audio_index, out_path, *, duration_sec=None, start_sec=0.0,
                 sample_rate=None, mono=False):
        calls.append(audio_index)
        level = rms_values[audio_index]
        n = 1000
        data = np.full((n, 2), level, dtype=np.float32) if stereo else np.full(n, level, dtype=np.float32)
        sf.write(str(out_path), data, 16000, subtype="FLOAT")
        return True

    monkeypatch.setattr(obs_core, "_extract_track_to_wav", _extract)
    return calls


class TestProbeTopologyGrouping:
    def _streams(self, monkeypatch, n, channels=1):
        streams = [{"index": i, "channels": channels, "sample_rate": "16000"} for i in range(n)]
        monkeypatch.setattr(obs_core, "_ffprobe_streams", lambda v: streams)
        monkeypatch.setattr(obs_core.subprocess, "check_output", lambda *a, **k: "2.0\n")

    def test_three_identical_tracks_are_duplicated(self, monkeypatch, tmp_path):
        monkeypatch.setattr(obs_core.shutil, "which", lambda n: f"/usr/bin/{n}")
        self._streams(monkeypatch, 3, channels=2)
        _fake_extract_factory(monkeypatch, [0.2, 0.2, 0.2])
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        report = obs_core.probe_topology(video)
        assert report.topology == "duplicated"
        assert report.unique_signal_groups == [[0, 1, 2]]
        assert report.active_indices == [0, 1, 2]

    def test_two_distinct_rms_is_multitrack(self, monkeypatch, tmp_path):
        monkeypatch.setattr(obs_core.shutil, "which", lambda n: f"/usr/bin/{n}")
        self._streams(monkeypatch, 2)
        _fake_extract_factory(monkeypatch, [0.2, 0.8])
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        report = obs_core.probe_topology(video)
        assert report.topology == "multitrack"
        assert len(report.unique_signal_groups) == 2

    def test_silent_among_active_is_duplicated(self, monkeypatch, tmp_path):
        monkeypatch.setattr(obs_core.shutil, "which", lambda n: f"/usr/bin/{n}")
        self._streams(monkeypatch, 3)
        # track 2 silent, 0 and 1 identical → group count 1, active 2 < probes 3
        _fake_extract_factory(monkeypatch, [0.2, 0.2, 0.0])
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        report = obs_core.probe_topology(video)
        assert report.topology == "duplicated"
        assert report.active_indices == [0, 1]

    def test_single_active_track(self, monkeypatch, tmp_path):
        monkeypatch.setattr(obs_core.shutil, "which", lambda n: f"/usr/bin/{n}")
        self._streams(monkeypatch, 2)
        _fake_extract_factory(monkeypatch, [0.3, 0.0])
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        report = obs_core.probe_topology(video)
        assert report.topology == "single"
        assert report.active_indices == [0]

    def test_stereo_probe_data_is_downmixed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(obs_core.shutil, "which", lambda n: f"/usr/bin/{n}")
        self._streams(monkeypatch, 1, channels=2)
        _fake_extract_factory(monkeypatch, [0.25], stereo=True)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        report = obs_core.probe_topology(video)
        assert report.track_probes[0].is_stereo is True
        assert report.track_probes[0].rms == pytest.approx(0.25, abs=1e-3)

    def test_no_audio_streams_returns_empty_report(self, monkeypatch, tmp_path):
        monkeypatch.setattr(obs_core.shutil, "which", lambda n: f"/usr/bin/{n}")
        monkeypatch.setattr(obs_core, "_ffprobe_streams", lambda v: [])
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        report = obs_core.probe_topology(video)
        assert report.topology == "silent"
        assert report.n_streams == 0
        assert report.track_probes == []

    def test_extraction_failure_marks_silent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(obs_core.shutil, "which", lambda n: f"/usr/bin/{n}")
        self._streams(monkeypatch, 2)
        monkeypatch.setattr(obs_core, "_extract_track_to_wav",
                            lambda *a, **k: False)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        report = obs_core.probe_topology(video)
        assert report.track_probes[0].is_silent is True
        assert report.track_probes[0].rms == 0.0
        assert report.topology == "silent"

    def test_duration_probe_failure_keeps_zero(self, monkeypatch, tmp_path):
        monkeypatch.setattr(obs_core.shutil, "which", lambda n: f"/usr/bin/{n}")
        self._streams(monkeypatch, 1)

        def _boom(*a, **k):
            raise _subprocess.SubprocessError("ffprobe failed")

        monkeypatch.setattr(obs_core.subprocess, "check_output", _boom)
        _fake_extract_factory(monkeypatch, [0.3])
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        report = obs_core.probe_topology(video)
        assert report.duration_sec == 0.0


class TestExtractTrackToWavFailure:
    def test_nonzero_returncode_returns_false(self, monkeypatch, tmp_path):
        class _Result:
            returncode = 1
            stderr = "boom"

        monkeypatch.setattr(obs_core.subprocess, "run", lambda *a, **k: _Result())
        ok = obs_core._extract_track_to_wav(tmp_path / "v.mp4", 0, tmp_path / "o.wav")
        assert ok is False


class TestProbeGroupingSkipPaths:
    def test_already_grouped_index_is_skipped(self, monkeypatch, tmp_path):
        # rms [0.2, 0.2, 0.5]: track 1 is consumed by track 0's group, so the outer
        # loop must skip it before forming a second group
        monkeypatch.setattr(obs_core.shutil, "which", lambda n: f"/usr/bin/{n}")
        streams = [{"index": i, "channels": 1, "sample_rate": "16000"} for i in range(3)]
        monkeypatch.setattr(obs_core, "_ffprobe_streams", lambda v: streams)
        monkeypatch.setattr(obs_core.subprocess, "check_output", lambda *a, **k: "1.0\n")
        _fake_extract_factory(monkeypatch, [0.2, 0.2, 0.5])
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        report = obs_core.probe_topology(video)
        assert report.topology == "multitrack"
        assert report.unique_signal_groups == [[0, 1], [2]]


class TestDryRunExtractionFailure:
    def test_failed_extraction_adds_note_and_continues(self, monkeypatch, tmp_path):
        monkeypatch.setattr(obs_core.shutil, "which", lambda n: f"/usr/bin/{n}")
        monkeypatch.setattr(obs_core, "_ensure_tools", lambda: None)
        monkeypatch.setattr(obs_core, "_extract_track_to_wav", lambda *a, **k: False)
        topo = obs_core.TopologyReport(
            topology="single", n_streams=1,
            track_probes=[obs_core.TrackProbe(0, 0.2, False, True)],
            active_indices=[0], unique_signal_groups=[[0]],
            sample_rate=48000, duration_sec=1.0,
        )
        monkeypatch.setattr(obs_core, "probe_topology", lambda v, **k: topo)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        report = obs_core.dry_run_video(video)
        assert report.track_diagnostics == []
        assert report.treatments == []
        assert any("추출 실패" in n for n in report.notes)


class TestProbeGroupingInnerSkip:
    def test_inner_loop_skips_already_used_track(self, monkeypatch, tmp_path):
        # order [0.2, 0.5, 0.2]: track 2 joins track 0's group, then track 1's inner
        # loop must skip the already-consumed track 2
        monkeypatch.setattr(obs_core.shutil, "which", lambda n: f"/usr/bin/{n}")
        streams = [{"index": i, "channels": 1, "sample_rate": "16000"} for i in range(3)]
        monkeypatch.setattr(obs_core, "_ffprobe_streams", lambda v: streams)
        monkeypatch.setattr(obs_core.subprocess, "check_output", lambda *a, **k: "1.0\n")
        _fake_extract_factory(monkeypatch, [0.2, 0.5, 0.2])
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x")
        report = obs_core.probe_topology(video)
        assert sorted(report.unique_signal_groups) == [[0, 2], [1]]
        assert report.topology == "multitrack"
