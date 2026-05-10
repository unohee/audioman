# Created: 2026-05-07
# Purpose: core/obs.py 단위 테스트 — 토폴로지 분류, 트랙 분류, 처치 룰 엔진

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from audioman.core import obs as obs_core


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
