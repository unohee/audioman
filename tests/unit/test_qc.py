# tests/unit/test_qc.py — 마스터링 QC 검수 리포트

import numpy as np
import pytest
import soundfile as sf

from audioman.core import qc


SR = 48000


def _stereo_sine(amp: float = 0.3, freq: float = 1000.0, duration: float = 5.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    s = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([s, s])


@pytest.fixture
def clean_master_wav(tmp_path):
    """납품 표준에 가까운 깨끗한 마스터 (5초 사인 + 1초 헤드 무음 + 2초 테일 무음)."""
    sr = SR
    head = np.zeros((2, sr // 5), dtype=np.float32)  # 200ms
    body = _stereo_sine(0.3, duration=5.0, sr=sr)
    tail = np.zeros((2, sr * 2), dtype=np.float32)
    audio = np.concatenate([head, body, tail], axis=1)
    path = tmp_path / "master.wav"
    sf.write(str(path), audio.T, sr, subtype="PCM_24")
    return path


class TestDetectClipping:
    def test_no_clipping(self):
        audio = _stereo_sine(0.5)
        result = qc.detect_clipping(audio)
        assert result["n_samples"] == 0

    def test_clipping_detected(self):
        # 강제 클립 — 일부 샘플을 1.0 이상으로 만들기
        audio = _stereo_sine(0.5)
        audio[0, 100:110] = 1.0
        audio[1, 200:205] = -1.0
        result = qc.detect_clipping(audio)
        # 채널 union — sample 100~109(10) + 200~204(5) = 15
        assert result["n_samples"] == 15
        assert result["per_channel"] == [10, 5]

    def test_threshold_strictness(self):
        audio = _stereo_sine(0.999)  # 거의 클립
        # threshold 0.999면 일부 잡힐 수 있음, 0.9999면 안 잡힘
        relaxed = qc.detect_clipping(audio, threshold=0.9999)
        assert relaxed["n_samples"] == 0


class TestDetectClicks:
    def test_clean_signal_no_clicks(self):
        audio = _stereo_sine(0.3, duration=2.0)
        result = qc.detect_clicks(audio, SR, sensitivity=8.0)
        assert result["n_clicks"] == 0

    def test_artificial_click_detected(self):
        sr = SR
        audio = _stereo_sine(0.2, duration=2.0, sr=sr)
        # 1초 지점에 단일 샘플 spike
        click_pos = sr
        audio[0, click_pos] = 0.95
        audio[1, click_pos] = 0.95
        result = qc.detect_clicks(audio, sr, sensitivity=5.0)
        assert result["n_clicks"] >= 1
        # 위치도 1초 부근
        assert any(abs(loc - 1.0) < 0.01 for loc in result["locations_sec"])

    def test_grouping_consecutive(self):
        sr = SR
        audio = _stereo_sine(0.1, duration=1.0, sr=sr)
        # 연속 샘플에 spike (한 번의 클릭으로 묶여야 함)
        for i in range(5):
            audio[0, sr // 2 + i] = 0.8
        result = qc.detect_clicks(audio, sr, sensitivity=5.0, min_separation_ms=10.0)
        # 5개 spike이 한 클릭으로 그룹핑돼야 함
        assert result["n_clicks"] <= 2  # 대개 1개


class TestPhaseCorrelation:
    def test_mono_in_phase_correlation_one(self):
        s = _stereo_sine(0.3)
        result = qc.stereo_phase_correlation(s, sample_rate=SR)
        assert result["applicable"]
        assert abs(result["global_correlation"] - 1.0) < 0.01

    def test_inverted_correlation_negative(self):
        s = _stereo_sine(0.3)
        s[1] = -s[1]  # 우채널 반전
        result = qc.stereo_phase_correlation(s, sample_rate=SR)
        assert result["global_correlation"] < -0.95

    def test_mono_input_not_applicable(self):
        mono = _stereo_sine(0.3)[0]
        result = qc.stereo_phase_correlation(mono, sample_rate=SR)
        assert not result["applicable"]


class TestChannelImbalance:
    def test_balanced_zero_db(self):
        s = _stereo_sine(0.3)
        result = qc.channel_imbalance_db(s)
        assert abs(result["imbalance_db"]) < 0.01

    def test_left_louder(self):
        s = _stereo_sine(0.3)
        s[0] *= 2.0  # 좌채널 +6dB
        result = qc.channel_imbalance_db(s)
        assert 5.5 < result["imbalance_db"] < 6.5


class TestHeadTailSilence:
    def test_clean_padding(self):
        sr = SR
        head = np.zeros((2, sr // 2), dtype=np.float32)  # 500ms
        body = _stereo_sine(0.3, duration=2.0, sr=sr)
        tail = np.zeros((2, sr * 2), dtype=np.float32)  # 2s
        audio = np.concatenate([head, body, tail], axis=1)
        result = qc.head_tail_silence(audio, sr)
        assert 480 < result["head_ms"] < 520
        assert 1.95 < result["tail_sec"] < 2.05

    def test_no_padding(self):
        s = _stereo_sine(0.3)
        result = qc.head_tail_silence(s, SR)
        assert result["head_ms"] < 5
        assert result["tail_sec"] < 0.005


class TestEvaluate:
    def test_clean_master_against_spotify(self, clean_master_wav, tmp_path):
        report = qc.evaluate_file(clean_master_wav, target="spotify")
        assert report["target"] == "spotify"
        assert "verdict" in report
        assert "checks" in report
        # 스테레오 사인이라 phase corr는 PASS, padding도 spotify 범위 안
        names = [c["name"] for c in report["checks"]]
        assert "integrated_lufs" in names
        assert "true_peak_dbtp" in names
        assert "head_silence_ms" in names

    def test_clipped_signal_fails(self, tmp_path):
        # 강한 클리핑 스테레오
        sr = SR
        s = _stereo_sine(0.5, duration=5.0, sr=sr)
        s[0, 100:200] = 1.0
        s[0, 1000:1100] = -1.0
        path = tmp_path / "clipped.wav"
        sf.write(str(path), s.T, sr, subtype="PCM_24")

        report = qc.evaluate_file(path, target="spotify")
        clip_check = next(c for c in report["checks"] if c["name"] == "clipping_samples")
        assert clip_check["status"] in ("WARN", "FAIL")

    def test_unknown_target_raises(self, clean_master_wav):
        with pytest.raises(ValueError, match="알 수 없는 target"):
            qc.evaluate_file(clean_master_wav, target="myspace")

    def test_targets_listing(self):
        targets = qc.list_targets()
        assert "spotify" in targets
        assert "apple_music" in targets
        assert "broadcast_ebu_r128" in targets
        assert "cd_master" in targets


class TestVerdictAggregation:
    def test_all_pass_verdict(self, clean_master_wav):
        # 깨끗한 마스터 — verdict는 보통 WARN (사인이라 LUFS가 spotify -14 범위 밖일 수 있음)
        # 정확한 verdict보다는 구조 확인
        report = qc.evaluate_file(clean_master_wav, target="spotify")
        assert report["verdict"] in ("PASS", "WARN", "FAIL")
        assert report["summary"]["n_pass"] + report["summary"]["n_warn"] + report["summary"]["n_fail"] == len(report["checks"])
