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

    def test_short_buffer_does_not_crash(self):
        result = qc.detect_clicks(np.array([0.0, 0.8, 0.0], dtype=np.float32), SR)
        assert "n_clicks" in result


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


# ---------------------------------------------------------------------------
# Boundary / branch coverage for the measurement and verdict helpers
# ---------------------------------------------------------------------------


class TestDetectClippingEdges:
    def test_mono_path(self):
        audio = np.zeros(100, dtype=np.float32)
        audio[10:13] = 1.0
        result = qc.detect_clipping(audio)
        assert result["n_samples"] == 3
        assert result["first_sample_locations"][0] == 10
        assert "per_channel" not in result


class TestDetectClicksEdges:
    def test_single_sample_buffer(self):
        audio = np.array([0.5], dtype=np.float32)
        assert qc.detect_clicks(audio, SR) == {"n_clicks": 0, "locations_sec": []}

    def test_short_buffer_scales_signal(self):
        # fewer samples than the analysis window → local RMS fallback path
        audio = np.full(50, 0.2, dtype=np.float32)
        audio[25] = 5.0  # huge step
        result = qc.detect_clicks(audio, SR, sensitivity=6.0)
        assert result["n_clicks"] >= 1
        assert "max_ratio" in result

    def test_short_buffer_clean_returns_zero(self):
        audio = np.full(50, 0.2, dtype=np.float32)
        result = qc.detect_clicks(audio, SR, sensitivity=6.0)
        assert result["n_clicks"] == 0
        assert "max_ratio" in result

    def test_clicks_grouped_by_min_separation(self):
        audio = np.concatenate([
            np.full(500, 0.2, dtype=np.float32),
            np.array([5.0], dtype=np.float32),
            np.full(500, 0.2, dtype=np.float32),
        ])
        result = qc.detect_clicks(audio, SR, sensitivity=6.0, min_separation_ms=100.0)
        assert result["n_clicks"] == 1  # everything within one separation window

    def test_window_far_shorter_than_the_buffer(self):
        # win << n: the rolling mean is much shorter than n - 1 and the symmetric
        # edge pad is what stretches it to the length `diff` needs.
        audio = np.full(2048 + 40, 0.3, dtype=np.float32)
        audio[1000] = 9.0
        result = qc.detect_clicks(audio, SR, window_ms=40.0, sensitivity=6.0)
        assert "n_clicks" in result


class TestStereoPhaseEdges:
    def test_not_stereo(self):
        assert qc.stereo_phase_correlation(np.zeros(100, dtype=np.float32), sample_rate=SR)["applicable"] is False

    def test_single_window_short_audio(self):
        # length between win and 2*win → n_windows == 1
        n = int(0.15 * SR)
        t = np.arange(n) / SR
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        stereo = np.stack([left, left])
        result = qc.stereo_phase_correlation(stereo, window_ms=100.0, sample_rate=SR)
        assert result["applicable"] is True
        assert result["negative_correlation_pct"] == 0.0

    def test_silent_channels_skipped(self):
        # one channel silent → window std check skips → no windows counted
        n = SR
        left = np.sin(2 * np.pi * 440 * np.arange(n) / SR).astype(np.float32)
        right = np.zeros(n, dtype=np.float32)
        result = qc.stereo_phase_correlation(np.stack([left, right]), sample_rate=SR)
        assert result["negative_correlation_pct"] == 0.0
        assert result["global_correlation"] == 0.0

    def test_out_of_phase_negative_percentage(self):
        n = SR
        t = np.arange(n) / SR
        sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        result = qc.stereo_phase_correlation(np.stack([sig, -sig]), sample_rate=SR)
        assert result["negative_correlation_pct"] > 0.0
        assert result["min_window_correlation"] < 0.0


class TestChannelImbalanceEdges:
    def test_not_stereo(self):
        assert qc.channel_imbalance_db(np.zeros(100, dtype=np.float32))["applicable"] is False

    def test_silent_channel(self):
        left = np.zeros(SR, dtype=np.float32)
        right = np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
        result = qc.channel_imbalance_db(np.stack([left, right]))
        assert result["imbalance_db"] is None
        assert result["reason"] == "silent channel"


class TestHeadTailSilenceEdges:
    def test_all_silence(self):
        result = qc.head_tail_silence(np.zeros((2, SR), dtype=np.float32), SR)
        assert result["all_silence"] is True
        assert result["head_ms"] is None

    def test_reports_head_ms_and_tail_sec(self):
        audio = np.zeros((2, SR), dtype=np.float32)
        audio[:, SR // 4: 3 * SR // 4] = 0.5
        result = qc.head_tail_silence(audio, SR)
        assert result["head_ms"] == pytest.approx(250.0, abs=1.0)
        assert result["tail_sec"] == pytest.approx(0.25, abs=0.01)


class TestFileFormatInfoSubtypes:
    @pytest.mark.parametrize("subtype,expected_depth", [
        ("PCM_16", 16), ("PCM_24", 24), ("PCM_32", 32), ("FLOAT", 32), ("DOUBLE", 64),
    ])
    def test_bit_depth_from_subtype(self, tmp_path, subtype, expected_depth):
        path = tmp_path / f"a_{subtype}.wav"
        sf.write(str(path), np.zeros((100, 2), dtype=np.float32), SR, subtype=subtype)
        info = qc.file_format_info(path)
        assert info["bit_depth"] == expected_depth
        assert info["channels"] == 2
        assert info["file_size_mb"] >= 0.0


class TestStatusHelpers:
    def test_lufs_none_is_fail(self):
        assert qc._status_for_lufs(None, (-14.0, -12.0)) == "FAIL"

    def test_lufs_in_range_pass(self):
        assert qc._status_for_lufs(-13.0, (-14.0, -12.0)) == "PASS"

    def test_lufs_within_half_lu_warn(self):
        assert qc._status_for_lufs(-11.7, (-14.0, -12.0)) == "WARN"

    def test_lufs_far_out_fail(self):
        assert qc._status_for_lufs(-5.0, (-14.0, -12.0)) == "FAIL"

    def test_tp_none_pass(self):
        assert qc._status_for_tp(None, -1.0) == "PASS"

    def test_tp_under_ceiling_pass(self):
        assert qc._status_for_tp(-2.0, -1.0) == "PASS"

    def test_tp_slightly_over_warn(self):
        assert qc._status_for_tp(-0.85, -1.0) == "WARN"

    def test_tp_far_over_fail(self):
        assert qc._status_for_tp(0.5, -1.0) == "FAIL"

    def test_silence_none_fail(self):
        assert qc._status_for_silence(None, (0.0, 1.0)) == "FAIL"

    def test_silence_in_range_pass(self):
        assert qc._status_for_silence(0.5, (0.0, 1.0)) == "PASS"

    def test_silence_out_of_range_warn(self):
        assert qc._status_for_silence(2.0, (0.0, 1.0)) == "WARN"


class TestEvaluateBranches:
    def test_target_object_instead_of_name(self):
        audio = _stereo_sine(0.3)
        target = qc.QCTarget(
            name="Custom", integrated_lufs=(-30.0, -10.0), max_true_peak_dbtp=0.0,
        )
        report = qc.evaluate(audio, SR, target=target)
        assert report["target"] == "Custom"
        assert report["target_profile"]["name"] == "Custom"

    def test_min_lra_check_emitted(self):
        audio = _stereo_sine(0.3)
        report = qc.evaluate(audio, SR, target="broadcast_ebu_r128")
        names = [c["name"] for c in report["checks"]]
        assert "loudness_range_lu" in names

    def test_file_path_yields_format_checks(self, clean_master_wav, tmp_path):
        # cd_master enforces sample_rate + min_bit_depth
        path = tmp_path / "cd.wav"
        sf.write(str(path), _stereo_sine(0.3).T, 44100, subtype="PCM_16")
        report = qc.evaluate_file(path, target="cd_master")
        names = [c["name"] for c in report["checks"]]
        assert "sample_rate" in names
        assert "bit_depth" in names

    def test_format_sample_rate_mismatch_fails(self, tmp_path):
        path = tmp_path / "wrong_sr.wav"
        sf.write(str(path), _stereo_sine(0.3).T, 48000, subtype="PCM_24")
        report = qc.evaluate_file(path, target="cd_master")
        sr_check = next(c for c in report["checks"] if c["name"] == "sample_rate")
        assert sr_check["status"] == "FAIL"
        assert report["verdict"] == "FAIL"

    def test_head_silence_none_fails_verdict(self, tmp_path):
        # a file with no silence at all → head_ms ~= 0; spotify wants 100-700ms → WARN
        path = tmp_path / "no_silence.wav"
        sf.write(str(path), _stereo_sine(0.3, duration=5.0).T, SR, subtype="PCM_24")
        report = qc.evaluate_file(path, target="spotify")
        head = next(c for c in report["checks"] if c["name"] == "head_silence_ms")
        assert head["status"] == "WARN"

    def test_clipped_signal_status_fail(self):
        audio = _stereo_sine(0.5)
        audio[:, 10:20] = 1.0  # > 5 clipped samples → FAIL
        report = qc.evaluate(audio, SR)
        clip_check = next(c for c in report["checks"] if c["name"] == "clipping_samples")
        assert clip_check["status"] == "FAIL"

    def test_dc_offset_checks(self):
        audio = _stereo_sine(0.3)
        audio[0] += 0.02  # > 0.01 → FAIL on that channel's contribution
        report = qc.evaluate(audio, SR)
        dc = next(c for c in report["checks"] if c["name"] == "dc_offset")
        assert dc["status"] == "FAIL"

    def test_summary_counts_match_checks(self):
        report = qc.evaluate(_stereo_sine(0.3), SR, target="youtube")
        s = report["summary"]
        assert s["n_pass"] + s["n_warn"] + s["n_fail"] == len(report["checks"])

    def test_targets_listing_sorted(self):
        assert qc.list_targets() == list(qc.TARGETS.keys())


class TestDetectClicksGroupingDeep:
    def test_short_buffer_click_grouping_dedups_by_separation(self):
        audio = np.full(50, 0.2, dtype=np.float32)
        audio[5] = 8.0
        audio[9] = 8.0
        audio[45] = 8.0
        # wide separation window → all three collapse into one click
        merged = qc.detect_clicks(audio, SR, sensitivity=2.0, min_separation_ms=1.0)
        assert merged["n_clicks"] == 1
        # tight window → each candidate kept separately
        split = qc.detect_clicks(audio, SR, sensitivity=2.0, min_separation_ms=0.1)
        assert split["n_clicks"] == 3

    def test_long_buffer_two_separate_clicks(self):
        audio = np.full(20000, 0.2, dtype=np.float32)
        audio[1000] = 9.0
        audio[15000] = 9.0
        result = qc.detect_clicks(audio, SR, sensitivity=6.0, min_separation_ms=5.0)
        assert result["n_clicks"] == 2


class TestStereoPhaseVeryShort:
    def test_audio_shorter_than_window(self):
        n = 2000  # < win (100 ms @ 48 kHz = 4800)
        stereo = np.zeros((2, n), dtype=np.float32)
        stereo[0] = 0.1
        result = qc.stereo_phase_correlation(stereo, window_ms=100.0, sample_rate=SR)
        assert result["applicable"] is True
        assert result["n_windows"] if "n_windows" in result else True
        assert result["negative_correlation_pct"] == 0.0


class TestClickDetectionOddWindowPadding:
    """Odd windows exercise the left/right split of the edge padding in detect_clicks.

    `pad = (n - 1 - len(rolling)) // 2` is an integer division, so for an odd `win`
    the gap `win - 2` is odd and the two sides of the pad end up unequal by one
    sample: the right width is `pad + 1`. The rolling RMS still has to line up with
    `np.diff`, so an odd window is where an off-by-one in that split would show up as
    a misaligned ratio.
    """

    @pytest.mark.parametrize("window_ms", [441 / 44100 * 1000.0, 2205 / 44100 * 1000.0])
    def test_odd_window_still_detects_the_click(self, window_ms):
        # win is odd at 44.1 kHz for these durations -> pad is short by one sample.
        sr = 44100
        win = int(window_ms / 1000.0 * sr)
        assert win % 2 == 1, "test premise: the window must be odd"

        audio = np.full(win * 3, 0.2, dtype=np.float32)
        audio[win * 2] = 9.0

        result = qc.detect_clicks(audio, sr, sensitivity=6.0, window_ms=window_ms)

        assert result["n_clicks"] == 1
        assert result["locations_sec"][0] == pytest.approx(win * 2 / sr, abs=1e-3)

    def test_padded_length_matches_the_difference_signal(self):
        # The pad exists so `diff / local_rms` is well defined; if the two widths
        # were wrong this would raise a broadcast error instead of returning a dict.
        sr = 44100
        window_ms = 441 / 44100 * 1000.0
        win = int(window_ms / 1000.0 * sr)
        audio = np.random.default_rng(0).normal(0, 0.1, win * 2 + 7).astype(np.float32)

        result = qc.detect_clicks(audio, sr, window_ms=window_ms)

        assert set(result) >= {"n_clicks", "locations_sec"}
        assert isinstance(result["n_clicks"], int)
