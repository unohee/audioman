# tests/unit/test_automix.py — automix 밴드별 RMS 분석 + gain 최적화 테스트

import numpy as np
import pytest
import soundfile as sf

from audioman.core.automix import (
    BandDefinition,
    DEFAULT_BANDS,
    AutomixResult,
    compute_band_rms,
    pink_noise_profile,
    reference_profile,
    compute_automix_gains,
    automix,
    k_weight_magnitude,
    K20_REF_LUFS,
    classify_tracks,
    genre_profile,
    compute_broadband_rms_db,
    _match_relative_keyword,
    GENRE_PROFILES,
)


class TestKWeighting:
    """ITU-R BS.1770 K-weighting 필터 검증"""

    def test_dc_is_zero(self):
        """0Hz(DC)는 완전 차단"""
        freqs = np.array([0.0, 100.0, 1000.0])
        mag = k_weight_magnitude(freqs)
        assert mag[0] == 0.0

    def test_1khz_near_unity(self):
        """1kHz에서 약 0dB (±2dB) — BS.1770 K-weighting"""
        freqs = np.array([1000.0])
        mag = k_weight_magnitude(freqs)
        mag_db = 20.0 * np.log10(mag[0])
        assert abs(mag_db) < 2.0, f"1kHz K-weight = {mag_db:.2f}dB (expected ~0dB)"

    def test_high_freq_boost(self):
        """고주파(~2-6kHz) 영역에서 부스트 — 두부 회절 보상"""
        freqs = np.array([2000.0, 4000.0])
        mag = k_weight_magnitude(freqs)
        # 2-4kHz 영역에서 부스트가 있어야 함
        mag_db_2k = 20.0 * np.log10(mag[0])
        assert mag_db_2k > 0.5, f"2kHz K-weight = {mag_db_2k:.2f}dB (expected boost)"

    def test_low_freq_attenuation(self):
        """저주파(<100Hz) 영역에서 감쇠"""
        freqs = np.array([30.0, 1000.0])
        mag = k_weight_magnitude(freqs)
        # 30Hz는 1kHz보다 크게 감쇠되어야 함
        assert mag[0] < mag[1] * 0.5, (
            f"30Hz({mag[0]:.4f}) should be << 1kHz({mag[1]:.4f})"
        )

    def test_overall_shape(self):
        """저주파 감쇠 + 중고역 부스트 형태"""
        freqs = np.array([50.0, 200.0, 1000.0, 4000.0])
        mag = k_weight_magnitude(freqs)
        # 50Hz < 200Hz < 1kHz (저역 → 중역으로 증가)
        assert mag[0] < mag[1] < mag[2]

    def test_response_uses_supplied_sample_rate(self):
        """동일 물리 주파수에서도 레이트별 디지털 필터 계수를 사용한다."""
        freqs = np.array([1000.0, 4000.0])
        at_44k = k_weight_magnitude(freqs, 44100)
        at_48k = k_weight_magnitude(freqs, 48000)
        assert not np.allclose(at_44k, at_48k)


class TestComputeBandRms:
    """밴드별 RMS 측정"""

    def test_sine_in_correct_band(self):
        """1kHz 사인파 → mid 밴드(800-4000Hz)에 에너지 집중"""
        sr = 48000
        duration = 1.0
        n = int(sr * duration)
        t = np.arange(n, dtype=np.float32) / sr
        # 1kHz 사인파
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)

        rms = compute_band_rms(audio, sr)

        # mid 밴드가 가장 높아야 함
        band_names = [b.name for b in DEFAULT_BANDS]
        mid_idx = band_names.index("mid")
        for i, b in enumerate(DEFAULT_BANDS):
            if i != mid_idx:
                assert rms[mid_idx] > rms[i] + 10, (
                    f"mid({rms[mid_idx]:.1f}) should be >> {b.name}({rms[i]:.1f})"
                )

    def test_low_frequency_sine(self):
        """100Hz 사인파 → sub 밴드(20-200Hz)에 에너지 집중"""
        sr = 48000
        n = int(sr * 1.0)
        t = np.arange(n, dtype=np.float32) / sr
        audio = 0.5 * np.sin(2 * np.pi * 100 * t)

        rms = compute_band_rms(audio, sr)

        band_names = [b.name for b in DEFAULT_BANDS]
        sub_idx = band_names.index("sub")
        for i, b in enumerate(DEFAULT_BANDS):
            if i != sub_idx:
                assert rms[sub_idx] > rms[i] + 10

    def test_silence_returns_very_low(self):
        """무음 → 모든 밴드 -100dB 이하"""
        sr = 48000
        audio = np.zeros(sr, dtype=np.float32)
        rms = compute_band_rms(audio, sr)

        for r in rms:
            assert r <= -100.0

    def test_stereo_input(self):
        """스테레오 입력도 정상 처리"""
        sr = 48000
        n = sr
        t = np.arange(n, dtype=np.float32) / sr
        mono = 0.5 * np.sin(2 * np.pi * 1000 * t)
        stereo = np.stack([mono, mono])

        rms = compute_band_rms(stereo, sr)
        assert len(rms) == 4  # 4밴드

    def test_k_weighted_attenuates_sub(self):
        """K-weighting 적용 시 sub 밴드 에너지가 감쇠됨"""
        sr = 48000
        n = sr
        t = np.arange(n, dtype=np.float32) / sr
        # 50Hz 사인파 — sub 밴드
        audio = 0.5 * np.sin(2 * np.pi * 50 * t)

        rms_raw = compute_band_rms(audio, sr, k_weighted=False)
        rms_kw = compute_band_rms(audio, sr, k_weighted=True)

        band_names = [b.name for b in DEFAULT_BANDS]
        sub_idx = band_names.index("sub")
        # K-weighted sub RMS는 raw보다 낮아야 함 (저역 감쇠)
        assert rms_kw[sub_idx] < rms_raw[sub_idx], (
            f"K-weighted sub({rms_kw[sub_idx]:.1f}) should be < raw({rms_raw[sub_idx]:.1f})"
        )

    def test_k_weighted_boosts_high(self):
        """K-weighting 적용 시 high 밴드 에너지가 부스트됨"""
        sr = 48000
        n = sr
        t = np.arange(n, dtype=np.float32) / sr
        # 6kHz 사인파 — high 밴드
        audio = 0.5 * np.sin(2 * np.pi * 6000 * t)

        rms_raw = compute_band_rms(audio, sr, k_weighted=False)
        rms_kw = compute_band_rms(audio, sr, k_weighted=True)

        band_names = [b.name for b in DEFAULT_BANDS]
        high_idx = band_names.index("high")
        # K-weighted high RMS는 raw보다 높아야 함 (고역 부스트)
        assert rms_kw[high_idx] > rms_raw[high_idx], (
            f"K-weighted high({rms_kw[high_idx]:.1f}) should be > raw({rms_raw[high_idx]:.1f})"
        )


class TestPinkNoiseProfile:
    """Pink noise (-3dB/oct) 프로파일"""

    def test_decreasing_with_frequency(self):
        """높은 밴드일수록 낮은 RMS"""
        profile = pink_noise_profile()
        for i in range(len(profile) - 1):
            assert profile[i] > profile[i + 1], (
                f"band {i}({profile[i]:.1f}) should be > band {i+1}({profile[i+1]:.1f})"
            )

    def test_slope_approximately_3db_per_octave(self):
        """옥타브당 약 -3dB (실제로는 -10*log10(f2/f1))"""
        # sub center ≈ 63Hz, low center ≈ 400Hz
        # 차이 ≈ -10*log10(400/63) ≈ -8dB (2.67 옥타브 × 3dB)
        profile = pink_noise_profile(ref_level_db=0.0)
        sub_to_low_diff = profile[0] - profile[1]  # 양수여야 함
        assert 5 < sub_to_low_diff < 12, f"sub-low diff = {sub_to_low_diff:.1f}dB"

    def test_ref_level_shifts_all(self):
        """ref_level 변경 → 전체 프로파일 시프트"""
        p1 = pink_noise_profile(ref_level_db=-20.0)
        p2 = pink_noise_profile(ref_level_db=-10.0)
        # 모든 밴드가 10dB 차이
        for a, b in zip(p1, p2):
            assert b - a == pytest.approx(10.0, abs=0.01)

    def test_custom_bands(self):
        """커스텀 밴드 정의 지원"""
        bands = [
            BandDefinition("lo", 20, 500),
            BandDefinition("hi", 500, 20000),
        ]
        profile = pink_noise_profile(bands)
        assert len(profile) == 2
        assert profile[0] > profile[1]


class TestComputeAutomixGains:
    """gain 최적화"""

    def test_single_track_matching(self):
        """단일 트랙 → gain이 타겟에 맞춰짐"""
        track_rms = [[-26.0, -26.0, -26.0, -26.0]]
        target_rms = [-20.0, -20.0, -20.0, -20.0]

        gains, residual, _ = compute_automix_gains(track_rms, target_rms)

        assert len(gains) == 1
        assert gains[0] == pytest.approx(6.0, abs=1.0)

    def test_two_tracks_complementary(self):
        """상보적인 2트랙 → 각각에 적절한 gain 배분 (flat 모드)"""
        track_rms = [
            [-10.0, -15.0, -25.0, -35.0],
            [-35.0, -25.0, -15.0, -10.0],
        ]
        target_rms = [-20.0, -20.0, -20.0, -20.0]

        gains, residual, _ = compute_automix_gains(track_rms, target_rms)

        assert len(gains) == 2
        for g in gains:
            assert -24.0 <= g <= 12.0

    def test_gain_clipping(self):
        """gain이 범위를 초과하면 클리핑"""
        track_rms = [[-80.0, -80.0, -80.0, -80.0]]
        target_rms = [-10.0, -10.0, -10.0, -10.0]

        gains, _, _ = compute_automix_gains(
            track_rms, target_rms, max_gain_db=12.0, min_gain_db=-24.0
        )
        assert gains[0] <= 12.0

    def test_empty_tracks(self):
        """빈 트랙 리스트 → 빈 결과"""
        gains, residual, groups = compute_automix_gains([], [-20.0])
        assert gains == []
        assert residual == 0.0


class TestReferenceProfile:
    """레퍼런스 프로파일 추출"""

    def test_reference_from_file(self, tmp_path):
        """WAV 파일에서 프로파일 추출"""
        sr = 48000
        n = sr
        t = np.arange(n, dtype=np.float32) / sr
        # 500Hz 사인파
        audio = 0.5 * np.sin(2 * np.pi * 500 * t)
        ref_path = tmp_path / "ref.wav"
        sf.write(str(ref_path), audio, sr, subtype="PCM_24")

        profile = reference_profile(ref_path)

        assert len(profile) == 4
        # 500Hz는 low 밴드(200-800Hz) → low가 가장 높아야 함
        band_names = [b.name for b in DEFAULT_BANDS]
        low_idx = band_names.index("low")
        assert profile[low_idx] == max(profile)


class TestAutomix:
    """automix 통합 테스트"""

    def test_automix_pink_noise_k20(self, tmp_path):
        """pink noise + K-20 타겟으로 automix (기본 설정)"""
        sr = 48000
        n = sr

        # 트랙1: 저역 중심 (100Hz)
        t = np.arange(n, dtype=np.float32) / sr
        t1 = 0.5 * np.sin(2 * np.pi * 100 * t)
        p1 = tmp_path / "bass.wav"
        sf.write(str(p1), t1, sr, subtype="PCM_24")

        # 트랙2: 고역 중심 (8kHz)
        t2 = 0.3 * np.sin(2 * np.pi * 8000 * t)
        p2 = tmp_path / "highs.wav"
        sf.write(str(p2), t2, sr, subtype="PCM_24")

        result = automix([str(p1), str(p2)], target="pink")

        assert isinstance(result, AutomixResult)
        assert len(result.gains_db) == 2
        assert len(result.band_analysis) == 2
        assert result.target_profile["type"] == "pink_noise"
        assert result.target_profile["k_weighted"] is True
        assert result.target_profile["ref_level_db"] == K20_REF_LUFS

    def test_automix_reference(self, tmp_path):
        """레퍼런스 트랙 기반 automix"""
        sr = 48000
        n = sr
        t = np.arange(n, dtype=np.float32) / sr

        # 레퍼런스: 1kHz
        ref = 0.5 * np.sin(2 * np.pi * 1000 * t)
        ref_path = tmp_path / "ref.wav"
        sf.write(str(ref_path), ref, sr, subtype="PCM_24")

        # 트랙: 1kHz (같은 주파수 대역)
        t1 = 0.3 * np.sin(2 * np.pi * 1000 * t)
        p1 = tmp_path / "track.wav"
        sf.write(str(p1), t1, sr, subtype="PCM_24")

        result = automix(
            [str(p1)],
            target="reference",
            reference_path=str(ref_path),
        )

        assert result.target_profile["type"] == "reference"
        assert len(result.gains_db) == 1


# ---------------------------------------------------------------------------
# Residual branches: classify_tracks, relative-level keyword match, grouped staging
# ---------------------------------------------------------------------------


class TestClassifyTracks:
    def test_mixed_filenames(self):
        groups = classify_tracks([
            "01_Kick.wav", "02_Snare.wav", "Bass DI.wav",
            "LeadVox_comp.wav", "Piano.wav", "mystery_thing.wav",
        ])
        assert groups["drums"] == [0, 1]
        assert groups["bass"] == [2]
        assert groups["vocals"] == [3]
        assert groups["keys"] == [4]
        assert groups["other"] == [5]

    def test_returns_only_nonempty_groups(self):
        groups = classify_tracks(["kick.wav"])
        assert list(groups.keys()) == ["drums"]

    def test_empty_input(self):
        assert classify_tracks([]) == {}


class TestMatchRelativeKeyword:
    def test_most_specific_longest_keyword_wins(self):
        levels = {"kick": 0.0, "oh": -8.0, "overhead": -10.0, "_default": -6.0}
        # "overhead" is longer than "oh" → -10.0
        assert _match_relative_keyword("OH_Overhead_L.wav", levels) == -10.0

    def test_fallback_default(self):
        assert _match_relative_keyword("whatever.wav", {"kick": 0.0, "_default": -5.0}) == -5.0

    def test_no_default_returns_zero(self):
        assert _match_relative_keyword("x.wav", {"kick": 0.0}) == 0.0


class TestGenreProfile:
    def test_unknown_genre_falls_back_to_default(self):
        prof = genre_profile("not_a_genre")
        default = genre_profile("default")
        assert prof == default

    def test_missing_band_name_uses_minus30(self):
        bands = [BandDefinition("unknown_band", 10, 20)]
        assert genre_profile("pop", bands) == [-30.0]

    def test_custom_band_order(self):
        bands = [BandDefinition("high", 4000, 20000), BandDefinition("sub", 20, 200)]
        prof = genre_profile("rock", bands)
        assert prof == [GENRE_PROFILES["rock"]["bands"]["high"],
                        GENRE_PROFILES["rock"]["bands"]["sub"]]


class TestKWeightSampleRateValidation:
    def test_zero_sample_rate_raises(self):
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            k_weight_magnitude(np.array([1000.0]), 0)


class TestComputeBroadbandRmsUnweighted:
    def test_unweighted_matches_plain_rms(self):
        t = np.arange(48000) / 48000
        audio = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        db = compute_broadband_rms_db(audio, 48000, k_weighted=False)
        expected = 20.0 * np.log10(0.5 / np.sqrt(2))
        assert db == pytest.approx(expected, abs=0.05)

    def test_unweighted_silence_returns_floor(self):
        # unweighted path floors at 1e-10 linear amplitude → -200 dBFS
        assert compute_broadband_rms_db(np.zeros(1000, dtype=np.float32), 48000,
                                        k_weighted=False) == pytest.approx(-200.0)

    def test_weighted_too_short_returns_floor(self):
        assert compute_broadband_rms_db(np.zeros(100, dtype=np.float32), 48000) == -120.0


class TestComputeAutomixGainsGrouped:
    def _synthetic(self):
        # 3 tracks across 2 groups (drums kick/snare, bass)
        tracks_band_rms = [
            [-20.0, -25.0, -30.0, -40.0],   # kick
            [-30.0, -28.0, -26.0, -42.0],   # snare
            [-18.0, -20.0, -34.0, -50.0],   # bass
        ]
        target = [-25.0, -26.0, -28.0, -35.0]
        paths = ["kick.wav", "snare.wav", "bass.wav"]
        rms_db = [-18.0, -22.0, -16.0]
        return tracks_band_rms, target, paths, rms_db

    def test_grouped_gains_and_groups_info(self):
        tracks, target, paths, rms_db = self._synthetic()
        gains, residual, groups = compute_automix_gains(
            tracks, target, track_paths=paths, track_rms_db=rms_db,
        )
        assert len(gains) == 3
        assert groups is not None
        assert "drums" in groups and "bass" in groups
        assert isinstance(residual, float)

    def test_group_balance_override(self):
        tracks, target, paths, rms_db = self._synthetic()
        gains, _res, _g = compute_automix_gains(
            tracks, target, track_paths=paths, track_rms_db=rms_db,
            group_balance={"drums": 0.0, "bass": 0.0, "other": 0.0},
        )
        assert len(gains) == 3

    def test_grouped_without_drums_uses_loudest_group(self):
        tracks = [[-20.0, -25.0, -30.0, -40.0], [-18.0, -20.0, -34.0, -50.0]]
        target = [-25.0, -26.0, -28.0, -35.0]
        _gains, _res, groups = compute_automix_gains(
            tracks, target, track_paths=["bass.wav", "keys.wav"],
            track_rms_db=[-18.0, -16.0],
        )
        assert "drums" not in groups

    def test_flat_fallback_when_paths_missing(self):
        tracks, target, _paths, _rms = self._synthetic()
        gains, residual, groups = compute_automix_gains(tracks, target)
        assert groups is None
        assert len(gains) == 3

    def test_zero_power_track_skips_flat_gain_assignment(self):
        # band power 10**(-120) < the 1e-20 guard → gain left at 0.0, not boosted
        tracks = [[-1200.0] * 4, [-20.0, -25.0, -30.0, -40.0]]
        target = [-25.0, -26.0, -28.0, -35.0]
        gains, _res, _g = compute_automix_gains(tracks, target)
        assert gains[0] == 0.0

    def test_gain_clamped_to_bounds(self):
        tracks = [[-120.0, -120.0, -120.0, -120.0]]
        target = [-25.0, -26.0, -28.0, -35.0]
        gains, _res, _g = compute_automix_gains(tracks, target, min_gain_db=-6.0, max_gain_db=3.0)
        assert -6.0 <= gains[0] <= 3.0


class TestAutomixEntryPoint:
    def _tracks(self, tmp_path):
        sr = 48000
        t = np.arange(sr * 2) / sr
        paths = []
        for name, freq in [("kick.wav", 60.0), ("bass.wav", 100.0)]:
            audio = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            p = tmp_path / name
            sf.write(str(p), audio, sr, subtype="FLOAT")
            paths.append(str(p))
        return paths

    def test_genre_target_path(self, tmp_path):
        result = automix(self._tracks(tmp_path), target="rock")
        assert result.target_profile["type"] == "genre"
        assert result.target_profile["genre"] == "rock"
        assert "bands" in result.target_profile
        assert result.groups is not None
        assert len(result.gains_db) == 2

    def test_reference_target_path(self, tmp_path):
        paths = self._tracks(tmp_path)
        result = automix(paths, target="reference", reference_path=paths[0])
        assert result.target_profile["type"] == "reference"
        assert result.target_profile["path"] == paths[0]

    def test_reference_target_without_path_falls_back_to_pink(self, tmp_path):
        result = automix(self._tracks(tmp_path), target="reference", reference_path=None)
        assert result.target_profile["type"] == "pink_noise"

    def test_custom_bands_respected(self, tmp_path):
        bands = [BandDefinition("low", 20, 1000), BandDefinition("high", 1000, 20000)]
        result = automix(self._tracks(tmp_path), target="pop", bands=bands)
        assert set(result.target_profile["bands"].keys()) == {"low", "high"}
        assert set(result.band_analysis[0]["bands"].keys()) == {"low", "high"}

    def test_to_dict_serializable(self, tmp_path):
        result = automix(self._tracks(tmp_path))
        d = result.to_dict()
        assert isinstance(d["gains_db"], list)
        assert d["target_profile"]["type"] == "pink_noise"
