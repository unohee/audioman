# tests/unit/test_dsp.py

import numpy as np
import pytest
from audioman.core.dsp import normalize, gate, trim_silence, fade_in, fade_out, gain, trim


class TestNormalize:
    def test_peak_normalize(self, test_audio):
        result = normalize(test_audio, peak_db=0.0)
        peak = np.max(np.abs(result))
        assert abs(peak - 1.0) < 0.01

    def test_peak_normalize_minus6(self, test_audio):
        result = normalize(test_audio, peak_db=-6.0)
        peak = np.max(np.abs(result))
        expected = 10 ** (-6.0 / 20.0)
        assert abs(peak - expected) < 0.02

    def test_rms_normalize(self, test_audio):
        result = normalize(test_audio, target_rms_db=-20.0)
        rms = np.sqrt(np.mean(result**2))
        expected = 10 ** (-20.0 / 20.0)
        assert abs(rms - expected) < 0.02

    def test_silent_unchanged(self, silent_audio):
        result = normalize(silent_audio, peak_db=0.0)
        assert np.max(np.abs(result)) == 0.0


class TestGate:
    def test_gate_removes_silence(self, sample_rate):
        audio = np.zeros((2, sample_rate), dtype=np.float32)
        t = np.linspace(0, 0.5, sample_rate // 2, dtype=np.float32)
        audio[:, sample_rate // 2:] = 0.5 * np.sin(2 * np.pi * 440 * t)
        result = gate(audio, sample_rate, threshold_db=-40.0)
        # the RMS of the leading silence region must be low
        rms_first = np.sqrt(np.mean(result[:, :sample_rate // 4]**2))
        assert rms_first < 0.01


class TestFade:
    def test_fade_in(self, test_audio, sample_rate):
        fade_samples = sample_rate // 10  # 0.1 s
        result = fade_in(test_audio, fade_samples)
        assert abs(result[0, 0]) < 0.01
        np.testing.assert_allclose(result[:, -1000:], test_audio[:, -1000:], atol=1e-6)

    def test_fade_out(self, test_audio, sample_rate):
        fade_samples = sample_rate // 10
        result = fade_out(test_audio, fade_samples)
        assert abs(result[0, -1]) < 0.01
        np.testing.assert_allclose(result[:, :1000], test_audio[:, :1000], atol=1e-6)

    @pytest.mark.parametrize("fade", [fade_in, fade_out])
    def test_zero_length_fade_preserves_audio(self, test_audio, fade):
        np.testing.assert_array_equal(fade(test_audio, 0), test_audio)


class TestGain:
    def test_gain_6db(self, test_audio):
        result = gain(test_audio, db=6.0)
        ratio = np.max(np.abs(result)) / np.max(np.abs(test_audio))
        expected = 10 ** (6.0 / 20.0)
        assert abs(ratio - expected) < 0.1

    def test_gain_zero(self, test_audio):
        result = gain(test_audio, db=0.0)
        np.testing.assert_allclose(result, test_audio, atol=1e-6)

    def test_gain_negative(self, test_audio):
        result = gain(test_audio, db=-6.0)
        assert np.max(np.abs(result)) < np.max(np.abs(test_audio))


class TestFadeCurves:
    def test_linear_curve_endpoints(self, test_audio, sample_rate):
        from audioman.core.dsp import fade_in, fade_out
        n = sample_rate // 10
        fi = fade_in(test_audio, n, curve="linear")
        fo = fade_out(test_audio, n, curve="linear")
        assert abs(fi[0, 0]) < 1e-6
        assert abs(fo[0, -1]) < 1e-6

    def test_cosine_curve_smooth(self, test_audio, sample_rate):
        from audioman.core.dsp import fade_in
        n = sample_rate // 10
        fi = fade_in(test_audio, n, curve="cosine")
        # cosine S-curve: the derivative must be 0 at the start and end to be smooth
        # the difference between the first two samples < that between the middle two samples
        diff_start = abs(fi[0, 1] - fi[0, 0])
        diff_mid = abs(fi[0, n // 2 + 1] - fi[0, n // 2])
        assert diff_start < diff_mid

    def test_equal_power_midpoint(self, test_audio, sample_rate):
        from audioman.core.dsp import fade_in
        n = sample_rate // 10
        fi = fade_in(test_audio, n, curve="equal_power")
        # equal_power midpoint: sqrt(0.5) ≈ 0.707 (linear gives 0.5)
        original_mid = test_audio[0, n // 2]
        if abs(original_mid) > 0.01:  # avoid silence
            ratio = fi[0, n // 2] / original_mid
            assert 0.65 < ratio < 0.75

    def test_unknown_curve_raises(self, test_audio):
        from audioman.core.dsp import fade_in
        with pytest.raises(ValueError, match="Unknown fade curve"):
            fade_in(test_audio, 100, curve="bouncy")

    def test_all_curves_accept(self, test_audio, sample_rate):
        from audioman.core.dsp import fade_in, fade_out, FADE_CURVES
        n = sample_rate // 100
        for curve in FADE_CURVES:
            fade_in(test_audio, n, curve=curve)
            fade_out(test_audio, n, curve=curve)


class TestPad:
    def test_pad_head_only(self, test_audio, sample_rate):
        from audioman.core.dsp import pad
        original_len = test_audio.shape[1]
        result = pad(test_audio, head_samples=sample_rate // 2)  # 0.5 s
        assert result.shape[1] == original_len + sample_rate // 2
        # the head padding is silent
        assert np.max(np.abs(result[:, :sample_rate // 2])) == 0.0
        # the original is preserved after it
        np.testing.assert_allclose(result[:, sample_rate // 2:], test_audio, atol=1e-6)

    def test_pad_tail_only(self, test_audio, sample_rate):
        from audioman.core.dsp import pad
        result = pad(test_audio, tail_samples=sample_rate)  # 1 s
        assert result.shape[1] == test_audio.shape[1] + sample_rate
        assert np.max(np.abs(result[:, -sample_rate:])) == 0.0

    def test_pad_both(self, test_audio, sample_rate):
        from audioman.core.dsp import pad
        head = sample_rate // 4
        tail = sample_rate // 2
        result = pad(test_audio, head_samples=head, tail_samples=tail)
        assert result.shape[1] == test_audio.shape[1] + head + tail
        assert np.max(np.abs(result[:, :head])) == 0.0
        assert np.max(np.abs(result[:, -tail:])) == 0.0

    def test_pad_mono(self, test_audio_mono, sample_rate):
        from audioman.core.dsp import pad
        result = pad(test_audio_mono, head_samples=100, tail_samples=200)
        assert result.shape == (test_audio_mono.shape[0] + 300,)

    def test_pad_zero_returns_copy(self, test_audio):
        from audioman.core.dsp import pad
        result = pad(test_audio, head_samples=0, tail_samples=0)
        np.testing.assert_array_equal(result, test_audio)
        assert result is not test_audio  # copy, not same object

    def test_pad_negative_raises(self, test_audio):
        from audioman.core.dsp import pad
        with pytest.raises(ValueError, match="cannot be negative"):
            pad(test_audio, head_samples=-1)


class TestRemoveDC:
    def test_remove_dc_offset(self, sample_rate):
        from audioman.core.dsp import remove_dc, measure_dc_offset
        # a different DC bias per channel
        t = np.linspace(0, 1, sample_rate, dtype=np.float32)
        sine = 0.3 * np.sin(2 * np.pi * 440 * t)
        ch_l = sine + 0.1   # +0.1 DC
        ch_r = sine - 0.05  # -0.05 DC
        stereo = np.stack([ch_l, ch_r])

        before = measure_dc_offset(stereo)
        assert abs(before[0] - 0.1) < 0.01
        assert abs(before[1] - (-0.05)) < 0.01

        cleaned = remove_dc(stereo)
        after = measure_dc_offset(cleaned)
        assert abs(after[0]) < 1e-6
        assert abs(after[1]) < 1e-6

    def test_remove_dc_preserves_signal_shape(self, test_audio):
        from audioman.core.dsp import remove_dc
        result = remove_dc(test_audio)
        assert result.shape == test_audio.shape
        # the original DC is nearly 0, so the signal is nearly unchanged
        np.testing.assert_allclose(result, test_audio, atol=1e-3)

    def test_remove_dc_mono(self, test_audio_mono):
        from audioman.core.dsp import remove_dc
        biased = test_audio_mono + 0.2
        cleaned = remove_dc(biased)
        assert abs(np.mean(cleaned)) < 1e-6


class TestTrim:
    def test_trim_basic(self, test_audio):
        result = trim(test_audio, start=100, end=200)
        assert result.shape[1] == 100

    def test_trim_silence(self, sample_rate):
        # 0.2 s silence + 0.6 s tone + 0.2 s silence
        audio = np.zeros((2, sample_rate), dtype=np.float32)
        start = int(0.2 * sample_rate)
        end = int(0.8 * sample_rate)
        t = np.linspace(0, 0.6, end - start, dtype=np.float32)
        audio[:, start:end] = 0.5 * np.sin(2 * np.pi * 440 * t)
        result = trim_silence(audio, sample_rate, threshold_db=-40.0)
        # the length must shrink after trimming
        assert result.shape[1] < audio.shape[1]
        # there must be sound at the start
        assert np.max(np.abs(result[:, :100])) > 0.01


class TestCutRegion:
    """cut_region: delete a region + join the remaining sides."""

    def test_removes_middle_segment(self):
        from audioman.core.dsp import cut_region
        audio = np.arange(10, dtype=np.float32)
        out = cut_region(audio, start=2, end=5)
        np.testing.assert_array_equal(out, np.array([0, 1, 5, 6, 7, 8, 9], dtype=np.float32))

    def test_stereo_removes_middle_segment(self):
        from audioman.core.dsp import cut_region
        audio = np.stack([np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32)])
        out = cut_region(audio, start=2, end=5)
        assert out.shape == (2, 7)
        np.testing.assert_array_equal(out[0], np.array([0, 1, 5, 6, 7, 8, 9], dtype=np.float32))

    def test_empty_range_returns_copy(self):
        from audioman.core.dsp import cut_region
        audio = np.arange(6, dtype=np.float32)
        out = cut_region(audio, start=3, end=3)
        np.testing.assert_array_equal(out, audio)
        assert out is not audio

    def test_out_of_range_clamped(self):
        from audioman.core.dsp import cut_region
        audio = np.arange(6, dtype=np.float32)
        # end > n, start < 0 → clamp to [0, n]
        out = cut_region(audio, start=-5, end=100)
        assert out.shape == (0,)

    def test_crossfade_overlaps_by_cf_samples(self):
        from audioman.core.dsp import cut_region
        audio = (np.arange(1000, dtype=np.float32) / 1000.0)
        out = cut_region(audio, start=200, end=400, crossfade_samples=50)
        # 200 cut out, plus the cf-sample tail/head overlap is consumed
        assert out.shape == (1000 - 200 - 50,)
        # 1000-200-50=750 samples: head ends at index 149, mixed spans 150..199
        mixed = out[150:200]
        # complementary blend of left tail (0.150..0.199) and right head (0.400..0.449)
        assert mixed[0] == pytest.approx(0.150, abs=1e-3)
        assert mixed[-1] == pytest.approx(0.449, abs=1e-3)
        assert np.all(np.diff(mixed) > 0)  # rising ramp stays monotonic

    def test_crossfade_zero_is_plain_concat(self):
        from audioman.core.dsp import cut_region
        audio = np.ones((2, 100), dtype=np.float32)
        out = cut_region(audio, start=10, end=20, crossfade_samples=0)
        assert out.shape == (2, 90)
        np.testing.assert_allclose(out, 1.0)

    def test_crossfade_mono(self):
        from audioman.core.dsp import cut_region
        audio = np.ones(100, dtype=np.float32)
        out = cut_region(audio, start=10, end=20, crossfade_samples=5)
        assert out.shape == (85,)

    def test_crossfade_larger_than_sides_clamped(self):
        from audioman.core.dsp import cut_region
        audio = np.ones((2, 100), dtype=np.float32)
        # cf clamps to min(left=20, right=30) = 20 → 100 - 50 - 20
        out = cut_region(audio, start=20, end=70, crossfade_samples=999)
        assert out.shape == (2, 30)

    def test_crossfade_at_start_has_no_left_to_blend(self):
        from audioman.core.dsp import cut_region
        audio = np.arange(100, dtype=np.float32)
        # start=0 → left is empty, clamp makes cf 0 → plain concat path
        out = cut_region(audio, start=0, end=30, crossfade_samples=10)
        assert out.shape == (70,)
        np.testing.assert_array_equal(out, audio[30:])


class TestSplice:
    def test_insert_at_position(self):
        from audioman.core.dsp import splice
        base = np.array([1, 2, 3], dtype=np.float32)
        ins = np.array([9, 9], dtype=np.float32)
        out = splice(base, ins, position=1, mode="insert")
        np.testing.assert_array_equal(out, np.array([1, 9, 9, 2, 3], dtype=np.float32))

    def test_insert_position_clamped_past_end(self):
        from audioman.core.dsp import splice
        base = np.array([1, 2, 3], dtype=np.float32)
        ins = np.array([9], dtype=np.float32)
        out = splice(base, ins, position=99, mode="insert")
        np.testing.assert_array_equal(out, np.array([1, 2, 3, 9], dtype=np.float32))

    def test_overwrite_keeps_length(self):
        from audioman.core.dsp import splice
        base = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        ins = np.array([9, 9], dtype=np.float32)
        out = splice(base, ins, position=1, mode="overwrite")
        np.testing.assert_array_equal(out, np.array([1, 9, 9, 4, 5], dtype=np.float32))

    def test_overwrite_past_end_truncates(self):
        from audioman.core.dsp import splice
        base = np.array([1, 2, 3], dtype=np.float32)
        ins = np.array([9, 9, 9, 9], dtype=np.float32)
        out = splice(base, ins, position=2, mode="overwrite")
        np.testing.assert_array_equal(out, np.array([1, 2, 9], dtype=np.float32))

    def test_overwrite_fully_past_end_is_noop(self):
        from audioman.core.dsp import splice
        base = np.array([1, 2, 3], dtype=np.float32)
        ins = np.array([9], dtype=np.float32)
        out = splice(base, ins, position=3, mode="overwrite")
        np.testing.assert_array_equal(out, base)

    def test_mix_sums_at_position(self):
        from audioman.core.dsp import splice
        base = np.array([1, 2, 3, 4], dtype=np.float32)
        ins = np.array([10, 10], dtype=np.float32)
        out = splice(base, ins, position=1, mode="mix")
        np.testing.assert_array_equal(out, np.array([1, 12, 13, 4], dtype=np.float32))

    def test_mix_fully_past_end_is_noop(self):
        from audioman.core.dsp import splice
        base = np.array([1, 2, 3], dtype=np.float32)
        out = splice(base, np.array([5], dtype=np.float32), position=3, mode="mix")
        np.testing.assert_array_equal(out, base)

    def test_channel_mismatch_raises(self):
        from audioman.core.dsp import splice
        base = np.ones((2, 10), dtype=np.float32)
        ins = np.ones(5, dtype=np.float32)
        with pytest.raises(ValueError, match="(?i)channel count mismatch"):
            splice(base, ins, position=0)

    def test_unknown_mode_raises(self):
        from audioman.core.dsp import splice
        base = np.ones(5, dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown splice mode"):
            splice(base, np.ones(2, dtype=np.float32), position=0, mode="bogus")

    def test_insert_with_crossfade_left_and_right(self):
        from audioman.core.dsp import splice
        base = np.arange(200, dtype=np.float32) / 200.0
        ins = np.zeros(100, dtype=np.float32)
        out = splice(base, ins, position=100, mode="insert", crossfade_samples=20)
        # cf consumed from both base sides: 80 + 100 + 80
        assert out.shape == (260,)
        # left boundary mixed region (indices 80..99) blends base tail into inserted zeros
        left_mixed = out[80:100]
        assert left_mixed[0] == pytest.approx(80 / 200.0, abs=1e-3)
        assert left_mixed[-1] == pytest.approx(0.0, abs=1e-3)
        # right boundary region (indices 160..179) blends inserted zeros into base head
        right_mixed = out[160:180]
        assert right_mixed[0] == pytest.approx(0.0, abs=1e-3)
        assert right_mixed[-1] == pytest.approx(119 / 200.0, abs=1e-3)

    def test_insert_mono_with_crossfade(self):
        from audioman.core.dsp import splice
        base = np.ones(200, dtype=np.float32)
        ins = np.ones(100, dtype=np.float32)
        out = splice(base, ins, position=100, mode="insert", crossfade_samples=10)
        assert out.shape == (280,)

    def test_overwrite_mono(self):
        from audioman.core.dsp import splice
        base = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        out = splice(base, np.array([9, 9], dtype=np.float32), position=1, mode="overwrite")
        np.testing.assert_array_equal(out, np.array([1, 9, 9, 4, 5], dtype=np.float32))

    def test_mix_mono(self):
        from audioman.core.dsp import splice
        base = np.array([1, 2, 3, 4], dtype=np.float32)
        out = splice(base, np.array([10, 10], dtype=np.float32), position=1, mode="mix")
        np.testing.assert_array_equal(out, np.array([1, 12, 13, 4], dtype=np.float32))

    def test_overwrite_stereo(self):
        from audioman.core.dsp import splice
        base = np.zeros((2, 10), dtype=np.float32)
        ins = np.ones((2, 3), dtype=np.float32)
        out = splice(base, ins, position=2, mode="overwrite")
        assert out.shape == (2, 10)
        np.testing.assert_array_equal(out[:, 2:5], 1.0)
        np.testing.assert_array_equal(out[:, :2], 0.0)
        np.testing.assert_array_equal(out[:, 5:], 0.0)

    def test_mix_stereo(self):
        from audioman.core.dsp import splice
        base = np.ones((2, 10), dtype=np.float32)
        ins = np.full((2, 3), 0.5, dtype=np.float32)
        out = splice(base, ins, position=2, mode="mix")
        assert out.shape == (2, 10)
        np.testing.assert_allclose(out[:, 2:5], 1.5)
        np.testing.assert_allclose(out[:, :2], 1.0)

    def test_insert_stereo_crossfade_both_boundaries(self):
        from audioman.core.dsp import splice
        base = np.stack([np.arange(100, dtype=np.float32) / 100.0] * 2)
        ins = np.full((2, 40), 0.5, dtype=np.float32)
        out = splice(base, ins, position=50, mode="insert", crossfade_samples=10)
        # left_head(40) + inserted(40) + right(40) with both 10-sample boundaries blended
        assert out.shape == (2, 120)
        # left boundary (indices 40..49) starts at base[40]=0.40 and reaches the insert level
        assert out[0, 40] == pytest.approx(0.40, abs=1e-3)
        assert out[0, 49] == pytest.approx(0.50, abs=1e-3)
        # right plain head starts at base[60]=0.60 after the blended insert tail
        assert out[0, 80] == pytest.approx(0.60, abs=1e-3)
        assert np.all(np.diff(out[0, 40:]) >= -1e-6)


class TestConcat:
    def test_empty_list_returns_empty(self):
        from audioman.core.dsp import concat
        out = concat([])
        assert out.shape == (0,)

    def test_simple_concat(self):
        from audioman.core.dsp import concat
        out = concat([np.array([1, 2], dtype=np.float32), np.array([3], dtype=np.float32)])
        np.testing.assert_array_equal(out, np.array([1, 2, 3], dtype=np.float32))

    def test_channel_mismatch_raises(self):
        from audioman.core.dsp import concat
        with pytest.raises(ValueError, match="clips\\[1\\] channel count mismatch"):
            concat([np.ones(5, dtype=np.float32), np.ones((2, 5), dtype=np.float32)])

    def test_crossfade_concatenation(self):
        from audioman.core.dsp import concat
        a = np.ones(100, dtype=np.float32)
        b = np.ones(100, dtype=np.float32)
        out = concat([a, b], crossfade_samples=10)
        assert out.shape == (190,)
        # boundary region blended (still 1.0 for equal signals but no click)
        np.testing.assert_allclose(out[:90], 1.0, atol=1e-6)

    def test_crossfade_zero_clip_pads_directly(self):
        from audioman.core.dsp import concat
        a = np.ones(50, dtype=np.float32)
        empty = np.zeros(0, dtype=np.float32)
        out = concat([a, empty], crossfade_samples=10)
        assert out.shape == (50,)


class TestFadeEdgeCases:
    def test_negative_fade_samples_raises(self, test_audio):
        from audioman.core.dsp import fade_in, fade_out
        with pytest.raises(ValueError, match="cannot be negative"):
            fade_in(test_audio, -1)
        with pytest.raises(ValueError, match="cannot be negative"):
            fade_out(test_audio, -1)

    def test_fade_mono(self, test_audio_mono):
        from audioman.core.dsp import fade_in, fade_out
        fi = fade_in(test_audio_mono, 100)
        assert abs(fi[0]) < 1e-6
        fo = fade_out(test_audio_mono, 100)
        assert abs(fo[-1]) < 1e-6

    def test_fade_longer_than_audio_clamps(self):
        from audioman.core.dsp import fade_in
        audio = np.ones(50, dtype=np.float32)
        out = fade_in(audio, 500)
        assert out.shape == (50,)
        assert abs(out[0]) < 1e-6

    def test_fade_curve_zero_length(self):
        from audioman.core.dsp import _fade_curve
        assert _fade_curve(0, "linear", "in").shape == (0,)

    def test_exponential_and_logarithmic_normalized_endpoints(self):
        from audioman.core.dsp import _fade_curve
        for kind in ("exponential", "logarithmic"):
            c = _fade_curve(64, kind, "in")
            assert abs(c[0]) < 1e-5
            assert abs(c[-1] - 1.0) < 1e-5
            assert np.all(np.diff(c) >= -1e-6)  # monotonic non-decreasing


class TestTrimBranches:
    def test_trim_mono(self):
        from audioman.core.dsp import trim
        audio = np.arange(10, dtype=np.float32)
        out = trim(audio, start=2, end=5)
        np.testing.assert_array_equal(out, np.array([2, 3, 4], dtype=np.float32))

    def test_trim_silence_mono(self, sample_rate):
        from audioman.core.dsp import trim_silence
        audio = np.zeros(sample_rate, dtype=np.float32)
        t = np.linspace(0, 0.5, sample_rate // 2, dtype=np.float32)
        audio[sample_rate // 4:sample_rate // 4 + len(t)] = 0.5 * np.sin(2 * np.pi * 440 * t)
        out = trim_silence(audio, sample_rate, threshold_db=-40.0)
        assert out.shape[0] < audio.shape[0]

    def test_trim_silence_all_silent_returns_original(self, sample_rate):
        from audioman.core.dsp import trim_silence
        audio = np.zeros((2, sample_rate), dtype=np.float32)
        out = trim_silence(audio, sample_rate)
        np.testing.assert_array_equal(out, audio)

    def test_trim_silence_pad_expands_window(self, sample_rate):
        from audioman.core.dsp import trim_silence
        audio = np.zeros((2, sample_rate), dtype=np.float32)
        audio[:, sample_rate // 2:sample_rate // 2 + 1000] = 0.5
        no_pad = trim_silence(audio, sample_rate, pad_samples=0)
        padded = trim_silence(audio, sample_rate, pad_samples=500)
        assert padded.shape[1] == no_pad.shape[1] + 1000


class TestNormalizeBranches:
    def test_rms_normalize_silent_returns_unchanged(self, silent_audio):
        from audioman.core.dsp import normalize
        out = normalize(silent_audio, target_rms_db=-20.0)
        assert np.max(np.abs(out)) == 0.0

    def test_no_target_returns_copy(self, test_audio):
        from audioman.core.dsp import normalize
        out = normalize(test_audio)
        np.testing.assert_allclose(out, test_audio, atol=1e-6)

    def test_measure_dc_offset_mono(self):
        from audioman.core.dsp import measure_dc_offset
        audio = np.full(100, 0.3, dtype=np.float32)
        assert measure_dc_offset(audio) == pytest.approx([0.3], abs=1e-6)


class TestGateBranches:
    def test_gate_mono(self, sample_rate):
        from audioman.core.dsp import gate
        audio = np.zeros(sample_rate, dtype=np.float32)
        t = np.linspace(0, 0.5, sample_rate // 2, dtype=np.float32)
        audio[sample_rate // 2:] = 0.5 * np.sin(2 * np.pi * 440 * t)
        out = gate(audio, sample_rate, threshold_db=-40.0)
        assert out.shape == audio.shape
        assert np.sqrt(np.mean(out[:sample_rate // 4] ** 2)) < 0.01

    def test_gate_stereo(self, sample_rate):
        from audioman.core.dsp import gate
        audio = np.zeros((2, sample_rate), dtype=np.float32)
        t = np.linspace(0, 0.5, sample_rate // 2, dtype=np.float32)
        audio[:, sample_rate // 2:] = 0.5 * np.sin(2 * np.pi * 440 * t)
        out = gate(audio, sample_rate, threshold_db=-40.0)
        assert out.shape == audio.shape


class TestHelpers:
    def test_length_and_channels(self):
        from audioman.core.dsp import _length, _channels
        mono = np.zeros(10, dtype=np.float32)
        stereo = np.zeros((2, 10), dtype=np.float32)
        assert _length(mono) == 10 and _channels(mono) == 1
        assert _length(stereo) == 10 and _channels(stereo) == 2

    def test_concat_time_all_empty_returns_zeros(self):
        from audioman.core.dsp import _concat_time
        out = _concat_time([np.zeros(0, dtype=np.float32)])
        assert out.shape == (0,)
