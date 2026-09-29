# tests/unit/test_residual_core.py — edge branches in small core modules:
# vad helpers, analysis, discontinuity, detectors, config paths/settings, rt_bench.
#
# These modules are mostly covered already; this file fills the remaining
# boundary/error paths without duplicating existing suites.

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# core/vad.py
# ---------------------------------------------------------------------------

from audioman.core.vad import (  # noqa: E402
    Segment,
    _to_mono_16k,
    detect_speech,
    invert_to_noise,
    merge_segments,
)


class TestToMono16k:
    def test_mono_input_skips_downmix(self):
        mono = np.zeros(16000, dtype=np.float32)
        out = _to_mono_16k(mono, 16000)
        assert out.ndim == 1
        assert len(out) == 16000

    def test_stereo_downmixed_and_resampled(self):
        stereo = np.stack([np.ones(48000), np.ones(48000)]).astype(np.float32)
        out = _to_mono_16k(stereo, 48000)
        assert out.ndim == 1
        assert len(out) == pytest.approx(16000, abs=2)

    def test_16k_stereo_not_resampled(self):
        stereo = np.stack([np.ones(1600), np.ones(1600)]).astype(np.float32)
        out = _to_mono_16k(stereo, 16000)
        assert len(out) == 1600


class TestDetectSpeech:
    """silero is stubbed — the conversion loop is what is under test."""

    @pytest.fixture
    def stub_silero(self, monkeypatch):
        timestamps = [{"start": 0, "end": 8000}, {"start": 8000, "end": 8000}]

        fake = types.ModuleType("silero_vad")
        fake.get_speech_timestamps = lambda tensor, model, **kw: timestamps
        fake.load_silero_vad = lambda: object()
        monkeypatch.setitem(sys.modules, "silero_vad", fake)

        from audioman.core import vad as vad_mod
        monkeypatch.setattr(vad_mod, "_SILERO_VAD_MODEL", object())
        return timestamps

    def test_timestamps_scaled_to_source_rate(self, stub_silero):
        audio = np.zeros(48000, dtype=np.float32)
        segs = detect_speech(audio, 48000)
        # 16k indices scaled by 48000/16000 == 3; the empty second entry is dropped
        assert len(segs) == 1
        assert segs[0].start == 0
        assert segs[0].end == 24000
        assert segs[0].kind == "speech"

    def test_end_clamped_to_audio_length(self, stub_silero):
        audio = np.zeros(1000, dtype=np.float32)
        segs = detect_speech(audio, 16000)
        assert segs[0].end <= 1000


class TestInvertToNoise:
    def test_gaps_become_noise_segments(self):
        speech = [Segment(100, 200, "speech"), Segment(400, 500, "speech")]
        noise = invert_to_noise(speech, 600)
        assert [(s.start, s.end) for s in noise] == [(0, 100), (200, 400), (500, 600)]
        assert all(s.kind == "noise" for s in noise)

    def test_overlapping_speech_extends_cursor(self):
        # second segment starts before the first ends → no noise gap between them
        speech = [Segment(0, 300, "speech"), Segment(100, 500, "speech")]
        noise = invert_to_noise(speech, 500)
        assert noise == []

    def test_full_coverage_no_noise(self):
        assert invert_to_noise([Segment(0, 1000, "speech")], 1000) == []


class TestMergeSegments:
    def test_sorted_timeline(self):
        speech = [Segment(200, 300, "speech")]
        noise = [Segment(0, 200, "noise")]
        merged = merge_segments(speech, noise)
        assert [s.start for s in merged] == [0, 200]

    def test_empty_inputs(self):
        assert merge_segments([], []) == []


# ---------------------------------------------------------------------------
# core/analysis.py
# ---------------------------------------------------------------------------

from audioman.core import analysis  # noqa: E402


class TestAnalysisToMono:
    def test_mono_passthrough(self):
        mono = np.ones(10, dtype=np.float32)
        assert analysis._to_mono(mono) is mono

    def test_stereo_averaged(self):
        stereo = np.stack([np.ones(10), np.zeros(10)]).astype(np.float32)
        assert analysis._to_mono(stereo) == pytest.approx(np.full(10, 0.5))


class TestComputeSummaryEmpty:
    def test_empty_metrics_use_zero_defaults(self):
        metrics = analysis.FrameMetrics()
        summary = analysis.compute_summary(metrics)
        assert summary["rms"] == {"mean": 0, "min": 0, "max": 0, "std": 0}


class TestLongTermSpectrumQuietFrames:
    def test_frames_below_min_rms_are_skipped(self):
        audio = np.zeros(48000, dtype=np.float32)
        _freqs, power, n_used = analysis.long_term_spectrum(
            audio, 48000, fft_size=1024, min_rms=0.01
        )
        assert n_used == 0
        assert not np.any(power)

    def test_loud_frames_accumulate(self):
        t = np.arange(48000) / 48000
        audio = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        _freqs, power, n_used = analysis.long_term_spectrum(
            audio, 48000, fft_size=1024, min_rms=0.01
        )
        assert n_used > 0
        assert np.any(power > 0)


class TestDetectHumEdges:
    def test_hum_frequency_above_nyquist_skipped(self):
        freqs = np.linspace(0, 100, 101)
        power = np.zeros(101)
        out = analysis.detect_hum(freqs, power, candidates=(500, 1000))
        assert out == []

    def test_zero_power_gives_no_hum(self):
        freqs = np.linspace(0, 20000, 2001)
        power = np.zeros(2001)
        out = analysis.detect_hum(freqs, power)
        assert all(not h["is_hum"] for h in out)

    def test_real_hum_peak_detected(self):
        freqs = np.linspace(0, 20000, 20001)
        power = np.full(20001, 1e-6)
        # inject a 60 Hz bin peak
        idx = int(np.argmin(np.abs(freqs - 60)))
        power[idx] = 1.0
        out = analysis.detect_hum(freqs, power)
        hum = next(h for h in out if h["frequency_hz"] == 60)
        assert hum["is_hum"] is True


class TestHfSlope:
    def test_band_outside_range_returns_none(self):
        freqs = np.linspace(0, 5000, 100)
        power = np.ones(100)
        # both bands above the available frequency range → empty masks
        result = analysis.hf_slope(freqs, power, mid_band=(10000.0, 12000.0),
                                   high_band=(15000.0, 16000.0))
        assert result == {"mid_db": None, "high_db": None, "slope_db": None}

    def test_zero_power_returns_none(self):
        freqs = np.linspace(0, 20000, 2001)
        power = np.zeros(2001)
        result = analysis.hf_slope(freqs, power)
        assert result["slope_db"] is None

    def test_rising_spectrum_positive_slope(self):
        freqs = np.linspace(0, 20000, 2001)
        power = np.where(freqs >= 10000, 1.0, 1e-6)
        result = analysis.hf_slope(freqs, power)
        assert result["slope_db"] > 0


# ---------------------------------------------------------------------------
# core/discontinuity.py
# ---------------------------------------------------------------------------

from audioman.core import discontinuity  # noqa: E402


class TestDiscontinuityMono:
    def test_mono_passthrough(self):
        mono = np.ones(8, dtype=np.float32)
        assert discontinuity._to_mono(mono) is mono

    def test_stereo_averaged(self):
        stereo = np.stack([np.ones(8), np.zeros(8)]).astype(np.float32)
        assert discontinuity._to_mono(stereo) == pytest.approx(np.full(8, 0.5))


class TestDetectDiscontinuities:
    def test_too_short_returns_empty(self):
        assert discontinuity.detect_discontinuities(np.zeros(3, dtype=np.float32), 48000) == []

    def test_no_spike_returns_empty(self):
        # smooth ramp has a constant first difference well below min_jump
        audio = np.linspace(0.0, 0.1, 1000).astype(np.float32)
        assert discontinuity.detect_discontinuities(audio, 48000) == []

    def test_isolated_spike_reported_without_block_size(self):
        audio = np.zeros(500, dtype=np.float32)
        audio[100] = 0.9
        findings = discontinuity.detect_discontinuities(audio, 48000, min_jump=0.1)
        assert len(findings) == 1
        assert findings[0].code.value == "CLICK_DENSITY"
        assert findings[0].measurement["block_aligned"] is False

    def test_separated_spikes_are_distinct_events(self):
        audio = np.zeros(500, dtype=np.float32)
        audio[100] = 0.9
        audio[400] = 0.9
        findings = discontinuity.detect_discontinuities(audio, 48000, min_jump=0.1)
        assert len(findings) == 2

    def test_block_aligned_spike_flagged_and_critical(self):
        block_size = 64
        audio = np.zeros(block_size * 8, dtype=np.float32)
        audio[block_size * 4] = 0.9  # exactly on a block boundary
        findings = discontinuity.detect_discontinuities(
            audio, 48000, block_size=block_size, min_jump=0.1
        )
        assert findings
        assert findings[0].measurement["block_aligned"] is True
        assert findings[0].measurement["nearest_block_edge"] == block_size * 4
        assert findings[0].severity.value == "critical"

    def test_unaligned_spike_flagged_as_content_click(self):
        block_size = 64
        audio = np.zeros(block_size * 8, dtype=np.float32)
        audio[block_size * 4 + 17] = 0.9  # far from any boundary
        findings = discontinuity.detect_discontinuities(
            audio, 48000, block_size=block_size, edge_tolerance=1, min_jump=0.1
        )
        assert findings
        assert findings[0].measurement["block_aligned"] is False
        assert findings[0].severity.value == "warn"


class TestNullTest:
    def test_identical_inputs_no_finding(self):
        audio = np.ones(1000, dtype=np.float32) * 0.5
        assert discontinuity.null_test(audio, audio.copy(), 48000) == []

    def test_large_difference_reported(self):
        ref = np.ones(1000, dtype=np.float32) * 0.5
        cand = np.ones(1000, dtype=np.float32) * 0.9
        findings = discontinuity.null_test(ref, cand, 48000)
        assert len(findings) == 1
        assert findings[0].severity.value == "critical"  # max_db > -20

    def test_latency_compensation_aligns_signals(self):
        ref = np.zeros(1000, dtype=np.float32)
        ref[100] = 1.0
        cand = np.zeros(1000, dtype=np.float32)
        cand[150] = 1.0  # 50 samples late
        assert discontinuity.null_test(ref, cand, 48000) != []
        assert discontinuity.null_test(ref, cand, 48000, latency_samples=50) == []

    def test_empty_after_latency_trim(self):
        ref = np.ones(10, dtype=np.float32)
        cand = np.ones(10, dtype=np.float32)
        assert discontinuity.null_test(ref, cand, 48000, latency_samples=10) == []

    def test_small_difference_below_threshold(self):
        ref = np.zeros(1000, dtype=np.float32)
        cand = np.full(1000, 1e-6, dtype=np.float32)
        assert discontinuity.null_test(ref, cand, 48000) == []


# ---------------------------------------------------------------------------
# core/detectors.py
# ---------------------------------------------------------------------------

from audioman.core import detectors  # noqa: E402


class TestDetectorsMono:
    def test_mono_passthrough(self):
        mono = np.ones(8, dtype=np.float32)
        assert detectors._to_mono(mono) is mono

    def test_stereo_averaged(self):
        stereo = np.stack([np.ones(8), np.zeros(8)]).astype(np.float32)
        assert detectors._to_mono(stereo) == pytest.approx(np.full(8, 0.5))


class TestDetectChannelImbalanceEdges:
    def test_non_stereo_returns_empty(self):
        assert detectors.detect_channel_imbalance(np.ones(100, dtype=np.float32), 48000) == []

    def test_silent_channel_returns_empty(self):
        stereo = np.stack([np.zeros(1000), np.ones(1000)]).astype(np.float32)
        assert detectors.detect_channel_imbalance(stereo, 48000) == []

    def test_imbalance_reported(self):
        stereo = np.stack([np.full(1000, 0.5), np.full(1000, 0.25)]).astype(np.float32)
        findings = detectors.detect_channel_imbalance(stereo, 48000)
        assert len(findings) == 1


class TestSilenceToFindings:
    def test_short_inner_gap_produces_no_finding(self):
        from audioman.core.analysis import SilenceRegion
        # 0.01 s gap, well below inner_min_sec → the `continue` branch
        # mid-file (not head/tail) and shorter than inner_min_sec → dropped
        region = SilenceRegion(start_sample=10000, end_sample=10480, duration_sec=0.01)
        findings = detectors.silence_to_findings([region], 100000, 48000)
        assert findings == []

    def test_long_inner_gap_reported(self):
        from audioman.core.analysis import SilenceRegion
        region = SilenceRegion(start_sample=10000, end_sample=58000, duration_sec=1.0)
        findings = detectors.silence_to_findings([region], 100000, 48000)
        assert [f.code.value for f in findings] == ["SILENCE_INNER"]

    def test_empty_region_list(self):
        assert detectors.silence_to_findings([], 100000, 48000) == []

    def test_leading_and_trailing(self):
        from audioman.core.analysis import SilenceRegion
        lead = SilenceRegion(start_sample=0, end_sample=4800, duration_sec=0.1)
        tail = SilenceRegion(start_sample=48000, end_sample=52800, duration_sec=0.1)
        findings = detectors.silence_to_findings([lead, tail], 52800, 48000)
        codes = {f.code.value for f in findings}
        assert "SILENCE_LEADING" in codes
        assert "SILENCE_TRAILING" in codes


class TestSpectrumToFindingsHfNoise:
    def test_flat_hf_energy_reports_noise_floor(self):
        spectrum = {
            "hf_slope": {"slope_db": -1.0, "high_db": -40.0, "mid_db": -39.0},
        }
        findings = detectors.spectrum_to_findings(spectrum)
        assert any(f.code.value == "HF_NOISE_FLOOR" for f in findings)

    def test_steep_hf_rolloff_no_finding(self):
        spectrum = {"hf_slope": {"slope_db": -30.0, "high_db": -80.0, "mid_db": -50.0}}
        findings = detectors.spectrum_to_findings(spectrum)
        assert not any(f.code.value == "HF_NOISE_FLOOR" for f in findings)


# ---------------------------------------------------------------------------
# config/paths.py + config/settings.py
# ---------------------------------------------------------------------------

from audioman.config import paths as cfg_paths  # noqa: E402


class TestSearchPathsPerPlatform:
    def _with_system(self, monkeypatch, system):
        monkeypatch.setattr(cfg_paths.platform, "system", lambda: system)

    def test_darwin_vst3(self, monkeypatch):
        self._with_system(monkeypatch, "Darwin")
        out = cfg_paths.get_vst3_search_paths()
        assert any("VST3" in str(p) for p in out)

    def test_linux_vst3(self, monkeypatch):
        self._with_system(monkeypatch, "Linux")
        out = cfg_paths.get_vst3_search_paths()
        assert any("vst3" in str(p) for p in out)

    def test_windows_vst3(self, monkeypatch):
        self._with_system(monkeypatch, "Windows")
        out = cfg_paths.get_vst3_search_paths()
        assert out == [cfg_paths.Path("C:/Program Files/Common Files/VST3")]

    def test_unknown_system_empty(self, monkeypatch):
        self._with_system(monkeypatch, "Plan9")
        assert cfg_paths.get_vst3_search_paths() == []

    def test_au_paths_only_on_darwin(self, monkeypatch):
        self._with_system(monkeypatch, "Linux")
        assert cfg_paths.get_au_search_paths() == []
        self._with_system(monkeypatch, "Darwin")
        assert len(cfg_paths.get_au_search_paths()) == 2


from audioman.config import settings as cfg_settings  # noqa: E402


class TestTomlSettingsSource:
    def test_missing_file_returns_empty(self, tmp_path):
        source = cfg_settings._TomlSettingsSource(cfg_settings.AudiomanSettings)
        cfg_settings.AudiomanSettings.config_file = tmp_path / "absent.toml"
        assert source() == {}

    def test_reads_toml_table(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('json_output = true\n')
        cfg_settings.AudiomanSettings.config_file = path
        source = cfg_settings._TomlSettingsSource(cfg_settings.AudiomanSettings)
        assert source() == {"json_output": True}

    def test_get_field_value_returns_default_and_flag(self, tmp_path):
        cfg_settings.AudiomanSettings.config_file = tmp_path / "absent.toml"
        source = cfg_settings._TomlSettingsSource(cfg_settings.AudiomanSettings)
        value, name, is_complex = source.get_field_value(None, "json_output")
        assert name == "json_output"
        assert is_complex is False


# ---------------------------------------------------------------------------
# core/rt_bench.py
# ---------------------------------------------------------------------------

from audioman.core import rt_bench  # noqa: E402
from audioman.core.streaming import BlockTiming, StreamResult  # noqa: E402


def _stream_result(n_blocks=20, process_sec=0.001, block_size=512, sample_rate=48000):
    timings = [
        BlockTiming(
            index=i, n_samples=block_size, process_sec=process_sec,
            deadline_sec=block_size / sample_rate,
        )
        for i in range(n_blocks)
    ]
    return StreamResult(
        audio=np.zeros((2, n_blocks * block_size), dtype=np.float32),
        sample_rate=sample_rate, block_size=block_size, reset_per_block=False,
        timings=timings,
    )


class TestRTBenchmark:
    def test_report_to_dict(self):
        report = rt_bench.benchmark(_stream_result())
        d = report.to_dict()
        assert d["block_size"] == 512
        assert d["blocks"] > 0
        assert isinstance(d["est_max_tracks"], int)

    def test_xruns_counted_when_over_deadline(self):
        # process time exceeds the block deadline → every block is an xrun
        report = rt_bench.benchmark(_stream_result(process_sec=0.02, block_size=512))
        assert report.xruns > 0
        assert report.xrun_ratio == pytest.approx(1.0)

    def test_warmup_blocks_dropped_when_enough_blocks(self):
        report = rt_bench.benchmark(_stream_result(n_blocks=20), warmup_blocks=2)
        assert report.blocks == 18

    def test_no_timings_raises(self):
        empty = StreamResult(
            audio=np.zeros((2, 100), dtype=np.float32),
            sample_rate=48000, block_size=512, reset_per_block=False, timings=[],
        )
        with pytest.raises(ValueError, match="no block timings"):
            rt_bench.benchmark(empty)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_zero_audio_seconds_gives_infinite_mean(self):
        timing = BlockTiming(index=0, n_samples=512, process_sec=0.001, deadline_sec=0.0)
        st = StreamResult(
            audio=np.zeros((2, 512), dtype=np.float32),
            sample_rate=48000, block_size=512, reset_per_block=False, timings=[timing],
        )
        report = rt_bench.benchmark(st)
        # total audio duration is 0 → the mean-ratio guard returns +inf
        assert report.rt_factor_mean == float("inf")


class TestVadModelLoader:
    def test_model_loaded_once_and_cached(self, monkeypatch):
        from audioman.core import vad as vad_mod

        calls = {"n": 0}

        def _load():
            calls["n"] += 1
            return {"model": True}

        fake = types.ModuleType("silero_vad")
        fake.load_silero_vad = _load
        monkeypatch.setitem(sys.modules, "silero_vad", fake)
        monkeypatch.setattr(vad_mod, "_SILERO_VAD_MODEL", None)

        first = vad_mod._get_silero_vad_model()
        second = vad_mod._get_silero_vad_model()
        assert first is second
        assert calls["n"] == 1


class TestSegmentDataclass:
    def test_duration_samples(self):
        assert Segment(10, 40, "speech").duration_samples == 30

    def test_to_dict(self):
        d = Segment(48000, 96000, "speech").to_dict(48000)
        assert d == {
            "start": 48000, "end": 96000,
            "start_sec": 1.0, "end_sec": 2.0, "duration_sec": 1.0,
            "kind": "speech",
        }


class TestDetectHumAllCandidatesAboveRange:
    def test_candidates_above_nyquist_are_skipped(self):
        # every default candidate exceeds freqs[-1] → the continue branch runs
        freqs = np.linspace(0, 30, 31)
        power = np.ones(31)
        assert analysis.detect_hum(freqs, power) == []

    def test_no_side_bins_near_spectrum_edges(self):
        # 50 Hz sits at both edges of a two-point spectrum → side window is empty
        freqs = np.array([0.0, 50.0])
        power = np.array([1.0, 1.0])
        assert analysis.detect_hum(freqs, power, candidates=(50,)) == []
