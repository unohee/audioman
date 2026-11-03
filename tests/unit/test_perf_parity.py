"""Parity of the vectorised loudness and screening paths against per-window references.

The fast paths (single-pass K-weighting for short-term LUFS, one batched STFT for
the aesthetic screeners) must report the same numbers as the definitions they
replace. Speed is asserted separately by `scripts/bench_perf.py`.
"""

from __future__ import annotations

import numpy as np
import pyloudnorm
import pytest

from audioman.core import aesthetic, loudness
from audioman.core.framewise import frame_features

SR = 48000


def _signals() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1234)
    t = np.arange(SR * 20) / SR
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    return {
        "tone": tone,
        "low_tone_40hz": 0.5 * np.sin(2 * np.pi * 40 * t),
        "noise": 0.1 * rng.standard_normal(len(t)),
        "gated_tone": np.where((t % 8) < 4, tone, 0.0),
        "dc_plus_tone": 0.2 + 0.1 * np.sin(2 * np.pi * 1000 * t),
        "rising_noise": (t / t[-1]) * 0.5 * rng.standard_normal(len(t)),
        "quiet_noise": 1e-4 * rng.standard_normal(len(t)),  # around the -70 LUFS gate
    }


def _stereo(x: np.ndarray) -> np.ndarray:
    return np.stack([x, 0.8 * x]).astype(np.float32)


# ---------------------------------------------------------------------------
# Short-term LUFS
# ---------------------------------------------------------------------------


def _gated_windows_reference(audio, sr, window_sec=3.0, hop_sec=0.1) -> np.ndarray:
    """pyloudnorm's own gating, one call per window, on the continuously K-weighted signal.

    A `custom` meter has no filters, so it measures exactly the samples it is given;
    feeding it windows of the once-filtered signal isolates the windowing and gating
    arithmetic of the fast path from the filter-state question.
    """
    weighted = loudness.k_weighted(audio, sr).T
    win, hop = int(window_sec * sr), int(hop_sec * sr)
    meter = pyloudnorm.Meter(sr, filter_class="custom", block_size=window_sec)
    out = []
    for start in range(0, weighted.shape[0] - win + 1, hop):
        value = meter.integrated_loudness(weighted[start:start + win])
        out.append(value if np.isfinite(value) else -np.inf)
    return np.array(out)


def _restarted_filter_reference(audio, sr, window_sec=3.0, hop_sec=0.1) -> np.ndarray:
    """The previous implementation: filter every window from a zero state."""
    data = audio.T
    win, hop = int(window_sec * sr), int(hop_sec * sr)
    meter = pyloudnorm.Meter(sr, block_size=window_sec)
    out = []
    for start in range(0, data.shape[0] - win + 1, hop):
        value = meter.integrated_loudness(data[start:start + win])
        out.append(value if np.isfinite(value) else -np.inf)
    return np.array(out)


@pytest.mark.parametrize("name", list(_signals()))
def test_short_term_matches_pyloudnorm_gating_on_continuous_filter(name):
    audio = _stereo(_signals()[name])
    fast = loudness.short_term_lufs(audio, SR)
    ref = _gated_windows_reference(audio, SR)
    assert fast.shape == ref.shape
    np.testing.assert_array_equal(np.isfinite(fast), np.isfinite(ref))
    finite = np.isfinite(ref)
    assert np.max(np.abs(fast[finite] - ref[finite]), initial=0.0) < 0.01


def test_short_term_mono_matches_reference():
    mono = _signals()["noise"].astype(np.float32)
    fast = loudness.short_term_lufs(mono, SR)
    ref = _gated_windows_reference(mono, SR)
    assert np.max(np.abs(fast - ref)) < 0.01


@pytest.mark.parametrize("name", ["tone", "low_tone_40hz", "noise", "dc_plus_tone", "rising_noise"])
def test_short_term_agrees_with_restarted_filter_on_steady_signals(name):
    """Against the old per-window filtering only the start-up transient differs."""
    audio = _stereo(_signals()[name])
    fast = loudness.short_term_lufs(audio, SR)
    old = _restarted_filter_reference(audio, SR)
    np.testing.assert_array_equal(np.isfinite(fast), np.isfinite(old))
    assert np.max(np.abs(fast - old)) < 0.03


def test_offset_windows_measure_the_filter_tail_instead_of_silence():
    """Known, intended divergence: a window that starts as the tone stops.

    The old path restarted the filter on all-zero input and reported -inf; the
    continuous filter still rings for a few ms and reports a finite, very low level.
    Every other window agrees with the old path to within 0.05 LU.
    """
    audio = _stereo(_signals()["gated_tone"])
    fast = loudness.short_term_lufs(audio, SR)
    old = _restarted_filter_reference(audio, SR)
    diverging = np.isfinite(fast) != np.isfinite(old)
    assert diverging.any()
    assert np.all(np.isneginf(old[diverging]))
    assert np.all(fast[diverging] < old[np.isfinite(old)].min() - 20.0)
    both = np.isfinite(fast) & np.isfinite(old)
    assert np.max(np.abs(fast[both] - old[both])) < 0.05


@pytest.mark.parametrize("name", ["tone", "noise", "rising_noise", "gated_tone"])
def test_lra_and_report_agree_with_old_path(name):
    audio = _stereo(_signals()[name])
    new_lra = loudness.loudness_range(loudness.short_term_lufs(audio, SR))
    old_lra = loudness.loudness_range(_restarted_filter_reference(audio, SR).astype(np.float32))
    assert abs(new_lra - old_lra) <= 0.05


def test_measure_uses_fast_short_term():
    report = loudness.measure(_stereo(_signals()["tone"]), SR)
    assert report.short_term_max == pytest.approx(report.short_term_min, abs=0.01)
    assert report.lra == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Batched frame features (screen)
# ---------------------------------------------------------------------------

_FRAME_CASES = [(20.0, 10.0), (35.0, 15.0), (100.0, 50.0)]
_BANDS = ((80.0, 700.0), (700.0, 3500.0), (1200.0, 8500.0), (4000.0, 10000.0), (30000.0, 40000.0))


def _speechlike(seconds: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.arange(int(SR * seconds)) / SR
    voiced = 0.2 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 3 * t) > 0)
    hiss = 0.05 * rng.standard_normal(len(t)) * (np.sin(2 * np.pi * 1.3 * t) > 0.6)
    return (voiced + hiss).astype(np.float32)


@pytest.mark.parametrize("frame_ms,hop_ms", _FRAME_CASES)
def test_frame_features_match_single_frame_helpers(frame_ms, hop_ms):
    mono = _speechlike()
    bounds = aesthetic._frame_bounds(len(mono), SR, frame_ms, hop_ms)
    feats = frame_features(mono, bounds, SR, bands=_BANDS, flatness=True, zcr=True)
    for i, (start, end) in enumerate(bounds):
        frame = mono[start:end]
        assert feats.rms[i] == pytest.approx(float(np.sqrt(np.mean(frame**2))), rel=1e-6, abs=1e-12)
        # Band power comes from a float32 FFT (as in `_band_power`), whose rounding
        # noise scales with the frame's energy, not with the band's own value: an
        # out-of-band bin at -120 dB carries float32 noise of the in-band level.
        energy_scale = float(np.sum(frame.astype(np.float64) ** 2))
        for k, band in enumerate(_BANDS):
            assert feats.band_power[k][i] == pytest.approx(
                aesthetic._band_power(frame, SR, band), rel=1e-6, abs=1e-6 * energy_scale + 1e-30
            )
        assert feats.flatness[i] == pytest.approx(aesthetic._spectral_flatness(frame), rel=1e-6)
        assert feats.zcr[i] == pytest.approx(aesthetic._zero_crossing_rate(frame), abs=1e-12)


def test_frame_features_chunking_is_invisible(monkeypatch):
    mono = _speechlike(1.0)
    bounds = aesthetic._frame_bounds(len(mono), SR, 20.0, 10.0)
    whole = frame_features(mono, bounds, SR, bands=_BANDS[:2], flatness=True, zcr=True)
    monkeypatch.setattr("audioman.core.framewise._CHUNK_FRAMES", 7)
    chunked = frame_features(mono, bounds, SR, bands=_BANDS[:2], flatness=True, zcr=True)
    np.testing.assert_array_equal(whole.rms, chunked.rms)
    for a, b in zip(whole.band_power, chunked.band_power, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-6)
    np.testing.assert_allclose(whole.flatness, chunked.flatness, rtol=1e-9)
    np.testing.assert_array_equal(whole.zcr, chunked.zcr)


def test_frame_features_edge_inputs():
    empty = frame_features(np.zeros(0, np.float32), [], SR, bands=_BANDS[:1], flatness=True, zcr=True)
    assert len(empty.rms) == 0 and empty.flatness is not None and len(empty.zcr) == 0

    one = np.array([0.5], dtype=np.float32)
    feats = frame_features(one, [(0, 1)], SR, bands=_BANDS[:1], flatness=True, zcr=True)
    assert feats.rms[0] == pytest.approx(0.5)
    assert feats.band_power[0][0] == aesthetic._band_power(one, SR, _BANDS[0]) == 0.0
    assert feats.flatness[0] == aesthetic._spectral_flatness(one) == 0.0
    assert feats.zcr[0] == aesthetic._zero_crossing_rate(one) == 0.0

    plain = frame_features(_speechlike(0.2), [(0, 960)], SR)
    assert plain.flatness is None and plain.zcr is None and plain.band_power == []


@pytest.mark.parametrize(
    "detector",
    [aesthetic.detect_sibilance_events, aesthetic.detect_breath_events, aesthetic.detect_background_noise_events],
)
def test_detectors_unchanged_against_per_frame_reference(detector, monkeypatch):
    """Events from the batched path equal those from the single-frame helpers."""
    mono = _speechlike(4.0)
    batched_events, _ = detector(mono, SR)

    def _per_frame(mono_in, bounds, sample_rate, *, bands=(), flatness=False, zcr=False):
        from audioman.core.framewise import FrameFeatures

        frames = [mono_in[s:e] for s, e in bounds]
        return FrameFeatures(
            rms=np.array([float(np.sqrt(np.mean(f**2))) for f in frames]),
            band_power=[np.array([aesthetic._band_power(f, sample_rate, b) for f in frames]) for b in bands],
            flatness=np.array([aesthetic._spectral_flatness(f) for f in frames]) if flatness else None,
            zcr=np.array([aesthetic._zero_crossing_rate(f) for f in frames]) if zcr else None,
        )

    monkeypatch.setattr(aesthetic, "frame_features", _per_frame)
    reference_events, _ = detector(mono, SR)
    assert [e.to_dict()["start_sec"] for e in batched_events] == [e.to_dict()["start_sec"] for e in reference_events]
    assert [e.to_dict()["end_sec"] for e in batched_events] == [e.to_dict()["end_sec"] for e in reference_events]
    assert len(batched_events) > 0 or len(reference_events) == 0


@pytest.mark.parametrize("name", ["tone", "noise", "gated_tone", "quiet_noise"])
def test_measure_shares_one_k_weighting_pass_without_changing_integrated(name):
    audio = _stereo(_signals()[name])
    report = loudness.measure(audio, SR)
    # `integrated_lufs` lets pyloudnorm write the filtered signal back into the
    # float32 input; the shared pass stays in float64. That is a ~1e-7 LU difference.
    expected = loudness.integrated_lufs(audio, SR)
    if np.isfinite(expected):
        assert report.integrated_lufs == pytest.approx(expected, abs=1e-6)
    else:
        assert report.integrated_lufs == expected
    st = loudness.short_term_lufs(audio, SR)
    finite = st[np.isfinite(st)]
    expected_max = float(finite.max()) if len(finite) else float("-inf")
    assert report.short_term_max == pytest.approx(expected_max, abs=1e-6)


def test_measure_mono_and_short_inputs():
    mono = _signals()["tone"][: SR * 5].astype(np.float32)
    assert loudness.measure(mono, SR).integrated_lufs == pytest.approx(loudness.integrated_lufs(mono, SR), abs=1e-6)
    short = _stereo(_signals()["tone"][: SR // 10])  # shorter than a 400 ms gating block
    report = loudness.measure(short, SR)
    assert report.integrated_lufs == float("-inf")
    assert report.lra == 0.0


def test_measure_six_channels_keeps_the_undefined_result():
    t = np.arange(SR * 4) / SR
    six = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 6).astype(np.float32)
    report = loudness.measure(six, SR)
    assert report.integrated_lufs == float("-inf")
    assert report.short_term_max == float("-inf")


@pytest.mark.parametrize("name", ["tone", "noise", "dc_plus_tone"])
def test_true_peak_float32_path_matches_float64(name):
    x = _stereo(_signals()[name][: SR * 5])
    assert loudness.true_peak_dbtp(x, SR) == pytest.approx(loudness.true_peak_dbtp(x.astype(np.float64), SR), abs=1e-4)


# ---------------------------------------------------------------------------
# Windowed stereo correlation (master qc)
# ---------------------------------------------------------------------------


def _phase_correlation_reference(audio, window_ms=100.0, sample_rate=SR):
    left, right = audio[0].astype(np.float64), audio[1].astype(np.float64)
    win = max(int(window_ms / 1000.0 * sample_rate), 256)
    corrs = []
    for i in range(len(left) // win):
        wl, wr = left[i * win:(i + 1) * win], right[i * win:(i + 1) * win]
        if np.std(wl) < 1e-8 or np.std(wr) < 1e-8:
            continue
        corrs.append(float(np.corrcoef(wl, wr)[0, 1]))
    return np.array(corrs)


def test_stereo_phase_correlation_matches_per_window_reference():
    from audioman.core.qc import stereo_phase_correlation

    rng = np.random.default_rng(3)
    t = np.arange(SR * 6) / SR
    left = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.05 * rng.standard_normal(len(t))
    right = np.where((t % 2) < 1, left, -left) + 0.02 * rng.standard_normal(len(t))
    right[SR: SR * 2] = 0.0  # silent windows are skipped by both
    audio = np.stack([left, right]).astype(np.float32)

    ref = _phase_correlation_reference(audio)
    got = stereo_phase_correlation(audio, sample_rate=SR)
    assert got["n_windows"] == len(ref)
    assert got["min_window_correlation"] == round(float(ref.min()), 4)
    assert got["mean_window_correlation"] == pytest.approx(round(float(ref.mean()), 4), abs=1e-4)
    assert got["negative_correlation_pct"] == round(100.0 * float((ref < 0).sum()) / len(ref), 2)


# ---------------------------------------------------------------------------
# RF-noise neighbourhood medians (screen)
# ---------------------------------------------------------------------------


def test_neighbourhood_medians_match_per_bin_definition():
    rng = np.random.default_rng(11)
    spectrum = rng.random(2049) ** 3
    spectrum[700] = 50.0  # a tone: excluded from its own floor, included in its neighbours'
    indices = np.arange(1, len(spectrum))  # covers both edges and the interior
    batched = aesthetic._neighbourhood_medians(spectrum, indices)
    reference = np.array([aesthetic._neighbourhood_median(spectrum, int(i)) for i in indices])
    np.testing.assert_array_equal(batched, reference)


def test_rf_noise_detector_unchanged_against_per_bin_reference(monkeypatch):
    t = np.arange(SR * 2) / SR
    rng = np.random.default_rng(5)
    audio = (0.01 * rng.standard_normal(len(t)) + 0.02 * np.sin(2 * np.pi * 9000 * t)
             + 0.02 * np.sin(2 * np.pi * 12500 * t)).astype(np.float32)
    batched, _ = aesthetic.detect_rf_noise_events(audio, SR)
    monkeypatch.setattr(
        aesthetic, "_neighbourhood_medians",
        lambda spec, idx: np.array([aesthetic._neighbourhood_median(spec, int(i)) for i in idx]),
    )
    reference, _ = aesthetic.detect_rf_noise_events(audio, SR)
    assert len(batched) == 1
    assert [e.to_dict() for e in batched] == [e.to_dict() for e in reference]
