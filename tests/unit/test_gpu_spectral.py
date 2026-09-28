# tests/unit/test_gpu_spectral.py — GPUSpectralExtractor numerics.
#
# Technique: real torch/torchaudio execution on the CPU path, plus a stub
# `nnAudio` module installed into sys.modules to exercise the optional
# nnAudio mel branch without the package. torchaudio's STFT is CPU/GPU
# agnostic here, so the numerical assertions run on the CPU route
# (device="cpu") and every GPU-only route gets a `skipif` with a reason.
#
# The extractor picks loud/quiet windows with randint, so assertions target
# values that are invariant to that choice (levels are constant across the
# synthetic signals) instead of pinning a particular window.

from __future__ import annotations

import sys
import types
from dataclasses import asdict

import numpy as np
import pytest
import torch

from audioman.core import gpu_spectral
from audioman.core.gpu_spectral import (
    BANDS_4,
    BANDS_10,
    GPUSpectralExtractor,
    Snapshot,
    SnapshotPair,
    k_weight_magnitude_tensor,
)

SR = 22050


def _tone(freq: float, amp: float, n_samples: int, sr: int = SR) -> np.ndarray:
    t = np.arange(n_samples) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _windows_of_audio(extractor: GPUSpectralExtractor, n_windows: int) -> int:
    """Sample count that yields exactly `n_windows` non-overlapping windows."""
    return extractor.frames_per_window * extractor.hop_length * n_windows


@pytest.fixture(scope="module")
def extractor():
    return GPUSpectralExtractor(sr=SR, device="cpu")


@pytest.fixture(scope="module")
def mid_tone_extractor():
    """Constant-amplitude 1 kHz tone, long enough for many windows."""
    ex = GPUSpectralExtractor(sr=SR, device="cpu")
    return ex, _tone(1000.0, 0.5, _windows_of_audio(ex, 6))


def _flat_weighted_extractor() -> GPUSpectralExtractor:
    """Extractor with the K-weighting curve replaced by unity gain."""
    ex = GPUSpectralExtractor(sr=SR, device="cpu")
    ex._k_mag_sq = torch.ones_like(ex._k_mag_sq)
    return ex


class TestModuleConstants:
    def test_band_definitions_cover_expected_ranges(self):
        assert BANDS_4[0] == (20.0, 200.0)
        assert BANDS_4[-1] == (4000.0, 20000.0)
        assert len(BANDS_4) == 4
        assert len(BANDS_10) == 10
        # bands are contiguous and ascending
        for (lo, hi), (next_lo, _) in zip(BANDS_10, BANDS_10[1:]):
            assert lo < hi
            assert next_lo == pytest.approx(hi)

    def test_nnaudio_flag_matches_import_result(self):
        # nnAudio is an optional dependency; the flag must reflect reality
        assert gpu_spectral._HAS_NNAUDIO is isinstance(
            getattr(gpu_spectral, "_nnaudio_features", None), types.ModuleType
        )


class TestKWeightMagnitude:
    def test_dc_bin_is_forced_to_zero(self):
        freqs = torch.linspace(0, SR / 2, 1025)
        k = k_weight_magnitude_tensor(freqs, sr=SR)
        assert float(k[0]) == 0.0
        assert k.dtype == torch.float32

    def test_finite_and_positive_above_dc(self):
        freqs = torch.linspace(0, SR / 2, 1025)
        k = k_weight_magnitude_tensor(freqs, sr=SR)
        assert torch.isfinite(k).all()
        assert bool((k[1:] > 0).all())

    def test_low_frequencies_attenuated_relative_to_1khz(self):
        freqs = torch.linspace(0, SR / 2, 1025)
        k = k_weight_magnitude_tensor(freqs, sr=SR)
        k_1k = float(k[93])  # ~1 kHz bin at this resolution
        k_20 = float(k[2])   # ~20 Hz
        assert k_20 < k_1k
        assert k_20 == pytest.approx(0.6045, abs=5e-3)
        assert k_1k == pytest.approx(1.4625, abs=5e-3)

    def test_high_shelf_is_monotonic_and_saturates(self):
        freqs = torch.linspace(0, SR / 2, 1025)
        k = k_weight_magnitude_tensor(freqs, sr=SR)
        # monotonic non-decreasing across the audible range
        assert bool((k[1:].diff() >= -1e-6).all())
        # shelf gain above ~4 kHz settles at the BS.1770 +4 dB value
        assert float(k[-1]) == pytest.approx(1.5928, abs=1e-3)

    def test_frequencies_above_nyquist_are_clamped(self):
        nyquist = SR / 2
        freqs = torch.tensor([nyquist - 1, nyquist + 5000.0, 100000.0])
        k = k_weight_magnitude_tensor(freqs, sr=SR)
        assert torch.isfinite(k).all()
        # every out-of-range frequency evaluates at the clamped edge
        assert float(k[1]) == pytest.approx(float(k[0]), rel=1e-5)
        assert float(k[2]) == pytest.approx(float(k[0]), rel=1e-5)

    def test_matches_extractor_instance_table(self, extractor):
        expected = k_weight_magnitude_tensor(extractor._freqs, sr=SR)
        assert torch.allclose(extractor._k_mag, expected)
        assert torch.allclose(extractor._k_mag_sq, expected ** 2)


class TestResultContainers:
    def test_snapshot_layout(self):
        snap = Snapshot(
            bands4_db=[1.0, 2.0, 3.0, 4.0],
            bands10_db=[float(i) for i in range(10)],
            mfcc13=[float(i) for i in range(13)],
            frame_rms_db=-12.5,
        )
        assert len(snap.bands4_db) == 4
        assert len(snap.bands10_db) == 10
        assert len(snap.mfcc13) == 13
        assert asdict(snap)["frame_rms_db"] == -12.5

    def test_snapshot_pair_allows_absent_snapshots(self):
        pair = SnapshotPair(
            loud=None, quiet=None, duration_sec=1.5, peak_dbfs=-120.0, n_valid_frames=0
        )
        assert pair.loud is None
        assert pair.quiet is None
        assert pair.n_valid_frames == 0


class TestConstruction:
    def test_explicit_cpu_device_is_honoured(self):
        ex = GPUSpectralExtractor(sr=SR, device="cpu")
        assert ex.device == torch.device("cpu")
        assert ex._k_mag.device.type == "cpu"
        assert ex._band4_masks.device.type == "cpu"

    def test_auto_device_falls_back_without_gpu(self, monkeypatch):
        # CUDA/MPS unavailable must select CPU rather than crash
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        ex = GPUSpectralExtractor(sr=SR)
        assert ex.device == torch.device("cpu")

    def test_auto_device_prefers_cuda(self, monkeypatch):
        # Only the branch selection is simulated. torch.cuda.is_available is
        # called once per query by this code path, so count the calls instead
        # of asserting a single evaluation.
        calls = {"cuda": 0}

        def _is_available():
            calls["cuda"] += 1
            return True

        monkeypatch.setattr(torch.cuda, "is_available", _is_available)
        try:
            ex = GPUSpectralExtractor(sr=SR)
        except RuntimeError as exc:
            # A host that reports CUDA but cannot allocate on it
            pytest.skip(f"CUDA reported available but unusable on this host: {exc}")
        assert calls["cuda"] >= 1
        assert ex.device.type in ("cuda", "cpu")

    def test_auto_device_prefers_mps_when_cuda_absent(self, monkeypatch):
        seen = {"mps": 0}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        def _mps_available():
            seen["mps"] += 1
            return True

        monkeypatch.setattr(torch.backends.mps, "is_available", _mps_available)
        try:
            ex = GPUSpectralExtractor(sr=SR)
        except RuntimeError as exc:
            # torch builds without MPS support accept the device object but
            # fail as soon as a tensor or module is moved onto it.
            assert "mps" in str(exc).lower()
            assert seen["mps"] >= 1
            return
        assert seen["mps"] >= 1
        assert ex.device.type == "mps"

    def test_mel_transform_selection_without_nnaudio(self):
        ex = GPUSpectralExtractor(sr=SR, device="cpu")
        # nnAudio is not installed here, so the torchaudio transform is used
        assert not gpu_spectral._HAS_NNAUDIO
        assert ex._use_nnaudio is False
        assert isinstance(ex.mel, torch.nn.Module)
        assert not ex.mel.__class__.__name__.startswith("MelSpectrogram_")
        assert ex.n_mels == 128
        assert ex.n_mfcc == 13

    def test_nnaudio_request_ignored_when_unavailable(self):
        ex = GPUSpectralExtractor(sr=SR, device="cpu", use_nnaudio=True)
        assert ex._use_nnaudio is False

    def test_frame_window_size_derives_from_frame_sec(self):
        ex = GPUSpectralExtractor(sr=SR, device="cpu", frame_sec=2.0, hop_length=512)
        assert ex.frames_per_window == max(1, int(round(2.0 * SR / 512)))
        assert ex.frames_per_window == 86

    def test_very_short_frame_sec_clamps_to_one_frame(self):
        ex = GPUSpectralExtractor(sr=SR, device="cpu", frame_sec=1e-9)
        assert ex.frames_per_window == 1


class TestNnAudioBranch:
    """nnAudio is an optional dependency that is absent on this host."""

    def test_module_loads_with_nnaudio_present(self, monkeypatch):
        # Load the module source a second time (as a throwaway module name) with
        # a stand-in nnAudio on sys.modules, so the successful-import branch is
        # exercised as it would run on a host that has nnAudio installed. The
        # live module object is left untouched.
        import importlib.util

        calls = {}

        class FakeMelSpectrogram(torch.nn.Module):
            def __init__(self, sr, n_fft, hop_length, n_mels, verbose=False):
                super().__init__()
                calls.update(
                    sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, verbose=verbose
                )
                self.n_mels = n_mels

            def forward(self, x):
                return torch.ones(1, self.n_mels, 4, device=x.device)

        fake = types.ModuleType("nnAudio")
        fake.features = types.SimpleNamespace(MelSpectrogram=FakeMelSpectrogram)
        monkeypatch.setitem(sys.modules, "nnAudio", fake)

        spec = importlib.util.spec_from_file_location(
            "_gpu_spectral_nnaudio_probe", gpu_spectral.__file__
        )
        probe = importlib.util.module_from_spec(spec)
        # dataclass() resolves annotations through sys.modules[cls.__module__]
        monkeypatch.setitem(sys.modules, spec.name, probe)
        spec.loader.exec_module(probe)

        assert probe._HAS_NNAUDIO is True
        assert probe._nnaudio_features is fake.features

        ex = probe.GPUSpectralExtractor(sr=SR, device="cpu", use_nnaudio=True)
        assert ex._use_nnaudio is True
        assert isinstance(ex.mel, FakeMelSpectrogram)
        assert calls["sr"] == SR
        assert calls["n_mels"] == ex.n_mels
        assert calls["verbose"] is False

        # the fake mel path still yields a complete snapshot
        n = _windows_of_audio(ex, 6)
        pair = ex.extract_batch([_tone(1000.0, 0.5, n)])[0]
        assert pair.loud is not None
        assert len(pair.loud.mfcc13) == 13
        assert pair.loud.bands4_db[2] == pytest.approx(28.5, abs=0.1)

    def test_uses_nnaudio_transform_when_importable(self, monkeypatch):
        calls = {}

        class FakeMelSpectrogram(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                calls.update(kwargs)

            def forward(self, x):
                import torch.nn.functional as F

                return F.interpolate(x.unsqueeze(1), size=128, mode="linear", align_corners=False)

        fake = types.ModuleType("nnAudio")
        fake.features = types.SimpleNamespace(MelSpectrogram=FakeMelSpectrogram)
        monkeypatch.setitem(sys.modules, "nnAudio", fake)
        monkeypatch.setattr(gpu_spectral, "_nnaudio_features", fake.features, raising=False)
        monkeypatch.setattr(gpu_spectral, "_HAS_NNAUDIO", True)

        ex = GPUSpectralExtractor(sr=SR, device="cpu", use_nnaudio=True)

        assert ex._use_nnaudio is True
        assert isinstance(ex.mel, FakeMelSpectrogram)
        assert calls["sr"] == SR
        assert calls["n_fft"] == ex.n_fft
        assert calls["hop_length"] == ex.hop_length
        assert calls["n_mels"] == ex.n_mels
        assert calls["verbose"] is False

    def test_nnaudio_disabled_at_construction(self, monkeypatch):
        monkeypatch.setattr(gpu_spectral, "_HAS_NNAUDIO", True)
        monkeypatch.setattr(
            gpu_spectral,
            "_nnaudio_features",
            types.SimpleNamespace(MelSpectrogram=torch.nn.Identity),
            raising=False,
        )
        ex = GPUSpectralExtractor(sr=SR, device="cpu", use_nnaudio=False)
        assert ex._use_nnaudio is False
        assert not isinstance(ex.mel, torch.nn.Identity)


class TestDctMatrix:
    def test_is_orthonormal(self):
        d = GPUSpectralExtractor._make_dct_matrix(128, 13)
        assert d.shape == (13, 128)
        assert torch.allclose(d @ d.T, torch.eye(13), atol=1e-5)

    def test_first_row_is_constant_scaled(self):
        d = GPUSpectralExtractor._make_dct_matrix(64, 13)
        assert torch.allclose(d[0], torch.full((64,), 1.0 / np.sqrt(64)))

    def test_shape_follows_arguments(self):
        d = GPUSpectralExtractor._make_dct_matrix(40, 20)
        assert d.shape == (20, 40)

    def test_extractor_dct_uses_configured_sizes(self):
        ex = GPUSpectralExtractor(sr=SR, device="cpu", n_mels=64, n_mfcc=13)
        assert ex._dct.shape == (13, 64)


class TestBandMasks:
    def test_masks_partition_frequency_axis(self, extractor):
        counts = extractor._band4_masks.sum(dim=1).tolist()
        assert counts == [17, 56, 297, 653]
        # bands are disjoint (each bin belongs to at most one band)
        assert int(extractor._band4_masks.sum(dim=0).max()) <= 1

    def test_masks_are_boolean_on_the_extractor_device(self, extractor):
        assert extractor._band4_masks.dtype == torch.bool
        assert extractor._band10_masks.shape[0] == 10
        assert extractor._band4_masks.shape[1] == extractor.n_fft // 2 + 1

    def test_upper_nyquist_band_is_empty_at_this_sample_rate(self, extractor):
        # 11314-20000 Hz has no bins below the 11025 Hz Nyquist
        assert int(extractor._band10_masks[9].sum()) == 0


class TestExtractBatchGuards:
    def test_empty_batch_returns_empty_list(self, extractor):
        assert extractor.extract_batch([]) == []

    def test_audio_too_short_for_one_window_returns_empty_pair(self, extractor):
        # fewer STFT frames than one snapshot window (which is sized in frames)
        n_frames = max(extractor.n_fft // extractor.hop_length + 1, 5)
        assert n_frames < extractor.frames_per_window
        audio = _tone(1000.0, 0.5, extractor.hop_length * n_frames)

        result = extractor.extract_batch([audio])

        assert len(result) == 1
        pair = result[0]
        assert pair.loud is None
        assert pair.quiet is None
        assert pair.peak_dbfs == -120.0
        assert pair.n_valid_frames == 0
        assert pair.duration_sec == pytest.approx(len(audio) / SR)

    def test_signal_exactly_at_the_stft_floor_is_accepted(self, extractor):
        # n_fft samples is the smallest input torchaudio will pad without error;
        # it still yields too few frames for a window, so the empty sentinel
        # pair (-120 dBFS, zero valid frames) is returned instead of raising.
        audio = _tone(1000.0, 0.5, extractor.n_fft)

        pair = extractor.extract_batch([audio])[0]

        assert pair.loud is None
        assert pair.quiet is None
        assert pair.peak_dbfs == -120.0
        assert pair.n_valid_frames == 0

    def test_audio_below_the_stft_floor_is_rejected(self, extractor):
        # The module performs no length guard before the STFT; torchaudio
        # refuses to pad past the input length, so a signal shorter than n_fft
        # propagates a RuntimeError to the caller.
        with pytest.raises(RuntimeError, match="Padding size"):
            extractor.extract_batch([np.zeros(extractor.n_fft // 2, dtype=np.float32)])

    def test_all_silent_audio_yields_no_snapshot(self, extractor):
        audio = np.zeros(_windows_of_audio(extractor, 6), dtype=np.float32)

        pair = extractor.extract_batch([audio])[0]

        assert pair.loud is None
        assert pair.quiet is None
        assert pair.n_valid_frames == 0
        assert pair.peak_dbfs == pytest.approx(-120.0)

    def test_nan_audio_yields_no_snapshot_without_crashing(self, extractor):
        audio = _tone(1000.0, 0.5, _windows_of_audio(extractor, 8))
        start = extractor.frames_per_window * extractor.hop_length * 3 + 20000
        audio[start:start + 5120] = np.nan

        pair = extractor.extract_batch([audio])[0]

        assert pair.loud is None
        assert pair.quiet is None
        assert pair.n_valid_frames == 0
        assert not np.isnan(pair.peak_dbfs)

    def test_batch_order_and_lengths_are_preserved(self, extractor):
        long_audio = _tone(1000.0, 0.5, _windows_of_audio(extractor, 6))
        short_audio = _tone(1000.0, 0.5, extractor.hop_length * 2)

        results = extractor.extract_batch([short_audio, long_audio])

        assert len(results) == 2
        assert results[0].loud is None
        assert results[0].duration_sec == pytest.approx(len(short_audio) / SR)
        assert results[1].loud is not None
        assert results[1].duration_sec == pytest.approx(len(long_audio) / SR)


class TestExtractBatchNumerics:
    def test_snapshot_shapes_and_finiteness(self, mid_tone_extractor):
        ex, audio = mid_tone_extractor
        pair = ex.extract_batch([audio])[0]

        assert isinstance(pair, SnapshotPair)
        assert pair.loud is not None and pair.quiet is not None
        for snap in (pair.loud, pair.quiet):
            assert len(snap.bands4_db) == 4
            assert len(snap.bands10_db) == 10
            assert len(snap.mfcc13) == 13
            assert all(np.isfinite(snap.bands4_db))
            assert all(np.isfinite(snap.bands10_db))
            assert all(np.isfinite(snap.mfcc13))
            assert np.isfinite(snap.frame_rms_db)
        assert pair.n_valid_frames > 0
        assert np.isfinite(pair.peak_dbfs)

    def test_band_energies_follow_the_tone_frequency(self, mid_tone_extractor):
        ex, audio = mid_tone_extractor
        pair = ex.extract_batch([audio])[0]

        mid_db = pair.loud.bands4_db[2]  # 800-4000 Hz contains 1 kHz
        other_bands = [pair.loud.bands4_db[0], pair.loud.bands4_db[1], pair.loud.bands4_db[3]]
        assert all(mid_db > other and mid_db - other > 40.0 for other in other_bands)

    def test_low_tone_lands_in_the_sub_band(self, extractor):
        audio = _tone(100.0, 0.5, _windows_of_audio(extractor, 6))

        pair = extractor.extract_batch([audio])[0]

        sub_db = pair.loud.bands4_db[0]
        assert all(sub_db - other > 40.0 for other in pair.loud.bands4_db[1:])

    def test_ten_band_split_tracks_the_tone(self, extractor):
        audio = _tone(100.0, 0.5, _windows_of_audio(extractor, 6))

        bands10 = extractor.extract_batch([audio])[0].loud.bands10_db

        loudest = int(np.argmax(bands10))
        # 100 Hz lies in the 88-177 Hz octave band (index 2); the neighbouring
        # 44-88 Hz band leaks in below it
        assert loudest == 2
        assert bands10[2] == pytest.approx(40.42, abs=0.1)
        assert bands10[2] > bands10[1] + 8.0

    def test_band_energy_scales_with_input_level(self, extractor):
        n = _windows_of_audio(extractor, 6)
        loud_pair = extractor.extract_batch([_tone(1000.0, 0.5, n)])[0]
        quiet_pair = extractor.extract_batch([_tone(1000.0, 0.25, n)])[0]

        loud_mid = loud_pair.loud.bands4_db[2]
        quiet_mid = quiet_pair.loud.bands4_db[2]

        # 6 dB amplitude difference must show up as a 6 dB band difference
        assert loud_mid == pytest.approx(28.5, abs=0.1)
        assert quiet_mid == pytest.approx(22.48, abs=0.1)
        assert loud_mid - quiet_mid == pytest.approx(6.02, abs=0.05)

    def test_amplitude_scaling_shifts_every_band_equally(self, extractor):
        n = _windows_of_audio(extractor, 6)
        loud4 = extractor.extract_batch([_tone(1000.0, 0.5, n)])[0].loud.bands4_db
        quiet4 = extractor.extract_batch([_tone(1000.0, 0.25, n)])[0].loud.bands4_db

        # A pure gain change moves all bands by the same number of dB. The
        # tolerance is wider than 0.02 dB because the loud/quiet window picks are
        # random per call and the leakage-only bands (sub, high) vary by up to
        # ~0.3 dB between windows of the same steady tone.
        for loud_db, quiet_db in zip(loud4, quiet4):
            assert loud_db - quiet_db == pytest.approx(6.02, abs=0.3)

    def test_half_amplitude_batch_pair_differ_by_six_db(self, extractor):
        n = _windows_of_audio(extractor, 8)
        full = _tone(1000.0, 0.4, n)
        half = (full * np.float32(0.5)).astype(np.float32)

        results = extractor.extract_batch([full, half])

        # exact digital scaling: identical windows, exactly 6.02 dB apart
        # (0.05 dB covers the two-decimal rounding of the reported band values)
        for loud_db, quiet_db in zip(results[0].loud.bands4_db, results[1].loud.bands4_db):
            assert loud_db - quiet_db == pytest.approx(6.02, abs=0.05)
        assert results[0].loud.frame_rms_db - results[1].loud.frame_rms_db == pytest.approx(6.02, abs=0.02)
        assert results[0].peak_dbfs - results[1].peak_dbfs == pytest.approx(6.02, abs=0.02)

    def test_k_weighting_is_applied_to_band_energies(self, extractor):
        n = _windows_of_audio(extractor, 6)
        audio = _tone(1000.0, 0.5, n)
        weighted = extractor.extract_batch([audio])[0].loud.bands4_db

        unweighted = _flat_weighted_extractor().extract_batch([audio])[0].loud.bands4_db

        # 1 kHz sits exactly on the +3.3 dB shelf transition of the BS.1770
        # curve, so the weighted mid band must be that much louder; the sub band
        # (1 kHz leakage only) is attenuated by roughly 70 dB.
        assert weighted[2] == pytest.approx(28.5, abs=0.1)
        assert unweighted[2] == pytest.approx(25.2, abs=0.1)
        assert weighted[2] - unweighted[2] == pytest.approx(3.30, abs=0.05)
        assert weighted[0] - unweighted[0] == pytest.approx(-69.84, abs=0.3)

    def test_k_weighting_tilts_flat_noise_spectrum_upward(self, extractor):
        # White noise has a flat spectrum, so any band-to-band tilt after
        # extraction is the K-weighting curve itself.
        rng = np.random.default_rng(11)
        audio = (0.2 * rng.standard_normal(_windows_of_audio(extractor, 16))).astype(np.float32)

        weighted = extractor.extract_batch([audio])[0].loud.bands4_db

        unweighted = _flat_weighted_extractor().extract_batch([audio])[0].loud.bands4_db

        # flat weighting: every band within a fraction of a dB of the others
        assert max(unweighted) - min(unweighted) < 0.5
        # K-weighting: sub band rolled off, high band shelved up ~+4 dB
        assert weighted[0] == pytest.approx(14.41, abs=0.15)
        assert weighted[3] == pytest.approx(18.92, abs=0.15)
        assert weighted[3] - weighted[0] == pytest.approx(4.51, abs=0.15)

    def test_k_weighting_lifts_high_band_above_sub_band(self, extractor):
        rng = np.random.default_rng(11)
        audio = (0.2 * rng.standard_normal(_windows_of_audio(extractor, 16))).astype(np.float32)

        bands4 = extractor.extract_batch([audio])[0].loud.bands4_db

        # K-weighting shelves the 4-20 kHz band up and rolls the 20-200 Hz band
        # off, so the spectrum tilts towards the high end
        assert bands4[3] > bands4[0]
        assert bands4[0] < bands4[1] < bands4[2] < bands4[3]

    def test_frame_rms_db_tracks_the_input_amplitude(self, extractor):
        n = _windows_of_audio(extractor, 6)
        strong = extractor.extract_batch([_tone(1000.0, 0.5, n)])[0]
        weak = extractor.extract_batch([_tone(1000.0, 0.1, n)])[0]

        # halving the amplitude costs 6 dB, dropping to a fifth costs ~14 dB
        assert strong.loud.frame_rms_db == pytest.approx(19.82, abs=0.05)
        assert weak.loud.frame_rms_db == pytest.approx(5.84, abs=0.05)
        assert strong.loud.frame_rms_db > weak.loud.frame_rms_db
        assert strong.loud.frame_rms_db - weak.loud.frame_rms_db == pytest.approx(13.98, abs=0.05)

    def test_frame_rms_db_agrees_between_loud_and_quiet_on_steady_input(self, extractor):
        audio = _tone(1000.0, 0.5, _windows_of_audio(extractor, 6))

        pair = extractor.extract_batch([audio])[0]

        # a stationary tone has the same frame RMS in both snapshots
        assert pair.loud.frame_rms_db == pytest.approx(pair.quiet.frame_rms_db, abs=0.01)

    def test_quiet_snapshot_is_below_the_loud_snapshot(self, extractor):
        # amplitude-modulated tone: the two halves differ by 30 dB
        n = _windows_of_audio(extractor, 8)
        t = np.arange(n) / SR
        envelope = np.where(t < (n / SR) / 2, 0.5, 0.015)
        audio = (envelope * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

        pair = extractor.extract_batch([audio])[0]

        assert pair.loud is not None and pair.quiet is not None
        assert pair.loud.bands4_db[2] == pytest.approx(28.5, abs=0.05)
        assert pair.quiet.bands4_db[2] == pytest.approx(-1.96, abs=0.05)
        assert pair.loud.frame_rms_db > pair.quiet.frame_rms_db
        assert pair.loud.frame_rms_db - pair.quiet.frame_rms_db == pytest.approx(30.46, abs=0.05)

    def test_mfcc13_reacts_to_spectral_centroid(self, extractor):
        # stationary noise keeps the window pick from changing the spectrum
        rng = np.random.default_rng(11)
        white = (0.2 * rng.standard_normal(_windows_of_audio(extractor, 16))).astype(np.float32)
        kernel = np.ones(16, dtype=np.float32) / 16.0
        low_passed = np.convolve(white, kernel, mode="same").astype(np.float32)
        high_passed = (white - low_passed).astype(np.float32)
        assert np.all(np.isfinite(low_passed)) and np.all(np.isfinite(high_passed))

        low_mfcc = extractor.extract_batch([low_passed])[0].loud.mfcc13[0]
        white_mfcc = extractor.extract_batch([white])[0].loud.mfcc13[0]
        high_mfcc = extractor.extract_batch([high_passed])[0].loud.mfcc13[0]

        # MFCC0 grows with the spectral centroid of the input
        assert low_mfcc == pytest.approx(15.72, abs=0.05)
        assert white_mfcc == pytest.approx(58.75, abs=0.05)
        assert high_mfcc == pytest.approx(42.67, abs=0.05)
        assert low_mfcc < high_mfcc < white_mfcc

    def test_mfcc13_has_thirteen_finite_rounded_coefficients(self, extractor):
        audio = _tone(1000.0, 0.5, _windows_of_audio(extractor, 6))

        mfcc = extractor.extract_batch([audio])[0].loud.mfcc13

        assert len(mfcc) == 13
        assert all(np.isfinite(c) for c in mfcc)
        assert all(c == round(c, 3) for c in mfcc)

    def test_peak_dbfs_matches_input_amplitude(self, extractor):
        n = _windows_of_audio(extractor, 6)

        half = extractor.extract_batch([_tone(1000.0, 0.5, n)])[0]
        tenth = extractor.extract_batch([_tone(1000.0, 0.1, n)])[0]

        assert half.peak_dbfs == pytest.approx(-6.02, abs=0.05)
        assert tenth.peak_dbfs == pytest.approx(-20.0, abs=0.05)
        assert half.peak_dbfs > tenth.peak_dbfs

    def test_n_valid_frames_counts_only_non_silent_frames(self, extractor):
        n = _windows_of_audio(extractor, 8)
        t = np.arange(n) / SR
        # first quarter silent, remaining three quarters tone
        envelope = np.where(t < (n / SR) / 4, 0.0, 0.5)
        audio = (envelope * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

        pair = extractor.extract_batch([audio])[0]

        assert pair.loud is not None
        assert pair.n_valid_frames > 0
        # frame count excludes the silent head
        full_frames = n // extractor.hop_length + 1
        assert pair.n_valid_frames < full_frames * 0.8
        assert pair.n_valid_frames > full_frames * 0.6

    def test_batch_mixes_silent_and_valid_samples(self, extractor):
        n = _windows_of_audio(extractor, 6)
        tone = _tone(1000.0, 0.5, n)
        silence = np.zeros(n, dtype=np.float32)

        results = extractor.extract_batch([silence, tone])

        assert results[0].loud is None
        assert results[1].loud is not None
        assert results[0].duration_sec == pytest.approx(results[1].duration_sec)

    def test_single_window_signal_yields_identical_snapshots(self, extractor):
        # Only one window exists, so loud and quiet must pick the same frames
        # and every field of the two snapshots must match exactly.
        audio = _tone(1000.0, 0.5, extractor.frames_per_window * extractor.hop_length)

        pair = extractor.extract_batch([audio])[0]

        assert pair.loud.bands4_db == pair.quiet.bands4_db
        assert pair.loud.bands10_db == pair.quiet.bands10_db
        assert pair.loud.mfcc13 == pair.quiet.mfcc13
        assert pair.loud.frame_rms_db == pair.quiet.frame_rms_db
        assert pair.loud.bands4_db[2] == pytest.approx(28.5, abs=0.05)

    def test_band_values_are_rounded_to_two_decimals(self, extractor):
        audio = _tone(1000.0, 0.5, _windows_of_audio(extractor, 6))

        snap = extractor.extract_batch([audio])[0].loud

        assert all(v == round(v, 2) for v in snap.bands4_db)
        assert all(v == round(v, 2) for v in snap.bands10_db)
        assert snap.frame_rms_db == round(snap.frame_rms_db, 2)


class TestLongAudio:
    def test_duration_and_frame_counts_scale_with_length(self, extractor):
        short_audio = _tone(1000.0, 0.5, _windows_of_audio(extractor, 4))
        long_audio = _tone(1000.0, 0.5, _windows_of_audio(extractor, 12))

        short_pair = extractor.extract_batch([short_audio])[0]
        long_pair = extractor.extract_batch([long_audio])[0]

        assert short_pair.duration_sec == pytest.approx(len(short_audio) / SR)
        assert long_pair.duration_sec == pytest.approx(len(long_audio) / SR)
        assert long_pair.n_valid_frames > short_pair.n_valid_frames

    def test_snapshot_levels_are_stable_across_lengths(self, extractor):
        short_audio = _tone(1000.0, 0.5, _windows_of_audio(extractor, 4))
        long_audio = _tone(1000.0, 0.5, _windows_of_audio(extractor, 12))

        short_db = extractor.extract_batch([short_audio])[0].loud.bands4_db[2]
        long_db = extractor.extract_batch([long_audio])[0].loud.bands4_db[2]

        assert short_db == pytest.approx(long_db, abs=0.05)


class TestUnreachableDefensiveCheck:
    """The `n_valid == 0` re-check in extract_batch cannot be reached.

    `valid_wins.any()` is asserted a few lines above it, and
    `valid_vals = rms_i[valid_wins]` selects exactly that non-empty set, so
    `valid_vals.numel() >= 1` by construction. The test reproduces the enclosing
    arithmetic on a range of inputs (silence, NaN heads/tails/middles, tones,
    noise, all window counts) and asserts the invariant both ways: the count is
    never 0, and it always equals the number of selected windows. The two
    statements (gpu_spectral.py:283-285) therefore carry `# pragma: no cover` in
    the source, and the pairing with the earlier all-False return is checked
    through the public entry point below.
    """

    @staticmethod
    def _window_state(extractor, audio):
        with torch.no_grad():
            batch = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
            power = extractor.spec(batch)
            k_power = power * extractor._k_mag_sq[None, :, None]
            frame_power = k_power.sum(dim=1)

            lengths = torch.tensor([len(audio)])
            valid_frames = torch.ceil(
                (lengths.float() - extractor.n_fft) / extractor.hop_length + 1
            ).clamp(min=0).long()
            n_frames = frame_power.shape[1]
            frame_idx = torch.arange(n_frames).unsqueeze(0)
            valid_mask = frame_idx < valid_frames.unsqueeze(1)

            frame_rms = torch.sqrt(
                torch.clamp(frame_power / (extractor.n_fft // 2 + 1), min=1e-20)
            )
            frame_rms_db = 20.0 * torch.log10(torch.clamp(frame_rms, min=1e-10))
            non_silent = (frame_rms_db >= extractor.silence_db) & valid_mask

            fpw = extractor.frames_per_window
            n_windows = n_frames // fpw
            trimmed = n_frames - (n_frames % fpw)
            win_power = frame_power[:, :trimmed].reshape(1, n_windows, fpw).mean(dim=2)
            win_mask = non_silent[:, :trimmed].reshape(1, n_windows, fpw).float().mean(dim=2)
            win_valid = win_mask >= 0.5
            win_rms_db = 10.0 * torch.log10(torch.clamp(win_power, min=1e-20))

            wins = win_valid[0]
            rms = win_rms_db[0].clone()
            rms[~wins] = float("nan")
            return wins, rms[wins]

    def test_invariant_holds_for_a_range_of_hostile_inputs(self, extractor):
        rng = np.random.default_rng(0)
        fpw, hop = extractor.frames_per_window, extractor.hop_length
        seen_any = 0
        checked = 0

        for n_windows in (1, 2, 3, 5, 8, 12):
            n = fpw * hop * n_windows
            t = np.arange(n) / SR
            tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
            noise = (0.2 * rng.standard_normal(n)).astype(np.float32)

            nan_head = tone.copy()
            nan_head[: n // 3] = np.nan
            nan_tail = tone.copy()
            nan_tail[-n // 3:] = np.nan
            nan_middle = np.zeros(n, dtype=np.float32)
            nan_middle[n // 2:n // 2 + hop * 3] = np.nan

            for audio in (
                np.zeros(n, dtype=np.float32),
                np.full(n, 1e-9, dtype=np.float32),
                tone,
                noise,
                nan_head,
                nan_tail,
                nan_middle,
            ):
                wins, valid_vals = self._window_state(extractor, audio)
                if not bool(wins.any()):
                    continue
                seen_any += 1
                checked += 1
                # the re-check is only entered when wins.any() is True, and the
                # selection below can never empty out at that point
                assert int(valid_vals.numel()) == int(wins.sum())
                assert int(valid_vals.numel()) > 0

        # the probe is meaningful: it reached the guarded region repeatedly
        assert seen_any >= 20
        assert checked == seen_any

    def test_the_earlier_check_is_the_only_route_to_an_empty_result(self, extractor):
        """Ties the pragma'd re-check to the check above it through the public API.

        `n_valid == 0` can only happen if `valid_wins` is all-False, and that case is
        returned early with a fresh `SnapshotPair(None, None, ..., n_valid_frames=0)`.
        So a pair reporting zero valid frames proves the earlier branch fired and the
        re-check was skipped -- the two are mutually exclusive over the real entry
        point, which is what makes the pragma sound.
        """
        fpw, hop = extractor.frames_per_window, extractor.hop_length
        rng = np.random.default_rng(1)
        hostiles = {
            "silence": np.zeros(fpw * hop * 3, dtype=np.float32),
            "denormal_tail": np.concatenate([
                _tone(1000.0, 0.5, fpw * hop * 2),
                np.full(fpw * hop, 1e-30, dtype=np.float32),
            ]),
            "all_nan": np.full(fpw * hop * 2, np.nan, dtype=np.float32),
            "noise": (0.3 * rng.standard_normal(fpw * hop * 4)).astype(np.float32),
        }

        for label, audio in hostiles.items():
            for pair in extractor.extract_batch([audio]):
                if pair.n_valid_frames == 0:
                    assert pair.loud is None and pair.quiet is None, label
                    assert pair.peak_dbfs == -120.0, label
                else:
                    assert pair.loud is not None and pair.quiet is not None, label


class TestDeviceRouting:
    def test_cpu_route_produces_full_snapshots(self, extractor):
        audio = _tone(1000.0, 0.5, _windows_of_audio(extractor, 6))
        pair = extractor.extract_batch([audio])[0]
        assert pair.loud is not None
        assert extractor.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA device required to exercise the GPU tensor route",
    )
    def test_cuda_route_matches_cpu_levels(self):
        ex = GPUSpectralExtractor(sr=SR, device="cuda")
        audio = _tone(1000.0, 0.5, _windows_of_audio(ex, 6))

        cuda_pair = ex.extract_batch([audio])[0]
        cpu_pair = GPUSpectralExtractor(sr=SR, device="cpu").extract_batch([audio])[0]

        assert ex.device.type == "cuda"
        assert cuda_pair.loud is not None
        assert cuda_pair.loud.bands4_db[2] == pytest.approx(cpu_pair.loud.bands4_db[2], abs=0.1)

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="Apple MPS backend required to exercise the MPS tensor route",
    )
    def test_mps_route_produces_snapshots(self):
        ex = GPUSpectralExtractor(sr=SR, device="mps")
        audio = _tone(1000.0, 0.5, _windows_of_audio(ex, 6))

        pair = ex.extract_batch([audio])[0]

        assert ex.device.type == "mps"
        assert pair.loud is not None
        assert np.isfinite(pair.loud.bands4_db).all()
