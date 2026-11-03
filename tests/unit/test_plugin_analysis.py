# tests/unit/test_plugin_analysis.py — PluginDoctor-style measurement engine
#
# The engine wraps a VST3 in VST3PluginWrapper, so every measurement is exercised here
# against a deterministic fake loader: the transform is a real numpy operation (identity,
# gain, clipper, saturator, delay, one-pole, compressor) and the assertions are the
# numeric properties the measurement claims to report — dB level shifts, harmonic growth
# with drive, phase slope of a delay, group delay, symmetry. Nothing here needs a real
# plugin, which the host does not have (AUD-1857).

import sys
import tempfile
import types

import numpy as np
import pytest

from audioman.core import plugin_analysis as pa
from audioman.core.test_signal import generate_log_sweep_deconv


# ---------------------------------------------------------------------------
# Fake plugin + transforms
# ---------------------------------------------------------------------------


class _FakePlugin:
    """Stand-in for a loaded plugin: a numpy transform plus call bookkeeping."""

    def __init__(self, transform):
        self.transform = transform
        self.process_shapes = []
        self.output_shapes = []
        self.sample_rates = []
        self.resets = 0
        self.parameter_sets = []
        self.load_calls = 0

    def load(self):
        self.load_calls += 1

    def process(self, audio, sample_rate, reset=True):
        audio = np.asarray(audio, dtype=np.float32)
        self.process_shapes.append(audio.shape)
        self.sample_rates.append(sample_rate)
        out = np.asarray(self.transform(audio, sample_rate), dtype=np.float32)
        self.output_shapes.append(out.shape)
        return out

    def reset(self):
        self.resets += 1

    def set_parameters(self, params):
        self.parameter_sets.append(params)


def _identity(audio, sample_rate):
    return audio


def _gain(factor):
    return lambda audio, sample_rate: audio * factor


def _hard_clip(threshold):
    return lambda audio, sample_rate: np.clip(audio, -threshold, threshold)


def _tanh_saturator(drive):
    return lambda audio, sample_rate: np.tanh(drive * audio)


def _shift_samples(n, gain=1.0):
    """Delay by ``n`` samples, wrapping — keeps the impulse inside the analysis frame."""
    return lambda audio, sample_rate: np.roll(audio, n, axis=1) * gain


def _delay_zero_padded(n, gain=1.0):
    """Delay by ``n`` samples, discarding the samples pushed past the end."""

    def f(audio, sample_rate):
        out = np.zeros_like(audio)
        if n > 0:
            out[:, n:] = audio[:, :-n]
        return out * gain

    return f


def _onepole(coefficient, gain=1.0):
    from scipy.signal import lfilter

    return lambda audio, sample_rate: lfilter(
        [gain * coefficient], [1.0, -(1.0 - coefficient)], audio, axis=1
    )


def _compressor(threshold=0.1, ratio=6.0, attack_ms=5.0, release_ms=200.0):
    """Feed-forward compressor with an asymmetric one-pole gain envelope."""

    def f(audio, sample_rate):
        attack = np.exp(-1.0 / (attack_ms / 1000.0 * sample_rate))
        release = np.exp(-1.0 / (release_ms / 1000.0 * sample_rate))
        env = np.zeros_like(audio)
        prev = 0.0
        for channel in range(audio.shape[0]):
            for i in range(audio.shape[1]):
                level = abs(float(audio[channel, i]))
                coeff = attack if level > prev else release
                prev = coeff * prev + (1.0 - coeff) * level
                env[channel, i] = prev
        gain = np.ones_like(env)
        over = env > threshold
        gain[over] = (threshold / env[over]) ** (1.0 - 1.0 / ratio)
        return audio * gain

    return f


def _mono_out(transform):
    return lambda audio, sample_rate: transform(audio, sample_rate)[0]


def _truncate(n_samples):
    return lambda audio, sample_rate: audio[:, :n_samples]


def _install(monkeypatch, *plugins):
    """Replace the module loader with a queue of prepared fake plugins.

    Returns the list of ``(path, params)`` pairs the engine asked for.
    """
    queue = list(plugins)
    calls = []

    def loader(path, params=None):
        calls.append((path, params))
        return queue.pop(0)

    monkeypatch.setattr(pa, "_load_plugin", loader)
    return calls


def _db(value):
    return 20.0 * np.log10(abs(value) + 1e-10)


# ---------------------------------------------------------------------------
# 0. Loader helpers
# ---------------------------------------------------------------------------


class TestLoadPlugin:
    def test_loads_then_sets_parameters(self, monkeypatch):
        seen = {}

        class Wrapper:
            def __init__(self, path):
                seen["path"] = path
                self.loaded = False

            def load(self):
                self.loaded = True

            def set_parameters(self, params):
                seen["params"] = params

        monkeypatch.setattr(pa, "VST3PluginWrapper", Wrapper)
        wrapper = pa._load_plugin("plugin.vst3", {"drive": 4.0})
        assert seen["path"] == "plugin.vst3"
        assert wrapper.loaded is True
        assert seen["params"] == {"drive": 4.0}

    def test_parameters_dict_is_only_applied_when_truthy(self, monkeypatch):
        """A missing dict is skipped, and so is an empty one — both leave the plugin on
        its own defaults."""
        calls = []

        class Wrapper:
            def __init__(self, path):
                pass

            def load(self):
                calls.append("load")

            def set_parameters(self, params):
                calls.append(params)

        monkeypatch.setattr(pa, "VST3PluginWrapper", Wrapper)
        pa._load_plugin("plugin.vst3")
        pa._load_plugin("plugin.vst3", {})
        assert calls == ["load", "load"]


class TestSetPluginParameter:
    def test_accepts_readable_attribute(self):
        class Plugin:
            pass

        plugin = Plugin()
        assert pa._set_plugin_parameter(plugin, "gain_db", 6.0) is True
        assert plugin.gain_db == 6.0

    def test_reports_each_rejection_kind(self):
        class Plugin:
            def __setattr__(self, name, value):
                if name == "unknown":
                    raise AttributeError(name)
                if name == "bad_type":
                    raise TypeError("incompatible function arguments")
                if name == "out_of_range":
                    raise ValueError("value out of range")
                object.__setattr__(self, name, value)

        plugin = Plugin()
        assert pa._set_plugin_parameter(plugin, "unknown", 1.0) is False
        assert pa._set_plugin_parameter(plugin, "bad_type", "loud") is False
        assert pa._set_plugin_parameter(plugin, "out_of_range", 900.0) is False

    def test_real_builtin_rejects_junk_and_accepts_valid(self):
        """The three exceptions the docstring claims are the real pedalboard behaviour."""
        pedalboard = pytest.importorskip("pedalboard")
        plugin = pedalboard.Reverb()
        assert pa._set_plugin_parameter(plugin, "no_such_parameter", 1.0) is False
        assert pa._set_plugin_parameter(plugin, "room_size", 0.8) is True
        assert plugin.room_size == pytest.approx(0.8, abs=1e-6)
        assert pa._set_plugin_parameter(plugin, "wet_level", 5.0) is False
        assert plugin.wet_level < 1.01


# ---------------------------------------------------------------------------
# 1. Linear analysis
# ---------------------------------------------------------------------------


class TestMeasureLinear:
    def test_impulse_identity_is_flat_with_zero_phase(self, monkeypatch):
        """The default delta sits at sample 0, where the window zeroes every bin."""
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_linear("p", fft_size=1024)
        assert result.method == "impulse"
        assert result.sample_rate == 44100
        assert result.fft_size == 1024
        assert len(result.frequencies) == 1024 // 2 + 1
        assert len(result.magnitude_db) == len(result.frequencies)
        assert len(result.phase_deg) == len(result.frequencies)
        # Hanning(0) == 0: the delta is multiplied away, so every bin is at the
        # floor. The tolerance is derived from float32 spacing rather than picked:
        # these spectra are computed in float32, and `np.spacing(np.float32(200.0))`
        # is 1.526e-5, so demanding 1e-6 here asks for more precision than the
        # arithmetic has. It passed on the dev box only because that NumPy/BLAS
        # build rounded the intermediate differently — CI caught the difference.
        assert max(result.magnitude_db) == pytest.approx(
            -200.0, abs=10 * float(np.spacing(np.float32(200.0)))
        )
        assert all(abs(p) < 1e-6 for p in result.phase_deg)

    def test_impulse_into_the_window_is_flat(self, monkeypatch):
        """A delta the analysis window can see gives a flat magnitude at the window value."""
        _install(monkeypatch, _FakePlugin(_shift_samples(100)))
        result = pa.measure_linear("p", fft_size=1024)
        magnitude = np.array(result.magnitude_db)
        assert magnitude.max() == pytest.approx(20 * np.log10(np.hanning(1024)[100] + 1e-10), abs=1e-3)
        assert magnitude.max() - magnitude.min() < 1e-3

    def test_delay_phase_slope_is_linear_in_frequency(self, monkeypatch):
        """A 100-sample delay gives a phase that turns by -360*100/fft_size per bin."""
        _install(monkeypatch, _FakePlugin(_shift_samples(100)))
        result = pa.measure_linear("p", fft_size=1024)
        phase = np.array(result.phase_deg)
        slope = np.polyfit(np.arange(1, 6), phase[1:6], 1)[0]
        assert slope == pytest.approx(-360.0 * 100 / 1024, rel=1e-3)
        # `np.angle` wraps to (-180, 180], so compare on the unit circle rather than the
        # raw numbers: bin 26 onwards has turned past half a revolution.
        bins = np.arange(len(phase))
        expected = np.exp(1j * np.radians(-360.0 * 100 * bins / 1024))
        measured = np.exp(1j * np.radians(phase))
        assert np.max(np.abs(measured - expected)) < 1e-6

    def test_noise_level_shift_matches_gain_in_db(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity), _FakePlugin(_gain(0.5)))
        reference = pa.measure_linear("p", fft_size=8192, method="noise")
        quiet = pa.measure_linear("p", fft_size=8192, method="noise")
        assert reference.method == "noise"
        diff = np.array(quiet.magnitude_db) - np.array(reference.magnitude_db)
        assert np.mean(diff) == pytest.approx(20 * np.log10(0.5), abs=1e-3)
        assert np.max(np.abs(diff - 20 * np.log10(0.5))) < 1e-2

    def test_noise_rolloff_is_monotone(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_onepole(0.05)))
        result = pa.measure_linear("p", fft_size=8192, method="noise")
        magnitude = np.array(result.magnitude_db)
        bands = [magnitude[10:100].mean(), magnitude[200:500].mean(),
                 magnitude[1000:3000].mean(), magnitude[4000:8000].mean()]
        assert bands[0] > bands[1] > bands[2] > bands[3]
        assert bands[0] - bands[3] > 20.0

    def test_impulse_response_decays_with_frequency_for_onepole(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_onepole(0.05)))
        result = pa.measure_linear("p", fft_size=8192)
        magnitude = np.array(result.magnitude_db)
        low = magnitude[5:50].mean()
        high = magnitude[4000:8000].mean()
        assert low > high + 80.0

    def test_mono_plugin_output_is_accepted(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_mono_out(_identity)))
        result = pa.measure_linear("p", fft_size=512)
        assert len(result.magnitude_db) == 257


# ---------------------------------------------------------------------------
# 2. Harmonic analysis
# ---------------------------------------------------------------------------


class TestMeasureThd:
    def test_clean_sine_has_low_thd_and_finds_harmonics(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_thd("p")
        assert result.method == "thd"
        assert result.fundamental_freq == 1000.0
        assert result.thd_percent < 0.05
        assert result.imd_percent is None
        # Harmonics 2..min(16, nyquist/fundamental) = 2..16, each at a multiple of 1 kHz.
        assert [h["order"] for h in result.harmonics] == list(range(2, 17))
        assert [h["freq"] for h in result.harmonics] == [1000.0 * h for h in range(2, 17)]
        assert all(h["db"] < result.fundamental_db for h in result.harmonics)

    def test_thd_is_scale_invariant_while_levels_track_gain(self, monkeypatch):
        """THD is a ratio: a linear gain moves every dB by the same amount."""
        _install(monkeypatch, _FakePlugin(_identity))
        unity = pa.measure_thd("p")
        _install(monkeypatch, _FakePlugin(_gain(0.5)))
        quiet = pa.measure_thd("p")
        _install(monkeypatch, _FakePlugin(_gain(2.0)))
        loud = pa.measure_thd("p")

        assert quiet.thd_percent == pytest.approx(unity.thd_percent, rel=1e-4)
        assert loud.thd_percent == pytest.approx(unity.thd_percent, rel=1e-4)
        assert quiet.fundamental_db == pytest.approx(unity.fundamental_db - 6.0206, abs=0.01)
        assert loud.fundamental_db == pytest.approx(unity.fundamental_db + 6.0206, abs=0.01)
        assert quiet.harmonics[0]["db"] == pytest.approx(unity.harmonics[0]["db"] - 6.0206, abs=0.01)

    def test_thd_and_harmonics_grow_monotonically_with_drive(self, monkeypatch):
        drives = [0.5, 1.0, 2.0, 5.0, 20.0]
        thds = []
        for drive in drives:
            _install(monkeypatch, _FakePlugin(_tanh_saturator(drive)))
            thds.append(pa.measure_thd("p").thd_percent)
        assert thds == sorted(thds)
        assert len(set(thds)) == len(thds)
        assert thds[0] < 1.0
        assert thds[-1] > 30.0

    def test_hard_clipper_raises_thd_with_drive(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_hard_clip(1.0)))
        soft = pa.measure_thd("p", level_db=-6.0)
        _install(monkeypatch, _FakePlugin(_hard_clip(0.25)))
        hard = pa.measure_thd("p", level_db=-6.0)
        assert hard.thd_percent > soft.thd_percent * 10
        assert hard.thd_plus_n_percent > soft.thd_plus_n_percent

    def test_short_output_is_padded_to_one_fft_frame(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_truncate(100)))
        result = pa.measure_thd("p")
        assert result.thd_percent == 0.0
        assert len(result.harmonics) == 15

    def test_mono_output_and_odd_fft_size(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_mono_out(_identity)))
        result = pa.measure_thd("p", frequency=1000.0, sample_rate=8000, fft_size=999)
        # Harmonics above the 4 kHz Nyquist stop the loop.
        assert [h["order"] for h in result.harmonics] == [2, 3]
        assert result.harmonics[0]["freq"] == 2000.0

    def test_odd_fft_size_includes_harmonic_up_to_nyquist(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_thd("p", frequency=1000.0, sample_rate=8000, fft_size=1001)
        assert [h["order"] for h in result.harmonics] == [2, 3, 4]

    def test_negative_fundamental_yields_no_harmonics(self, monkeypatch):
        """Frequency < 0 makes the harmonic count negative: the loop never runs."""
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_thd("p", frequency=-1000.0, sample_rate=8000, fft_size=1024)
        assert result.harmonics == []
        assert result.thd_percent == 0.0


class TestMeasureImd:
    def test_sidebands_sit_at_high_plus_minus_n_low(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_imd("p")
        assert result.method == "imd"
        assert result.fundamental_freq == 7000.0
        assert result.thd_percent == 0.0
        assert result.thd_plus_n_percent == 0.0
        offsets = [h["order"] for h in result.harmonics]
        assert offsets[:4] == ["±1", "±1", "±2", "±2"]
        assert result.harmonics[0]["freq"] == pytest.approx(6940.0)
        assert result.harmonics[1]["freq"] == pytest.approx(7060.0)
        assert len(result.harmonics) == 20

    def test_imd_rises_with_saturation_drive(self, monkeypatch):
        imds = []
        for drive in [0.5, 1.0, 2.0, 4.0, 8.0]:
            _install(monkeypatch, _FakePlugin(_tanh_saturator(drive)))
            imds.append(pa.measure_imd("p", freq_low=1000.0).imd_percent)
        assert imds == sorted(imds)
        assert len(set(imds)) == len(imds)

    def test_clipper_raises_imd_above_clean_baseline(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        clean = pa.measure_imd("p", freq_low=1000.0)
        _install(monkeypatch, _FakePlugin(_hard_clip(1.0)))
        clipped = pa.measure_imd("p", freq_low=1000.0)
        assert clipped.imd_percent > clean.imd_percent
        # The high tone itself is compressed by the clipper.
        assert clipped.fundamental_db < clean.fundamental_db

    def test_imd_is_scale_invariant_for_linear_gain(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        unity = pa.measure_imd("p", freq_low=1000.0)
        _install(monkeypatch, _FakePlugin(_gain(0.05)))
        quiet = pa.measure_imd("p", freq_low=1000.0)
        assert quiet.imd_percent == pytest.approx(unity.imd_percent, rel=1e-6)

    def test_sidebands_outside_the_audio_band_are_skipped(self, monkeypatch):
        """A low tone too high to produce in-band sidebands leaves none to measure."""
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_imd("p", freq_low=30000.0)
        assert result.harmonics == []
        assert result.imd_percent == 0.0

    def test_sideband_bin_past_nyquist_is_skipped(self, monkeypatch):
        """An odd FFT size can round a valid sideband frequency onto bin len(spectrum)."""
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_imd("p", freq_low=2.0, freq_high=22045.0,
                                sample_rate=44100, fft_size=999)
        freqs = [h["freq"] for h in result.harmonics]
        assert not any(f >= 22050.0 for f in freqs)
        assert len(freqs) < 20

    def test_sideband_rounding_onto_the_bin_count_is_skipped(self, monkeypatch):
        """The out-of-range bin guard fires for a sideband that passes the Nyquist test.

        `sb_freq < sample_rate / 2` is checked in real arithmetic, but the bin index is
        `int(round(sb_freq * fft_size / sample_rate))`. With an odd `fft_size` the exact
        quotient sits at the `k + 0.5` tie, and ties-to-even rounds it up to
        `fft_size // 2 + 1` -- one past the last bin, which is `len(spectrum)`.

        Here the high tone is placed so its first lower sideband lands on the largest
        frequency still below Nyquist; that sideband is skipped while the remaining nine
        are measured a step further down the band. The test pins the guard's observable
        effect (the band edge is missing from the list) rather than the rounding itself.
        """
        _install(monkeypatch, _FakePlugin(_identity))
        sample_rate, fft_size, freq_low = 8000.1, 2051, 60.0
        boundary = float(np.nextafter(sample_rate / 2.0, -np.inf))
        freq_high = boundary + freq_low

        # Premise: the boundary sideband is legal by the frequency test but out of
        # range once it is rounded to a bin index.
        assert boundary < sample_rate / 2
        assert int(round(boundary * fft_size / sample_rate)) == fft_size // 2 + 1

        result = pa.measure_imd("p", freq_low=freq_low, freq_high=freq_high,
                                sample_rate=sample_rate, fft_size=fft_size)
        freqs = [float(f) for f in (h["freq"] for h in result.harmonics)]
        # The n = 1 lower sideband is the one the guard drops.
        assert round(boundary, 1) not in freqs
        assert freqs == [round(freq_high - n * freq_low, 1) for n in range(2, 11)]
        # The dropped sideband carries no energy, so IMD still reflects the rest.
        assert np.isfinite(result.imd_percent)

    def test_padded_frame_finds_the_clean_tone(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_truncate(5000)))
        result = pa.measure_imd("p")
        assert len(result.harmonics) == 20
        assert np.isfinite(result.imd_percent)

    def test_mono_output_and_silence(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_mono_out(_identity)))
        assert pa.measure_imd("p").imd_percent > 0.0
        _install(monkeypatch, _FakePlugin(lambda audio, sample_rate: np.zeros_like(audio)))
        silent = pa.measure_imd("p")
        assert silent.imd_percent == 0.0
        # Same float32-spacing reasoning as the linear case: 1e-6 is below the
        # resolution of a float32 computation near 200.
        assert silent.fundamental_db == pytest.approx(
            -200.0, abs=10 * float(np.spacing(np.float32(200.0)))
        )


# ---------------------------------------------------------------------------
# 3. Sweep analysis
# ---------------------------------------------------------------------------


class TestMeasureSweep:
    def test_spectrogram_shape_and_axes(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_sweep("p", duration_sec=0.5, fft_size=2048, hop_size=512)
        n_frames = (22050 - 2048) // 512 + 1
        assert result.spectrogram.shape == (n_frames, 2048 // 2 + 1)
        assert len(result.time_axis) == n_frames
        assert len(result.freq_axis) == 2048 // 2 + 1
        assert result.time_axis[1] == pytest.approx(512 / 44100)
        assert result.time_axis == sorted(result.time_axis)
        assert np.isfinite(result.spectrogram).all()
        assert result.freq_axis[-1] == pytest.approx(22050.0)

    def test_sweep_tracks_its_own_instantaneous_frequency(self, monkeypatch):
        """The frame's peak bin advances with the log sweep, never above Nyquist/2."""
        _install(monkeypatch, _FakePlugin(_identity), _FakePlugin(_identity))
        default = pa.measure_sweep("p", duration_sec=0.5, fft_size=2048, hop_size=512)
        late_start = pa.measure_sweep("p", freq_start=2000.0, duration_sec=0.5,
                                      fft_size=2048, hop_size=512)
        freqs = np.array(default.frequencies)
        assert freqs[0] == pytest.approx(20.0)
        assert np.all(np.diff(freqs) > 0)
        assert max(freqs) <= 44100 / 4
        assert len(default.gain_per_freq) == len(freqs)
        assert len(default.thd_per_freq) == len(freqs)
        assert len(default.frequencies) == len(default.time_axis)
        # Starting the sweep late leaves the last frames above Nyquist/2, where the THD
        # measurement stops: fewer frequency points than spectrogram frames.
        assert late_start.frequencies[0] == pytest.approx(2000.0)
        assert max(late_start.frequencies) <= 44100 / 4
        assert len(late_start.frequencies) < late_start.spectrogram.shape[0]

    def test_gain_per_frame_shift_matches_a_linear_gain(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        unity = pa.measure_sweep("p", duration_sec=0.5)
        _install(monkeypatch, _FakePlugin(_gain(0.5)))
        quiet = pa.measure_sweep("p", duration_sec=0.5)
        assert len(quiet.frequencies) == len(unity.frequencies)
        diff = np.array(quiet.gain_per_freq, dtype=float) - np.array(unity.gain_per_freq, dtype=float)
        assert np.allclose(diff, 20 * np.log10(0.5), atol=0.05)
        # THD is a ratio between harmonics and the fundamental: gain cancels.
        assert np.allclose(quiet.thd_per_freq, unity.thd_per_freq, rtol=1e-3)

    def test_thd_rises_for_a_hard_clipper(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        unity = pa.measure_sweep("p", duration_sec=0.5, fft_size=2048, hop_size=512)
        _install(monkeypatch, _FakePlugin(_hard_clip(0.2)))
        clipped = pa.measure_sweep("p", duration_sec=0.5, fft_size=2048, hop_size=512)
        # The clipper removes level from almost every frame and the THD rises by an
        # order of magnitude. Frames that pass a sample close to zero are unaffected,
        # so this is a mean rather than an all().
        assert np.mean(clipped.gain_per_freq) < np.mean(unity.gain_per_freq)
        assert np.median(clipped.thd_per_freq) > 5 * np.median(unity.thd_per_freq)
        assert np.count_nonzero(
            np.array(clipped.gain_per_freq, dtype=float) < np.array(unity.gain_per_freq, dtype=float)
        ) > 0.9 * len(unity.gain_per_freq)

    def test_clipper_compresses_more_as_input_level_rises(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_hard_clip(0.2)), _FakePlugin(_hard_clip(0.2)))
        quiet = pa.measure_sweep("p", duration_sec=0.5, level_db=-40.0)
        loud = pa.measure_sweep("p", duration_sec=0.5, level_db=0.0)
        # -40 dBFS in stays linear, so the measured gain is the true one;
        # 0 dBFS in is clipped flat, so the measured gain collapses at the end.
        assert quiet.gain_per_freq[0] > quiet.gain_per_freq[-1]
        assert loud.gain_per_freq[0] > loud.gain_per_freq[-1]
        assert quiet.gain_per_freq[-1] < loud.gain_per_freq[0]

    def test_frame_below_the_first_bin_reports_zero(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_sweep("p", freq_start=1.0, freq_end=50.0,
                                  duration_sec=0.2, fft_size=4096, hop_size=1024)
        assert result.frequencies[:4] == [pytest.approx(f) for f in [1.0, 1.6, 2.5, 3.9]]
        assert result.thd_per_freq[:4] == [0.0, 0.0, 0.0, 0.0]
        assert result.gain_per_freq[:4] == [0.0, 0.0, 0.0, 0.0]
        assert result.gain_per_freq[-1] != 0.0

    def test_mono_output(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_mono_out(_identity)))
        result = pa.measure_sweep("p", duration_sec=0.2, fft_size=1024, hop_size=512)
        assert len(result.frequencies) > 0


# ---------------------------------------------------------------------------
# 4. Dynamics
# ---------------------------------------------------------------------------


class TestMeasureDynamicsRamp:
    def test_linear_plugin_tracks_the_input_curve(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_dynamics_ramp("p", level_start_db=-40.0, level_end_db=0.0, step_db=10.0)
        assert result.method == "ramp"
        assert result.input_levels_db == [-40.0, -30.0, -20.0, -10.0, 0.0]
        assert result.output_levels_db == [pytest.approx(level, abs=0.05) for level in result.input_levels_db]
        assert result.gain_reduction_db == [pytest.approx(0.0, abs=0.05)] * 5

    def test_gain_shift_is_the_same_at_every_level(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_gain(0.5)))
        result = pa.measure_dynamics_ramp("p", level_start_db=-40.0, level_end_db=0.0, step_db=10.0)
        assert result.gain_reduction_db == [pytest.approx(-6.0206, abs=0.01)] * 5
        assert result.output_levels_db == [
            pytest.approx(level - 6.0206, abs=0.01) for level in result.input_levels_db
        ]

    def test_clipper_only_reduces_gain_above_threshold(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_hard_clip(0.2)))
        result = pa.measure_dynamics_ramp("p", level_start_db=-40.0, level_end_db=0.0, step_db=10.0)
        reduction = result.gain_reduction_db
        assert reduction[0] == pytest.approx(0.0, abs=0.05)
        assert reduction[1] == pytest.approx(0.0, abs=0.05)
        assert reduction[-1] == pytest.approx(_db(0.2), abs=0.05)
        assert reduction[-1] < reduction[-2] < 0.0

    def test_compressor_curve_flattens_the_top(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_compressor(threshold=0.1)))
        result = pa.measure_dynamics_ramp("p", level_start_db=-40.0, level_end_db=0.0, step_db=10.0)
        output = result.output_levels_db
        assert output[0] == pytest.approx(-40.0, abs=0.1)
        assert output[3] - output[2] < 10.0
        assert output[4] - output[3] < 10.0
        assert result.gain_reduction_db[-1] < -3.0

    def test_mono_output(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_mono_out(_gain(0.5))))
        result = pa.measure_dynamics_ramp("p", level_start_db=-20.0, level_end_db=-10.0, step_db=10.0)
        assert len(result.output_levels_db) == len(result.input_levels_db)


class TestMeasureDynamicsAr:
    def test_three_level_envelope_of_a_linear_plugin(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_dynamics_ar("p")
        assert result.method == "attack_release"
        assert result.input_levels_db == [-30.0, 0.0, -30.0]
        assert result.gain_reduction_db == []
        assert result.attack_release_audio.shape == (2, 88200)
        envelope = np.array(result.output_levels_db, dtype=float)
        hop = 256
        first = envelope[: int(0.5 * 44100) // hop]
        loud = envelope[int(0.5 * 44100) // hop: int(1.5 * 44100) // hop]
        tail = envelope[int(1.5 * 44100) // hop:]
        # RMS of a sine at level L is L/sqrt(2); the envelope reports exactly that.
        assert first.mean() == pytest.approx(_db(10 ** (-30 / 20) / np.sqrt(2)), abs=0.1)
        assert loud.mean() == pytest.approx(_db(1 / np.sqrt(2)), abs=0.1)
        assert tail[-20:].mean() == pytest.approx(_db(10 ** (-30 / 20) / np.sqrt(2)), abs=0.1)
        # The three-level input is echoed back in the reported input levels.
        assert loud.mean() > first.mean() + 25.0

    def test_compressor_pulls_the_loud_section_down_and_recovers(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_compressor(threshold=0.1, ratio=6.0)))
        result = pa.measure_dynamics_ar("p")
        envelope = np.array(result.output_levels_db, dtype=float)
        hop = 256
        quiet_end = int(0.5 * 44100) // hop
        loud_end = int(1.5 * 44100) // hop
        below = envelope[: quiet_end - 2]
        # Skip the attack transient so the comparison is between settled levels.
        loud = envelope[quiet_end + 2: loud_end]
        tail = envelope[loud_end:]
        # 30 dB of input difference comes out as far less than 30 dB.
        assert below.mean() - loud[-3:].mean() < 20.0
        assert loud[0] > loud[-1]
        assert loud.max() - loud.min() > 2.0
        # When the loud section ends the compressor overshoots then releases back up.
        assert tail.min() < tail[-1] - 10.0
        assert tail[-1] > tail.min()

    def test_mono_output(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_mono_out(_identity)))
        result = pa.measure_dynamics_ar("p")
        assert result.attack_release_audio is not None
        assert len(result.output_levels_db) > 0


# ---------------------------------------------------------------------------
# 5. Oscilloscope / waveshaper
# ---------------------------------------------------------------------------


class TestMeasureWaveshaper:
    def test_transfer_curve_of_a_linear_plugin_is_the_diagonal(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_waveshaper("p", frequency=100.0, level_db=0.0)
        assert len(result.waveshaper_input) == 441
        assert len(result.waveshaper_output) == 441
        assert result.input_signal.shape == (441,)
        assert result.output_signal.shape == (441,)
        # Inputs are sorted, so the curve can be plotted against x.
        assert np.all(np.diff(result.waveshaper_input) >= 0)
        assert np.allclose(result.waveshaper_input, result.waveshaper_output, atol=1e-5)
        assert min(result.waveshaper_input) > -1.0
        assert max(result.waveshaper_input) < 1.0

    def test_transfer_curve_reproduces_a_memoryless_saturator(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_tanh_saturator(5.0)))
        result = pa.measure_waveshaper("p", frequency=100.0, level_db=0.0)
        measured_in = np.array(result.waveshaper_input)
        measured_out = np.array(result.waveshaper_output)
        assert np.allclose(measured_out, np.tanh(5.0 * measured_in), atol=1e-5)
        # The curve is compressed relative to the diagonal.
        assert np.max(np.abs(measured_out)) < np.max(np.abs(measured_in))
        assert abs(measured_out[-1]) < 1.0

    def test_cycle_length_follows_the_test_frequency(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_mono_out(_identity)))
        result = pa.measure_waveshaper("p", frequency=1000.0)
        assert len(result.waveshaper_input) == 44
        assert result.input_signal.shape == (44,)


class TestMeasureWaveshaperV2:
    def test_default_levels_cover_the_full_input_range(self, monkeypatch):
        plugin = _FakePlugin(_identity)
        _install(monkeypatch, plugin)
        result = pa.measure_waveshaper_v2("p", frequency=1000.0)
        assert result.levels_db == [-24.0, -18.0, -12.0, -6.0, -3.0, -1.0, 0.0]
        assert result.n_points == 256
        assert len(result.input_values) == 256
        assert len(result.output_values) == 256
        assert result.input_values[0] == pytest.approx(-1.0)
        assert result.input_values[-1] == pytest.approx(1.0)
        assert result.input_coverage > 0.99
        assert result.is_symmetric is True
        assert len(result.raw_pairs) == 7
        # Each level is measured from a clean plugin state.
        assert plugin.resets == 7
        assert plugin.resets == len(result.raw_pairs)

    def test_linear_plugin_maps_input_to_output(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_waveshaper_v2("p", frequency=1000.0, levels_db=[-6.0, 0.0])
        assert np.max(np.abs(result.input_values - result.output_values)) < 2e-3
        for pair_in, pair_out in result.raw_pairs:
            assert np.allclose(pair_in, pair_out, atol=2e-3)

    def test_level_range_sets_the_coverage(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity), _FakePlugin(_identity))
        quiet = pa.measure_waveshaper_v2("p", frequency=1000.0, levels_db=[-24.0])
        loud = pa.measure_waveshaper_v2("p", frequency=1000.0, levels_db=[0.0])
        assert quiet.input_coverage == pytest.approx(10 ** (-24 / 20), abs=0.01)
        assert loud.input_coverage > 0.99
        assert loud.input_coverage > quiet.input_coverage

    def test_asymmetric_transfer_curve_is_detected(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(lambda audio, sample_rate: np.tanh(3.0 * audio) + 0.05))
        asymmetric = pa.measure_waveshaper_v2("p", frequency=1000.0, levels_db=[0.0])
        _install(monkeypatch, _FakePlugin(_tanh_saturator(3.0)))
        symmetric = pa.measure_waveshaper_v2("p", frequency=1000.0, levels_db=[0.0])
        assert asymmetric.is_symmetric is False
        assert symmetric.is_symmetric is True

    def test_preroll_and_cycle_count_shape_the_averaged_period(self, monkeypatch):
        plugin = _FakePlugin(_identity)
        _install(monkeypatch, plugin)
        result = pa.measure_waveshaper_v2("p", frequency=100.0, levels_db=[0.0],
                                          n_cycles=2, n_points=64, preroll_sec=0.0)
        assert result.n_points == 64
        assert len(result.input_values) == 64
        # One period at 100 Hz: each raw pair is a single averaged cycle.
        assert len(result.raw_pairs) == 1
        assert result.raw_pairs[0][0].shape == (441,)
        # The stimulus carries the requested preroll of silence.
        assert plugin.process_shapes[0][1] == 441 * 5

    def test_short_output_ends_the_cycle_loop_early(self, monkeypatch):
        period = 44
        cut = int(0.2 * 44100) + period + period + period // 2
        _install(monkeypatch, _FakePlugin(_truncate(cut)))
        result = pa.measure_waveshaper_v2("p", frequency=1000.0, levels_db=[0.0], n_cycles=3)
        assert len(result.raw_pairs) == 1
        assert result.raw_pairs[0][0].shape == (period,)

    def test_unusable_level_is_skipped_with_a_warning(self, monkeypatch, caplog):
        """A level whose output is too short for one period is dropped, not fatal."""

        class ShortFirstLevel(_FakePlugin):
            def __init__(self, transform):
                super().__init__(transform)
                self.calls = 0

            def process(self, audio, sample_rate, reset=True):
                out = super().process(audio, sample_rate, reset)
                self.calls += 1
                if self.calls == 1:
                    return out[:, :1]
                return out

        plugin = ShortFirstLevel(_identity)
        _install(monkeypatch, plugin)
        with caplog.at_level("WARNING"):
            result = pa.measure_waveshaper_v2("p", frequency=1000.0, levels_db=[-12.0, 0.0])
        assert len(result.raw_pairs) == 1
        # The skipped level stays in the requested list; only the data is missing.
        assert result.levels_db == [-12.0, 0.0]
        assert result.input_coverage > 0.99
        assert plugin.calls == 2
        assert plugin.resets == 2
        assert [r.levelname for r in caplog.records if r.levelno >= 30] == ["WARNING"]

    def test_all_levels_failing_raises(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_truncate(1)))
        with pytest.raises(RuntimeError):
            pa.measure_waveshaper_v2("p", frequency=1000.0, levels_db=[0.0])


# ---------------------------------------------------------------------------
# 6. Performance
# ---------------------------------------------------------------------------


class _FakeClock:
    """Deterministic perf_counter: yields the given stamps, fails when over-consumed."""

    def __init__(self, ticks):
        self.ticks = list(ticks)
        self.calls = 0

    def __call__(self):
        if not self.ticks:
            raise AssertionError("perf_counter called more often than the measurement needs")
        self.calls += 1
        return self.ticks.pop(0)


class TestMeasurePerformance:
    def test_reports_median_of_the_measured_call_times(self, monkeypatch):
        clock = _FakeClock([0.000, 0.002, 0.002, 0.004, 0.004, 0.006])
        monkeypatch.setattr(pa.time, "perf_counter", clock)
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_performance("p", buffer_sizes=[100], n_iterations=3)
        assert clock.calls == 6
        assert result.buffer_sizes == [100]
        assert result.process_times_ms == [2.0]
        assert result.samples_per_second == [50000]
        assert result.realtime_ratio == [pytest.approx(50000 / 44100, abs=0.01)]

    def test_zero_measured_time_reports_zero_rates(self, monkeypatch):
        monkeypatch.setattr(pa.time, "perf_counter", _FakeClock([0.0] * 4))
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_performance("p", buffer_sizes=[64], n_iterations=2)
        assert result.process_times_ms == [0.0]
        assert result.samples_per_second == [0]
        assert result.realtime_ratio == [0.0]

    def test_non_positive_sample_rate_reports_zero_realtime_ratio(self, monkeypatch):
        monkeypatch.setattr(pa.time, "perf_counter", _FakeClock([0.0, 0.001, 0.001, 0.002]))
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_performance("p", buffer_sizes=[64], n_iterations=2,
                                        sample_rate=-1)
        assert result.samples_per_second == [64000]
        assert result.realtime_ratio == [0]

    def test_stimulus_is_the_block_size_at_the_calibrated_level(self, monkeypatch):
        plugin = _FakePlugin(_identity)
        monkeypatch.setattr(pa.time, "perf_counter", _FakeClock([0.0, 0.001] * 4))
        _install(monkeypatch, plugin)
        result = pa.measure_performance("p", buffer_sizes=[64, 128], n_iterations=2)
        assert result.buffer_sizes == [64, 128]
        assert plugin.process_shapes == [(2, 64), (2, 64), (2, 128), (2, 128)]
        assert plugin.sample_rates == [44100] * 4
        assert np.isfinite(result.process_times_ms).all()
        assert all(t >= 0 for t in result.process_times_ms)

    def test_default_buffer_sizes_span_small_to_large_blocks(self, monkeypatch):
        monkeypatch.setattr(pa.time, "perf_counter", _FakeClock([0.0, 0.001] * 7))
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_performance("p", n_iterations=1)
        assert result.buffer_sizes == [64, 128, 256, 512, 1024, 2048, 4096]
        assert len(result.process_times_ms) == 7
        assert len(result.samples_per_second) == 7
        assert len(result.realtime_ratio) == 7


# ---------------------------------------------------------------------------
# 7. Two-plugin comparison
# ---------------------------------------------------------------------------


class TestCompareLinear:
    def test_gain_difference_shows_up_as_a_flat_db_offset(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_shift_samples(100)), _FakePlugin(_shift_samples(100, 0.5)))
        comparison = pa.compare_linear("a", "b")
        assert set(comparison) == {"plugin_1", "plugin_2", "diff_magnitude_db", "diff_phase_deg"}
        diff = np.array(comparison["diff_magnitude_db"])
        assert np.allclose(diff, 6.0206, atol=1e-3)
        assert np.allclose(comparison["diff_phase_deg"], 0.0, atol=1e-6)
        assert comparison["plugin_1"]["magnitude_db"][100] == pytest.approx(
            comparison["plugin_2"]["magnitude_db"][100] + 6.0206, abs=1e-3
        )
        assert comparison["plugin_1"]["method"] == "impulse"
        assert comparison["plugin_2"]["fft_size"] == 16384
        assert comparison["plugin_2"]["sample_rate"] == 44100
        assert len(diff) == len(comparison["plugin_1"]["frequencies"])
        assert len(diff) == len(comparison["diff_phase_deg"])

    def test_one_sample_of_extra_delay_shows_up_in_the_phase_difference(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_shift_samples(100)), _FakePlugin(_shift_samples(101)))
        comparison = pa.compare_linear("a", "b")
        diff_phase = np.array(comparison["diff_phase_deg"])
        fft_size = comparison["plugin_1"]["fft_size"]
        for bin_index in (1, 10, 100):
            # The second plugin is one sample later, so its phase leads by 360*f/fft.
            assert diff_phase[bin_index] == pytest.approx(
                360.0 * bin_index / fft_size, abs=1e-3
            )
        assert np.max(np.abs(comparison["diff_magnitude_db"])) < 1.0


# ---------------------------------------------------------------------------
# 8. CLAP embedding profile
# ---------------------------------------------------------------------------


class _FakeClapModule:
    """Minimal laion_clap stand-in: load_ckpt plus a deterministic embedding."""

    def __init__(self):
        self.loaded_ckpt = 0
        self.batch_sizes = []

    def CLAP_Module(self, enable_fusion=False):
        module = self

        class _Model:
            def __init__(self):
                module.fusion = enable_fusion

            def load_ckpt(self, *args, **kwargs):
                module.loaded_ckpt += 1

            def get_audio_embedding_from_filelist(self, batch, use_tensor=False):
                module.batch_sizes.append(len(batch))
                return np.tile(
                    np.arange(4, dtype=np.float32), (len(batch), 1)
                ) + len(batch)

        return _Model()


class _FakePedalboardPlugin:
    """Parameter assignment over a fixed allow-list, mirroring the pybind behaviour."""

    def __init__(self, allowed=("drive", "low_gain", "style"), gain=0.5):
        self.allowed = set(allowed)
        self.gain = gain
        self.params = {}
        self.process_calls = []

    def __setattr__(self, name, value):
        if name in ("allowed", "gain", "params", "process_calls"):
            object.__setattr__(self, name, value)
            return
        if name not in self.allowed:
            raise AttributeError(f"no parameter {name!r}")
        self.params[name] = value

    def process(self, audio, sample_rate, reset=True):
        audio = np.asarray(audio, dtype=np.float32)
        self.process_calls.append(audio.shape)
        return audio * self.gain


class TestMeasureClapProfile:
    def test_missing_clap_dependency_raises_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "laion_clap", None)
        with pytest.raises(ImportError):
            pa.measure_clap_profile("plugin.vst3", {"drive": [0]})

    def test_sweep_writes_one_render_per_combination_and_cleans_up(self, monkeypatch):
        clap = _FakeClapModule()
        module = types.ModuleType("laion_clap")
        module.CLAP_Module = clap.CLAP_Module
        monkeypatch.setitem(sys.modules, "laion_clap", module)

        plugin = _FakePedalboardPlugin()
        import pedalboard

        monkeypatch.setattr(pedalboard, "load_plugin", lambda path: plugin)

        made_dirs = []
        real_mkdtemp = tempfile.mkdtemp

        def recording_mkdtemp(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            made_dirs.append(path)
            return path

        monkeypatch.setattr(tempfile, "mkdtemp", recording_mkdtemp)
        seen_params = _install(monkeypatch, _FakePlugin(_identity))

        result = pa.measure_clap_profile(
            "plugin.vst3",
            {"drive": [0, 25], "style": [0.0, 1.0]},
            base_params={"mix": 100.0},
            duration_sec=0.05,
            sample_rate=8000,
        )

        assert result["n_settings"] == 4
        assert result["labels"] == [
            "drive=0, style=0.0", "drive=0, style=1.0",
            "drive=25, style=0.0", "drive=25, style=1.0",
        ]
        assert result["params"][-1] == {"drive": 25, "style": 1.0}
        assert result["embeddings_npy"].shape == (4, 4)
        assert result["embedding_dim"] == 4
        assert clap.loaded_ckpt == 1
        assert clap.batch_sizes == [4]
        # One render per combination, mono test tone through the wrapper once.
        assert len(plugin.process_calls) == 4
        assert plugin.process_calls[0] == (2, 400)
        assert seen_params == [("plugin.vst3", {"mix": 100.0})]
        # The rendered WAVs live in a temp directory that is removed again.
        assert len(made_dirs) == 1
        import os

        assert not os.path.exists(made_dirs[0])

    def test_rejected_parameters_fall_back_then_warn(self, monkeypatch, caplog):
        clap = _FakeClapModule()
        module = types.ModuleType("laion_clap")
        module.CLAP_Module = clap.CLAP_Module
        monkeypatch.setitem(sys.modules, "laion_clap", module)

        plugin = _FakePedalboardPlugin()
        import pedalboard

        monkeypatch.setattr(pedalboard, "load_plugin", lambda path: plugin)
        _install(monkeypatch, _FakePlugin(_identity))

        with caplog.at_level("WARNING"):
            result = pa.measure_clap_profile(
                "plugin.vst3",
                {"low gain": [3.0], "nope": [1.0]},
                base_params={"missing": 5.0},
                duration_sec=0.05,
                sample_rate=8000,
            )

        # "low gain" is rejected under that spelling but accepted as low_gain;
        # "nope" and the base parameter are rejected outright.
        assert plugin.params == {"low_gain": 3.0}
        assert result["labels"] == ["low gain=3.0, nope=1.0"]
        assert result["n_settings"] == 1

    def test_mono_render_is_written_as_one_channel(self, monkeypatch):
        clap = _FakeClapModule()
        module = types.ModuleType("laion_clap")
        module.CLAP_Module = clap.CLAP_Module
        monkeypatch.setitem(sys.modules, "laion_clap", module)

        class MonoPlugin(_FakePedalboardPlugin):
            def process(self, audio, sample_rate, reset=True):
                return np.asarray(audio, dtype=np.float32)[0]

        plugin = MonoPlugin()
        import pedalboard

        monkeypatch.setattr(pedalboard, "load_plugin", lambda path: plugin)
        _install(monkeypatch, _FakePlugin(_identity))
        result = pa.measure_clap_profile("plugin.vst3", {"drive": [0]},
                                         duration_sec=0.05, sample_rate=8000)
        assert result["embeddings_npy"].shape[0] == 1


# ---------------------------------------------------------------------------
# 9. EQ profiling — internals
# ---------------------------------------------------------------------------


class TestDeconvolve:
    @pytest.fixture(scope="class")
    def sweep(self):
        return generate_log_sweep_deconv(sample_rate=44100, duration_sec=0.3, level_db=-12.0)

    def test_impulse_response_is_zero_padded_to_a_power_of_two(self, sweep):
        sweep_audio, inverse = sweep
        ir = pa._deconvolve(sweep_audio[0], inverse[0])
        linear_length = len(sweep_audio[0]) + len(inverse[0]) - 1
        assert ir.dtype == np.float32
        assert len(ir) & (len(ir) - 1) == 0
        assert len(ir) >= linear_length
        assert len(ir) < 2 * linear_length

    def test_convolution_is_linear_in_the_output(self, sweep):
        sweep_audio, inverse = sweep
        single = pa._deconvolve(sweep_audio[0], inverse[0])
        doubled = pa._deconvolve(sweep_audio[0] * 2.0, inverse[0])
        assert np.allclose(doubled, single * 2.0, atol=1e-3)

    def test_delayed_output_shifts_the_ir_peak_by_the_delay(self, sweep):
        sweep_audio, inverse = sweep
        reference = pa._deconvolve(sweep_audio[0], inverse[0])
        reference_peak = int(np.argmax(np.abs(reference)))
        for delay in (1, 100, 1000):
            delayed = np.concatenate([np.zeros(delay, np.float32), sweep_audio[0][:-delay]])
            ir = pa._deconvolve(delayed, inverse[0])
            peak = int(np.argmax(np.abs(ir)))
            assert peak - reference_peak == delay

    def test_ir_energy_is_concentrated_in_the_peak(self, sweep):
        sweep_audio, inverse = sweep
        ir = pa._deconvolve(sweep_audio[0], inverse[0])
        peak = np.max(np.abs(ir))
        assert peak > 100 * np.median(np.abs(ir))


class TestCheckMinimumPhase:
    """``_check_minimum_phase`` compares the measured phase against the Hilbert
    transform of the same magnitude, so the test vectors have to be self-consistent
    in length: a phase array longer than the magnitude array cannot be compared."""

    @staticmethod
    def _hilbert_phase(magnitude_db):
        log_mag = np.log(10 ** (np.asarray(magnitude_db) / 20.0) + 1e-10)
        length = len(log_mag)
        frequencies = np.fft.rfftfreq(2 * length - 1, 1.0)
        signs = np.sign(frequencies)
        signs[0] = 0.0
        transformed = np.fft.irfft(
            1j * signs * np.fft.rfft(log_mag, n=2 * length - 1), n=2 * length - 1
        )
        return np.degrees(-np.imag(transformed))[:length]

    def test_phase_derived_from_the_magnitude_is_minimum_phase(self):
        n = 512
        magnitude_db = 20 * np.log10(np.abs(np.fft.rfft(np.hanning(n))) + 1e-6)
        assert len(magnitude_db) == 257
        assert pa._check_minimum_phase(
            magnitude_db, self._hilbert_phase(magnitude_db)
        ) is True

    def test_unrelated_phase_is_not_minimum_phase(self):
        n = 512
        magnitude_db = 20 * np.log10(np.abs(np.fft.rfft(np.hanning(n))) + 1e-6)
        rng = np.random.RandomState(7)
        random_phase = rng.uniform(-180.0, 180.0, len(magnitude_db))
        assert pa._check_minimum_phase(magnitude_db, random_phase) is False

    def test_phase_trimmed_away_entirely_is_treated_as_minimum_phase(self):
        """`trim` is len(magnitude)//20 = 10 here, so a 20-point phase leaves nothing
        to compare against and the function reports minimum phase."""
        assert pa._check_minimum_phase(np.zeros(200), np.zeros(20)) is True

    def test_mismatched_lengths_raise_rather_than_guess(self):
        """Phase and magnitude must describe the same frequency axis: a 100-point phase
        cannot be compared against a 200-point magnitude."""
        with pytest.raises(ValueError):
            pa._check_minimum_phase(np.zeros(200), np.zeros(100))

    def test_short_arrays_are_treated_as_minimum_phase(self):
        assert pa._check_minimum_phase(np.zeros(2), np.zeros(2)) is True
        assert pa._check_minimum_phase(np.zeros(3), np.zeros(3)) is True


# ---------------------------------------------------------------------------
# 9b. EQ profiling — measurements
# ---------------------------------------------------------------------------


class TestMeasureEqResponse:
    def test_identical_settings_measure_a_flat_zero_response(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity), _FakePlugin(_identity),
                 _FakePlugin(_identity))
        result = pa.measure_eq_response("p", fft_size=4096, sweep_duration=0.5)
        assert result.sample_rate == 44100
        assert result.fft_size == 4096
        assert len(result.frequencies) == 4096 // 2 + 1
        assert np.allclose(result.magnitude_db, 0.0, atol=1e-6)
        assert np.allclose(result.phase_deg, 0.0, atol=1e-6)
        assert np.allclose(result.group_delay_ms, 0.0, atol=1e-6)
        assert result.is_minimum_phase is True
        assert result.thd_at_1k < 0.05
        assert result.params == {}

    def test_gain_difference_is_flat_and_phase_free(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity), _FakePlugin(_gain(0.5)),
                 _FakePlugin(_identity))
        result = pa.measure_eq_response("p", params={"gain": 6.0}, bypass_params={"gain": 0.0},
                                        fft_size=4096, sweep_duration=0.5)
        assert np.allclose(result.magnitude_db, 20 * np.log10(0.5), atol=1e-3)
        assert np.allclose(result.group_delay_ms, 0.0, atol=1e-3)
        assert result.params == {"gain": 6.0}
        # A flat relative response is trivially minimum phase.
        assert result.is_minimum_phase is True

    def test_filter_shape_is_measured_as_a_tilt(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity),
                 _FakePlugin(_onepole(0.5)), _FakePlugin(_identity))
        result = pa.measure_eq_response("p", fft_size=4096, sweep_duration=0.3)
        magnitude = np.array(result.magnitude_db)
        low_band = magnitude[5:50].mean()
        high_band = magnitude[1500:2000].mean()
        # A one-pole low-pass at 0.5/(1 - 0.5 z^-1) is ~0 dB at DC and rolls off above.
        assert low_band == pytest.approx(0.0, abs=0.3)
        assert high_band < low_band - 8.0
        assert isinstance(result.is_minimum_phase, bool)

    def test_short_sweep_is_zero_padded_up_to_the_fft_size(self, monkeypatch):
        """A sweep shorter than fft_size forces the clamped window and the pad."""
        _install(monkeypatch, _FakePlugin(_identity), _FakePlugin(_gain(0.5)),
                 _FakePlugin(_identity))
        result = pa.measure_eq_response("p", fft_size=32768, sweep_duration=0.1)
        assert len(result.frequencies) == 32768 // 2 + 1
        assert np.allclose(result.magnitude_db, 20 * np.log10(0.5), atol=1e-3)
        assert np.isfinite(result.group_delay_ms).all()

    def test_late_target_peak_uses_the_clamped_window(self, monkeypatch):
        """A delay larger than half the window pushes the target IR peak past the end of
        the deconvolved buffer, so the centred window has to be clamped to fit."""
        _install(monkeypatch, _FakePlugin(_identity), _FakePlugin(_delay_zero_padded(10000)),
                 _FakePlugin(_identity))
        result = pa.measure_eq_response("p", fft_size=16384, sweep_duration=0.3)
        assert len(result.frequencies) == 16384 // 2 + 1
        assert np.isfinite(np.array(result.magnitude_db)).all()
        assert np.isfinite(np.array(result.phase_deg)).all()
        # A 10 ksample offset decorrelates the two IRs almost everywhere.
        assert np.mean(result.magnitude_db) < -50.0
        assert min(result.magnitude_db) < -100.0
        assert result.is_minimum_phase is False

    def test_moderate_delay_tilts_the_relative_response(self, monkeypatch):
        """Below the clamp threshold the IR peak is still inside the window, and the
        delay shows up as a bounded phase ramp rather than decorrelation."""
        _install(monkeypatch, _FakePlugin(_identity), _FakePlugin(_delay_zero_padded(1000)),
                 _FakePlugin(_identity))
        result = pa.measure_eq_response("p", fft_size=16384, sweep_duration=0.3)
        assert np.mean(result.magnitude_db) > -30.0
        assert np.max(np.abs(result.phase_deg)) > 100.0

    def test_thd_at_1k_follows_the_target_parameters(self, monkeypatch):
        _install(monkeypatch, _FakePlugin(_identity), _FakePlugin(_hard_clip(0.02)),
                 _FakePlugin(_hard_clip(0.02)))
        result = pa.measure_eq_response("p", params={"drive": 1.0},
                                        fft_size=4096, sweep_duration=0.3, level_db=-6.0)
        assert result.thd_at_1k > 10.0


class TestMeasureEqParameterSweep:
    def test_each_parameter_value_is_measured_with_its_own_response(self, monkeypatch):
        calls = []

        def loader(path, params=None):
            calls.append((path, params))
            gain = 10 ** (float(params["band1_gain"]) / 20.0) if params else 1.0
            return _FakePlugin(_gain(gain))

        monkeypatch.setattr(pa, "_load_plugin", loader)
        config = {
            "gain_sweep": {
                "param": "band1_gain",
                "values": [-12, -6, 0, 6, 12],
                "fixed": {"band1_freq": 1000.0},
            }
        }
        results = pa.measure_eq_parameter_sweep("p", config, bypass_params={"band1_gain": 0},
                                                fft_size=2048)
        assert len(results) == 5
        assert [r.params for r in results] == [
            {"band1_freq": 1000.0, "band1_gain": value} for value in [-12, -6, 0, 6, 12]
        ]
        for value, result in zip([-12, -6, 0, 6, 12], results):
            assert np.mean(result.magnitude_db) == pytest.approx(value, abs=0.02)
        assert calls[0] == ("p", {"band1_gain": 0})
        assert calls[1] == ("p", {"band1_freq": 1000.0, "band1_gain": -12})

    def test_failing_measurement_is_skipped_without_aborting_the_sweep(self, monkeypatch):
        _install(monkeypatch)  # empty loader queue: every load raises IndexError
        config = {"sweep": {"param": "gain", "values": [1.0, 2.0]}}
        results = pa.measure_eq_parameter_sweep("p", config, fft_size=2048)
        assert results == []

    def test_multiple_sweeps_are_concatenated(self, monkeypatch):
        def loader(path, params=None):
            return _FakePlugin(_identity)

        monkeypatch.setattr(pa, "_load_plugin", loader)
        results = pa.measure_eq_parameter_sweep(
            "p",
            {"a": {"param": "one", "values": [1.0, 2.0]},
             "b": {"param": "two", "values": [3.0]}},
            fft_size=2048,
        )
        assert len(results) == 3
        assert [r.params for r in results] == [{"one": 1.0}, {"one": 2.0}, {"two": 3.0}]


class TestMeasureEqNonlinearity:
    def test_level_dependent_compression_is_reported_per_level(self, monkeypatch):
        def loader(path, params=None):
            if params and "threshold" in params:
                return _FakePlugin(_hard_clip(float(params["threshold"])))
            return _FakePlugin(_identity)

        monkeypatch.setattr(pa, "_load_plugin", loader)
        results = pa.measure_eq_nonlinearity("p", params={"threshold": 0.05},
                                             levels_db=[-36.0, -24.0, -12.0, 0.0],
                                             fft_size=2048)
        assert [r.params["_input_level_db"] for r in results] == [-36.0, -24.0, -12.0, 0.0]
        for result in results:
            assert result.params["threshold"] == 0.05
        magnitudes = [float(np.mean(r.magnitude_db)) for r in results]
        thds = [r.thd_at_1k for r in results]
        # Louder input is compressed harder, and the clipping harmonics grow.
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert magnitudes[0] == pytest.approx(0.0, abs=0.05)
        assert magnitudes[-1] < -10.0
        assert thds == sorted(thds)
        assert thds[0] < 0.05
        assert thds[-1] > 10.0

    def test_linear_plugin_shows_no_level_dependence(self, monkeypatch):
        monkeypatch.setattr(pa, "_load_plugin", lambda path, params=None: _FakePlugin(_identity))
        results = pa.measure_eq_nonlinearity("p", levels_db=[-24.0, -6.0], fft_size=2048)
        assert len(results) == 2
        for result in results:
            assert np.allclose(result.magnitude_db, 0.0, atol=1e-6)
            assert result.thd_at_1k < 0.05

    def test_default_levels_cover_the_useful_range(self, monkeypatch):
        monkeypatch.setattr(pa, "_load_plugin", lambda path, params=None: _FakePlugin(_identity))
        results = pa.measure_eq_nonlinearity("p", fft_size=2048)
        assert [r.params["_input_level_db"] for r in results] == [
            -36.0, -24.0, -18.0, -12.0, -6.0, -3.0, 0.0
        ]

    def test_failing_measurement_is_skipped_without_aborting(self, monkeypatch):
        _install(monkeypatch)
        results = pa.measure_eq_nonlinearity("p", levels_db=[-12.0, -6.0], fft_size=2048)
        assert results == []
