# Created: 2026-03-25
# Purpose: plugin analysis engine — PluginDoctor style measurements
#
# Measurements:
# 1. Linear: impulse response -> frequency response (magnitude + phase)
# 2. Harmonic: THD, THD+N, IMD
# 3. Sweep: THD vs frequency, 2D spectrogram (aliasing detection)
# 4. Dynamics: ramp (I/O curve), attack/release
# 5. Oscilloscope: waveshaper curve
# 6. Performance: processing time measurement

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np

from audioman.core.test_signal import (
    generate_impulse, generate_sine, generate_two_tone,
    generate_white_noise, generate_sweep,
    generate_dynamics_ramp, generate_dynamics_attack_release,
    generate_log_sweep_deconv, generate_multitone,
    to_mid_side, from_mid_side,
)
from audioman.plugins.vst3 import VST3PluginWrapper

logger = logging.getLogger(__name__)


@dataclass
class LinearResult:
    """Frequency response measurement result"""
    frequencies: list[float]      # Hz
    magnitude_db: list[float]     # dB
    phase_deg: list[float]        # degrees
    sample_rate: int
    fft_size: int
    method: str                   # "impulse" or "noise"


@dataclass
class HarmonicResult:
    """Harmonic distortion measurement result"""
    thd_percent: float
    thd_plus_n_percent: float
    fundamental_freq: float
    fundamental_db: float
    harmonics: list[dict]         # [{freq, db, order}]
    imd_percent: Optional[float] = None
    method: str = "thd"


@dataclass
class SweepResult:
    """Sweep analysis result"""
    frequencies: list[float]
    thd_per_freq: list[float]     # THD vs frequency
    gain_per_freq: list[float]    # dB gain vs frequency
    spectrogram: Optional[np.ndarray] = None  # 2D (time, freq) for aliasing
    time_axis: Optional[list[float]] = None
    freq_axis: Optional[list[float]] = None


@dataclass
class DynamicsResult:
    """Dynamics measurement result"""
    input_levels_db: list[float]
    output_levels_db: list[float]
    gain_reduction_db: list[float]
    method: str = "ramp"          # "ramp" or "attack_release"
    attack_release_audio: Optional[np.ndarray] = None


@dataclass
class OscilloscopeResult:
    """Oscilloscope / waveshaper result"""
    input_signal: np.ndarray
    output_signal: np.ndarray
    # waveshaper: input->output mapping
    waveshaper_input: list[float]
    waveshaper_output: list[float]


@dataclass
class PerformanceResult:
    """Performance measurement result"""
    buffer_sizes: list[int]
    process_times_ms: list[float]
    samples_per_second: list[float]
    realtime_ratio: list[float]


def _load_plugin(plugin_path: str, params: Optional[dict] = None) -> VST3PluginWrapper:
    wrapper = VST3PluginWrapper(plugin_path)
    wrapper.load()
    if params:
        wrapper.set_parameters(params)
    return wrapper


# Performance stimulus level, ~-20 dBFS: hot enough to load a plugin's dynamics path,
# quiet enough to stay clear of clipping in the measurement.
_PERFORMANCE_STIMULUS_LEVEL_DB = -20.0


def _set_plugin_parameter(plugin: Any, name: str, value: Any) -> bool:
    """Set one pedalboard parameter, returning False when the plugin rejected it.

    Absorbs only the two ways `setattr` on a pedalboard plugin signals "no such
    parameter or unusable value" — verified against pedalboard 0.9.22: AttributeError
    for an unknown name (VST3Plugin delegates to object.__setattr__, Reverb raises
    directly) and TypeError/ValueError from a parameter's own coercion (a pybind
    signature mismatch on built-ins, a range or type message on external plugins).
    Callers turn False into a warning; a plugin that cannot take the parameter runs
    with its own default, which is a fact worth reporting rather than dropping.
    """
    try:
        setattr(plugin, name, value)
        return True
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug(f"parameter {name}={value!r} rejected: {exc}")
        return False


# =============================================================================
# 1. Linear Analysis
# =============================================================================


def measure_linear(
    plugin_path: str,
    params: Optional[dict] = None,
    sample_rate: int = 44100,
    fft_size: int = 16384,
    method: str = "impulse",
    level_db: float = 0.0,
) -> LinearResult:
    """Frequency response measurement (magnitude + phase)

    method: "impulse" (delta) or "noise" (white noise averaged)
    """
    wrapper = _load_plugin(plugin_path, params)

    if method == "impulse":
        test = generate_impulse(sample_rate, duration_sec=fft_size / sample_rate + 0.1,
                                level_db=level_db)
    else:
        test = generate_white_noise(sample_rate, duration_sec=2.0, level_db=level_db)

    output = wrapper.process(test, sample_rate)

    # convert to mono
    mono = output[0] if output.ndim == 2 else output

    # FFT
    window = np.hanning(fft_size).astype(np.float32)
    frame = mono[:fft_size] * window
    spectrum = np.fft.rfft(frame)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum, deg=True)

    # convert to dB
    mag_db = 20 * np.log10(magnitude + 1e-10)

    return LinearResult(
        frequencies=freqs.tolist(),
        magnitude_db=mag_db.tolist(),
        phase_deg=phase.tolist(),
        sample_rate=sample_rate,
        fft_size=fft_size,
        method=method,
    )


# =============================================================================
# 2. Harmonic Analysis (THD, IMD)
# =============================================================================


def measure_thd(
    plugin_path: str,
    params: Optional[dict] = None,
    frequency: float = 1000.0,
    level_db: float = -6.0,
    sample_rate: int = 44100,
    fft_size: int = 16384,
) -> HarmonicResult:
    """THD + THD+N measurement"""
    wrapper = _load_plugin(plugin_path, params)

    duration = fft_size / sample_rate + 0.5
    test = generate_sine(frequency, sample_rate, duration, level_db)
    output = wrapper.process(test, sample_rate)

    mono = output[0] if output.ndim == 2 else output
    # use the stable region (skip the first 0.1 s)
    skip = int(0.1 * sample_rate)
    frame = mono[skip:skip + fft_size]
    if len(frame) < fft_size:
        frame = np.pad(frame, (0, fft_size - len(frame)))

    window = np.hanning(fft_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frame * window))
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    # fundamental peak
    fund_bin = int(round(frequency * fft_size / sample_rate))
    search_range = max(3, fund_bin // 20)
    fund_region = spectrum[max(0, fund_bin - search_range):fund_bin + search_range]
    fund_peak = np.max(fund_region)
    fund_db = 20 * np.log10(fund_peak + 1e-10)

    # find the harmonic peaks
    harmonics = []
    harmonic_energy = 0.0
    max_harmonic = min(16, int(sample_rate / 2 / frequency))

    for h in range(2, max_harmonic + 1):
        h_bin = int(round(h * frequency * fft_size / sample_rate))
        if h_bin >= len(spectrum):
            break
        sr = max(3, h_bin // 50)
        region = spectrum[max(0, h_bin - sr):min(len(spectrum), h_bin + sr)]
        # Unreachable: line 223 breaks as soon as h_bin >= len(spectrum), so h_bin <
        # bins here and the slice reaches at least min(h_bin + sr, bins) > h_bin >= 0.
        if len(region) == 0:  # pragma: no cover - h_bin < len(spectrum), sr >= 3
            continue
        h_peak = np.max(region)
        h_db = 20 * np.log10(h_peak + 1e-10)
        harmonic_energy += h_peak**2
        harmonics.append({"freq": round(h * frequency, 1), "db": round(h_db, 2), "order": h})

    # THD = sqrt(sum(harmonics^2)) / fundamental
    thd = np.sqrt(harmonic_energy) / (fund_peak + 1e-10) * 100

    # THD+N = sqrt(sum(everything_except_fundamental^2)) / fundamental
    total_energy = np.sum(spectrum**2)
    fund_energy = fund_peak**2
    thd_n = np.sqrt(max(0, total_energy - fund_energy)) / (fund_peak + 1e-10) * 100

    return HarmonicResult(
        thd_percent=round(thd, 4),
        thd_plus_n_percent=round(thd_n, 4),
        fundamental_freq=frequency,
        fundamental_db=round(fund_db, 2),
        harmonics=harmonics,
        method="thd",
    )


def measure_imd(
    plugin_path: str,
    params: Optional[dict] = None,
    freq_low: float = 60.0,
    freq_high: float = 7000.0,
    sample_rate: int = 44100,
    fft_size: int = 16384,
) -> HarmonicResult:
    """IMD (intermodulation distortion) measurement"""
    wrapper = _load_plugin(plugin_path, params)

    duration = fft_size / sample_rate + 0.5
    test = generate_two_tone(freq_low, freq_high, sample_rate, duration)
    output = wrapper.process(test, sample_rate)

    mono = output[0] if output.ndim == 2 else output
    skip = int(0.1 * sample_rate)
    frame = mono[skip:skip + fft_size]
    if len(frame) < fft_size:
        frame = np.pad(frame, (0, fft_size - len(frame)))

    window = np.hanning(fft_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frame * window))
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    # 7kHz peak
    high_bin = int(round(freq_high * fft_size / sample_rate))
    sr = max(3, high_bin // 50)
    high_peak = np.max(spectrum[max(0, high_bin - sr):high_bin + sr])

    # IMD sidebands: 7000 ± N*60 Hz
    imd_energy = 0.0
    harmonics = []
    for n in range(1, 11):
        for sign in [-1, 1]:
            sb_freq = freq_high + sign * n * freq_low
            if sb_freq <= 0 or sb_freq >= sample_rate / 2:
                continue
            sb_bin = int(round(sb_freq * fft_size / sample_rate))
            if sb_bin >= len(spectrum):
                continue
            sr2 = max(2, sb_bin // 100)
            region = spectrum[max(0, sb_bin - sr2):min(len(spectrum), sb_bin + sr2)]
            # Unreachable: the `sb_bin >= len(spectrum)` check above skips every bin
            # that is out of range, so sb_bin is in range here and the slice reaches
            # min(sb_bin + sr2, bins) > sb_bin >= 0, which is never empty.
            if len(region) == 0:  # pragma: no cover - sb_bin < len(spectrum), sr2 >= 2
                continue
            sb_peak = np.max(region)
            imd_energy += sb_peak**2
            sb_db = 20 * np.log10(sb_peak + 1e-10)
            harmonics.append({"freq": round(sb_freq, 1), "db": round(sb_db, 2), "order": f"±{n}"})

    imd = np.sqrt(imd_energy) / (high_peak + 1e-10) * 100

    return HarmonicResult(
        thd_percent=0.0,
        thd_plus_n_percent=0.0,
        fundamental_freq=freq_high,
        fundamental_db=round(20 * np.log10(high_peak + 1e-10), 2),
        harmonics=harmonics,
        imd_percent=round(imd, 4),
        method="imd",
    )


# =============================================================================
# 3. Sweep Analysis
# =============================================================================


def measure_sweep(
    plugin_path: str,
    params: Optional[dict] = None,
    freq_start: float = 20.0,
    freq_end: float = 20000.0,
    sample_rate: int = 44100,
    duration_sec: float = 6.0,
    level_db: float = -6.0,
    fft_size: int = 4096,
    hop_size: int = 1024,
) -> SweepResult:
    """Frequency sweep -> THD vs freq + 2D spectrogram"""
    wrapper = _load_plugin(plugin_path, params)

    test = generate_sweep(freq_start, freq_end, sample_rate, duration_sec, level_db)
    output = wrapper.process(test, sample_rate)

    mono = output[0] if output.ndim == 2 else output
    n = len(mono)

    # STFT → 2D spectrogram
    window = np.hanning(fft_size).astype(np.float32)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    n_frames = (n - fft_size) // hop_size + 1

    spectrogram = np.zeros((n_frames, len(freqs)), dtype=np.float32)
    time_axis = []

    for i in range(n_frames):
        start = i * hop_size
        frame = mono[start:start + fft_size] * window
        spectrum = np.abs(np.fft.rfft(frame))
        spectrogram[i] = 20 * np.log10(spectrum + 1e-10)
        time_axis.append(start / sample_rate)

    # compute the fundamental frequency at each sweep instant
    sweep_freqs = []
    thd_per_freq = []
    gain_per_freq = []

    for i in range(n_frames):
        t = time_axis[i]
        # current sweep frequency (exponential)
        ratio = t / duration_sec
        current_freq = freq_start * (freq_end / freq_start) ** ratio

        if current_freq > sample_rate / 4:  # above Nyquist/2 THD is meaningless
            break

        sweep_freqs.append(round(current_freq, 1))

        # fundamental peak
        fund_bin = int(round(current_freq * fft_size / sample_rate))
        if fund_bin >= len(freqs) or fund_bin < 1:
            thd_per_freq.append(0.0)
            gain_per_freq.append(0.0)
            continue

        spec = 10 ** (spectrogram[i] / 20)  # linear
        sr = max(2, fund_bin // 20)
        fund_peak = np.max(spec[max(0, fund_bin - sr):min(len(spec), fund_bin + sr)])

        # harmonic energy
        h_energy = 0.0
        for h in range(2, 8):
            h_bin = int(round(h * current_freq * fft_size / sample_rate))
            if h_bin >= len(spec):
                break
            sr2 = max(2, h_bin // 30)
            h_energy += np.max(spec[max(0, h_bin - sr2):min(len(spec), h_bin + sr2)])**2

        thd = np.sqrt(h_energy) / (fund_peak + 1e-10) * 100
        thd_per_freq.append(round(thd, 4))
        gain_per_freq.append(round(20 * np.log10(fund_peak + 1e-10), 2))

    return SweepResult(
        frequencies=sweep_freqs,
        thd_per_freq=thd_per_freq,
        gain_per_freq=gain_per_freq,
        spectrogram=spectrogram,
        time_axis=time_axis,
        freq_axis=freqs.tolist(),
    )


# =============================================================================
# 4. Dynamics
# =============================================================================


def measure_dynamics_ramp(
    plugin_path: str,
    params: Optional[dict] = None,
    frequency: float = 1000.0,
    sample_rate: int = 44100,
    level_start_db: float = -80.0,
    level_end_db: float = 0.0,
    step_db: float = 1.0,
) -> DynamicsResult:
    """Measure output level per input level (compressor I/O curve)"""
    wrapper = _load_plugin(plugin_path, params)

    test, levels = generate_dynamics_ramp(
        frequency, sample_rate, level_start_db, level_end_db, step_db,
    )
    output = wrapper.process(test, sample_rate)

    mono = output[0] if output.ndim == 2 else output
    step_samples = int(0.5 * sample_rate)

    output_levels = []
    for i in range(len(levels)):
        start = i * step_samples
        end = start + step_samples
        segment = mono[start:end]
        peak = np.max(np.abs(segment))
        out_db = 20 * np.log10(peak + 1e-10)
        output_levels.append(round(out_db, 2))

    gain_reduction = [round(o - i, 2) for i, o in zip(levels, output_levels)]

    return DynamicsResult(
        input_levels_db=levels,
        output_levels_db=output_levels,
        gain_reduction_db=gain_reduction,
        method="ramp",
    )


def measure_dynamics_ar(
    plugin_path: str,
    params: Optional[dict] = None,
    frequency: float = 1000.0,
    sample_rate: int = 44100,
    level_below_db: float = -30.0,
    level_above_db: float = 0.0,
) -> DynamicsResult:
    """Attack/release response measurement"""
    wrapper = _load_plugin(plugin_path, params)

    test = generate_dynamics_attack_release(
        frequency, sample_rate, level_below_db, level_above_db,
    )
    output = wrapper.process(test, sample_rate)

    # extract the RMS envelope
    hop = 256
    mono = output[0] if output.ndim == 2 else output
    rms_env = []
    for i in range(0, len(mono) - hop, hop):
        rms = np.sqrt(np.mean(mono[i:i + hop]**2))
        rms_env.append(round(20 * np.log10(rms + 1e-10), 2))

    return DynamicsResult(
        input_levels_db=[level_below_db, level_above_db, level_below_db],
        output_levels_db=rms_env,
        gain_reduction_db=[],
        method="attack_release",
        attack_release_audio=output,
    )


# =============================================================================
# 5. Oscilloscope / Waveshaper
# =============================================================================


@dataclass
class WaveshaperV2Result:
    """Multi-amplitude waveshaper measurement result (v2)"""
    input_values: np.ndarray       # (n_points,) uniform distribution over [-1, +1]
    output_values: np.ndarray      # (n_points,) mapped output
    n_points: int
    levels_db: list[float]
    input_coverage: float          # 0~1 (how much of [-1,+1] the input covers)
    is_symmetric: bool             # f(-x) ≈ -f(x) symmetry
    raw_pairs: Optional[list[tuple[np.ndarray, np.ndarray]]] = None


def measure_waveshaper(
    plugin_path: str,
    params: Optional[dict] = None,
    frequency: float = 100.0,
    level_db: float = 0.0,
    sample_rate: int = 44100,
) -> OscilloscopeResult:
    """Extract the input->output waveshaper curve"""
    wrapper = _load_plugin(plugin_path, params)

    test = generate_sine(frequency, sample_rate, 0.5, level_db)
    output = wrapper.process(test, sample_rate)

    in_mono = test[0]
    out_mono = output[0] if output.ndim == 2 else output

    # extract one stable cycle
    period = int(sample_rate / frequency)
    skip = int(0.1 * sample_rate)
    in_cycle = in_mono[skip:skip + period]
    out_cycle = out_mono[skip:skip + period]

    # sort by input value -> waveshaper curve
    sort_idx = np.argsort(in_cycle)
    ws_input = in_cycle[sort_idx].tolist()
    ws_output = out_cycle[sort_idx].tolist()

    return OscilloscopeResult(
        input_signal=in_cycle,
        output_signal=out_cycle,
        waveshaper_input=ws_input,
        waveshaper_output=ws_output,
    )


def measure_waveshaper_v2(
    plugin_path: str,
    params: Optional[dict] = None,
    frequency: float = 100.0,
    sample_rate: int = 44100,
    levels_db: Optional[list[float]] = None,
    n_cycles: int = 3,
    n_points: int = 256,
    preroll_sec: float = 0.2,
) -> WaveshaperV2Result:
    """Extract a multi-amplitude-level waveshaper curve (v2)

    Addresses the limitations of measure_waveshaper():
    - single level -> multiple levels (7 by default) covering the whole input range
    - 1 cycle -> averaging over several cycles to reduce noise/transient influence
    - variable point count -> uniform 256-point resampling

    Args:
        plugin_path: VST3 path
        params: plugin parameters
        frequency: test sine frequency (Hz)
        sample_rate: sample rate
        levels_db: dBFS levels to measure (default: [-24, -18, -12, -6, -3, -1, 0])
        n_cycles: number of cycles to average (default 3)
        n_points: final resampling point count (default 256)
        preroll_sec: length of the silence preroll in seconds — latency compensation

    Returns:
        WaveshaperV2Result
    """
    if levels_db is None:
        levels_db = [-24.0, -18.0, -12.0, -6.0, -3.0, -1.0, 0.0]

    wrapper = _load_plugin(plugin_path, params)
    period_samples = int(sample_rate / frequency)

    # collect input->output pairs for each level
    all_inputs = []
    all_outputs = []
    raw_pairs = []

    for level_db in levels_db:
        # reset plugin state (measure each level independently)
        wrapper.reset()

        # needed span: preroll + settling (skip 1 cycle) + measurement (n_cycles cycles)
        # generate a signal long enough to secure a stable region
        settle_cycles = 1  # skipped cycles to avoid the transient response
        total_cycles_needed = settle_cycles + n_cycles
        test_duration = preroll_sec + (total_cycles_needed + 2) * (1.0 / frequency)

        # generate the sine (preroll silence included)
        preroll_samples = int(preroll_sec * sample_rate)
        sine_duration = test_duration - preroll_sec
        sine_signal = generate_sine(frequency, sample_rate, sine_duration, level_db)

        # concatenate the silence preroll with the sine
        silence = np.zeros((sine_signal.shape[0], preroll_samples), dtype=np.float32)
        test_signal = np.concatenate([silence, sine_signal], axis=1)

        # run through the plugin
        output = wrapper.process(test_signal, sample_rate)

        # extract mono
        in_mono = test_signal[0]
        out_mono = output[0] if output.ndim == 2 else output

        # stable region start: preroll + settle_cycles periods in
        stable_start = preroll_samples + settle_cycles * period_samples

        # extract n_cycles periods, then average per period
        level_in_cycles = []
        level_out_cycles = []

        for c in range(n_cycles):
            start = stable_start + c * period_samples
            end = start + period_samples
            if end > len(in_mono) or end > len(out_mono):
                break
            level_in_cycles.append(in_mono[start:end])
            level_out_cycles.append(out_mono[start:end])

        if not level_in_cycles:
            logger.warning(f"level {level_db}dB: not enough complete cycles, skipping")
            continue

        # average per period (reduces transient response and noise)
        avg_in = np.mean(level_in_cycles, axis=0)
        avg_out = np.mean(level_out_cycles, axis=0)

        raw_pairs.append((avg_in.copy(), avg_out.copy()))

        # sort by input value
        sort_idx = np.argsort(avg_in)
        all_inputs.append(avg_in[sort_idx])
        all_outputs.append(avg_out[sort_idx])

    if not all_inputs:
        raise RuntimeError("measurement failed at every level — not enough data could be extracted")

    # merge the data from all levels and sort by input value
    combined_in = np.concatenate(all_inputs)
    combined_out = np.concatenate(all_outputs)
    global_sort = np.argsort(combined_in)
    combined_in = combined_in[global_sort]
    combined_out = combined_out[global_sort]

    # resample to a uniform n_points grid
    x_uniform = np.linspace(-1.0, 1.0, n_points)

    # interpolate only within the actual data range (no extrapolation outside it)
    in_min, in_max = combined_in[0], combined_in[-1]
    output_values = np.interp(x_uniform, combined_in, combined_out)

    # coverage: how much of [-1, +1] the input covers
    input_coverage = float((in_max - in_min) / 2.0)  # relative to the full 2.0 range

    # symmetry check: f(-x) ≈ -f(x) means odd-harmonic symmetry
    # compare both sides around the center (0)
    n_half = n_points // 2
    f_neg_x = output_values[:n_half][::-1]   # f(-x) reversed
    neg_f_x = -output_values[n_points - n_half:]  # -f(x)

    # symmetry error (normalized)
    max_output = np.max(np.abs(output_values)) + 1e-10
    symmetry_error = np.mean(np.abs(f_neg_x - neg_f_x)) / max_output
    is_symmetric = bool(symmetry_error < 0.05)  # 5% or less counts as symmetric

    logger.info(
        f"Waveshaper v2: {len(levels_db)} levels, coverage={input_coverage:.1%}, "
        f"symmetric={is_symmetric} (error={symmetry_error:.4f})"
    )

    return WaveshaperV2Result(
        input_values=x_uniform.astype(np.float32),
        output_values=output_values.astype(np.float32),
        n_points=n_points,
        levels_db=levels_db,
        input_coverage=round(input_coverage, 4),
        is_symmetric=is_symmetric,
        raw_pairs=raw_pairs,
    )


# =============================================================================
# 6. Performance
# =============================================================================


def measure_performance(
    plugin_path: str,
    params: Optional[dict] = None,
    sample_rate: int = 44100,
    buffer_sizes: Optional[list[int]] = None,
    n_iterations: int = 100,
) -> PerformanceResult:
    """Processing callback time measurement"""
    wrapper = _load_plugin(plugin_path, params)

    if buffer_sizes is None:
        buffer_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    process_times = []
    sps_list = []
    rt_ratios = []

    for bs in buffer_sizes:
        # Deterministic seeded white noise as the timing stimulus: the same signal on
        # every run keeps block-size timings comparable, and the full band exercises
        # more of the plugin than a single tone would. One extra sample is requested
        # because generate_white_noise sizes itself by duration and
        # int(sample_rate * duration) can land a sample short at an arbitrary sample
        # rate; the trim to the block size is then exact.
        duration_sec = (bs + 1) / sample_rate
        test = generate_white_noise(
            sample_rate,
            duration_sec=duration_sec,
            level_db=_PERFORMANCE_STIMULUS_LEVEL_DB,
        )[:, :bs]
        times = []

        for _ in range(n_iterations):
            t0 = time.perf_counter()
            wrapper.process(test, sample_rate)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        avg_ms = np.median(times)
        process_times.append(round(avg_ms, 4))

        # samples per second
        sps = bs / (avg_ms / 1000) if avg_ms > 0 else 0
        sps_list.append(round(sps))

        # realtime ratio
        rt = sps / sample_rate if sample_rate > 0 else 0
        rt_ratios.append(round(rt, 2))

    return PerformanceResult(
        buffer_sizes=buffer_sizes,
        process_times_ms=process_times,
        samples_per_second=sps_list,
        realtime_ratio=rt_ratios,
    )


# =============================================================================
# Integration: compare 2 plugins
# =============================================================================


def compare_linear(
    plugin_path_1: str,
    plugin_path_2: str,
    params_1: Optional[dict] = None,
    params_2: Optional[dict] = None,
    sample_rate: int = 44100,
) -> dict:
    """Compare the frequency responses (difference) of 2 plugins"""
    r1 = measure_linear(plugin_path_1, params_1, sample_rate)
    r2 = measure_linear(plugin_path_2, params_2, sample_rate)

    diff_db = [round(a - b, 4) for a, b in zip(r1.magnitude_db, r2.magnitude_db)]
    diff_phase = [round(a - b, 4) for a, b in zip(r1.phase_deg, r2.phase_deg)]

    return {
        "plugin_1": asdict(r1),
        "plugin_2": asdict(r2),
        "diff_magnitude_db": diff_db,
        "diff_phase_deg": diff_phase,
    }


# =============================================================================
# 7. CLAP embedding profiling
# =============================================================================


def measure_clap_profile(
    plugin_path: str,
    param_sweeps: dict[str, list],
    base_params: Optional[dict] = None,
    sample_rate: int = 44100,
    duration_sec: float = 2.0,
    test_frequency: float = 1000.0,
    test_level_db: float = -6.0,
) -> dict:
    """Generate CLAP embeddings per parameter sweep — a "fingerprint" of saturation

    Args:
        plugin_path: VST3 path
        param_sweeps: {"drive": [0, 25, 50, 75, 100], "style": ["Soft", "Hard"]}
        base_params: base parameters {"mix": 100.0, ...}

    Returns: {
        "embeddings": [(param_values, embedding_512d), ...],
        "labels": ["drive=0", "drive=25", ...],
        "embeddings_npy": np.ndarray (N, 512),
    }
    """
    try:
        import laion_clap
    except ImportError:
        raise ImportError("CLAP required: pip install laion-clap")

    import soundfile as sf
    import tempfile
    import os

    wrapper = _load_plugin(plugin_path, base_params)

    # test signal
    n_samples = int(duration_sec * sample_rate)
    test = generate_sine(test_frequency, sample_rate, duration_sec, test_level_db)

    # build every parameter combination
    import itertools
    param_names = list(param_sweeps.keys())
    param_values_list = list(param_sweeps.values())
    combinations = list(itertools.product(*param_values_list))

    # load the plugin once, only swap parameters
    from pedalboard import load_plugin as pb_load
    plugin = pb_load(plugin_path)
    if base_params:
        for k, v in base_params.items():
            if not _set_plugin_parameter(plugin, k, v):
                logger.warning(f"base parameter {k}={v!r} rejected by {plugin_path}")

    tmpdir = tempfile.mkdtemp()
    wav_paths = []
    labels = []
    param_records = []

    for combo in combinations:
        # apply parameters (same instance reused)
        param_dict = dict(zip(param_names, combo))
        for k, v in param_dict.items():
            fallback = k.replace(" ", "_")
            if not (_set_plugin_parameter(plugin, k, v)
                    or _set_plugin_parameter(plugin, fallback, v)):
                logger.warning(
                    f"parameter {k}={v!r} (and {fallback!r}) rejected by {plugin_path}; "
                    f"this combination runs with the plugin default for it"
                )

        output = plugin.process(test, sample_rate)

        # save WAV
        label = ", ".join(f"{k}={v}" for k, v in param_dict.items())
        labels.append(label)
        param_records.append(param_dict)

        wav_path = os.path.join(tmpdir, f"{len(wav_paths):04d}.wav")
        if output.ndim == 2:
            sf.write(wav_path, output.T, sample_rate, subtype='FLOAT')
        else:
            sf.write(wav_path, output, sample_rate, subtype='FLOAT')
        wav_paths.append(wav_path)

    # CLAP encoding
    logger.info(f"CLAP encoding: {len(wav_paths)} settings")
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    batch_size = 32
    all_emb = []
    for i in range(0, len(wav_paths), batch_size):
        batch = wav_paths[i:i + batch_size]
        emb = model.get_audio_embedding_from_filelist(batch, use_tensor=False)
        all_emb.append(emb)

    embeddings = np.concatenate(all_emb, axis=0)

    # cleanup
    for p in wav_paths:
        os.unlink(p)
    os.rmdir(tmpdir)

    return {
        "labels": labels,
        "params": param_records,
        "embeddings_npy": embeddings,
        "n_settings": len(combinations),
        "embedding_dim": embeddings.shape[1],
    }


# =============================================================================
# 8. EQ profiling
# =============================================================================


@dataclass
class EQResponseResult:
    """EQ frequency/phase response measurement result"""
    frequencies: list[float]       # Hz
    magnitude_db: list[float]      # dB (relative to bypass)
    phase_deg: list[float]         # degrees
    group_delay_ms: list[float]    # ms
    params: dict                   # parameters used for the measurement
    sample_rate: int
    fft_size: int
    is_minimum_phase: bool
    thd_at_1k: float               # nonlinearity indicator (%)


def _deconvolve(output: np.ndarray, inverse_filter: np.ndarray) -> np.ndarray:
    """Extract the impulse response by applying the inverse filter to the sweep output (FFT convolution)"""
    n = len(output) + len(inverse_filter) - 1
    # pad to a power of two (FFT efficiency)
    n_fft = 1
    while n_fft < n:
        n_fft *= 2

    O = np.fft.rfft(output, n=n_fft)
    I = np.fft.rfft(inverse_filter, n=n_fft)
    ir = np.fft.irfft(O * I, n=n_fft)
    return ir.astype(np.float32)


def _check_minimum_phase(magnitude_db: np.ndarray, phase_deg: np.ndarray) -> bool:
    """Determine minimum phase via the Hilbert transform

    Minimum-phase system: phase = -Hilbert(ln|H(f)|)
    If the difference between the measured phase and the Hilbert-derived phase is
    small, the system is minimum phase.
    """
    # log magnitude -> Hilbert transform -> minimum phase
    log_mag = np.log(10 ** (magnitude_db / 20.0) + 1e-10)
    # Hilbert transform (discrete)
    n = len(log_mag)
    if n < 4:
        return True

    spectrum = np.fft.rfft(log_mag)
    # compute minimum phase: imag(Hilbert(log|H|))
    min_phase_rad = -np.imag(np.fft.irfft(
        1j * np.sign(np.fft.rfftfreq(2 * n - 1, 1.0)) * np.fft.rfft(log_mag, n=2 * n - 1),
        n=2 * n - 1,
    ))[:n]
    min_phase_deg = np.degrees(min_phase_rad)

    # compare with the measured phase (excluding regions near DC and Nyquist)
    trim = max(1, n // 20)
    measured = np.array(phase_deg[trim:-trim])
    expected = min_phase_deg[trim:-trim]

    if len(measured) == 0:
        return True

    error = np.mean(np.abs(measured - expected))
    return bool(error < 15.0)  # within 15 degrees counts as minimum phase


def measure_eq_response(
    plugin_path: str,
    params: Optional[dict] = None,
    bypass_params: Optional[dict] = None,
    sample_rate: int = 44100,
    fft_size: int = 32768,
    sweep_duration: float = 6.0,
    level_db: float = -12.0,
) -> EQResponseResult:
    """Measure EQ frequency/phase/group delay — log sweep deconvolution

    1. Sweep in the bypass state -> extract the reference IR
    2. Sweep with the target parameters -> extract the target IR
    3. Compute the difference in the frequency domain -> response relative to bypass

    Args:
        plugin_path: VST3 path
        params: EQ parameters to measure
        bypass_params: bypass-state parameters (loads without parameters when None)
        sample_rate: sample rate
        fft_size: FFT size (for low-frequency resolution, default 32768)
        sweep_duration: sweep length in seconds
        level_db: input level (dBFS)

    Returns:
        EQResponseResult
    """
    sweep_audio, inverse_filter = generate_log_sweep_deconv(
        sample_rate=sample_rate,
        duration_sec=sweep_duration,
        level_db=level_db,
    )
    inv_mono = inverse_filter[0]

    # 1) bypass measurement (reference)
    wrapper_bypass = _load_plugin(plugin_path, bypass_params)
    bypass_output = wrapper_bypass.process(sweep_audio, sample_rate)
    bypass_mono = bypass_output[0] if bypass_output.ndim == 2 else bypass_output
    bypass_ir = _deconvolve(bypass_mono, inv_mono)

    # 2) measurement with the target parameters
    wrapper_target = _load_plugin(plugin_path, params)
    target_output = wrapper_target.process(sweep_audio, sample_rate)
    target_mono = target_output[0] if target_output.ndim == 2 else target_output
    target_ir = _deconvolve(target_mono, inv_mono)

    # 3) FFT — response relative to bypass
    window = np.hanning(fft_size).astype(np.float32)

    # locate the IR peak (where the linear response concentrates in the deconvolution result)
    bypass_peak = int(np.argmax(np.abs(bypass_ir)))
    target_peak = int(np.argmax(np.abs(target_ir)))

    # extract an fft_size window centered on the peak
    def _extract_ir_window(ir, peak_idx):
        half = fft_size // 2
        start = max(0, peak_idx - half // 4)  # slightly before the peak
        end = start + fft_size
        if end > len(ir):
            start = max(0, len(ir) - fft_size)
            end = start + fft_size
        segment = ir[start:end]
        if len(segment) < fft_size:
            segment = np.pad(segment, (0, fft_size - len(segment)))
        return segment * window

    bypass_frame = _extract_ir_window(bypass_ir, bypass_peak)
    target_frame = _extract_ir_window(target_ir, target_peak)

    bypass_spectrum = np.fft.rfft(bypass_frame)
    target_spectrum = np.fft.rfft(target_frame)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    # relative response: H_eq = H_target / H_bypass
    bypass_mag = np.abs(bypass_spectrum) + 1e-10
    target_mag = np.abs(target_spectrum)
    relative_mag = target_mag / bypass_mag
    magnitude_db = (20 * np.log10(relative_mag + 1e-10)).tolist()

    # phase (relative)
    bypass_phase = np.angle(bypass_spectrum)
    target_phase = np.angle(target_spectrum)
    relative_phase = np.degrees(target_phase - bypass_phase)
    # unwrap
    relative_phase_unwrapped = np.unwrap(np.radians(relative_phase))
    phase_deg = np.degrees(relative_phase_unwrapped).tolist()

    # group delay: -d(phase)/d(omega)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    d_phase = np.gradient(relative_phase_unwrapped, 2 * np.pi * df)
    group_delay_ms = (-d_phase * 1000).tolist()

    # minimum-phase determination
    is_min_phase = _check_minimum_phase(
        np.array(magnitude_db), np.array(phase_deg),
    )

    # THD @ 1kHz (nonlinearity indicator)
    thd_result = measure_thd(plugin_path, params, frequency=1000.0,
                             level_db=level_db, sample_rate=sample_rate)
    thd_at_1k = thd_result.thd_percent

    return EQResponseResult(
        frequencies=freqs.tolist(),
        magnitude_db=magnitude_db,
        phase_deg=phase_deg,
        group_delay_ms=group_delay_ms,
        params=params or {},
        sample_rate=sample_rate,
        fft_size=fft_size,
        is_minimum_phase=is_min_phase,
        thd_at_1k=thd_at_1k,
    )


def measure_eq_parameter_sweep(
    plugin_path: str,
    sweep_config: dict[str, dict],
    bypass_params: Optional[dict] = None,
    sample_rate: int = 44100,
    fft_size: int = 32768,
    level_db: float = -12.0,
) -> list[EQResponseResult]:
    """Batch frequency response measurement for EQ parameter combinations

    Args:
        plugin_path: VST3 path
        sweep_config: sweep configuration dict
            {
                "gain_sweep": {
                    "param": "band1_gain",
                    "values": [-12, -6, 0, 6, 12],
                    "fixed": {"band1_freq": 1000, "band1_q": 1.0}
                },
                "freq_sweep": {
                    "param": "band1_freq",
                    "values": [100, 500, 1000, 5000, 10000],
                    "fixed": {"band1_gain": 6.0, "band1_q": 1.0}
                },
            }
        bypass_params: bypass-state parameters
        sample_rate: sample rate
        fft_size: FFT size
        level_db: input level

    Returns:
        list[EQResponseResult] — measurement result for each parameter combination
    """
    results = []

    for sweep_name, config in sweep_config.items():
        param_name = config["param"]
        values = config["values"]
        fixed = config.get("fixed", {})

        logger.info(f"EQ sweep '{sweep_name}': {param_name} = {values}")

        for value in values:
            # combine the fixed parameters with the swept parameter
            params = dict(fixed)
            params[param_name] = value

            try:
                result = measure_eq_response(
                    plugin_path, params, bypass_params,
                    sample_rate=sample_rate,
                    fft_size=fft_size,
                    level_db=level_db,
                )
                results.append(result)
                logger.info(
                    f"  {param_name}={value}: peak={max(result.magnitude_db):.1f}dB, "
                    f"min_phase={result.is_minimum_phase}, thd={result.thd_at_1k:.4f}%"
                )
            except Exception as e:
                logger.warning(f"  {param_name}={value}: measurement failed — {e}")

    return results


def measure_eq_nonlinearity(
    plugin_path: str,
    params: Optional[dict] = None,
    bypass_params: Optional[dict] = None,
    levels_db: Optional[list[float]] = None,
    sample_rate: int = 44100,
    fft_size: int = 32768,
) -> list[EQResponseResult]:
    """Measure EQ nonlinearity (level dependence)

    The same EQ setting is measured at different input levels.
    Analog-modeled EQs change their response with level (saturation).

    Args:
        plugin_path: VST3 path
        params: EQ parameters
        bypass_params: bypass-state parameters
        levels_db: input levels to measure (dBFS)
        sample_rate: sample rate
        fft_size: FFT size

    Returns:
        list[EQResponseResult] — response result per level
    """
    if levels_db is None:
        levels_db = [-36.0, -24.0, -18.0, -12.0, -6.0, -3.0, 0.0]

    results = []
    for level in levels_db:
        try:
            result = measure_eq_response(
                plugin_path, params, bypass_params,
                sample_rate=sample_rate,
                fft_size=fft_size,
                level_db=level,
            )
            # record the level in params
            result.params = dict(result.params)
            result.params["_input_level_db"] = level
            results.append(result)
            logger.info(f"  level={level}dB: thd={result.thd_at_1k:.4f}%")
        except Exception as e:
            logger.warning(f"  level={level}dB: measurement failed — {e}")

    if len(results) >= 2:
        # check the response difference across levels
        ref = np.array(results[0].magnitude_db)
        max_deviation = 0.0
        for r in results[1:]:
            diff = np.max(np.abs(np.array(r.magnitude_db) - ref))
            max_deviation = max(max_deviation, diff)
        logger.info(f"  max response deviation across levels: {max_deviation:.2f} dB "
                     f"({'nonlinear' if max_deviation > 0.5 else 'linear'})")

    return results
