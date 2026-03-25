# Created: 2026-03-25
# Purpose: 플러그인 분석 엔진 — PluginDoctor 스타일 측정
#
# 측정 항목:
# 1. Linear: impulse response → frequency response (magnitude + phase)
# 2. Harmonic: THD, THD+N, IMD
# 3. Sweep: THD vs frequency, 2D spectrogram (앨리어싱 감지)
# 4. Dynamics: ramp (I/O 곡선), attack/release
# 5. Oscilloscope: waveshaper 곡선
# 6. Performance: 처리 시간 측정

import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from audioman.core.test_signal import (
    generate_impulse, generate_sine, generate_two_tone,
    generate_white_noise, generate_sweep,
    generate_dynamics_ramp, generate_dynamics_attack_release,
    to_mid_side, from_mid_side,
)
from audioman.plugins.vst3 import VST3PluginWrapper

logger = logging.getLogger(__name__)


@dataclass
class LinearResult:
    """주파수 응답 측정 결과"""
    frequencies: list[float]      # Hz
    magnitude_db: list[float]     # dB
    phase_deg: list[float]        # degrees
    sample_rate: int
    fft_size: int
    method: str                   # "impulse" or "noise"


@dataclass
class HarmonicResult:
    """하모닉 왜곡 측정 결과"""
    thd_percent: float
    thd_plus_n_percent: float
    fundamental_freq: float
    fundamental_db: float
    harmonics: list[dict]         # [{freq, db, order}]
    imd_percent: Optional[float] = None
    method: str = "thd"


@dataclass
class SweepResult:
    """스윕 분석 결과"""
    frequencies: list[float]
    thd_per_freq: list[float]     # THD vs frequency
    gain_per_freq: list[float]    # dB gain vs frequency
    spectrogram: Optional[np.ndarray] = None  # 2D (time, freq) for aliasing
    time_axis: Optional[list[float]] = None
    freq_axis: Optional[list[float]] = None


@dataclass
class DynamicsResult:
    """다이내믹스 측정 결과"""
    input_levels_db: list[float]
    output_levels_db: list[float]
    gain_reduction_db: list[float]
    method: str = "ramp"          # "ramp" or "attack_release"
    attack_release_audio: Optional[np.ndarray] = None


@dataclass
class OscilloscopeResult:
    """오실로스코프/웨이브셰이퍼 결과"""
    input_signal: np.ndarray
    output_signal: np.ndarray
    # waveshaper: input→output 매핑
    waveshaper_input: list[float]
    waveshaper_output: list[float]


@dataclass
class PerformanceResult:
    """성능 측정 결과"""
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
    """주파수 응답 측정 (magnitude + phase)

    method: "impulse" (delta) or "noise" (white noise averaged)
    """
    wrapper = _load_plugin(plugin_path, params)

    if method == "impulse":
        test = generate_impulse(sample_rate, duration_sec=fft_size / sample_rate + 0.1,
                                level_db=level_db)
    else:
        test = generate_white_noise(sample_rate, duration_sec=2.0, level_db=level_db)

    output = wrapper.process(test, sample_rate)

    # 모노 변환
    mono = output[0] if output.ndim == 2 else output

    # FFT
    window = np.hanning(fft_size).astype(np.float32)
    frame = mono[:fft_size] * window
    spectrum = np.fft.rfft(frame)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum, deg=True)

    # dB 변환
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
    """THD + THD+N 측정"""
    wrapper = _load_plugin(plugin_path, params)

    duration = fft_size / sample_rate + 0.5
    test = generate_sine(frequency, sample_rate, duration, level_db)
    output = wrapper.process(test, sample_rate)

    mono = output[0] if output.ndim == 2 else output
    # 안정 구간 사용 (처음 0.1초 제외)
    skip = int(0.1 * sample_rate)
    frame = mono[skip:skip + fft_size]
    if len(frame) < fft_size:
        frame = np.pad(frame, (0, fft_size - len(frame)))

    window = np.hanning(fft_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frame * window))
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    # 기본 주파수 피크
    fund_bin = int(round(frequency * fft_size / sample_rate))
    search_range = max(3, fund_bin // 20)
    fund_region = spectrum[max(0, fund_bin - search_range):fund_bin + search_range]
    fund_peak = np.max(fund_region)
    fund_db = 20 * np.log10(fund_peak + 1e-10)

    # 하모닉 피크 찾기
    harmonics = []
    harmonic_energy = 0.0
    max_harmonic = min(16, int(sample_rate / 2 / frequency))

    for h in range(2, max_harmonic + 1):
        h_bin = int(round(h * frequency * fft_size / sample_rate))
        if h_bin >= len(spectrum):
            break
        sr = max(3, h_bin // 50)
        region = spectrum[max(0, h_bin - sr):min(len(spectrum), h_bin + sr)]
        if len(region) == 0:
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
    """IMD (상호변조 왜곡) 측정"""
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

    # 7kHz 피크
    high_bin = int(round(freq_high * fft_size / sample_rate))
    sr = max(3, high_bin // 50)
    high_peak = np.max(spectrum[max(0, high_bin - sr):high_bin + sr])

    # IMD 사이드밴드: 7000 ± N*60 Hz
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
            if len(region) == 0:
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
    """주파수 스윕 → THD vs freq + 2D spectrogram"""
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

    # 스윕 시 각 시점의 기본 주파수 계산
    sweep_freqs = []
    thd_per_freq = []
    gain_per_freq = []

    for i in range(n_frames):
        t = time_axis[i]
        # 현재 스윕 주파수 (exponential)
        ratio = t / duration_sec
        current_freq = freq_start * (freq_end / freq_start) ** ratio

        if current_freq > sample_rate / 4:  # Nyquist/2 이상은 THD 의미 없음
            break

        sweep_freqs.append(round(current_freq, 1))

        # 기본 주파수 피크
        fund_bin = int(round(current_freq * fft_size / sample_rate))
        if fund_bin >= len(freqs) or fund_bin < 1:
            thd_per_freq.append(0.0)
            gain_per_freq.append(0.0)
            continue

        spec = 10 ** (spectrogram[i] / 20)  # linear
        sr = max(2, fund_bin // 20)
        fund_peak = np.max(spec[max(0, fund_bin - sr):min(len(spec), fund_bin + sr)])

        # 하모닉 에너지
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
    """입력 레벨별 출력 레벨 측정 (컴프레서 I/O 곡선)"""
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
    """Attack/Release 응답 측정"""
    wrapper = _load_plugin(plugin_path, params)

    test = generate_dynamics_attack_release(
        frequency, sample_rate, level_below_db, level_above_db,
    )
    output = wrapper.process(test, sample_rate)

    # RMS envelope 추출
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


def measure_waveshaper(
    plugin_path: str,
    params: Optional[dict] = None,
    frequency: float = 100.0,
    level_db: float = 0.0,
    sample_rate: int = 44100,
) -> OscilloscopeResult:
    """입력→출력 웨이브셰이퍼 곡선 추출"""
    wrapper = _load_plugin(plugin_path, params)

    test = generate_sine(frequency, sample_rate, 0.5, level_db)
    output = wrapper.process(test, sample_rate)

    in_mono = test[0]
    out_mono = output[0] if output.ndim == 2 else output

    # 안정 구간 1주기 추출
    period = int(sample_rate / frequency)
    skip = int(0.1 * sample_rate)
    in_cycle = in_mono[skip:skip + period]
    out_cycle = out_mono[skip:skip + period]

    # 정렬: 입력값 기준 정렬 → waveshaper 곡선
    sort_idx = np.argsort(in_cycle)
    ws_input = in_cycle[sort_idx].tolist()
    ws_output = out_cycle[sort_idx].tolist()

    return OscilloscopeResult(
        input_signal=in_cycle,
        output_signal=out_cycle,
        waveshaper_input=ws_input,
        waveshaper_output=ws_output,
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
    """프로세싱 콜백 시간 측정"""
    wrapper = _load_plugin(plugin_path, params)

    if buffer_sizes is None:
        buffer_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    process_times = []
    sps_list = []
    rt_ratios = []

    for bs in buffer_sizes:
        test = np.random.randn(2, bs).astype(np.float32) * 0.1
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
# 통합: 2 플러그인 비교
# =============================================================================


def compare_linear(
    plugin_path_1: str,
    plugin_path_2: str,
    params_1: Optional[dict] = None,
    params_2: Optional[dict] = None,
    sample_rate: int = 44100,
) -> dict:
    """2 플러그인 주파수 응답 비교 (차이)"""
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
