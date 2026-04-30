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
    generate_log_sweep_deconv, generate_multitone,
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


@dataclass
class WaveshaperV2Result:
    """다중 진폭 웨이브셰이퍼 측정 결과 (v2)"""
    input_values: np.ndarray       # (n_points,) 균등 분포 [-1, +1]
    output_values: np.ndarray      # (n_points,) 매핑된 출력
    n_points: int
    levels_db: list[float]
    input_coverage: float          # 0~1 (입력이 [-1,+1] 중 얼마를 커버하는지)
    is_symmetric: bool             # f(-x) ≈ -f(x) 대칭성
    raw_pairs: Optional[list[tuple[np.ndarray, np.ndarray]]] = None


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
    """다중 진폭 레벨 웨이브셰이퍼 곡선 추출 (v2)

    기존 measure_waveshaper()의 한계 극복:
    - 단일 레벨 → 다중 레벨 (기본 7단계)로 입력 범위 전체 커버
    - 1주기 → 복수 주기 평균으로 노이즈/과도 응답 영향 감소
    - 가변 포인트 수 → 256포인트 균등 리샘플링

    Args:
        plugin_path: VST3 경로
        params: 플러그인 파라미터
        frequency: 테스트 사인파 주파수 (Hz)
        sample_rate: 샘플레이트
        levels_db: 측정할 dBFS 레벨 목록 (기본: [-24, -18, -12, -6, -3, -1, 0])
        n_cycles: 평균할 주기 수 (기본 3)
        n_points: 최종 리샘플링 포인트 수 (기본 256)
        preroll_sec: silence 프리롤 길이 (초) — 레이턴시 보상

    Returns:
        WaveshaperV2Result
    """
    if levels_db is None:
        levels_db = [-24.0, -18.0, -12.0, -6.0, -3.0, -1.0, 0.0]

    wrapper = _load_plugin(plugin_path, params)
    period_samples = int(sample_rate / frequency)

    # 각 레벨별로 입력→출력 쌍 수집
    all_inputs = []
    all_outputs = []
    raw_pairs = []

    for level_db in levels_db:
        # 플러그인 상태 리셋 (레벨 간 독립 측정)
        wrapper.reset()

        # 필요한 구간: 프리롤 + 안정화(1주기 스킵) + 측정(n_cycles 주기)
        # 안정 구간 확보를 위해 충분한 길이의 신호 생성
        settle_cycles = 1  # 과도 응답 회피용 스킵 주기
        total_cycles_needed = settle_cycles + n_cycles
        test_duration = preroll_sec + (total_cycles_needed + 2) * (1.0 / frequency)

        # 사인파 생성 (프리롤 silence 포함)
        preroll_samples = int(preroll_sec * sample_rate)
        sine_duration = test_duration - preroll_sec
        sine_signal = generate_sine(frequency, sample_rate, sine_duration, level_db)

        # silence 프리롤 + 사인파 결합
        silence = np.zeros((sine_signal.shape[0], preroll_samples), dtype=np.float32)
        test_signal = np.concatenate([silence, sine_signal], axis=1)

        # 플러그인 처리
        output = wrapper.process(test_signal, sample_rate)

        # 모노 추출
        in_mono = test_signal[0]
        out_mono = output[0] if output.ndim == 2 else output

        # 안정 구간 시작점: 프리롤 + settle_cycles 주기 이후
        stable_start = preroll_samples + settle_cycles * period_samples

        # n_cycles 주기 추출 후 주기별 평균
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
            logger.warning(f"레벨 {level_db}dB: 충분한 주기를 추출할 수 없음, 건너뜀")
            continue

        # 주기별 평균 (과도 응답, 노이즈 감소)
        avg_in = np.mean(level_in_cycles, axis=0)
        avg_out = np.mean(level_out_cycles, axis=0)

        raw_pairs.append((avg_in.copy(), avg_out.copy()))

        # 입력값 기준 정렬
        sort_idx = np.argsort(avg_in)
        all_inputs.append(avg_in[sort_idx])
        all_outputs.append(avg_out[sort_idx])

    if not all_inputs:
        raise RuntimeError("모든 레벨에서 측정 실패 — 충분한 데이터를 추출할 수 없음")

    # 모든 레벨의 데이터를 합쳐서 입력값 기준 정렬
    combined_in = np.concatenate(all_inputs)
    combined_out = np.concatenate(all_outputs)
    global_sort = np.argsort(combined_in)
    combined_in = combined_in[global_sort]
    combined_out = combined_out[global_sort]

    # 균등 분포 n_points로 리샘플링
    x_uniform = np.linspace(-1.0, 1.0, n_points)

    # 실제 데이터 범위 내에서만 보간 (범위 밖은 외삽 방지)
    in_min, in_max = combined_in[0], combined_in[-1]
    output_values = np.interp(x_uniform, combined_in, combined_out)

    # 커버리지: 입력이 [-1, +1] 중 얼마를 커버하는지
    input_coverage = float((in_max - in_min) / 2.0)  # 전체 범위 2.0 대비

    # 대칭성 검증: f(-x) ≈ -f(x) 이면 홀수 하모닉 대칭
    # 중심(0)을 기준으로 양쪽 비교
    n_half = n_points // 2
    f_neg_x = output_values[:n_half][::-1]   # f(-x) reversed
    neg_f_x = -output_values[n_points - n_half:]  # -f(x)

    # 대칭 오차 계산 (정규화)
    max_output = np.max(np.abs(output_values)) + 1e-10
    symmetry_error = np.mean(np.abs(f_neg_x - neg_f_x)) / max_output
    is_symmetric = bool(symmetry_error < 0.05)  # 5% 이하면 대칭으로 판단

    logger.info(
        f"Waveshaper v2: {len(levels_db)}레벨, 커버리지={input_coverage:.1%}, "
        f"대칭={is_symmetric} (오차={symmetry_error:.4f})"
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


# =============================================================================
# 7. CLAP 임베딩 프로파일링
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
    """파라미터 스윕별 CLAP 임베딩 생성 — 새추레이션 "지문"

    Args:
        plugin_path: VST3 경로
        param_sweeps: {"drive": [0, 25, 50, 75, 100], "style": ["Soft", "Hard"]}
        base_params: 기본 파라미터 {"mix": 100.0, ...}

    Returns: {
        "embeddings": [(param_values, embedding_512d), ...],
        "labels": ["drive=0", "drive=25", ...],
        "embeddings_npy": np.ndarray (N, 512),
    }
    """
    try:
        import laion_clap
    except ImportError:
        raise ImportError("CLAP 필요: pip install laion-clap")

    import soundfile as sf
    import tempfile
    import os

    wrapper = _load_plugin(plugin_path, base_params)

    # 테스트 신호
    n_samples = int(duration_sec * sample_rate)
    test = generate_sine(test_frequency, sample_rate, duration_sec, test_level_db)

    # 모든 파라미터 조합 생성
    import itertools
    param_names = list(param_sweeps.keys())
    param_values_list = list(param_sweeps.values())
    combinations = list(itertools.product(*param_values_list))

    # 플러그인 한 번만 로딩, 파라미터만 교체
    from pedalboard import load_plugin as pb_load
    plugin = pb_load(plugin_path)
    if base_params:
        for k, v in base_params.items():
            try:
                setattr(plugin, k, v)
            except Exception:
                pass

    tmpdir = tempfile.mkdtemp()
    wav_paths = []
    labels = []
    param_records = []

    for combo in combinations:
        # 파라미터 적용 (같은 인스턴스 재사용)
        param_dict = dict(zip(param_names, combo))
        for k, v in param_dict.items():
            try:
                setattr(plugin, k, v)
            except Exception:
                try:
                    setattr(plugin, k.replace(' ', '_'), v)
                except Exception:
                    pass

        output = plugin.process(test, sample_rate)

        # WAV 저장
        label = ", ".join(f"{k}={v}" for k, v in param_dict.items())
        labels.append(label)
        param_records.append(param_dict)

        wav_path = os.path.join(tmpdir, f"{len(wav_paths):04d}.wav")
        if output.ndim == 2:
            sf.write(wav_path, output.T, sample_rate, subtype='FLOAT')
        else:
            sf.write(wav_path, output, sample_rate, subtype='FLOAT')
        wav_paths.append(wav_path)

    # CLAP 인코딩
    logger.info(f"CLAP 인코딩: {len(wav_paths)}개 설정")
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()

    batch_size = 32
    all_emb = []
    for i in range(0, len(wav_paths), batch_size):
        batch = wav_paths[i:i + batch_size]
        emb = model.get_audio_embedding_from_filelist(batch, use_tensor=False)
        all_emb.append(emb)

    embeddings = np.concatenate(all_emb, axis=0)

    # 정리
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
# 8. EQ 프로파일링
# =============================================================================


@dataclass
class EQResponseResult:
    """EQ 주파수/위상 응답 측정 결과"""
    frequencies: list[float]       # Hz
    magnitude_db: list[float]      # dB (bypass 대비 상대값)
    phase_deg: list[float]         # degrees
    group_delay_ms: list[float]    # ms
    params: dict                   # 측정 시 파라미터
    sample_rate: int
    fft_size: int
    is_minimum_phase: bool
    thd_at_1k: float               # 비선형성 지표 (%)


def _deconvolve(output: np.ndarray, inverse_filter: np.ndarray) -> np.ndarray:
    """스윕 출력에 역필터 적용하여 임펄스 응답 추출 (FFT 컨볼루션)"""
    n = len(output) + len(inverse_filter) - 1
    # 2의 거듭제곱으로 패딩 (FFT 효율)
    n_fft = 1
    while n_fft < n:
        n_fft *= 2

    O = np.fft.rfft(output, n=n_fft)
    I = np.fft.rfft(inverse_filter, n=n_fft)
    ir = np.fft.irfft(O * I, n=n_fft)
    return ir.astype(np.float32)


def _check_minimum_phase(magnitude_db: np.ndarray, phase_deg: np.ndarray) -> bool:
    """Hilbert 변환으로 최소위상 여부 판별

    최소위상 시스템: phase = -Hilbert(ln|H(f)|)
    측정 위상과 Hilbert 유도 위상의 차이가 작으면 최소위상.
    """
    # log magnitude → Hilbert transform → minimum phase
    log_mag = np.log(10 ** (magnitude_db / 20.0) + 1e-10)
    # Hilbert 변환 (이산)
    n = len(log_mag)
    if n < 4:
        return True

    spectrum = np.fft.rfft(log_mag)
    # 최소위상 계산: imag(Hilbert(log|H|))
    min_phase_rad = -np.imag(np.fft.irfft(
        1j * np.sign(np.fft.rfftfreq(2 * n - 1, 1.0)) * np.fft.rfft(log_mag, n=2 * n - 1),
        n=2 * n - 1,
    ))[:n]
    min_phase_deg = np.degrees(min_phase_rad)

    # 측정 위상과 비교 (DC, Nyquist 근처 제외)
    trim = max(1, n // 20)
    measured = np.array(phase_deg[trim:-trim])
    expected = min_phase_deg[trim:-trim]

    if len(measured) == 0:
        return True

    error = np.mean(np.abs(measured - expected))
    return bool(error < 15.0)  # 15도 이내면 최소위상


def measure_eq_response(
    plugin_path: str,
    params: Optional[dict] = None,
    bypass_params: Optional[dict] = None,
    sample_rate: int = 44100,
    fft_size: int = 32768,
    sweep_duration: float = 6.0,
    level_db: float = -12.0,
) -> EQResponseResult:
    """EQ 주파수/위상/그룹딜레이 측정 — 로그 스윕 디컨볼루션 방식

    1. bypass 상태로 스윕 → 레퍼런스 IR 추출
    2. 타겟 파라미터로 스윕 → 타겟 IR 추출
    3. 주파수 도메인에서 차이 계산 → bypass 대비 상대 응답

    Args:
        plugin_path: VST3 경로
        params: 측정할 EQ 파라미터
        bypass_params: bypass 상태 파라미터 (None이면 파라미터 없이 로드)
        sample_rate: 샘플레이트
        fft_size: FFT 크기 (저주파 해상도용, 기본 32768)
        sweep_duration: 스윕 길이 (초)
        level_db: 입력 레벨 (dBFS)

    Returns:
        EQResponseResult
    """
    sweep_audio, inverse_filter = generate_log_sweep_deconv(
        sample_rate=sample_rate,
        duration_sec=sweep_duration,
        level_db=level_db,
    )
    inv_mono = inverse_filter[0]

    # 1) Bypass 측정 (레퍼런스)
    wrapper_bypass = _load_plugin(plugin_path, bypass_params)
    bypass_output = wrapper_bypass.process(sweep_audio, sample_rate)
    bypass_mono = bypass_output[0] if bypass_output.ndim == 2 else bypass_output
    bypass_ir = _deconvolve(bypass_mono, inv_mono)

    # 2) 타겟 파라미터 측정
    wrapper_target = _load_plugin(plugin_path, params)
    target_output = wrapper_target.process(sweep_audio, sample_rate)
    target_mono = target_output[0] if target_output.ndim == 2 else target_output
    target_ir = _deconvolve(target_mono, inv_mono)

    # 3) FFT — bypass 대비 상대 응답
    window = np.hanning(fft_size).astype(np.float32)

    # IR의 피크 위치 찾기 (디컨볼루션 결과에서 선형 응답이 집중되는 지점)
    bypass_peak = int(np.argmax(np.abs(bypass_ir)))
    target_peak = int(np.argmax(np.abs(target_ir)))

    # 피크 중심으로 fft_size 윈도우 추출
    def _extract_ir_window(ir, peak_idx):
        half = fft_size // 2
        start = max(0, peak_idx - half // 4)  # 피크 약간 앞부터
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

    # 상대 응답: H_eq = H_target / H_bypass
    bypass_mag = np.abs(bypass_spectrum) + 1e-10
    target_mag = np.abs(target_spectrum)
    relative_mag = target_mag / bypass_mag
    magnitude_db = (20 * np.log10(relative_mag + 1e-10)).tolist()

    # 위상 (상대)
    bypass_phase = np.angle(bypass_spectrum)
    target_phase = np.angle(target_spectrum)
    relative_phase = np.degrees(target_phase - bypass_phase)
    # unwrap
    relative_phase_unwrapped = np.unwrap(np.radians(relative_phase))
    phase_deg = np.degrees(relative_phase_unwrapped).tolist()

    # 그룹 딜레이: -d(phase)/d(omega)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    d_phase = np.gradient(relative_phase_unwrapped, 2 * np.pi * df)
    group_delay_ms = (-d_phase * 1000).tolist()

    # 최소위상 판별
    is_min_phase = _check_minimum_phase(
        np.array(magnitude_db), np.array(phase_deg),
    )

    # THD @ 1kHz (비선형성 지표)
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
    """EQ 파라미터 조합별 일괄 주파수 응답 측정

    Args:
        plugin_path: VST3 경로
        sweep_config: 스윕 설정 dict
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
        bypass_params: bypass 상태 파라미터
        sample_rate: 샘플레이트
        fft_size: FFT 크기
        level_db: 입력 레벨

    Returns:
        list[EQResponseResult] — 각 파라미터 조합에 대한 측정 결과
    """
    results = []

    for sweep_name, config in sweep_config.items():
        param_name = config["param"]
        values = config["values"]
        fixed = config.get("fixed", {})

        logger.info(f"EQ sweep '{sweep_name}': {param_name} = {values}")

        for value in values:
            # 고정 파라미터 + 스윕 파라미터 결합
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
                logger.warning(f"  {param_name}={value}: 측정 실패 — {e}")

    return results


def measure_eq_nonlinearity(
    plugin_path: str,
    params: Optional[dict] = None,
    bypass_params: Optional[dict] = None,
    levels_db: Optional[list[float]] = None,
    sample_rate: int = 44100,
    fft_size: int = 32768,
) -> list[EQResponseResult]:
    """EQ 비선형성(레벨 의존성) 측정

    동일한 EQ 설정을 다른 입력 레벨에서 측정.
    아날로그 모델링 EQ는 레벨에 따라 응답이 변함 (saturation).

    Args:
        plugin_path: VST3 경로
        params: EQ 파라미터
        bypass_params: bypass 상태 파라미터
        levels_db: 측정할 입력 레벨 목록 (dBFS)
        sample_rate: 샘플레이트
        fft_size: FFT 크기

    Returns:
        list[EQResponseResult] — 레벨별 응답 결과
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
            # 레벨 정보를 params에 추가
            result.params = dict(result.params)
            result.params["_input_level_db"] = level
            results.append(result)
            logger.info(f"  level={level}dB: thd={result.thd_at_1k:.4f}%")
        except Exception as e:
            logger.warning(f"  level={level}dB: 측정 실패 — {e}")

    if len(results) >= 2:
        # 레벨 간 응답 차이 확인
        ref = np.array(results[0].magnitude_db)
        max_deviation = 0.0
        for r in results[1:]:
            diff = np.max(np.abs(np.array(r.magnitude_db) - ref))
            max_deviation = max(max_deviation, diff)
        logger.info(f"  레벨 간 최대 응답 편차: {max_deviation:.2f} dB "
                     f"({'비선형' if max_deviation > 0.5 else '선형'})")

    return results
