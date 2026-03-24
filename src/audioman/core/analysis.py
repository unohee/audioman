# Created: 2026-03-21
# Purpose: 오디오 분석 함수 (스펙트럼, silence 감지 등)

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FrameMetrics:
    """프레임 단위 분석 결과"""
    rms: list[float] = field(default_factory=list)
    peak: list[float] = field(default_factory=list)
    spectral_centroid: list[float] = field(default_factory=list)
    spectral_entropy: list[float] = field(default_factory=list)
    zero_crossing_rate: list[float] = field(default_factory=list)


@dataclass
class SilenceRegion:
    start_sample: int
    end_sample: int
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "duration_sec": round(self.duration_sec, 6),
        }


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """(channels, samples) → mono 1D"""
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def compute_frame_metrics(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> FrameMetrics:
    """프레임 단위 오디오 메트릭 계산"""
    mono = _to_mono(audio)
    n_samples = len(mono)
    metrics = FrameMetrics()

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = mono[start : start + frame_size]

        # RMS
        rms = float(np.sqrt(np.mean(frame**2)))
        metrics.rms.append(rms)

        # Peak
        metrics.peak.append(float(np.max(np.abs(frame))))

        # Zero crossing rate
        zcr = float(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_size))
        metrics.zero_crossing_rate.append(zcr)

        # Spectrum
        window = np.hanning(frame_size)
        spectrum = np.abs(np.fft.rfft(frame * window))
        spectrum_sum = np.sum(spectrum)

        # Spectral centroid
        if spectrum_sum > 1e-10:
            centroid = float(np.sum(freqs * spectrum) / spectrum_sum)
        else:
            centroid = 0.0
        metrics.spectral_centroid.append(centroid)

        # Spectral entropy
        if spectrum_sum > 1e-10:
            prob = spectrum / spectrum_sum
            prob = prob[prob > 1e-10]
            entropy = float(-np.sum(prob * np.log2(prob)))
        else:
            entropy = 0.0
        metrics.spectral_entropy.append(entropy)

    return metrics


def compute_summary(metrics: FrameMetrics) -> dict:
    """프레임 메트릭의 요약 통계 (mean/min/max/std)"""
    summary = {}
    for name in ["rms", "peak", "spectral_centroid", "spectral_entropy", "zero_crossing_rate"]:
        values = np.array(getattr(metrics, name))
        if len(values) == 0:
            summary[name] = {"mean": 0, "min": 0, "max": 0, "std": 0}
        else:
            summary[name] = {
                "mean": round(float(np.mean(values)), 6),
                "min": round(float(np.min(values)), 6),
                "max": round(float(np.max(values)), 6),
                "std": round(float(np.std(values)), 6),
            }
    return summary


def detect_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_duration_sec: float = 0.1,
    frame_size: int = 1024,
    hop_size: int = 512,
) -> list[SilenceRegion]:
    """RMS 임계값 이하 구간을 silence로 감지"""
    mono = _to_mono(audio)
    threshold_linear = 10 ** (threshold_db / 20.0)
    n_samples = len(mono)

    regions = []
    in_silence = False
    silence_start = 0

    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = mono[start : start + frame_size]
        rms = np.sqrt(np.mean(frame**2))

        if rms < threshold_linear:
            if not in_silence:
                silence_start = start
                in_silence = True
        else:
            if in_silence:
                duration = (start - silence_start) / sample_rate
                if duration >= min_duration_sec:
                    regions.append(SilenceRegion(
                        start_sample=silence_start,
                        end_sample=start,
                        duration_sec=duration,
                    ))
                in_silence = False

    # 파일 끝까지 silence인 경우
    if in_silence:
        duration = (n_samples - silence_start) / sample_rate
        if duration >= min_duration_sec:
            regions.append(SilenceRegion(
                start_sample=silence_start,
                end_sample=n_samples,
                duration_sec=duration,
            ))

    return regions


def compute_audio_features(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> dict:
    """Oneshot 오디오 특성 추출 — 데이터셋 임베딩용

    Returns: 고정 키 딕셔너리 (모든 값 float)
        - rms_mean, rms_max, peak
        - spectral_centroid_mean/std (밝기)
        - spectral_bandwidth_mean (넓이)
        - spectral_rolloff (에너지 집중도)
        - spectral_flatness_mean (노이즈/토널 비율)
        - zero_crossing_rate_mean
        - attack_time (초, 10%→90% rise)
        - release_time (초, 90%→10% fall)
        - spectral_entropy_mean (복잡도)
        - fundamental_freq (Hz, 기본 주파수 추정)
        - harmonic_ratio (하모닉/전체 비율)
        - crest_factor (peak/rms)
        - duration_effective (무음 제외 실효 길이)
    """
    mono = _to_mono(audio)
    n_samples = len(mono)
    duration = n_samples / sample_rate

    # 기본 통계
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    crest_factor = peak / rms if rms > 1e-10 else 0.0

    # 프레임 기반 분석
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    window = np.hanning(frame_size).astype(np.float32)

    centroids = []
    bandwidths = []
    rolloffs = []
    flatnesses = []
    entropies = []
    zcrs = []
    rms_frames = []

    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = mono[start : start + frame_size]

        # RMS
        frame_rms = float(np.sqrt(np.mean(frame**2)))
        rms_frames.append(frame_rms)

        # ZCR
        zcr = float(np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_size))
        zcrs.append(zcr)

        # Spectrum
        spectrum = np.abs(np.fft.rfft(frame * window))
        spectrum_sum = np.sum(spectrum)
        power = spectrum**2
        power_sum = np.sum(power)

        if spectrum_sum > 1e-10:
            # Spectral centroid (Hz)
            centroid = float(np.sum(freqs * spectrum) / spectrum_sum)
            centroids.append(centroid)

            # Spectral bandwidth (Hz)
            bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / spectrum_sum))
            bandwidths.append(bw)

            # Spectral rolloff (85% 에너지 주파수)
            cumsum = np.cumsum(power)
            rolloff_idx = np.searchsorted(cumsum, 0.85 * power_sum)
            rolloff_idx = min(rolloff_idx, len(freqs) - 1)
            rolloffs.append(float(freqs[rolloff_idx]))

            # Spectral flatness (기하평균/산술평균, 0=tonal, 1=noise)
            log_spectrum = np.log(spectrum + 1e-10)
            geo_mean = np.exp(np.mean(log_spectrum))
            arith_mean = np.mean(spectrum)
            flatness = float(geo_mean / arith_mean) if arith_mean > 1e-10 else 0.0
            flatnesses.append(flatness)

            # Spectral entropy
            prob = spectrum / spectrum_sum
            prob = prob[prob > 1e-10]
            entropy = float(-np.sum(prob * np.log2(prob)))
            entropies.append(entropy)
        else:
            centroids.append(0.0)
            bandwidths.append(0.0)
            rolloffs.append(0.0)
            flatnesses.append(0.0)
            entropies.append(0.0)

    # Attack/Release 시간 추정 (RMS envelope 기반)
    rms_arr = np.array(rms_frames) if rms_frames else np.array([0.0])
    rms_max = np.max(rms_arr) if len(rms_arr) > 0 else 0.0

    attack_time = 0.0
    release_time = 0.0

    if rms_max > 1e-10:
        thresh_10 = rms_max * 0.1
        thresh_90 = rms_max * 0.9

        # Attack: 첫 10% → 90%
        above_10 = np.where(rms_arr >= thresh_10)[0]
        above_90 = np.where(rms_arr >= thresh_90)[0]
        if len(above_10) > 0 and len(above_90) > 0:
            attack_frames = above_90[0] - above_10[0]
            attack_time = max(0, float(attack_frames * hop_size / sample_rate))

        # Release: 마지막 90% → 10%
        last_above_90 = np.where(rms_arr >= thresh_90)[0]
        last_below_10 = np.where(rms_arr[last_above_90[-1]:] < thresh_10)[0] if len(last_above_90) > 0 else np.array([])
        if len(last_above_90) > 0 and len(last_below_10) > 0:
            release_frames = last_below_10[0]
            release_time = float(release_frames * hop_size / sample_rate)

    # Effective duration (무음 제외)
    if len(rms_arr) > 0 and rms_max > 1e-10:
        active_frames = np.sum(rms_arr > rms_max * 0.01)
        duration_effective = float(active_frames * hop_size / sample_rate)
    else:
        duration_effective = 0.0

    # 기본 주파수 추정 (autocorrelation)
    fundamental_freq = _estimate_f0(mono, sample_rate)

    # Harmonic ratio 추정
    harmonic_ratio = _estimate_harmonic_ratio(mono, sample_rate, frame_size)

    result = {
        "peak": round(peak, 6),
        "rms_mean": round(rms, 6),
        "rms_max": round(float(rms_max), 6),
        "crest_factor": round(crest_factor, 4),
        "spectral_centroid_mean": round(float(np.mean(centroids)), 2) if centroids else 0.0,
        "spectral_centroid_std": round(float(np.std(centroids)), 2) if centroids else 0.0,
        "spectral_bandwidth_mean": round(float(np.mean(bandwidths)), 2) if bandwidths else 0.0,
        "spectral_rolloff_mean": round(float(np.mean(rolloffs)), 2) if rolloffs else 0.0,
        "spectral_flatness_mean": round(float(np.mean(flatnesses)), 6) if flatnesses else 0.0,
        "spectral_entropy_mean": round(float(np.mean(entropies)), 4) if entropies else 0.0,
        "zero_crossing_rate_mean": round(float(np.mean(zcrs)), 6) if zcrs else 0.0,
        "attack_time": round(attack_time, 4),
        "release_time": round(release_time, 4),
        "fundamental_freq": round(fundamental_freq, 2),
        "harmonic_ratio": round(harmonic_ratio, 4),
        "duration_effective": round(duration_effective, 4),
        "duration_total": round(duration, 4),
    }

    # Envelope curvature 분석
    env_curve = compute_envelope_curvature(audio, sample_rate, hop_size)
    result.update(env_curve)

    # === 시간축 스펙트럴 변화 분석 ===
    temporal = _compute_temporal_spectral_features(
        centroids, bandwidths, rolloffs, flatnesses, rms_frames, hop_size, sample_rate
    )
    result.update(temporal)

    # === 하모닉 시리즈 분석 ===
    harmonic = _analyze_harmonic_series(mono, sample_rate, fundamental_freq, frame_size)
    result.update(harmonic)

    # === Peak frequency trajectory (filter sweep 감지) ===
    peak_traj = _compute_peak_freq_trajectory(mono, sample_rate, frame_size, hop_size, rms_frames)
    result.update(peak_traj)

    return result


def _compute_temporal_spectral_features(
    centroids: list, bandwidths: list, rolloffs: list,
    flatnesses: list, rms_frames: list,
    hop_size: int, sample_rate: int,
) -> dict:
    """시간축 스펙트럴 변화량 — filter envelope, pitch envelope 추정

    핵심 아이디어:
    - centroid가 시간에 따라 올라가면 → filter envelope opening
    - centroid가 내려가면 → filter envelope closing
    - bandwidth 변화 → 음역대 넓어짐/좁아짐
    """
    if len(centroids) < 4:
        return {
            "centroid_attack_slope": 0.0, "centroid_decay_slope": 0.0,
            "centroid_range": 0.0, "bandwidth_attack_slope": 0.0,
            "bandwidth_range": 0.0, "brightness_trajectory": 0.0,
            "spectral_flux_mean": 0.0, "spectral_flux_std": 0.0,
        }

    c = np.array(centroids, dtype=np.float32)
    bw = np.array(bandwidths, dtype=np.float32)
    ro = np.array(rolloffs, dtype=np.float32)
    fl = np.array(flatnesses, dtype=np.float32)
    rms = np.array(rms_frames, dtype=np.float32)
    n = len(c)

    # RMS 피크 위치 (attack 끝 = sustain 시작)
    rms_max = np.max(rms)
    if rms_max < 1e-10:
        peak_idx = n // 4
    else:
        peak_idx = int(np.argmax(rms))

    # 시간 → 초 변환
    frame_to_sec = hop_size / sample_rate

    # --- Centroid 시간 변화 (filter envelope 지표) ---
    # Attack phase: 시작 → 피크 구간의 centroid 변화율
    atk_end = max(2, peak_idx)
    if atk_end > 1:
        # 선형 기울기 (Hz/s)
        t_atk = np.arange(atk_end) * frame_to_sec
        if len(t_atk) > 1:
            slope_atk = float(np.polyfit(t_atk, c[:atk_end], 1)[0])
        else:
            slope_atk = 0.0
    else:
        slope_atk = 0.0

    # Decay phase: 피크 → 끝 구간
    decay_start = peak_idx
    decay_len = n - decay_start
    if decay_len > 2:
        t_dec = np.arange(decay_len) * frame_to_sec
        slope_dec = float(np.polyfit(t_dec, c[decay_start:], 1)[0])
    else:
        slope_dec = 0.0

    # Centroid 전체 범위
    c_active = c[rms > rms_max * 0.05]  # 유효 구간만
    centroid_range = float(np.max(c_active) - np.min(c_active)) if len(c_active) > 0 else 0.0

    # --- Bandwidth 시간 변화 (음역대 넓어짐/좁아짐) ---
    bw_active = bw[rms > rms_max * 0.05] if len(bw) > 0 else bw
    if len(bw_active) > 2 and atk_end > 1:
        t_bw = np.arange(min(atk_end, len(bw_active))) * frame_to_sec
        bw_slope = float(np.polyfit(t_bw, bw_active[:len(t_bw)], 1)[0])
    else:
        bw_slope = 0.0
    bw_range = float(np.max(bw_active) - np.min(bw_active)) if len(bw_active) > 0 else 0.0

    # --- Brightness trajectory (전체 궤적 방향) ---
    # 양수 = 점점 밝아짐, 음수 = 점점 어두워짐
    if len(c_active) > 2:
        t_all = np.arange(len(c_active)) * frame_to_sec
        brightness_traj = float(np.polyfit(t_all, c_active, 1)[0])
    else:
        brightness_traj = 0.0

    # --- Spectral flux (프레임 간 스펙트럴 변화량) ---
    if len(c) > 1:
        flux = np.abs(np.diff(c))
        spectral_flux_mean = float(np.mean(flux))
        spectral_flux_std = float(np.std(flux))
    else:
        spectral_flux_mean = spectral_flux_std = 0.0

    return {
        "centroid_attack_slope": round(slope_atk, 2),      # Hz/s, + = filter opening
        "centroid_decay_slope": round(slope_dec, 2),        # Hz/s, - = filter closing
        "centroid_range": round(centroid_range, 2),          # Hz, 전체 변화 폭
        "bandwidth_attack_slope": round(bw_slope, 2),       # Hz/s, 음역대 변화
        "bandwidth_range": round(bw_range, 2),               # Hz
        "brightness_trajectory": round(brightness_traj, 2),  # Hz/s, 전체 밝기 방향
        "spectral_flux_mean": round(spectral_flux_mean, 2),  # 프레임 간 변화량 평균
        "spectral_flux_std": round(spectral_flux_std, 2),    # 변화량 표준편차
    }


def _analyze_harmonic_series(
    mono: np.ndarray,
    sample_rate: int,
    f0: float,
    frame_size: int = 4096,
) -> dict:
    """F0 대비 하모닉 시리즈 분석

    - 하모닉 에너지 분포 (어디에 에너지가 집중?)
    - 하모닉 간격 규칙성 (정수배 vs 비정수배)
    - 홀수/짝수 하모닉 비율 (파형 대칭성)
    """
    if f0 < 20 or len(mono) < frame_size:
        return {
            "harmonic_centroid_ratio": 0.0,
            "harmonic_spread": 0.0,
            "odd_even_ratio": 0.5,
            "harmonic_decay_rate": 0.0,
            "inharmonicity": 0.0,
        }

    # 안정 구간 (중간 1/3)
    n = len(mono)
    segment = mono[n // 3: 2 * n // 3]
    if len(segment) < frame_size:
        segment = mono[:frame_size]

    window = np.hanning(frame_size).astype(np.float32)
    frame = segment[:frame_size] * window
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)

    # 하모닉 피크 찾기 (f0의 정수배 ±5% 범위)
    max_harmonic = min(16, int(sample_rate / 2 / f0))
    harmonic_amps = []
    harmonic_freqs_actual = []

    for h in range(1, max_harmonic + 1):
        target = f0 * h
        tolerance = f0 * 0.05  # ±5%
        mask = (freqs >= target - tolerance) & (freqs <= target + tolerance)
        if np.any(mask):
            peak_idx = np.argmax(spectrum[mask])
            amp = float(spectrum[mask][peak_idx])
            actual_freq = float(freqs[mask][peak_idx])
            harmonic_amps.append(amp)
            harmonic_freqs_actual.append(actual_freq)
        else:
            harmonic_amps.append(0.0)
            harmonic_freqs_actual.append(target)

    if not harmonic_amps or max(harmonic_amps) < 1e-10:
        return {
            "harmonic_centroid_ratio": 0.0,
            "harmonic_spread": 0.0,
            "odd_even_ratio": 0.5,
            "harmonic_decay_rate": 0.0,
            "inharmonicity": 0.0,
        }

    amps = np.array(harmonic_amps, dtype=np.float32)
    amps_norm = amps / np.max(amps)

    # --- 하모닉 무게중심 (몇 번째 하모닉에 에너지 집중?) ---
    h_indices = np.arange(1, len(amps) + 1, dtype=np.float32)
    h_centroid = float(np.sum(h_indices * amps) / np.sum(amps))
    # f0 대비 비율로 정규화
    harmonic_centroid_ratio = h_centroid / max_harmonic

    # --- 하모닉 분산 (에너지가 넓게/좁게 퍼져있나) ---
    harmonic_spread = float(np.sqrt(np.sum(((h_indices - h_centroid) ** 2) * amps) / np.sum(amps)))

    # --- 홀수/짝수 비율 (0=짝수만, 0.5=균등, 1=홀수만) ---
    odd_energy = float(np.sum(amps[0::2]))   # 1st, 3rd, 5th...
    even_energy = float(np.sum(amps[1::2]))  # 2nd, 4th, 6th...
    total = odd_energy + even_energy
    odd_even_ratio = odd_energy / total if total > 1e-10 else 0.5

    # --- 하모닉 감쇠율 (고차 하모닉이 얼마나 빨리 줄어드나) ---
    if len(amps_norm) > 2:
        log_amps = np.log(amps_norm + 1e-10)
        decay_rate = -float(np.polyfit(h_indices[:len(log_amps)], log_amps, 1)[0])
    else:
        decay_rate = 0.0

    # --- 비정수배 정도 (inharmonicity) ---
    # 실제 피크 주파수와 정수배 f0의 편차
    deviations = []
    for h, actual in enumerate(harmonic_freqs_actual, 1):
        ideal = f0 * h
        if ideal > 0:
            dev = abs(actual - ideal) / ideal
            deviations.append(dev)
    inharmonicity = float(np.mean(deviations)) if deviations else 0.0

    return {
        "harmonic_centroid_ratio": round(harmonic_centroid_ratio, 4),  # 에너지 집중 위치 (0~1)
        "harmonic_spread": round(harmonic_spread, 4),                  # 에너지 분산
        "odd_even_ratio": round(odd_even_ratio, 4),                    # 홀수/짝수 (0.5=균등, 1=홀수만=square)
        "harmonic_decay_rate": round(decay_rate, 4),                   # 감쇠율 (높을수록 어두운 톤)
        "inharmonicity": round(inharmonicity, 6),                      # 비정수배 정도 (0=순수, 높을수록 벨/메탈릭)
    }


def compute_envelope_curvature(
    audio: np.ndarray,
    sample_rate: int,
    hop_size: int = 512,
) -> dict:
    """실제 오디오 엔벨로프와 linear ADSR 윈도우를 비교하여 커브 특성 분석

    Returns:
        - attack_curvature: <0 = log/concave, 0 = linear, >0 = exp/convex
        - decay_curvature: 동일
        - release_curvature: 동일
        - attack_r2: linear 피팅 R² (1.0이면 완벽한 직선)
        - decay_r2: 동일
        - release_r2: 동일
        - envelope_rms_error: linear ADSR 대비 전체 오차
    """
    mono = _to_mono(audio)

    # RMS envelope 추출
    n = len(mono)
    rms_env = []
    for start in range(0, n - hop_size, hop_size):
        frame = mono[start : start + hop_size]
        rms_env.append(float(np.sqrt(np.mean(frame**2))))

    rms_env = np.array(rms_env, dtype=np.float32)
    if len(rms_env) < 10 or np.max(rms_env) < 1e-10:
        return {
            "attack_curvature": 0.0, "decay_curvature": 0.0, "release_curvature": 0.0,
            "attack_r2": 0.0, "decay_r2": 0.0, "release_r2": 0.0,
            "envelope_rms_error": 0.0,
        }

    # 정규화
    env_max = np.max(rms_env)
    env_norm = rms_env / env_max

    # ADSR 구간 감지
    peak_idx = int(np.argmax(env_norm))

    # Attack: 시작 → 피크
    attack_start = 0
    # 10% 이상인 첫 프레임부터
    above_thresh = np.where(env_norm > 0.1)[0]
    if len(above_thresh) > 0:
        attack_start = above_thresh[0]

    # Sustain level 추정 (피크 이후 안정 구간의 평균)
    post_peak = env_norm[peak_idx:]
    if len(post_peak) > 10:
        sustain_level = float(np.median(post_peak[len(post_peak)//4 : 3*len(post_peak)//4]))
    else:
        sustain_level = float(np.mean(post_peak)) if len(post_peak) > 0 else 0.0

    # Decay: 피크 → sustain level 도달
    decay_end = peak_idx
    if sustain_level > 0.05:
        for i in range(peak_idx, len(env_norm)):
            if env_norm[i] <= sustain_level * 1.05:
                decay_end = i
                break
        else:
            decay_end = len(env_norm) - 1

    # Release: sustain 끝 → 10% 이하
    release_start = decay_end
    # note_off 지점 추정 (마지막 sustain level 이상인 프레임)
    above_sustain = np.where(env_norm[decay_end:] >= sustain_level * 0.5)[0]
    if len(above_sustain) > 0:
        release_start = decay_end + above_sustain[-1]

    release_end = len(env_norm) - 1
    below_thresh = np.where(env_norm[release_start:] < 0.05)[0]
    if len(below_thresh) > 0:
        release_end = release_start + below_thresh[0]

    # 각 구간 curvature 계산
    attack_curve = _measure_curvature(env_norm, attack_start, peak_idx)
    decay_curve = _measure_curvature(env_norm, peak_idx, decay_end, descending=True)
    release_curve = _measure_curvature(env_norm, release_start, release_end, descending=True)

    # Linear ADSR 윈도우 생성 + 전체 비교
    linear_env = _build_linear_adsr(
        len(env_norm), attack_start, peak_idx, decay_end,
        release_start, release_end, sustain_level
    )

    # 전체 RMS error
    error = float(np.sqrt(np.mean((env_norm - linear_env) ** 2)))

    return {
        "attack_curvature": round(attack_curve["curvature"], 4),
        "decay_curvature": round(decay_curve["curvature"], 4),
        "release_curvature": round(release_curve["curvature"], 4),
        "attack_r2": round(attack_curve["r2"], 4),
        "decay_r2": round(decay_curve["r2"], 4),
        "release_r2": round(release_curve["r2"], 4),
        "sustain_level": round(sustain_level, 4),
        "envelope_rms_error": round(error, 6),
    }


def _measure_curvature(env: np.ndarray, start: int, end: int, descending: bool = False) -> dict:
    """구간의 curvature 측정

    curvature > 0: exponential/convex (빠르게 변화 후 느려짐)
    curvature ≈ 0: linear
    curvature < 0: logarithmic/concave (느리게 시작 후 빨라짐)

    방법: log-linear 피팅으로 지수 추정
    """
    if end <= start or end - start < 3:
        return {"curvature": 0.0, "r2": 1.0}

    segment = env[start:end].copy()
    n = len(segment)
    t = np.linspace(0, 1, n)

    if descending:
        segment = segment[::-1]

    # 정규화 (0→1 범위)
    seg_min, seg_max = np.min(segment), np.max(segment)
    if seg_max - seg_min < 1e-10:
        return {"curvature": 0.0, "r2": 1.0}

    seg_norm = (segment - seg_min) / (seg_max - seg_min)

    # Linear fit
    linear = t
    residual_linear = seg_norm - linear
    ss_res_linear = float(np.sum(residual_linear**2))
    ss_tot = float(np.sum((seg_norm - np.mean(seg_norm))**2))
    r2 = 1.0 - ss_res_linear / ss_tot if ss_tot > 1e-10 else 1.0

    # Curvature: 실제 곡선과 직선의 편차 방향
    # 중간점에서의 편차 — 양수면 convex(exp), 음수면 concave(log)
    mid = n // 2
    curvature = float(np.mean(seg_norm[:mid] - linear[:mid]) - np.mean(seg_norm[mid:] - linear[mid:]))

    return {"curvature": curvature, "r2": r2}


def _build_linear_adsr(
    length: int, attack_start: int, peak_idx: int,
    decay_end: int, release_start: int, release_end: int,
    sustain_level: float,
) -> np.ndarray:
    """Linear ADSR 윈도우 생성"""
    env = np.zeros(length, dtype=np.float32)

    # Attack (0 → 1)
    if peak_idx > attack_start:
        env[attack_start:peak_idx] = np.linspace(0, 1, peak_idx - attack_start)

    # Decay (1 → sustain)
    if decay_end > peak_idx:
        env[peak_idx:decay_end] = np.linspace(1, sustain_level, decay_end - peak_idx)

    # Sustain
    if release_start > decay_end:
        env[decay_end:release_start] = sustain_level

    # Release (sustain → 0)
    if release_end > release_start:
        env[release_start:release_end] = np.linspace(sustain_level, 0, release_end - release_start)

    return env


def _estimate_f0(mono: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 2000.0) -> float:
    """Autocorrelation 기반 기본 주파수 추정"""
    # 안정 구간 (중간 1/3) 사용
    n = len(mono)
    segment = mono[n // 3 : 2 * n // 3]
    if len(segment) < 1024:
        return 0.0

    # Autocorrelation
    corr = np.correlate(segment, segment, mode="full")
    corr = corr[len(corr) // 2 :]
    corr = corr / (corr[0] + 1e-10)

    # 탐색 범위
    min_lag = int(sr / fmax)
    max_lag = min(int(sr / fmin), len(corr) - 1)

    if min_lag >= max_lag or max_lag >= len(corr):
        return 0.0

    # 첫 번째 피크 찾기
    search = corr[min_lag:max_lag]
    if len(search) < 3:
        return 0.0

    peaks = []
    for i in range(1, len(search) - 1):
        if search[i] > search[i - 1] and search[i] > search[i + 1] and search[i] > 0.3:
            peaks.append((i + min_lag, search[i]))

    if not peaks:
        return 0.0

    # 가장 강한 피크
    best_lag = max(peaks, key=lambda x: x[1])[0]
    return float(sr / best_lag)


def _estimate_harmonic_ratio(mono: np.ndarray, sr: int, frame_size: int = 4096) -> float:
    """하모닉 에너지 비율 추정 (0=noise, 1=pure tone)"""
    n = len(mono)
    segment = mono[n // 3 : 2 * n // 3]
    if len(segment) < frame_size:
        return 0.0

    window = np.hanning(frame_size)
    frame = segment[:frame_size] * window
    spectrum = np.abs(np.fft.rfft(frame))
    power = spectrum**2
    total_power = np.sum(power)

    if total_power < 1e-10:
        return 0.0

    # 상위 10개 피크의 에너지 합
    top_indices = np.argsort(power)[-10:]
    harmonic_power = np.sum(power[top_indices])

    return float(harmonic_power / total_power)


def _compute_peak_freq_trajectory(
    mono: np.ndarray,
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    rms_frames: list = None,
) -> dict:
    """Peak frequency spectrogram → filter envelope trajectory

    각 프레임에서 가장 에너지가 높은 주파수를 추적.
    이 궤적의 변화가 filter cutoff envelope을 직접 반영.

    Returns:
        peak_freq_start: 시작 구간 피크 주파수 (Hz)
        peak_freq_peak: RMS 피크 구간 피크 주파수
        peak_freq_sustain: 서스테인 구간 피크 주파수
        peak_freq_sweep_range: 전체 피크 주파수 변화 폭 (Hz)
        peak_freq_attack_ratio: start→peak 변화 비율 (>1=filter opening)
        peak_freq_decay_ratio: peak→sustain 변화 비율 (<1=filter closing)
        spectral_peak_stability: 피크 주파수 안정도 (0=많이 변함, 1=안정)
    """
    n_samples = len(mono)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    window = np.hanning(frame_size).astype(np.float32)

    peak_freqs = []
    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = mono[start: start + frame_size]
        spectrum = np.abs(np.fft.rfft(frame * window))
        # DC 제외 (20Hz 이상)
        min_bin = max(1, int(20 * frame_size / sample_rate))
        peak_bin = min_bin + np.argmax(spectrum[min_bin:])
        peak_freqs.append(float(freqs[peak_bin]))

    if len(peak_freqs) < 4:
        return {
            "peak_freq_start": 0.0, "peak_freq_peak": 0.0,
            "peak_freq_sustain": 0.0, "peak_freq_sweep_range": 0.0,
            "peak_freq_attack_ratio": 1.0, "peak_freq_decay_ratio": 1.0,
            "spectral_peak_stability": 1.0,
        }

    pf = np.array(peak_freqs, dtype=np.float32)
    n = len(pf)

    # RMS 기반 구간 분리
    if rms_frames and len(rms_frames) >= n:
        rms = np.array(rms_frames[:n], dtype=np.float32)
    else:
        rms = np.ones(n, dtype=np.float32)

    rms_max = np.max(rms)
    if rms_max < 1e-10:
        peak_idx = n // 4
    else:
        peak_idx = int(np.argmax(rms))

    # 유효 구간만 (소리가 나는 프레임)
    active = rms > rms_max * 0.05

    # 시작 구간 (처음 10%)
    start_end = max(1, n // 10)
    start_pf = float(np.median(pf[:start_end])) if start_end > 0 else pf[0]

    # 피크 구간 (RMS 피크 ±5프레임)
    pk_start = max(0, peak_idx - 5)
    pk_end = min(n, peak_idx + 5)
    peak_pf = float(np.median(pf[pk_start:pk_end]))

    # 서스테인 구간 (피크 이후 중간)
    sus_start = min(peak_idx + n // 10, n - 1)
    sus_end = max(sus_start + 1, n - n // 10)
    sustain_pf = float(np.median(pf[sus_start:sus_end])) if sus_end > sus_start else peak_pf

    # 변화 비율
    attack_ratio = peak_pf / max(start_pf, 1.0)
    decay_ratio = sustain_pf / max(peak_pf, 1.0)

    # 전체 스윕 범위
    active_pf = pf[active] if np.any(active) else pf
    sweep_range = float(np.max(active_pf) - np.min(active_pf))

    # 안정도 (변동 계수의 역수)
    pf_std = float(np.std(active_pf))
    pf_mean = float(np.mean(active_pf))
    stability = 1.0 / (1.0 + pf_std / max(pf_mean, 1.0))

    return {
        "peak_freq_start": round(start_pf, 1),
        "peak_freq_peak": round(peak_pf, 1),
        "peak_freq_sustain": round(sustain_pf, 1),
        "peak_freq_sweep_range": round(sweep_range, 1),
        "peak_freq_attack_ratio": round(attack_ratio, 4),    # >1 = filter opening
        "peak_freq_decay_ratio": round(decay_ratio, 4),      # <1 = filter closing
        "spectral_peak_stability": round(stability, 4),
    }
