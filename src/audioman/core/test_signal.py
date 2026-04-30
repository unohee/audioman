# Created: 2026-03-25
# Purpose: 플러그인 분석용 테스트 신호 생성

import numpy as np


def generate_impulse(
    sample_rate: int = 44100,
    duration_sec: float = 1.0,
    level_db: float = 0.0,
    channels: int = 2,
) -> np.ndarray:
    """델타 임펄스 — linear analysis (IR 측정)

    Returns: (channels, samples) float32
    """
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)
    audio = np.zeros((channels, n), dtype=np.float32)
    audio[:, 0] = amp
    return audio


def generate_sine(
    frequency: float = 1000.0,
    sample_rate: int = 44100,
    duration_sec: float = 1.0,
    level_db: float = 0.0,
    channels: int = 2,
) -> np.ndarray:
    """순수 사인파 — THD/oscilloscope 측정

    Returns: (channels, samples) float32
    """
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)
    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = amp * np.sin(2 * np.pi * frequency * t)
    return np.stack([mono] * channels)


def generate_two_tone(
    freq_low: float = 60.0,
    freq_high: float = 7000.0,
    sample_rate: int = 44100,
    duration_sec: float = 1.0,
    level_db_low: float = 0.0,
    level_db_high: float = -12.0,
    channels: int = 2,
) -> np.ndarray:
    """2톤 테스트 신호 — IMD 측정 (SMPTE 표준: 60Hz + 7kHz)

    Returns: (channels, samples) float32
    """
    n = int(sample_rate * duration_sec)
    t = np.arange(n, dtype=np.float32) / sample_rate
    amp_low = 10 ** (level_db_low / 20.0)
    amp_high = 10 ** (level_db_high / 20.0)
    mono = amp_low * np.sin(2 * np.pi * freq_low * t) + amp_high * np.sin(2 * np.pi * freq_high * t)
    return np.stack([mono] * channels)


def generate_white_noise(
    sample_rate: int = 44100,
    duration_sec: float = 2.0,
    level_db: float = 0.0,
    channels: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """화이트 노이즈 — linear analysis (평균화)

    Returns: (channels, samples) float32
    """
    rng = np.random.RandomState(seed)
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)
    audio = amp * rng.randn(channels, n).astype(np.float32)
    # 피크 정규화
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (amp / peak)
    return audio


def generate_sweep(
    freq_start: float = 20.0,
    freq_end: float = 20000.0,
    sample_rate: int = 44100,
    duration_sec: float = 6.0,
    level_db: float = -6.0,
    exponential: bool = True,
    channels: int = 2,
) -> np.ndarray:
    """주파수 스윕 — 2D sweep analysis, THD vs frequency

    Returns: (channels, samples) float32
    """
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)
    t = np.arange(n, dtype=np.float64) / sample_rate

    if exponential:
        # 로그 스윕 (Novak 방법 — 앨리어싱 감지에 최적)
        L = duration_sec / np.log(freq_end / freq_start)
        phase = 2 * np.pi * freq_start * L * (np.exp(t / L) - 1)
    else:
        # 선형 스윕
        freq_rate = (freq_end - freq_start) / duration_sec
        phase = 2 * np.pi * (freq_start * t + 0.5 * freq_rate * t**2)

    mono = (amp * np.sin(phase)).astype(np.float32)
    return np.stack([mono] * channels)


def generate_dynamics_ramp(
    frequency: float = 1000.0,
    sample_rate: int = 44100,
    level_start_db: float = -100.0,
    level_end_db: float = 0.0,
    step_db: float = 1.0,
    step_duration_sec: float = 0.5,
    channels: int = 2,
) -> tuple[np.ndarray, list[float]]:
    """다이내믹스 램프 — 입력 레벨별 출력 측정 (컴프레서 곡선)

    Returns: (audio, level_list_db)
    """
    levels = np.arange(level_start_db, level_end_db + step_db, step_db)
    step_samples = int(step_duration_sec * sample_rate)
    n = step_samples * len(levels)

    t_step = np.arange(step_samples, dtype=np.float32) / sample_rate
    audio = np.zeros((channels, n), dtype=np.float32)

    for i, level in enumerate(levels):
        amp = 10 ** (level / 20.0)
        start = i * step_samples
        segment = amp * np.sin(2 * np.pi * frequency * t_step)
        audio[:, start:start + step_samples] = segment

    return audio, levels.tolist()


def generate_dynamics_attack_release(
    frequency: float = 1000.0,
    sample_rate: int = 44100,
    level_below_db: float = -30.0,
    level_above_db: float = 0.0,
    t1_sec: float = 0.5,
    t2_sec: float = 1.0,
    t3_sec: float = 0.5,
    channels: int = 2,
) -> np.ndarray:
    """Attack/Release 테스트 — 3단계 레벨 (below → above → below)

    Returns: (channels, samples) float32
    """
    n1 = int(t1_sec * sample_rate)
    n2 = int(t2_sec * sample_rate)
    n3 = int(t3_sec * sample_rate)
    n = n1 + n2 + n3

    t = np.arange(n, dtype=np.float32) / sample_rate
    sine = np.sin(2 * np.pi * frequency * t)

    amp_below = 10 ** (level_below_db / 20.0)
    amp_above = 10 ** (level_above_db / 20.0)

    envelope = np.concatenate([
        np.full(n1, amp_below, dtype=np.float32),
        np.full(n2, amp_above, dtype=np.float32),
        np.full(n3, amp_below, dtype=np.float32),
    ])

    mono = (sine * envelope).astype(np.float32)
    return np.stack([mono] * channels)


def generate_log_sweep_deconv(
    freq_start: float = 20.0,
    freq_end: float = 20000.0,
    sample_rate: int = 44100,
    duration_sec: float = 6.0,
    level_db: float = -12.0,
    channels: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Farina method 로그 스윕 + 역필터 — EQ 주파수 응답 디컨볼루션용

    일반 임펄스 대비 장점:
    - SNR이 훨씬 높음 (에너지가 시간에 걸쳐 분산)
    - 저주파 shelving EQ의 긴 임펄스 응답 정확 캡처
    - 비선형 왜곡 성분을 시간축에서 분리 가능

    Returns: (sweep_audio, inverse_filter)
        sweep_audio: (channels, samples) float32
        inverse_filter: (1, samples) float32 — 디컨볼루션용 역필터
    """
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)
    t = np.arange(n, dtype=np.float64) / sample_rate

    # Farina log sweep
    L = duration_sec / np.log(freq_end / freq_start)
    phase = 2 * np.pi * freq_start * L * (np.exp(t / L) - 1)
    sweep = (amp * np.sin(phase)).astype(np.float32)

    # 역필터: 시간 반전 + 진폭 보상 (고주파로 갈수록 에너지 감소 보상)
    # Farina의 역필터는 sweep를 시간 반전 후 주파수 의존 진폭 보상 적용
    inverse = sweep[::-1].copy()

    # 주파수 의존 진폭 보상: exp(-t/L) envelope
    t_inv = np.arange(n, dtype=np.float64) / sample_rate
    envelope = np.exp(-t_inv / L).astype(np.float32)
    # 정규화: 디컨볼루션 결과가 단위 임펄스가 되도록
    envelope /= np.sum(sweep ** 2) / n + 1e-10
    inverse *= envelope

    sweep_audio = np.stack([sweep] * channels)
    inverse_filter = inverse.reshape(1, -1)

    return sweep_audio, inverse_filter


def generate_multitone(
    n_tones: int = 64,
    freq_start: float = 20.0,
    freq_end: float = 20000.0,
    sample_rate: int = 44100,
    duration_sec: float = 4.0,
    level_db: float = -18.0,
    channels: int = 2,
) -> np.ndarray:
    """Schroeder-phase 멀티톤 — EQ 단일 패스 주파수 응답 측정용

    로그 분포된 n_tones개의 사인파를 동시 생성.
    Schroeder phase를 적용하여 crest factor를 최소화.

    Returns: (channels, samples) float32
    """
    n = int(sample_rate * duration_sec)
    amp_per_tone = 10 ** (level_db / 20.0) / np.sqrt(n_tones)

    # 로그 분포 주파수
    freqs = np.geomspace(freq_start, freq_end, n_tones)

    t = np.arange(n, dtype=np.float64) / sample_rate
    signal = np.zeros(n, dtype=np.float64)

    for k, freq in enumerate(freqs):
        # Schroeder phase: phi_k = -k*(k-1)*pi/n_tones
        # crest factor를 줄여서 동일 peak에서 더 많은 에너지 전달
        phase = -k * (k - 1) * np.pi / n_tones
        signal += amp_per_tone * np.sin(2 * np.pi * freq * t + phase)

    # 피크 정규화 (목표 레벨 유지)
    target_amp = 10 ** (level_db / 20.0)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal *= target_amp / peak

    mono = signal.astype(np.float32)
    return np.stack([mono] * channels)


def generate_pink_noise(
    sample_rate: int = 44100,
    duration_sec: float = 3.0,
    level_db: float = -12.0,
    channels: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """핑크 노이즈 (1/f) — 음악적 스펙트럼에 가까운 EQ 테스트 신호

    Returns: (channels, samples) float32
    """
    rng = np.random.RandomState(seed)
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)

    # 주파수 도메인에서 1/sqrt(f) 스펙트럼 생성
    white = rng.randn(n).astype(np.float64)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    # DC 제외, 1/sqrt(f) 적용
    freqs[0] = 1.0  # DC 보호
    pink_filter = 1.0 / np.sqrt(freqs)
    spectrum *= pink_filter

    pink = np.fft.irfft(spectrum, n=n).astype(np.float32)

    # 피크 정규화
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink *= amp / peak

    return np.stack([pink] * channels)


def generate_band_limited_noise(
    freq_low: float = 200.0,
    freq_high: float = 2000.0,
    sample_rate: int = 44100,
    duration_sec: float = 3.0,
    level_db: float = -12.0,
    channels: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """밴드 제한 노이즈 — 특정 주파수 대역의 EQ 응답 테스트용

    Returns: (channels, samples) float32
    """
    rng = np.random.RandomState(seed)
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)

    white = rng.randn(n).astype(np.float64)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    # 밴드패스 마스크
    mask = np.zeros_like(freqs)
    mask[(freqs >= freq_low) & (freqs <= freq_high)] = 1.0
    spectrum *= mask

    band_noise = np.fft.irfft(spectrum, n=n).astype(np.float32)

    # 피크 정규화
    peak = np.max(np.abs(band_noise))
    if peak > 0:
        band_noise *= amp / peak

    return np.stack([band_noise] * channels)


def generate_impulse_train(
    rate_hz: float = 10.0,
    sample_rate: int = 44100,
    duration_sec: float = 2.0,
    level_db: float = -6.0,
    channels: int = 2,
) -> np.ndarray:
    """임펄스 트레인 — EQ 과도 응답 + 주파수 착색 테스트

    Returns: (channels, samples) float32
    """
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)
    audio = np.zeros(n, dtype=np.float32)

    period = int(sample_rate / rate_hz)
    for i in range(0, n, period):
        audio[i] = amp

    return np.stack([audio] * channels)


def to_mid_side(audio: np.ndarray) -> np.ndarray:
    """L/R → M/S 변환. audio: (2, samples)"""
    if audio.shape[0] != 2:
        raise ValueError("M/S 변환은 스테레오만 지원")
    mid = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5
    return np.stack([mid, side])


def from_mid_side(audio: np.ndarray) -> np.ndarray:
    """M/S → L/R 변환"""
    if audio.shape[0] != 2:
        raise ValueError("M/S 변환은 스테레오만 지원")
    left = audio[0] + audio[1]
    right = audio[0] - audio[1]
    return np.stack([left, right])
