# Created: 2026-03-25
# Purpose: Test signal generation for plugin analysis

import numpy as np


def generate_impulse(
    sample_rate: int = 44100,
    duration_sec: float = 1.0,
    level_db: float = 0.0,
    channels: int = 2,
) -> np.ndarray:
    """Delta impulse — linear analysis (IR measurement)

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
    """Pure sine wave — THD/oscilloscope measurement

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
    """Two-tone test signal — IMD measurement (SMPTE standard: 60 Hz + 7 kHz)

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
    """White noise — linear analysis (averaging)

    Returns: (channels, samples) float32
    """
    rng = np.random.RandomState(seed)
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)
    audio = amp * rng.randn(channels, n).astype(np.float32)
    # peak normalization
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
    """Frequency sweep — 2D sweep analysis, THD vs frequency

    Returns: (channels, samples) float32
    """
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)
    t = np.arange(n, dtype=np.float64) / sample_rate

    if exponential:
        # log sweep (Novak method — best for aliasing detection)
        L = duration_sec / np.log(freq_end / freq_start)
        phase = 2 * np.pi * freq_start * L * (np.exp(t / L) - 1)
    else:
        # linear sweep
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
    """Dynamics ramp — output measurement per input level (compressor curve)

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
    """Attack/Release test — three-level ramp (below → above → below)

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
    """Farina-method log sweep + inverse filter — for EQ frequency-response deconvolution

    Advantages over a plain impulse:
    - much higher SNR (energy is spread over time)
    - accurately captures the long impulse response of low-frequency shelving EQ
    - nonlinear distortion products can be separated on the time axis

    Returns: (sweep_audio, inverse_filter)
        sweep_audio: (channels, samples) float32
        inverse_filter: (1, samples) float32 — inverse filter for deconvolution
    """
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)
    t = np.arange(n, dtype=np.float64) / sample_rate

    # Farina log sweep
    L = duration_sec / np.log(freq_end / freq_start)
    phase = 2 * np.pi * freq_start * L * (np.exp(t / L) - 1)
    sweep = (amp * np.sin(phase)).astype(np.float32)

    # inverse filter: time reversal + amplitude compensation (compensate the energy roll-off at high frequencies)
    # Farina's inverse filter time-reverses the sweep, then applies frequency-dependent amplitude compensation
    inverse = sweep[::-1].copy()

    # frequency-dependent amplitude compensation: exp(-t/L) envelope
    t_inv = np.arange(n, dtype=np.float64) / sample_rate
    envelope = np.exp(-t_inv / L).astype(np.float32)
    # normalize so the deconvolution result becomes a unit impulse
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
    """Schroeder-phase multitone — for single-pass EQ frequency-response measurement

    Generates n_tones logarithmically spaced sine waves simultaneously.
    A Schroeder phase is applied to minimize the crest factor.

    Returns: (channels, samples) float32
    """
    n = int(sample_rate * duration_sec)
    amp_per_tone = 10 ** (level_db / 20.0) / np.sqrt(n_tones)

    # logarithmically spaced frequencies
    freqs = np.geomspace(freq_start, freq_end, n_tones)

    t = np.arange(n, dtype=np.float64) / sample_rate
    signal = np.zeros(n, dtype=np.float64)

    for k, freq in enumerate(freqs):
        # Schroeder phase: phi_k = -k*(k-1)*pi/n_tones
        # lower the crest factor so more energy fits under the same peak
        phase = -k * (k - 1) * np.pi / n_tones
        signal += amp_per_tone * np.sin(2 * np.pi * freq * t + phase)

    # peak normalization (keep the target level)
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
    """Pink noise (1/f) — EQ test signal closer to a musical spectrum

    Returns: (channels, samples) float32
    """
    rng = np.random.RandomState(seed)
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)

    # build a 1/sqrt(f) spectrum in the frequency domain
    white = rng.randn(n).astype(np.float64)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    # exclude DC, apply 1/sqrt(f)
    freqs[0] = 1.0  # guard DC
    pink_filter = 1.0 / np.sqrt(freqs)
    spectrum *= pink_filter

    pink = np.fft.irfft(spectrum, n=n).astype(np.float32)

    # peak normalization
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
    """Band-limited noise — for testing EQ response in a specific frequency band

    Returns: (channels, samples) float32
    """
    rng = np.random.RandomState(seed)
    n = int(sample_rate * duration_sec)
    amp = 10 ** (level_db / 20.0)

    white = rng.randn(n).astype(np.float64)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    # band-pass mask
    mask = np.zeros_like(freqs)
    mask[(freqs >= freq_low) & (freqs <= freq_high)] = 1.0
    spectrum *= mask

    band_noise = np.fft.irfft(spectrum, n=n).astype(np.float32)

    # peak normalization
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
    """Impulse train — EQ transient response + frequency coloration test

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
    """L/R → M/S conversion. audio: (2, samples)"""
    if audio.shape[0] != 2:
        raise ValueError("M/S conversion only supports stereo")
    mid = (audio[0] + audio[1]) * 0.5
    side = (audio[0] - audio[1]) * 0.5
    return np.stack([mid, side])


def from_mid_side(audio: np.ndarray) -> np.ndarray:
    """M/S → L/R conversion"""
    if audio.shape[0] != 2:
        raise ValueError("M/S conversion only supports stereo")
    left = audio[0] + audio[1]
    right = audio[0] - audio[1]
    return np.stack([left, right])
