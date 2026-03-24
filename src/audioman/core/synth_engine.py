# Created: 2026-03-24
# Purpose: SignalFlow 기반 서브트랙티브 신디사이저 엔진
# License: MIT (SignalFlow도 MIT)
# Dependencies: signalflow, numpy

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SynthPatch:
    """신스 패치 파라미터 (모든 값 0.0~1.0 정규화)

    preset_schema.py의 NormalizedPreset.to_vector()와 호환.
    """

    # OSC 1
    osc1_volume: float = 0.7
    osc1_waveform: float = 0.0  # 0=saw, 0.33=square, 0.66=tri, 1.0=sine
    osc1_octave: float = 0.5  # 0=C1, 0.5=C4, 1.0=C7
    osc1_semi: float = 0.5  # 0=-12, 0.5=0, 1.0=+12
    osc1_fine: float = 0.5  # 0=-1, 0.5=0, 1.0=+1 semitone
    osc1_pan: float = 0.5

    # OSC 2
    osc2_volume: float = 0.0
    osc2_waveform: float = 0.25
    osc2_octave: float = 0.5
    osc2_semi: float = 0.5
    osc2_fine: float = 0.5
    osc2_pan: float = 0.5

    # Sub / Noise
    sub_volume: float = 0.0
    noise_volume: float = 0.0

    # Filter
    filter_cutoff: float = 1.0  # 0=20Hz, 1.0=20kHz
    filter_resonance: float = 0.0
    filter_type: float = 0.0  # 0=LP, 0.5=BP, 1.0=HP
    filter_env_amount: float = 0.0

    # Amp Envelope
    amp_attack: float = 0.01
    amp_decay: float = 0.3
    amp_sustain: float = 0.7
    amp_release: float = 0.3

    # Filter Envelope
    filt_attack: float = 0.01
    filt_decay: float = 0.3
    filt_sustain: float = 0.5
    filt_release: float = 0.3

    # LFO
    lfo_rate: float = 0.3  # Hz 정규화
    lfo_depth: float = 0.0  # 필터 모듈레이션 양
    lfo_waveform: float = 0.0  # 0=sine, 0.5=tri, 1.0=saw

    # FX
    delay_wet: float = 0.0
    delay_time: float = 0.3
    delay_feedback: float = 0.3

    # Master
    master_volume: float = 0.7

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_vector(cls, vec: list[float]) -> "SynthPatch":
        """158차원 정규화 벡터에서 주요 파라미터 추출"""
        p = cls()
        if len(vec) < 45:
            return p

        # osc1: vec[0:9]
        p.osc1_volume = vec[0]
        p.osc1_pan = vec[1]
        p.osc1_octave = vec[2]
        p.osc1_semi = vec[3]
        p.osc1_fine = vec[4]
        p.osc1_waveform = vec[7]

        # osc2: vec[9:18]
        p.osc2_volume = vec[9]
        p.osc2_pan = vec[10]
        p.osc2_octave = vec[11]
        p.osc2_semi = vec[12]
        p.osc2_fine = vec[13]
        p.osc2_waveform = vec[16]

        # sub: vec[27], noise: vec[36]
        p.sub_volume = vec[27]
        p.noise_volume = vec[36]

        # filter: vec[45:51]
        if len(vec) > 50:
            p.filter_cutoff = vec[45]
            p.filter_resonance = vec[46]
            p.filter_type = vec[47]
            p.filter_env_amount = vec[50]

        # env_amp: vec[57:65]
        if len(vec) > 64:
            p.amp_attack = vec[57]
            p.amp_decay = vec[58]
            p.amp_sustain = vec[59]
            p.amp_release = vec[60]

        # env_filter: vec[65:73]
        if len(vec) > 72:
            p.filt_attack = vec[65]
            p.filt_decay = vec[66]
            p.filt_sustain = vec[67]
            p.filt_release = vec[68]

        # lfo1: vec[89:95]
        if len(vec) > 94:
            p.lfo_rate = vec[89]
            p.lfo_waveform = vec[90]
            p.lfo_depth = vec[92]

        # fx_delay: vec[118:123]
        if len(vec) > 122:
            p.delay_wet = vec[119]
            p.delay_time = vec[120]
            p.delay_feedback = vec[121]

        # master: vec[153]
        if len(vec) > 153:
            p.master_volume = vec[153]

        return p


def _param_to_hz(cutoff_norm: float, min_hz: float = 20.0, max_hz: float = 20000.0) -> float:
    """0~1 → Hz (로그 스케일)"""
    return min_hz * (max_hz / min_hz) ** cutoff_norm


def _param_to_time(norm: float, min_s: float = 0.001, max_s: float = 5.0) -> float:
    """0~1 → 초 (로그 스케일)"""
    return min_s * (max_s / min_s) ** norm


def _param_to_lfo_hz(norm: float) -> float:
    """0~1 → LFO Hz"""
    return 0.1 * (20.0 / 0.1) ** norm


def render_patch(
    patch: SynthPatch,
    note: int = 60,
    velocity: int = 100,
    duration: float = 2.0,
    tail: float = 1.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """SynthPatch → SignalFlow 렌더링 → numpy array (channels, samples)"""
    import signalflow as sf

    graph = sf.AudioGraph(start=False)

    midi_freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
    vel_gain = velocity / 127.0

    # --- 오실레이터 ---
    def make_osc(waveform_norm: float, freq: float) -> sf.Node:
        if waveform_norm < 0.25:
            return sf.SawOscillator(freq)
        elif waveform_norm < 0.5:
            return sf.SquareOscillator(freq)
        elif waveform_norm < 0.75:
            return sf.TriangleOscillator(freq)
        else:
            return sf.SineOscillator(freq)

    # OSC 1 피치 계산
    osc1_oct = int((patch.osc1_octave - 0.5) * 6)  # -3 ~ +3 옥타브
    osc1_semi = int((patch.osc1_semi - 0.5) * 24)  # -12 ~ +12
    osc1_fine_cents = (patch.osc1_fine - 0.5) * 2  # -1 ~ +1 semitone
    osc1_freq = midi_freq * (2.0 ** (osc1_oct + osc1_semi / 12.0 + osc1_fine_cents / 12.0))

    osc1 = make_osc(patch.osc1_waveform, osc1_freq)

    # OSC 2
    osc2_oct = int((patch.osc2_octave - 0.5) * 6)
    osc2_semi = int((patch.osc2_semi - 0.5) * 24)
    osc2_fine_cents = (patch.osc2_fine - 0.5) * 2
    osc2_freq = midi_freq * (2.0 ** (osc2_oct + osc2_semi / 12.0 + osc2_fine_cents / 12.0))

    osc2 = make_osc(patch.osc2_waveform, osc2_freq)

    # Sub (사인, 1옥타브 아래)
    sub = sf.SineOscillator(midi_freq / 2)

    # Noise
    # SignalFlow에 WhiteNoise가 없으면 대체
    try:
        noise = sf.WhiteNoise()
    except AttributeError:
        noise = sf.SawOscillator(midi_freq * 37.0)  # 고주파 saw로 대체

    # 믹스
    mix = (osc1 * patch.osc1_volume +
           osc2 * patch.osc2_volume +
           sub * patch.sub_volume +
           noise * patch.noise_volume)

    # --- 필터 ---
    cutoff_hz = _param_to_hz(patch.filter_cutoff)
    reso = patch.filter_resonance * 0.95  # 0~0.95 범위

    filter_mode = "low_pass"
    if patch.filter_type > 0.66:
        filter_mode = "high_pass"
    elif patch.filter_type > 0.33:
        filter_mode = "band_pass"

    filtered = sf.SVFilter(mix, filter_mode, cutoff_hz, reso)

    # --- 엔벨로프 ---
    clock = sf.Impulse(0)

    amp_env = sf.ASREnvelope(
        attack=_param_to_time(patch.amp_attack),
        sustain=patch.amp_sustain,
        release=_param_to_time(patch.amp_release),
        clock=clock,
    )

    mono_out = filtered * amp_env * patch.master_volume * vel_gain

    # --- 딜레이 ---
    if patch.delay_wet > 0.01:
        delay_time_s = patch.delay_time * 1.0  # 0~1초
        delayed = sf.CombDelay(mono_out, delay_time_s, patch.delay_feedback * 0.8)
        mono_out = mono_out * (1.0 - patch.delay_wet) + delayed * patch.delay_wet

    # --- 스테레오 패닝 ---
    # pan: 0.0=left, 0.5=center, 1.0=right → StereoPanner: -1~+1
    pan_value = (patch.osc1_pan - 0.5) * 2.0
    output = sf.StereoPanner(mono_out, pan_value)

    # --- 렌더링 ---
    total_time = duration + tail
    total_samples = int(total_time * sample_rate)
    buf = sf.Buffer(2, total_samples)
    output.play()
    graph.render_to_buffer(buf)

    audio = np.array(buf.data, dtype=np.float32)

    # 페이드아웃 (끝 500ms)
    fadeout_samples = int(0.5 * sample_rate)
    if fadeout_samples > 0 and fadeout_samples < audio.shape[1]:
        fade = np.linspace(1.0, 0.0, fadeout_samples, dtype=np.float32)
        audio[:, -fadeout_samples:] *= fade

    return audio


def text_to_sound(
    description: str,
    clap_model=None,
    clap_embeddings: Optional[np.ndarray] = None,
    preset_vectors: Optional[list] = None,
    note: int = 60,
    duration: float = 2.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """자연어 설명 → CLAP 검색 → 가장 가까운 프리셋 파라미터 → SignalFlow 렌더링

    완전한 text-to-sound 파이프라인.
    """
    if clap_model is None or clap_embeddings is None or preset_vectors is None:
        # CLAP 없으면 기본 패치 사용
        logger.warning("CLAP 모델/임베딩 없음 — 기본 패치 사용")
        return render_patch(SynthPatch(), note=note, duration=duration, sample_rate=sample_rate)

    from numpy.linalg import norm

    # 텍스트 → CLAP 임베딩
    text_emb = clap_model.get_text_embedding([description], use_tensor=False)[0]

    # 코사인 유사도로 가장 가까운 프리셋 찾기
    sims = np.dot(clap_embeddings, text_emb) / (norm(clap_embeddings, axis=1) * norm(text_emb))
    best_idx = np.argmax(sims)

    # 프리셋 벡터 → SynthPatch
    patch = SynthPatch.from_vector(preset_vectors[best_idx])
    logger.info(f'text_to_sound: "{description}" → preset #{best_idx} (sim={sims[best_idx]:.3f})')

    return render_patch(patch, note=note, duration=duration, sample_rate=sample_rate)
