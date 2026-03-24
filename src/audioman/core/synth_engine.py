# Created: 2026-03-24
# Purpose: SignalFlow 기반 신디사이저 엔진 v2
# License: MIT (SignalFlow도 MIT)
# Dependencies: signalflow, numpy
#
# 아키텍처:
#   OSC1 (VA/WT/FM) ─┐
#   OSC2 (VA/WT/FM) ─┼─► Mixer ─► Filter (SVF) ─► Amp ─► FX ─► Stereo Out
#   Sub (Sine)       ─┤                ↑              ↑
#   Noise (W/P)      ─┘           ENV2+LFO        ENV1+lag
#                                                 ENV3 → free
#                          FM: OSC2 → OSC1 freq
#                          4 LFO, 3 ENV (ADSR + lag)

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# SynthPatch v2 — 약 65 파라미터
# =============================================================================


@dataclass
class SynthPatch:
    """신스 패치 파라미터 (모든 값 0.0~1.0 정규화)"""

    # --- OSC 1 ---
    osc1_mode: float = 0.0        # 0=VA, 0.5=wavetable, 1.0=FM carrier
    osc1_waveform: float = 0.0    # VA: 0=saw 0.25=sqr 0.5=tri 0.75=sin / WT: position
    osc1_volume: float = 0.7
    osc1_pan: float = 0.5         # 0=L, 0.5=C, 1=R
    osc1_octave: float = 0.5      # 0=-3oct, 0.5=0, 1=+3oct
    osc1_semi: float = 0.5        # 0=-12, 0.5=0, 1=+12
    osc1_fine: float = 0.5        # 0=-1st, 0.5=0, 1=+1st

    # --- OSC 2 ---
    osc2_mode: float = 0.0
    osc2_waveform: float = 0.25   # square
    osc2_volume: float = 0.0
    osc2_pan: float = 0.5
    osc2_octave: float = 0.5
    osc2_semi: float = 0.5
    osc2_fine: float = 0.5

    # --- FM ---
    fm_ratio: float = 0.5         # 0=0.25x, 0.5=1x, 1=8x (mod/carrier 비율)
    fm_depth: float = 0.0         # 0=no FM, 1=max (500Hz deviation)

    # --- Sub / Noise ---
    sub_volume: float = 0.0
    noise_volume: float = 0.0
    noise_color: float = 0.0      # 0=white, 1=pink

    # --- Filter ---
    filter_cutoff: float = 1.0    # 0=20Hz, 1=20kHz (log)
    filter_resonance: float = 0.0 # 0~0.95
    filter_type: float = 0.0      # 0=LP, 0.33=BP, 0.66=HP
    filter_key_track: float = 0.0 # 0=none, 1=full key tracking
    filter_env_amount: float = 0.0  # ENV2 → cutoff (bipolar: 0.5=none, 0=full neg, 1=full pos)
    filter_lfo_amount: float = 0.0  # LFO1 → cutoff depth

    # --- ENV 1 (Amp) ---
    env1_attack: float = 0.01
    env1_decay: float = 0.3
    env1_sustain: float = 0.7
    env1_release: float = 0.3
    env1_lag: float = 0.0         # 0=instant(digital), 0.99=slow(analog)

    # --- ENV 2 (Filter) ---
    env2_attack: float = 0.01
    env2_decay: float = 0.3
    env2_sustain: float = 0.5
    env2_release: float = 0.3
    env2_lag: float = 0.0

    # --- ENV 3 (Mod) ---
    env3_attack: float = 0.01
    env3_decay: float = 0.5
    env3_sustain: float = 0.0
    env3_release: float = 0.5
    env3_lag: float = 0.0
    env3_target: float = 0.0      # 0=pitch, 0.33=osc2vol, 0.66=pan, 1=fm_depth
    env3_amount: float = 0.0

    # --- LFO 1 ---
    lfo1_rate: float = 0.3        # 0=0.1Hz, 1=20Hz (log)
    lfo1_depth: float = 0.0
    lfo1_waveform: float = 0.0    # 0=sin, 0.25=tri, 0.5=saw, 0.75=sqr, 1.0=s&h
    lfo1_target: float = 0.0      # 0=filter_cutoff, 0.33=pitch, 0.66=volume, 1=pan

    # --- LFO 2 ---
    lfo2_rate: float = 0.4
    lfo2_depth: float = 0.0
    lfo2_waveform: float = 0.0
    lfo2_target: float = 0.33     # pitch

    # --- LFO 3 ---
    lfo3_rate: float = 0.2
    lfo3_depth: float = 0.0
    lfo3_waveform: float = 0.25
    lfo3_target: float = 0.66     # volume

    # --- LFO 4 ---
    lfo4_rate: float = 0.5
    lfo4_depth: float = 0.0
    lfo4_waveform: float = 0.5
    lfo4_target: float = 1.0      # pan

    # --- FX ---
    drive: float = 0.0            # 0=clean, 1=heavy saturation
    delay_wet: float = 0.0
    delay_time: float = 0.3       # 0~1초
    delay_feedback: float = 0.3

    # --- Master ---
    master_volume: float = 0.7

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "SynthPatch":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    @classmethod
    def from_json(cls, path: str) -> "SynthPatch":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_vector(cls, vec: list[float]) -> "SynthPatch":
        """158차원 정규화 벡터(preset_schema)에서 주요 파라미터 추출"""
        p = cls()
        if len(vec) < 45:
            return p
        p.osc1_volume = vec[0]; p.osc1_pan = vec[1]; p.osc1_octave = vec[2]
        p.osc1_semi = vec[3]; p.osc1_fine = vec[4]; p.osc1_waveform = vec[7]
        if len(vec) > 16:
            p.osc2_volume = vec[9]; p.osc2_pan = vec[10]; p.osc2_octave = vec[11]
            p.osc2_semi = vec[12]; p.osc2_fine = vec[13]; p.osc2_waveform = vec[16]
        if len(vec) > 50:
            p.filter_cutoff = vec[45]; p.filter_resonance = vec[46]
            p.filter_type = vec[47]; p.filter_env_amount = vec[50]
        if len(vec) > 64:
            p.env1_attack = vec[57]; p.env1_decay = vec[58]
            p.env1_sustain = vec[59]; p.env1_release = vec[60]
        if len(vec) > 72:
            p.env2_attack = vec[65]; p.env2_decay = vec[66]
            p.env2_sustain = vec[67]; p.env2_release = vec[68]
        if len(vec) > 94:
            p.lfo1_rate = vec[89]; p.lfo1_waveform = vec[90]; p.lfo1_depth = vec[92]
        if len(vec) > 153:
            p.master_volume = vec[153]
        return p


# =============================================================================
# 파라미터 변환 유틸
# =============================================================================


def _p2hz(norm: float, lo: float = 20.0, hi: float = 20000.0) -> float:
    """0~1 → Hz (로그)"""
    return lo * (hi / lo) ** max(0.0, min(1.0, norm))


def _p2time(norm: float, lo: float = 0.001, hi: float = 5.0) -> float:
    """0~1 → 초 (로그)"""
    return lo * (hi / lo) ** max(0.0, min(1.0, norm))


def _p2lfo_hz(norm: float) -> float:
    """0~1 → LFO Hz (0.1~20)"""
    return 0.1 * (20.0 / 0.1) ** max(0.0, min(1.0, norm))


def _p2fm_ratio(norm: float) -> float:
    """0~1 → FM ratio (0.25~8)"""
    return 0.25 * (8.0 / 0.25) ** max(0.0, min(1.0, norm))


def _midi2freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _pitch_offset(octave: float, semi: float, fine: float) -> float:
    """정규화된 oct/semi/fine → 세미톤 오프셋"""
    oct_st = (octave - 0.5) * 72    # -3~+3 옥타브 = -36~+36 st
    semi_st = (semi - 0.5) * 24     # -12~+12 st
    fine_st = (fine - 0.5) * 2      # -1~+1 st
    return oct_st + semi_st + fine_st


# =============================================================================
# SignalFlow 그래프 빌더
# =============================================================================


def render_patch(
    patch: SynthPatch,
    note: int = 60,
    velocity: int = 100,
    duration: float = 2.0,
    tail: float = 1.0,
    sample_rate: int = 44100,
    wavetable_data: Optional[np.ndarray] = None,
) -> np.ndarray:
    """SynthPatch → SignalFlow 오디오 그래프 → numpy (2, samples)

    Args:
        wavetable_data: 커스텀 웨이브테이블 (1D numpy, 1주기 2048 샘플)
    """
    import signalflow as sf

    graph = sf.AudioGraph(start=False)
    base_freq = _midi2freq(note)
    vel_gain = velocity / 127.0
    total_samples = int((duration + tail) * sample_rate)

    # Gate 신호: duration 동안 1, 이후 0 (ADSR note-on/off)
    gate_data = np.zeros(total_samples, dtype=np.float32)
    gate_data[:int(duration * sample_rate)] = 1.0
    gate_buf = sf.Buffer(gate_data.tolist())
    gate = sf.BufferPlayer(gate_buf, loop=False)

    # ===== LFO 1~4 =====
    def make_lfo(rate_n, wf_n, depth_n):
        hz = _p2lfo_hz(rate_n)
        if wf_n < 0.2:
            lfo = sf.SineOscillator(hz)
        elif wf_n < 0.4:
            lfo = sf.TriangleOscillator(hz)
        elif wf_n < 0.6:
            lfo = sf.SawOscillator(hz)
        elif wf_n < 0.8:
            lfo = sf.SquareOscillator(hz)
        else:
            lfo = sf.RandomImpulse(hz)
        return lfo * depth_n

    lfo1 = make_lfo(patch.lfo1_rate, patch.lfo1_waveform, patch.lfo1_depth)
    lfo2 = make_lfo(patch.lfo2_rate, patch.lfo2_waveform, patch.lfo2_depth)
    lfo3 = make_lfo(patch.lfo3_rate, patch.lfo3_waveform, patch.lfo3_depth)
    lfo4 = make_lfo(patch.lfo4_rate, patch.lfo4_waveform, patch.lfo4_depth)

    # LFO 라우팅 수집 (타겟별 합산)
    lfo_cutoff = sf.Constant(0)
    lfo_pitch = sf.Constant(0)
    lfo_volume = sf.Constant(0)
    lfo_pan = sf.Constant(0)

    for lfo, target in [(lfo1, patch.lfo1_target), (lfo2, patch.lfo2_target),
                        (lfo3, patch.lfo3_target), (lfo4, patch.lfo4_target)]:
        if target < 0.17:
            lfo_cutoff = lfo_cutoff + lfo
        elif target < 0.5:
            lfo_pitch = lfo_pitch + lfo
        elif target < 0.83:
            lfo_volume = lfo_volume + lfo
        else:
            lfo_pan = lfo_pan + lfo

    # ===== 오실레이터 =====
    def make_va_osc(wf_n: float, freq) -> sf.Node:
        if wf_n < 0.25:
            return sf.SawOscillator(freq)
        elif wf_n < 0.5:
            return sf.SquareOscillator(freq)
        elif wf_n < 0.75:
            return sf.TriangleOscillator(freq)
        else:
            return sf.SineOscillator(freq)

    def make_wt_osc(freq, wt_data: Optional[np.ndarray] = None) -> sf.Node:
        if wt_data is not None:
            buf = sf.Buffer(wt_data.tolist())
        else:
            # 기본 웨이브테이블: 사인 + 하모닉스
            size = 2048
            t = np.linspace(0, 2 * np.pi, size, dtype=np.float32)
            wave = (np.sin(t) + 0.5 * np.sin(2 * t) + 0.3 * np.sin(3 * t)).astype(np.float32)
            wave /= np.max(np.abs(wave))
            buf = sf.Buffer(wave.tolist())
        return sf.Wavetable(buf, freq)

    # OSC1 피치
    osc1_st = _pitch_offset(patch.osc1_octave, patch.osc1_semi, patch.osc1_fine)
    osc1_freq_base = base_freq * (2.0 ** (osc1_st / 12.0))
    # LFO pitch 모듈레이션 (±1 semitone per unit)
    osc1_freq = osc1_freq_base + lfo_pitch * osc1_freq_base * 0.05946  # 1 semitone ratio

    # OSC2 피치
    osc2_st = _pitch_offset(patch.osc2_octave, patch.osc2_semi, patch.osc2_fine)
    osc2_freq_base = base_freq * (2.0 ** (osc2_st / 12.0))
    osc2_freq = osc2_freq_base + lfo_pitch * osc2_freq_base * 0.05946

    # FM 모듈레이션: OSC2 → OSC1 freq
    fm_carrier_freq = osc1_freq
    if patch.fm_depth > 0.01 and patch.osc1_mode > 0.7:
        fm_ratio = _p2fm_ratio(patch.fm_ratio)
        fm_mod_freq = osc1_freq_base * fm_ratio
        fm_deviation = patch.fm_depth * 500.0  # 0~500Hz
        modulator = sf.SineOscillator(fm_mod_freq) * fm_deviation
        fm_carrier_freq = osc1_freq + modulator

    # OSC1 빌드
    if patch.osc1_mode < 0.33:
        osc1 = make_va_osc(patch.osc1_waveform, fm_carrier_freq)
    elif patch.osc1_mode < 0.66:
        osc1 = make_wt_osc(fm_carrier_freq, wavetable_data)
    else:
        osc1 = make_va_osc(patch.osc1_waveform, fm_carrier_freq)  # FM은 위에서 처리됨

    # OSC2 빌드
    if patch.osc2_mode < 0.33:
        osc2 = make_va_osc(patch.osc2_waveform, osc2_freq)
    elif patch.osc2_mode < 0.66:
        osc2 = make_wt_osc(osc2_freq, wavetable_data)
    else:
        osc2 = make_va_osc(patch.osc2_waveform, osc2_freq)

    # Sub + Noise
    sub_osc = sf.SineOscillator(base_freq / 2)
    noise = sf.WhiteNoise() if patch.noise_color < 0.5 else sf.PinkNoise()

    # 믹서
    mix = (osc1 * patch.osc1_volume +
           osc2 * patch.osc2_volume +
           sub_osc * patch.sub_volume +
           noise * patch.noise_volume)

    # ===== ENV 2 (Filter) =====
    env2 = sf.ADSREnvelope(
        attack=_p2time(patch.env2_attack),
        decay=_p2time(patch.env2_decay),
        sustain=patch.env2_sustain,
        release=_p2time(patch.env2_release),
        gate=gate,
    )
    if patch.env2_lag > 0.01:
        env2 = sf.Smooth(env2, patch.env2_lag)

    # ===== 필터 =====
    cutoff_base = _p2hz(patch.filter_cutoff)

    # Key tracking
    kt_offset = 0.0
    if patch.filter_key_track > 0.01:
        kt_offset = (note - 60) * 100.0 * patch.filter_key_track  # Hz per semitone

    # ENV2 모듈레이션 (bipolar: 0.5=none)
    env2_mod_hz = (patch.filter_env_amount - 0.5) * 2.0 * 5000.0  # ±5000Hz

    # 최종 cutoff (Abs로 음수 방지 — SVFilter NaN 방지)
    cutoff_raw = cutoff_base + kt_offset + env2 * env2_mod_hz + lfo_cutoff * 2000.0
    cutoff = sf.Abs(cutoff_raw) + 20.0  # 최소 20Hz

    filter_mode = "low_pass"
    if patch.filter_type > 0.55:
        filter_mode = "high_pass"
    elif patch.filter_type > 0.22:
        filter_mode = "band_pass"

    reso = min(patch.filter_resonance * 0.95, 0.95)
    filtered = sf.SVFilter(mix, filter_mode, cutoff, reso)

    # ===== Drive (waveshaper) =====
    if patch.drive > 0.01:
        drive_amount = 1.0 + patch.drive * 10.0
        filtered = sf.Tanh(filtered * drive_amount) * (1.0 / drive_amount)

    # ===== ENV 1 (Amp) =====
    env1 = sf.ADSREnvelope(
        attack=_p2time(patch.env1_attack),
        decay=_p2time(patch.env1_decay),
        sustain=patch.env1_sustain,
        release=_p2time(patch.env1_release),
        gate=gate,
    )
    if patch.env1_lag > 0.01:
        env1 = sf.Smooth(env1, patch.env1_lag)

    # ===== ENV 3 (Mod) — 지금은 미사용, 확장용 =====
    # env3 타겟 라우팅은 추후 모듈레이션 매트릭스로 확장

    # LFO volume 모듈레이션
    vol_mod = 1.0 + lfo_volume * 0.5  # ±50% volume modulation

    mono_out = filtered * env1 * patch.master_volume * vel_gain * vol_mod

    # ===== 딜레이 =====
    if patch.delay_wet > 0.01:
        delay_s = max(0.01, patch.delay_time)
        delayed = sf.CombDelay(mono_out, delay_s, patch.delay_feedback * 0.85)
        mono_out = mono_out * (1.0 - patch.delay_wet) + delayed * patch.delay_wet

    # ===== 스테레오 패닝 =====
    pan = (patch.osc1_pan - 0.5) * 2.0 + lfo_pan * 0.5
    output = sf.StereoPanner(mono_out, pan)

    # ===== 렌더링 =====
    buf = sf.Buffer(2, total_samples)
    output.play()
    graph.render_to_buffer(buf)

    audio = np.array(buf.data, dtype=np.float32)

    # 페이드아웃
    fade_n = min(int(0.5 * sample_rate), audio.shape[1])
    if fade_n > 0:
        audio[:, -fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)

    return audio


# =============================================================================
# text-to-sound 파이프라인
# =============================================================================


def text_to_sound(
    description: str,
    clap_model=None,
    clap_embeddings: Optional[np.ndarray] = None,
    preset_vectors: Optional[list] = None,
    note: int = 60,
    duration: float = 2.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """자연어 → CLAP 검색 → 가장 가까운 프리셋 → SignalFlow 렌더링"""
    if clap_model is None or clap_embeddings is None or preset_vectors is None:
        logger.warning("CLAP 없음 — 기본 패치")
        return render_patch(SynthPatch(), note=note, duration=duration, sample_rate=sample_rate)

    from numpy.linalg import norm
    text_emb = clap_model.get_text_embedding([description], use_tensor=False)[0]
    sims = np.dot(clap_embeddings, text_emb) / (norm(clap_embeddings, axis=1) * norm(text_emb))
    best_idx = int(np.argmax(sims))
    patch = SynthPatch.from_vector(preset_vectors[best_idx])
    logger.info(f'text_to_sound: "{description}" → #{best_idx} (sim={sims[best_idx]:.3f})')
    return render_patch(patch, note=note, duration=duration, sample_rate=sample_rate)
