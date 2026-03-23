# Created: 2026-03-23
# Purpose: 범용 신디사이저 프리셋 파라미터 정규화 스키마 (pydantic)
# 모든 신스를 공통 구조로 매핑하여 크로스-플러그인 비교/임베딩 가능

import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# 정규화된 파라미터 그룹
# =============================================================================


class OscillatorParams(BaseModel):
    """오실레이터 파라미터 (최대 3개 OSC + Sub + Noise)"""

    volume: float = 0.0
    pan: float = 0.5
    octave: float = 0.5
    semi: float = 0.5
    fine: float = 0.5
    unison_voices: float = 0.0
    unison_detune: float = 0.0
    waveform: float = 0.0  # 웨이브테이블 위치 또는 파형 선택
    phase: float = 0.5


class FilterParams(BaseModel):
    """필터 파라미터"""

    cutoff: float = 1.0
    resonance: float = 0.0
    filter_type: float = 0.0  # LP/HP/BP/Notch 등
    drive: float = 0.0
    key_tracking: float = 0.0
    env_amount: float = 0.0


class EnvelopeParams(BaseModel):
    """ADSR 엔벨로프"""

    attack: float = 0.0
    decay: float = 0.5
    sustain: float = 1.0
    release: float = 0.2
    hold: float = 0.0
    attack_curve: float = 0.5
    decay_curve: float = 0.5
    release_curve: float = 0.5


class LFOParams(BaseModel):
    """LFO 파라미터"""

    rate: float = 0.5
    waveform: float = 0.0
    sync: float = 0.0  # free/sync
    depth: float = 0.0
    phase: float = 0.0
    delay: float = 0.0


class FXParams(BaseModel):
    """이펙트 파라미터 (on/off + wet + 주요 파라미터)"""

    enabled: float = 0.0
    wet: float = 0.0
    param_1: float = 0.0
    param_2: float = 0.0
    param_3: float = 0.0


class ModulationParams(BaseModel):
    """모듈레이션 라우팅"""

    amount: float = 0.0
    source: float = 0.0
    destination: float = 0.0


class MasterParams(BaseModel):
    """마스터/글로벌 파라미터"""

    volume: float = 0.7
    tune: float = 0.5
    portamento: float = 0.0
    voices: float = 0.0  # polyphony
    pitch_bend: float = 0.0


class NormalizedPreset(BaseModel):
    """정규화된 신디사이저 프리셋 — 모든 플러그인 공통 구조

    모든 값은 0.0 ~ 1.0 정규화. 플러그인별 고유 파라미터는 extra에 저장.
    """

    # 메타데이터
    plugin: str = ""
    preset_name: str = ""
    preset_path: str = ""

    # 오실레이터 (최대 3개)
    osc_1: OscillatorParams = Field(default_factory=OscillatorParams)
    osc_2: OscillatorParams = Field(default_factory=OscillatorParams)
    osc_3: OscillatorParams = Field(default_factory=OscillatorParams)
    sub: OscillatorParams = Field(default_factory=OscillatorParams)
    noise: OscillatorParams = Field(default_factory=OscillatorParams)

    # 필터 (최대 2개)
    filter_1: FilterParams = Field(default_factory=FilterParams)
    filter_2: FilterParams = Field(default_factory=FilterParams)

    # 엔벨로프 (최대 4개: amp, filter, mod1, mod2)
    env_amp: EnvelopeParams = Field(default_factory=EnvelopeParams)
    env_filter: EnvelopeParams = Field(default_factory=EnvelopeParams)
    env_mod_1: EnvelopeParams = Field(default_factory=EnvelopeParams)
    env_mod_2: EnvelopeParams = Field(default_factory=EnvelopeParams)

    # LFO (최대 4개)
    lfo_1: LFOParams = Field(default_factory=LFOParams)
    lfo_2: LFOParams = Field(default_factory=LFOParams)
    lfo_3: LFOParams = Field(default_factory=LFOParams)
    lfo_4: LFOParams = Field(default_factory=LFOParams)

    # 이펙트
    fx_chorus: FXParams = Field(default_factory=FXParams)
    fx_delay: FXParams = Field(default_factory=FXParams)
    fx_reverb: FXParams = Field(default_factory=FXParams)
    fx_distortion: FXParams = Field(default_factory=FXParams)
    fx_phaser: FXParams = Field(default_factory=FXParams)
    fx_flanger: FXParams = Field(default_factory=FXParams)
    fx_compressor: FXParams = Field(default_factory=FXParams)
    fx_eq: FXParams = Field(default_factory=FXParams)

    # 마스터
    master: MasterParams = Field(default_factory=MasterParams)

    # 모듈레이션 (최대 8개 슬롯)
    mod_slots: list[ModulationParams] = Field(default_factory=list)

    # 플러그인 고유 파라미터 (정규화 매핑 안 된 것)
    extra: dict[str, float] = Field(default_factory=dict)

    # 원본 파라미터 수 / 매핑 성공률
    original_param_count: int = 0
    mapped_param_count: int = 0

    def to_vector(self) -> list[float]:
        """정규화된 파라미터를 고정 길이 벡터로 변환 (임베딩용)

        구조: [osc1×9, osc2×9, osc3×9, sub×9, noise×9,
               fil1×6, fil2×6,
               env_amp×8, env_fil×8, env_m1×8, env_m2×8,
               lfo1×6, lfo2×6, lfo3×6, lfo4×6,
               fx×8×5, master×5]
        = 9*5 + 6*2 + 8*4 + 6*4 + 5*8 + 5 = 45+12+32+24+40+5 = 158 dims
        """
        vec = []

        for osc in [self.osc_1, self.osc_2, self.osc_3, self.sub, self.noise]:
            vec.extend([osc.volume, osc.pan, osc.octave, osc.semi, osc.fine,
                        osc.unison_voices, osc.unison_detune, osc.waveform, osc.phase])

        for fil in [self.filter_1, self.filter_2]:
            vec.extend([fil.cutoff, fil.resonance, fil.filter_type, fil.drive,
                        fil.key_tracking, fil.env_amount])

        for env in [self.env_amp, self.env_filter, self.env_mod_1, self.env_mod_2]:
            vec.extend([env.attack, env.decay, env.sustain, env.release,
                        env.hold, env.attack_curve, env.decay_curve, env.release_curve])

        for lfo in [self.lfo_1, self.lfo_2, self.lfo_3, self.lfo_4]:
            vec.extend([lfo.rate, lfo.waveform, lfo.sync, lfo.depth, lfo.phase, lfo.delay])

        for fx in [self.fx_chorus, self.fx_delay, self.fx_reverb, self.fx_distortion,
                   self.fx_phaser, self.fx_flanger, self.fx_compressor, self.fx_eq]:
            vec.extend([fx.enabled, fx.wet, fx.param_1, fx.param_2, fx.param_3])

        vec.extend([self.master.volume, self.master.tune, self.master.portamento,
                    self.master.voices, self.master.pitch_bend])

        return vec

    @staticmethod
    def vector_dim() -> int:
        return 158


# =============================================================================
# 플러그인별 매핑 규칙
# =============================================================================

# 매핑 규칙: (원본 파라미터 이름 패턴, 정규화 필드 경로)
# 패턴은 정규식, 대소문자 무시

SERUM_MAPPING: list[tuple[str, str]] = [
    # Master
    (r"^MasterVol$", "master.volume"),
    (r"^Mast\.Tun$", "master.tune"),
    (r"^PortTime$", "master.portamento"),
    (r"^Pitch Bend$", "master.pitch_bend"),
    # OSC A → osc_1
    (r"^A Vol$", "osc_1.volume"),
    (r"^A Pan$", "osc_1.pan"),
    (r"^A Octave$", "osc_1.octave"),
    (r"^A Semi$", "osc_1.semi"),
    (r"^A Fine$", "osc_1.fine"),
    (r"^A Unison$", "osc_1.unison_voices"),
    (r"^A UniDet$", "osc_1.unison_detune"),
    (r"^A WTPos$", "osc_1.waveform"),
    (r"^A Phase$", "osc_1.phase"),
    # OSC B → osc_2
    (r"^B Vol$", "osc_2.volume"),
    (r"^B Pan$", "osc_2.pan"),
    (r"^B Octave$", "osc_2.octave"),
    (r"^B Semi$", "osc_2.semi"),
    (r"^B Fine$", "osc_2.fine"),
    (r"^B Unison$", "osc_2.unison_voices"),
    (r"^B UniDet$", "osc_2.unison_detune"),
    (r"^B WTPos$", "osc_2.waveform"),
    (r"^B Phase$", "osc_2.phase"),
    # Sub/Noise
    (r"^Sub Osc Level$", "sub.volume"),
    (r"^Sub Osc Pan$", "sub.pan"),
    (r"^Noise Level$", "noise.volume"),
    (r"^Noise Pitch$", "noise.pan"),  # pitch → pan 슬롯 재사용
    # Filter
    (r"^Fil Cutoff$", "filter_1.cutoff"),
    (r"^Fil Reso$", "filter_1.resonance"),
    (r"^Fil Type$", "filter_1.filter_type"),
    # Envelopes (Env1 = amp, Env2 = filter, Env3 = mod)
    (r"^Env1 Atk$", "env_amp.attack"),
    (r"^Env1 Dec$", "env_amp.decay"),
    (r"^Env1 Sus$", "env_amp.sustain"),
    (r"^Env1 Rel$", "env_amp.release"),
    (r"^Env1 Hold$", "env_amp.hold"),
    (r"^Env2 Atk$", "env_filter.attack"),
    (r"^Env2 Dec$", "env_filter.decay"),
    (r"^Env2 Sus$", "env_filter.sustain"),
    (r"^Env2 Rel$", "env_filter.release"),
    (r"^Env3 Atk$", "env_mod_1.attack"),
    (r"^Env3 Dec$", "env_mod_1.decay"),
    (r"^Env3 Sus$", "env_mod_1.sustain"),
    (r"^Env3 Rel$", "env_mod_1.release"),
    # LFOs
    (r"^LFO1 Rate$", "lfo_1.rate"),
    (r"^LFO2 Rate$", "lfo_2.rate"),
    (r"^LFO3 Rate$", "lfo_3.rate"),
    (r"^LFO4 Rate$", "lfo_4.rate"),
    # FX
    (r"^Cho_Wet$", "fx_chorus.wet"),
    (r"^Dly_Wet$", "fx_delay.wet"),
    (r"^Verb Wet$", "fx_reverb.wet"),
    (r"^Dist_Wet$", "fx_distortion.wet"),
    (r"^Dist_Drv$", "fx_distortion.param_1"),
    (r"^Phs_Wet$", "fx_phaser.wet"),
    (r"^Flg_Wet$", "fx_flanger.wet"),
    (r"^Cmp_Thr$", "fx_compressor.param_1"),
]

VITAL_MAPPING: list[tuple[str, str]] = [
    # Master
    (r"^volume$", "master.volume"),
    (r"^voice_tune$", "master.tune"),
    (r"^polyphony$", "master.voices"),
    # OSC 1/2/3
    (r"^osc_1_level$", "osc_1.volume"),
    (r"^osc_1_pan$", "osc_1.pan"),
    (r"^osc_1_transpose$", "osc_1.octave"),
    (r"^osc_1_tune$", "osc_1.fine"),
    (r"^osc_1_unison_voices$", "osc_1.unison_voices"),
    (r"^osc_1_unison_detune$", "osc_1.unison_detune"),
    (r"^osc_1_frame$", "osc_1.waveform"),
    (r"^osc_1_phase$", "osc_1.phase"),
    (r"^osc_2_level$", "osc_2.volume"),
    (r"^osc_2_pan$", "osc_2.pan"),
    (r"^osc_2_transpose$", "osc_2.octave"),
    (r"^osc_2_tune$", "osc_2.fine"),
    (r"^osc_2_unison_voices$", "osc_2.unison_voices"),
    (r"^osc_2_unison_detune$", "osc_2.unison_detune"),
    (r"^osc_2_frame$", "osc_2.waveform"),
    (r"^osc_3_level$", "osc_3.volume"),
    (r"^osc_3_pan$", "osc_3.pan"),
    (r"^osc_3_transpose$", "osc_3.octave"),
    (r"^sample_level$", "noise.volume"),
    # Filter
    (r"^filter_1_cutoff$", "filter_1.cutoff"),
    (r"^filter_1_resonance$", "filter_1.resonance"),
    (r"^filter_1_drive$", "filter_1.drive"),
    (r"^filter_2_cutoff$", "filter_2.cutoff"),
    (r"^filter_2_resonance$", "filter_2.resonance"),
    # Envelopes
    (r"^env_1_attack$", "env_amp.attack"),
    (r"^env_1_decay$", "env_amp.decay"),
    (r"^env_1_sustain$", "env_amp.sustain"),
    (r"^env_1_release$", "env_amp.release"),
    (r"^env_2_attack$", "env_filter.attack"),
    (r"^env_2_decay$", "env_filter.decay"),
    (r"^env_2_sustain$", "env_filter.sustain"),
    (r"^env_2_release$", "env_filter.release"),
    (r"^env_3_attack$", "env_mod_1.attack"),
    (r"^env_3_decay$", "env_mod_1.decay"),
    (r"^env_3_sustain$", "env_mod_1.sustain"),
    (r"^env_3_release$", "env_mod_1.release"),
    # LFOs
    (r"^lfo_1_frequency$", "lfo_1.rate"),
    (r"^lfo_2_frequency$", "lfo_2.rate"),
    (r"^lfo_3_frequency$", "lfo_3.rate"),
    (r"^lfo_4_frequency$", "lfo_4.rate"),
    # FX
    (r"^chorus_dry_wet$", "fx_chorus.wet"),
    (r"^chorus_on$", "fx_chorus.enabled"),
    (r"^delay_dry_wet$", "fx_delay.wet"),
    (r"^delay_on$", "fx_delay.enabled"),
    (r"^reverb_dry_wet$", "fx_reverb.wet"),
    (r"^reverb_on$", "fx_reverb.enabled"),
    (r"^distortion_drive$", "fx_distortion.param_1"),
    (r"^distortion_on$", "fx_distortion.enabled"),
    (r"^phaser_dry_wet$", "fx_phaser.wet"),
    (r"^flanger_dry_wet$", "fx_flanger.wet"),
    (r"^compressor_on$", "fx_compressor.enabled"),
]

DIVA_MAPPING: list[tuple[str, str]] = [
    # Master
    (r"^main\.CcOp$", "master.volume"),
    (r"^VCC\.Voices$", "master.voices"),
    (r"^VCC\.Porta$", "master.portamento"),
    (r"^VCC\.Trsp$", "master.tune"),
    # OSC
    (r"^OSC\.Tune1$", "osc_1.octave"),
    (r"^OSC\.Tune2$", "osc_2.octave"),
    (r"^OSC\.Tune3$", "osc_3.octave"),
    (r"^OSC\.Vol1$", "osc_1.volume"),
    (r"^OSC\.Vol2$", "osc_2.volume"),
    (r"^OSC\.Vol3$", "osc_3.volume"),
    (r"^OSC\.Nois$", "noise.volume"),
    # Filter
    (r"^VCF1\.Freq$", "filter_1.cutoff"),
    (r"^VCF1\.Res$", "filter_1.resonance"),
    (r"^HPF\.Freq$", "filter_2.cutoff"),
    (r"^HPF\.Res$", "filter_2.resonance"),
    # Envelope
    (r"^ENV1\.Atk$", "env_amp.attack"),
    (r"^ENV1\.Dec$", "env_amp.decay"),
    (r"^ENV1\.Sus$", "env_amp.sustain"),
    (r"^ENV1\.Rel$", "env_amp.release"),
    (r"^ENV2\.Atk$", "env_filter.attack"),
    (r"^ENV2\.Dec$", "env_filter.decay"),
    (r"^ENV2\.Sus$", "env_filter.sustain"),
    (r"^ENV2\.Rel$", "env_filter.release"),
    # LFO
    (r"^LFO1\.Rate$", "lfo_1.rate"),
    (r"^LFO1\.Wave$", "lfo_1.waveform"),
    (r"^LFO2\.Rate$", "lfo_2.rate"),
    (r"^LFO2\.Wave$", "lfo_2.waveform"),
    # FX
    (r"^VCA1\.Vol$", "master.volume"),
    (r"^VCA1\.Pan$", "osc_1.pan"),
]

# 플러그인 ID → 매핑 테이블
PLUGIN_MAPPINGS: dict[str, list[tuple[str, str]]] = {
    "0x58667358": SERUM_MAPPING,  # Xfer Serum
    "Vital": VITAL_MAPPING,
    "Diva": DIVA_MAPPING,
    "Bazille": DIVA_MAPPING,  # Bazille도 u-he 섹션 구조 (유사)
}


def normalize_preset(
    parameters: dict[str, Any],
    plugin_id: str,
    preset_name: str = "",
    preset_path: str = "",
) -> NormalizedPreset:
    """플러그인 고유 파라미터를 정규화된 공통 구조로 변환"""

    mapping = PLUGIN_MAPPINGS.get(plugin_id, [])
    result = NormalizedPreset(
        plugin=plugin_id,
        preset_name=preset_name,
        preset_path=preset_path,
        original_param_count=len(parameters),
    )

    mapped_keys = set()

    for param_name, value in parameters.items():
        if not isinstance(value, (int, float)):
            continue

        matched = False
        for pattern, field_path in mapping:
            if re.match(pattern, param_name, re.IGNORECASE):
                _set_nested(result, field_path, float(value))
                mapped_keys.add(param_name)
                matched = True
                break

        if not matched:
            result.extra[param_name] = float(value)

    result.mapped_param_count = len(mapped_keys)
    return result


def _set_nested(obj: BaseModel, path: str, value: float) -> None:
    """중첩 필드 설정: "osc_1.volume" → obj.osc_1.volume = value"""
    parts = path.split(".")
    current = obj
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)
