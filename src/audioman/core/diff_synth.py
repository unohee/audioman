# Created: 2026-03-25
# Purpose: Differentiable synthesizer in PyTorch — end-to-end gradient optimization
# License: MIT
#
# SignalFlow 신스와 동일한 아키텍처를 PyTorch로 구현.
# 모든 파라미터에 gradient가 흐르므로, 타겟 오디오에 대해
# 직접 gradient descent로 파라미터를 최적화할 수 있음.
#
# 용도:
# 1. 타겟 오디오 → 파라미터 역추론 (gradient-based sound matching)
# 2. 학습 시 render-in-the-loop (렌더링 결과를 직접 비교)
# 3. 텍스트 → CLAP → 파라미터 직접 최적화

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DiffOscillator(nn.Module):
    """Differentiable 오실레이터 — 파형 연속 모핑"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        freq: torch.Tensor,          # (batch,) Hz
        waveform: torch.Tensor,       # (batch,) 0~1
        n_samples: int,
        sample_rate: int = 44100,
    ) -> torch.Tensor:
        """Returns: (batch, n_samples)"""
        batch = freq.shape[0]
        t = torch.arange(n_samples, device=freq.device, dtype=torch.float32) / sample_rate  # (samples,)
        phase = 2 * math.pi * freq.unsqueeze(1) * t.unsqueeze(0)  # (batch, samples)

        # 4개 파형을 연속 보간
        sine = torch.sin(phase)
        # Saw: 하모닉 합성 (aliasing-free)
        saw = torch.zeros_like(phase)
        for h in range(1, 16):
            saw = saw + torch.sin(h * phase) / h
        saw = saw * (2.0 / math.pi)
        # Square: 홀수 하모닉
        square = torch.zeros_like(phase)
        for h in range(1, 16, 2):
            square = square + torch.sin(h * phase) / h
        square = square * (4.0 / math.pi)
        # Triangle
        tri = torch.zeros_like(phase)
        for h in range(0, 8):
            n = 2 * h + 1
            tri = tri + ((-1.0) ** h) * torch.sin(n * phase) / (n * n)
        tri = tri * (8.0 / (math.pi ** 2))

        # 연속 보간: 0=saw, 0.33=square, 0.66=tri, 1.0=sine
        w = waveform.unsqueeze(1)  # (batch, 1)
        # 4개 파형 사이 smooth 보간
        out = (
            saw * torch.clamp(1.0 - w * 3.0, 0, 1) +
            square * torch.clamp(1.0 - torch.abs(w - 0.33) * 3.0, 0, 1) +
            tri * torch.clamp(1.0 - torch.abs(w - 0.66) * 3.0, 0, 1) +
            sine * torch.clamp((w - 0.66) * 3.0, 0, 1)
        )
        return out


class DiffFilter(nn.Module):
    """Differentiable SVF (State Variable Filter) — 1차 근사"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,             # (batch, samples)
        cutoff: torch.Tensor,         # (batch,) 0~1
        resonance: torch.Tensor,      # (batch,) 0~1
        sample_rate: int = 44100,
    ) -> torch.Tensor:
        """IIR 근사 — frequency domain에서 LP 필터"""
        batch, n = x.shape

        # cutoff → Hz → 정규화 주파수
        cutoff_hz = 20.0 * (20000.0 / 20.0) ** cutoff  # (batch,) log scale
        w = cutoff_hz / sample_rate  # (batch,) 0~0.5

        # FFT 기반 필터링 (differentiable)
        X = torch.fft.rfft(x)  # (batch, n//2+1)
        freqs = torch.fft.rfftfreq(n, 1.0 / sample_rate).to(x.device)  # (n//2+1,)

        # 2차 LP 응답: H(f) = 1 / (1 + j*f/(fc*Q))^2
        Q = 0.5 + resonance * 10.0  # Q: 0.5 ~ 10.5
        fc = cutoff_hz.unsqueeze(1)  # (batch, 1)
        Q = Q.unsqueeze(1)

        f_ratio = freqs.unsqueeze(0) / (fc + 1e-6)  # (batch, freqs)
        # Butterworth 2nd order magnitude response (더 안정적)
        H_mag_sq = 1.0 / (1.0 + f_ratio.pow(4))
        # Resonance peak
        H_mag = torch.sqrt(H_mag_sq + 1e-10) * (1.0 + resonance.unsqueeze(1) * f_ratio.pow(2) * 0.5 / (f_ratio.pow(2) + 0.01))
        H_mag = torch.clamp(H_mag, 0, 10)  # 안전 클램프

        Y = X * H_mag
        return torch.fft.irfft(Y, n=n)


class DiffADSR(nn.Module):
    """Differentiable ADSR envelope"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        attack: torch.Tensor,        # (batch,) 0~1
        decay: torch.Tensor,
        sustain: torch.Tensor,
        release: torch.Tensor,
        n_samples: int,
        note_duration: float = 2.0,
        sample_rate: int = 44100,
    ) -> torch.Tensor:
        """Returns: (batch, n_samples)"""
        batch = attack.shape[0]
        device = attack.device

        # 0~1 → 실제 시간 (로그 스케일)
        a_time = 0.001 * (5.0 / 0.001) ** attack  # 1ms ~ 5s
        d_time = 0.001 * (5.0 / 0.001) ** decay
        r_time = 0.001 * (5.0 / 0.001) ** release

        t = torch.arange(n_samples, device=device, dtype=torch.float32) / sample_rate  # (samples,)
        t = t.unsqueeze(0).expand(batch, -1)  # (batch, samples)

        # Attack phase: 0 → 1 (exp rise)
        a_env = 1.0 - torch.exp(-t / (a_time.unsqueeze(1) + 1e-6))

        # Decay phase: 1 → sustain (after attack)
        a_end = a_time.unsqueeze(1)
        t_after_a = torch.clamp(t - a_end, min=0)
        d_env = sustain.unsqueeze(1) + (1.0 - sustain.unsqueeze(1)) * torch.exp(
            -t_after_a / (d_time.unsqueeze(1) + 1e-6)
        )

        # Release phase (after note_duration)
        note_off = torch.tensor(note_duration, device=device)
        t_after_off = torch.clamp(t - note_off, min=0)
        r_env = torch.exp(-t_after_off / (r_time.unsqueeze(1) + 1e-6))

        # 결합: attack → decay/sustain → release
        # attack이 끝나기 전: a_env, 끝난 후: d_env
        env = torch.where(t < a_end, a_env, d_env)
        # note off 이후: release
        env = torch.where(t > note_off, env * r_env, env)

        return env


class DiffSynth(nn.Module):
    """Differentiable subtractive synthesizer

    SignalFlow synth_engine.py와 동일한 구조:
    OSC1 + OSC2 → Filter → Amp(ADSR) → Output
    """

    def __init__(self, sample_rate: int = 44100):
        super().__init__()
        self.sample_rate = sample_rate
        self.osc = DiffOscillator()
        self.filt = DiffFilter()
        self.amp_env = DiffADSR()
        self.filt_env = DiffADSR()

    def forward(
        self,
        params: torch.Tensor,        # (batch, n_params)
        note: int = 60,
        duration: float = 2.0,
        tail: float = 1.0,
    ) -> torch.Tensor:
        """params → audio (batch, n_samples)

        params indices (mapped from SynthPatch):
            0: osc1_waveform     1: osc1_volume
            2: osc2_waveform     3: osc2_volume
            4: osc2_semi         5: sub_volume
            6: noise_volume
            7: filter_cutoff     8: filter_resonance  9: filter_env_amount
            10: env1_attack      11: env1_decay       12: env1_sustain    13: env1_release
            14: env2_attack      15: env2_decay       16: env2_sustain    17: env2_release
            18: drive             19: master_volume
        """
        batch = params.shape[0]
        n_samples = int((duration + tail) * self.sample_rate)
        base_freq = 440.0 * (2.0 ** ((note - 69) / 12.0))

        # 파라미터 추출 (sigmoid로 0~1 보장)
        p = torch.sigmoid(params)

        osc1_wf = p[:, 0]
        osc1_vol = p[:, 1]
        osc2_wf = p[:, 2]
        osc2_vol = p[:, 3]
        osc2_semi = (p[:, 4] - 0.5) * 24  # -12 ~ +12 semitones
        sub_vol = p[:, 5]
        noise_vol = p[:, 6]
        filt_cutoff = p[:, 7]
        filt_reso = p[:, 8]
        filt_env_amt = (p[:, 9] - 0.5) * 2.0  # -1 ~ +1
        env1_a, env1_d, env1_s, env1_r = p[:, 10], p[:, 11], p[:, 12], p[:, 13]
        env2_a, env2_d, env2_s, env2_r = p[:, 14], p[:, 15], p[:, 16], p[:, 17]
        drive = p[:, 18]
        master = p[:, 19]

        # OSC1
        freq1 = torch.full((batch,), base_freq, device=params.device)
        osc1 = self.osc(freq1, osc1_wf, n_samples, self.sample_rate)

        # OSC2 (detuned)
        freq2 = base_freq * (2.0 ** (osc2_semi / 12.0))
        osc2 = self.osc(freq2.unsqueeze(0).expand(batch) if osc2_semi.dim() == 0 else
                         torch.full((batch,), base_freq, device=params.device) * (2.0 ** (osc2_semi / 12.0)),
                         osc2_wf, n_samples, self.sample_rate)

        # Sub (sine, 1oct down)
        sub_freq = torch.full((batch,), base_freq / 2, device=params.device)
        sub = self.osc(sub_freq, torch.ones(batch, device=params.device), n_samples, self.sample_rate)

        # Noise
        noise = torch.randn(batch, n_samples, device=params.device) * 0.3

        # Mix
        mix = (osc1 * osc1_vol.unsqueeze(1) +
               osc2 * osc2_vol.unsqueeze(1) +
               sub * sub_vol.unsqueeze(1) +
               noise * noise_vol.unsqueeze(1))

        # Filter envelope
        filt_env = self.filt_env(env2_a, env2_d, env2_s, env2_r,
                                 n_samples, duration, self.sample_rate)
        # Cutoff modulation
        mod_cutoff = filt_cutoff + filt_env_amt * filt_env.mean(dim=1) * 0.3
        mod_cutoff = torch.clamp(mod_cutoff, 0.01, 0.99)

        # Filter
        filtered = self.filt(mix, mod_cutoff, filt_reso, self.sample_rate)

        # Drive (tanh)
        if True:
            drive_amt = 1.0 + drive.unsqueeze(1) * 10.0
            filtered = torch.tanh(filtered * drive_amt) / drive_amt

        # Amp envelope
        amp = self.amp_env(env1_a, env1_d, env1_s, env1_r,
                           n_samples, duration, self.sample_rate)

        # Output
        out = filtered * amp * master.unsqueeze(1)

        return out


def optimize_params(
    target_audio: np.ndarray,
    note: int = 60,
    duration: float = 2.0,
    n_steps: int = 500,
    lr: float = 0.01,
    sample_rate: int = 44100,
) -> dict:
    """타겟 오디오 → gradient descent로 synth 파라미터 직접 최적화

    Returns: 최적화된 파라미터 dict
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 모노 변환
    if target_audio.ndim == 2:
        target = target_audio.mean(axis=0)
    else:
        target = target_audio

    n_samples = int((duration + 1.0) * sample_rate)
    target = target[:n_samples]
    if len(target) < n_samples:
        target = np.pad(target, (0, n_samples - len(target)))

    target_t = torch.tensor(target, dtype=torch.float32, device=device).unsqueeze(0)  # (1, samples)

    # Synth
    synth = DiffSynth(sample_rate).to(device)

    # 최적화할 파라미터 (20개)
    params = torch.randn(1, 20, device=device) * 0.1  # 초기값 ~sigmoid(0)=0.5 근처
    params.requires_grad_(True)

    optimizer = torch.optim.Adam([params], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # 타겟 mel spectrogram (주파수 도메인 손실)
    target_spec = torch.stft(target_t.squeeze(0), n_fft=2048, hop_length=512,
                              return_complex=True, window=torch.hann_window(2048, device=device))
    target_mag = torch.abs(target_spec)

    best_loss = float('inf')
    best_params = None

    for step in range(n_steps):
        optimizer.zero_grad()
        audio = synth(params, note=note, duration=duration)

        # 멀티 스케일 손실
        # 1. 시간 도메인 MSE
        time_loss = F.mse_loss(audio, target_t)

        # 2. 스펙트로그램 손실 (log-magnitude)
        pred_spec = torch.stft(audio.squeeze(0), n_fft=2048, hop_length=512,
                                return_complex=True, window=torch.hann_window(2048, device=device))
        pred_mag = torch.abs(pred_spec)
        spec_loss = F.mse_loss(torch.log(pred_mag + 1e-6), torch.log(target_mag + 1e-6))

        # 3. Envelope 손실 (RMS)
        hop = 512
        target_rms = target_t.unfold(1, hop, hop).pow(2).mean(-1).sqrt()
        pred_rms = audio.unfold(1, hop, hop).pow(2).mean(-1).sqrt()
        min_len = min(target_rms.shape[1], pred_rms.shape[1])
        env_loss = F.mse_loss(pred_rms[:, :min_len], target_rms[:, :min_len])

        loss = time_loss + spec_loss * 2.0 + env_loss * 3.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_([params], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = torch.sigmoid(params).detach().cpu().numpy()[0]

        if (step + 1) % 50 == 0:
            print(f'  step {step+1}/{n_steps}: loss={loss.item():.5f} '
                  f'(time={time_loss.item():.5f} spec={spec_loss.item():.5f} env={env_loss.item():.5f})',
                  flush=True)

    # 파라미터 이름 매핑
    param_names = [
        'osc1_waveform', 'osc1_volume', 'osc2_waveform', 'osc2_volume',
        'osc2_semi', 'sub_volume', 'noise_volume',
        'filter_cutoff', 'filter_resonance', 'filter_env_amount',
        'env1_attack', 'env1_decay', 'env1_sustain', 'env1_release',
        'env2_attack', 'env2_decay', 'env2_sustain', 'env2_release',
        'drive', 'master_volume',
    ]

    return {name: float(best_params[i]) for i, name in enumerate(param_names)}
