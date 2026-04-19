# Created: 2026-04-19
# Purpose: GPU-accelerated spectral snapshot extraction (MPS/CUDA)
# Dependencies: torch, torchaudio, nnAudio (optional), numpy

"""GPU 배치 스펙트럼 스냅샷 추출기.

설계 원칙:
  - 의존성 최소화 (torch + torchaudio만 필수, nnAudio는 optional)
  - MPS/CUDA/CPU 모두 지원
  - 곡 단위가 아닌 배치 단위 처리로 GPU 활용률 극대화
  - loud / quiet 두 지점 스냅샷 + 4/10밴드 K-weighted RMS + MFCC13

automix 호환:
  - 4밴드 정의가 DEFAULT_BANDS (sub/low/mid/high)와 동일
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchaudio.transforms as T

try:
    from nnAudio import features as _nnaudio_features
    _HAS_NNAUDIO = True
except ImportError:
    _HAS_NNAUDIO = False


# ────────────────────────────────────────────────────────
# 밴드 정의 (automix.py와 호환)
# ────────────────────────────────────────────────────────

BANDS_4 = [(20.0, 200.0), (200.0, 800.0), (800.0, 4000.0), (4000.0, 20000.0)]

# 옥타브 10밴드 (31.5Hz ~ 16kHz 중심)
BANDS_10 = [
    (22.0, 44.0), (44.0, 88.0), (88.0, 177.0), (177.0, 354.0),
    (354.0, 707.0), (707.0, 1414.0), (1414.0, 2828.0), (2828.0, 5657.0),
    (5657.0, 11314.0), (11314.0, 20000.0),
]


# ────────────────────────────────────────────────────────
# K-weighting (ITU-R BS.1770) — GPU tensor 버전
# ────────────────────────────────────────────────────────

def k_weight_magnitude_tensor(freqs: torch.Tensor, sr: float = 48000.0) -> torch.Tensor:
    """K-weighting magnitude, torch 구현 (any device)."""
    b1 = torch.tensor([1.53512485958697, -2.69169618940638, 1.19839281085285],
                      dtype=freqs.dtype, device=freqs.device)
    a1 = torch.tensor([1.0, -1.69065929318241, 0.73248077421585],
                      dtype=freqs.dtype, device=freqs.device)
    b2 = torch.tensor([1.0, -2.0, 1.0], dtype=freqs.dtype, device=freqs.device)
    a2 = torch.tensor([1.0, -1.99004745483398, 0.99007225036621],
                      dtype=freqs.dtype, device=freqs.device)

    w = 2.0 * torch.pi * torch.minimum(freqs, torch.tensor(sr / 2 - 1, device=freqs.device)) / sr

    def biquad(b, a, w):
        ejw = torch.exp(-1j * w)
        ejw2 = torch.exp(-2j * w)
        num = b[0] + b[1] * ejw + b[2] * ejw2
        den = a[0] + a[1] * ejw + a[2] * ejw2
        return num / den

    k = torch.abs(biquad(b1, a1, w) * biquad(b2, a2, w))
    k[freqs == 0] = 0.0
    return k.to(torch.float32)


# ────────────────────────────────────────────────────────
# 결과 컨테이너
# ────────────────────────────────────────────────────────

@dataclass
class Snapshot:
    bands4_db: list[float]      # 4밴드 K-weighted RMS dBFS
    bands10_db: list[float]     # 10밴드 옥타브 K-weighted RMS
    mfcc13: list[float]         # MFCC 13 계수
    frame_rms_db: float         # 이 스냅샷 프레임의 broadband RMS


@dataclass
class SnapshotPair:
    loud: Optional[Snapshot]
    quiet: Optional[Snapshot]
    duration_sec: float
    peak_dbfs: float
    n_valid_frames: int          # 무음 제외 프레임 수


# ────────────────────────────────────────────────────────
# GPU Spectral Extractor
# ────────────────────────────────────────────────────────

class GPUSpectralExtractor:
    """배치 오디오에서 loud/quiet 스냅샷 추출.

    Args:
        sr: 샘플레이트 (22050 권장)
        device: 'mps' / 'cuda' / 'cpu' (None이면 자동 감지)
        n_fft: STFT 윈도우 크기
        hop_length: STFT hop
        frame_sec: 스냅샷 단위 프레임 길이 (초)
        silence_db: 무음 판정 임계 (frame RMS dBFS)
        use_nnaudio: nnAudio 사용 (설치되어 있고 mel 경로를 쓸 때)
    """

    def __init__(
        self,
        sr: int = 22050,
        device: Optional[str] = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        frame_sec: float = 2.0,
        silence_db: float = -50.0,
        n_mels: int = 128,
        n_mfcc: int = 13,
        use_nnaudio: bool = True,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_sec = frame_sec
        self.silence_db = silence_db
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # STFT (power spectrum)
        self.spec = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_length, power=2.0,
        ).to(self.device)

        # Mel spectrogram — nnAudio 우선, 없으면 torchaudio
        self._use_nnaudio = use_nnaudio and _HAS_NNAUDIO
        if self._use_nnaudio:
            self.mel = _nnaudio_features.MelSpectrogram(
                sr=sr, n_fft=n_fft, hop_length=hop_length,
                n_mels=n_mels, verbose=False,
            ).to(self.device)
        else:
            self.mel = T.MelSpectrogram(
                sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
                n_mels=n_mels, power=2.0,
            ).to(self.device)

        # DCT matrix for MFCC (type-II, orthonormal)
        self._dct = self._make_dct_matrix(n_mels, n_mfcc).to(self.device)

        # 주파수 bins 및 밴드 마스크 (power spectrum frequency축용)
        freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1, device=self.device)
        self._freqs = freqs
        self._k_mag = k_weight_magnitude_tensor(freqs, sr=sr)
        self._k_mag_sq = self._k_mag ** 2  # power domain

        self._band4_masks = self._build_band_masks(freqs, BANDS_4)
        self._band10_masks = self._build_band_masks(freqs, BANDS_10)

        # 프레임 윈도우 크기 (STFT 프레임 기준)
        self.frames_per_window = max(1, int(round(frame_sec * sr / hop_length)))

    @staticmethod
    def _make_dct_matrix(n_mels: int, n_mfcc: int) -> torch.Tensor:
        """Orthonormal DCT-II matrix (n_mfcc x n_mels)."""
        n = torch.arange(n_mels, dtype=torch.float32)
        k = torch.arange(n_mfcc, dtype=torch.float32).unsqueeze(1)
        d = torch.cos(torch.pi * k * (2 * n + 1) / (2 * n_mels))
        d[0] *= 1.0 / np.sqrt(n_mels)
        d[1:] *= np.sqrt(2.0 / n_mels)
        return d

    def _build_band_masks(self, freqs: torch.Tensor, bands: list[tuple[float, float]]) -> torch.Tensor:
        """(n_bands, n_freq_bins) boolean mask."""
        masks = torch.zeros((len(bands), freqs.shape[0]), dtype=torch.bool, device=freqs.device)
        for i, (lo, hi) in enumerate(bands):
            masks[i] = (freqs >= lo) & (freqs < hi)
        return masks

    @torch.no_grad()
    def extract_batch(self, audio_batch: list[np.ndarray]) -> list[SnapshotPair]:
        """배치로 스냅샷 추출.

        Args:
            audio_batch: list of 1D mono float32 numpy arrays (샘플레이트=self.sr 가정)

        Returns:
            list of SnapshotPair (배치 순서 유지)
        """
        if not audio_batch:
            return []

        B = len(audio_batch)
        # 최대 길이로 zero-pad
        max_len = max(len(a) for a in audio_batch)
        batch = torch.zeros((B, max_len), dtype=torch.float32)
        lengths = torch.zeros(B, dtype=torch.long)
        for i, a in enumerate(audio_batch):
            n = len(a)
            batch[i, :n] = torch.from_numpy(a)
            lengths[i] = n

        batch = batch.to(self.device)

        # STFT power spectrum: (B, F, T_frames)
        power = self.spec(batch)

        # K-weighting 적용한 power (평가용)
        k_power = power * self._k_mag_sq[None, :, None]

        # Frame RMS (broadband, K-weighted) → (B, T_frames)
        # frame power = sum over freq
        frame_power = k_power.sum(dim=1)  # (B, T_frames)
        frame_rms = torch.sqrt(torch.clamp(frame_power / max(1, self.n_fft // 2 + 1), min=1e-20))
        frame_rms_db = 20.0 * torch.log10(torch.clamp(frame_rms, min=1e-10))

        # 유효 프레임 마스크 (zero-pad된 영역 제외)
        # sample i의 실제 프레임 개수
        valid_frames_per_sample = torch.ceil(
            (lengths.to(self.device).float() - self.n_fft) / self.hop_length + 1
        ).clamp(min=0).long()  # (B,)

        T_frames = frame_rms_db.shape[1]
        frame_idx = torch.arange(T_frames, device=self.device).unsqueeze(0)  # (1, T)
        valid_mask = frame_idx < valid_frames_per_sample.unsqueeze(1)  # (B, T)

        # 무음 마스크
        non_silent = (frame_rms_db >= self.silence_db) & valid_mask  # (B, T)

        # Window 단위로 프레임 그룹화 (frames_per_window 연속)
        # window_rms[i, w] = 윈도우 내 평균 power RMS
        # 간단화: stride=frames_per_window, 비겹침
        fpw = self.frames_per_window
        n_windows = T_frames // fpw
        if n_windows < 1:
            # 너무 짧은 오디오 — 빈 결과
            return [SnapshotPair(None, None, float(l / self.sr), -120.0, 0)
                    for l in lengths.cpu().tolist()]

        # (B, n_windows, fpw)
        trimmed = T_frames - (T_frames % fpw)
        win_power = frame_power[:, :trimmed].reshape(B, n_windows, fpw).mean(dim=2)  # (B, W)
        win_mask = non_silent[:, :trimmed].reshape(B, n_windows, fpw).float().mean(dim=2)  # (B, W)
        # 윈도우 내 50% 이상이 유효해야 채택
        win_valid = win_mask >= 0.5  # (B, W)
        win_rms_db = 10.0 * torch.log10(torch.clamp(win_power, min=1e-20))

        # 결과 수집
        results: list[SnapshotPair] = []

        # 배치 전체의 loud/quiet 프레임 인덱스 수집 후 일괄 Mel/MFCC 계산
        loud_frame_indices: list[tuple[int, int]] = []   # (batch_idx, start_frame)
        quiet_frame_indices: list[tuple[int, int]] = []
        per_sample_meta = []  # 재조립용

        for i in range(B):
            valid_wins = win_valid[i]
            if not valid_wins.any():
                per_sample_meta.append(None)
                continue

            rms_i = win_rms_db[i].clone()
            rms_i[~valid_wins] = float("nan")

            # 상위 10% / 하위 10% 중 각각 하나씩 랜덤
            valid_vals = rms_i[valid_wins]
            n_valid = int(valid_vals.numel())
            if n_valid == 0:
                per_sample_meta.append(None)
                continue

            # 정렬
            sorted_vals, _ = torch.sort(valid_vals)
            top_n = max(1, n_valid // 10)
            bot_n = max(1, n_valid // 10)
            loud_threshold = sorted_vals[-top_n]
            quiet_threshold = sorted_vals[bot_n - 1]

            loud_candidates = torch.nonzero(valid_wins & (win_rms_db[i] >= loud_threshold), as_tuple=False).flatten()
            quiet_candidates = torch.nonzero(valid_wins & (win_rms_db[i] <= quiet_threshold), as_tuple=False).flatten()

            if loud_candidates.numel() == 0 or quiet_candidates.numel() == 0:
                per_sample_meta.append(None)
                continue

            # 랜덤 선택 (torch.randperm 사용, device=CPU로 충분)
            loud_pick = loud_candidates[torch.randint(0, loud_candidates.numel(), (1,)).item()].item()
            quiet_pick = quiet_candidates[torch.randint(0, quiet_candidates.numel(), (1,)).item()].item()

            # 윈도우 중앙 프레임 → 대표 프레임 1개로
            loud_frame = loud_pick * fpw + fpw // 2
            quiet_frame = quiet_pick * fpw + fpw // 2

            loud_frame_indices.append((i, loud_frame))
            quiet_frame_indices.append((i, quiet_frame))

            per_sample_meta.append({
                "loud_frame": loud_frame,
                "quiet_frame": quiet_frame,
                "duration_sec": float(lengths[i].item() / self.sr),
                "peak_dbfs": float(20.0 * torch.log10(torch.clamp(batch[i, :lengths[i]].abs().max(), min=1e-10)).item()),
                "n_valid_frames": int(non_silent[i].sum().item()),
                "loud_rms_db": float(win_rms_db[i, loud_pick].item()),
                "quiet_rms_db": float(win_rms_db[i, quiet_pick].item()),
            })

        # 선택된 프레임들에 대해 Mel/MFCC/밴드 계산
        # 프레임 단위 power (1 frame의 주파수 분포): power[i, :, t]
        def extract_snapshot(i: int, t: int) -> Snapshot:
            # 프레임 주변 1 윈도우 전체의 평균 power 사용 (단일 frame은 노이즈 심함)
            half = max(1, fpw // 2)
            t0 = max(0, t - half)
            t1 = min(power.shape[2], t + half)
            frame_pow = power[i, :, t0:t1].mean(dim=1)  # (F,)

            # K-weighted
            k_frame_pow = frame_pow * self._k_mag_sq

            # 밴드 RMS
            def band_rms_db(masks: torch.Tensor) -> list[float]:
                # masks: (n_bands, F)
                band_pow = (k_frame_pow.unsqueeze(0) * masks.float()).sum(dim=1)  # (n_bands,)
                # normalize by bins in band
                bins = masks.float().sum(dim=1).clamp(min=1.0)
                band_mean = band_pow / bins
                return (10.0 * torch.log10(torch.clamp(band_mean, min=1e-20))).cpu().tolist()

            bands4 = band_rms_db(self._band4_masks)
            bands10 = band_rms_db(self._band10_masks)

            # Mel → log → MFCC: 윈도우 mean을 mel로 변환하려면 time-domain 필요
            # → 윈도우의 오디오 세그먼트를 mel transform
            start_sample = t0 * self.hop_length
            end_sample = min(lengths[i].item(), t1 * self.hop_length + self.n_fft)
            seg = batch[i, start_sample:end_sample].unsqueeze(0)  # (1, T)
            mel_s = self.mel(seg)  # (1, n_mels, t')
            mel_mean = mel_s.mean(dim=-1).squeeze(0)  # (n_mels,)
            log_mel = torch.log(torch.clamp(mel_mean, min=1e-10))
            mfcc = (self._dct @ log_mel).cpu().tolist()

            frame_rms_broadband = 10.0 * torch.log10(torch.clamp(frame_pow.sum() / frame_pow.numel(), min=1e-20))
            return Snapshot(
                bands4_db=[round(float(x), 2) for x in bands4],
                bands10_db=[round(float(x), 2) for x in bands10],
                mfcc13=[round(float(x), 3) for x in mfcc],
                frame_rms_db=round(float(frame_rms_broadband.item()), 2),
            )

        for i in range(B):
            meta = per_sample_meta[i]
            if meta is None:
                results.append(SnapshotPair(
                    loud=None, quiet=None,
                    duration_sec=float(lengths[i].item() / self.sr),
                    peak_dbfs=-120.0,
                    n_valid_frames=0,
                ))
                continue

            loud_snap = extract_snapshot(i, meta["loud_frame"])
            quiet_snap = extract_snapshot(i, meta["quiet_frame"])
            results.append(SnapshotPair(
                loud=loud_snap, quiet=quiet_snap,
                duration_sec=meta["duration_sec"],
                peak_dbfs=round(meta["peak_dbfs"], 2),
                n_valid_frames=meta["n_valid_frames"],
            ))

        return results
