# Created: 2026-03-24
# Purpose: Audio → SynthPatch 파라미터 추정 모델
# License: MIT
#
# 입력: audio features (21) + CLAP embedding (512) = 533 dim
# 출력: SynthPatch 파라미터 (61 dim)
# 손실함수: 파라미터 그룹별 가중 MSE
#   - 파형 선택 (waveform, mode): 높은 가중치 (소리 캐릭터 결정)
#   - 필터 (cutoff, reso, type): 높은 가중치 (음색 핵심)
#   - 엔벨로프 ADSR: 높은 가중치 (시간적 특성)
#   - LFO: 중간 가중치 (모듈레이션)
#   - FX/기타: 낮은 가중치

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 입력 feature 키
FEATURE_KEYS = [
    # 기본 (21)
    'peak', 'rms_mean', 'rms_max', 'crest_factor',
    'spectral_centroid_mean', 'spectral_centroid_std',
    'spectral_bandwidth_mean', 'spectral_rolloff_mean',
    'spectral_flatness_mean', 'spectral_entropy_mean',
    'zero_crossing_rate_mean',
    'attack_time', 'release_time',
    'fundamental_freq', 'harmonic_ratio',
    'duration_effective',
    'attack_curvature', 'decay_curvature', 'release_curvature',
    'sustain_level', 'envelope_rms_error',
    # 시간축 스펙트럴 변화 (8) — filter envelope 직접 측정
    'centroid_attack_slope', 'centroid_decay_slope', 'centroid_range',
    'bandwidth_attack_slope', 'bandwidth_range',
    'brightness_trajectory', 'spectral_flux_mean', 'spectral_flux_std',
    # Peak frequency trajectory (7) — filter sweep
    'peak_freq_start', 'peak_freq_peak', 'peak_freq_sustain',
    'peak_freq_sweep_range', 'peak_freq_attack_ratio', 'peak_freq_decay_ratio',
    'spectral_peak_stability',
    # 하모닉 시리즈 (5) — 파형/음색 특성
    'harmonic_centroid_ratio', 'harmonic_spread', 'odd_even_ratio',
    'harmonic_decay_rate', 'inharmonicity',
]

# 출력 파라미터 키 + 그룹별 가중치
PARAM_GROUPS = {
    # 파형/모드/웨이브테이블 — 소리의 기본 캐릭터 (가중치 3.0)
    'waveform': {
        'keys': ['osc1_mode', 'osc1_waveform', 'osc1_table',
                 'osc2_mode', 'osc2_waveform', 'osc2_table'],
        'weight': 3.0,
    },
    # 오실레이터 볼륨/피치 — 믹스 밸런스 (가중치 2.0)
    'osc_mix': {
        'keys': ['osc1_volume', 'osc2_volume', 'osc1_octave', 'osc1_semi',
                 'osc2_octave', 'osc2_semi', 'sub_volume', 'noise_volume'],
        'weight': 2.0,
    },
    # 필터 — 음색 핵심 (가중치 4.0)
    'filter': {
        'keys': ['filter_cutoff', 'filter_resonance', 'filter_type',
                 'filter_env_amount', 'filter_lfo_amount'],
        'weight': 4.0,
    },
    # Amp 엔벨로프 — 시간적 특성 (가중치 4.0)
    'env_amp': {
        'keys': ['env1_attack', 'env1_decay', 'env1_sustain', 'env1_release', 'env1_lag'],
        'weight': 4.0,
    },
    # Filter 엔벨로프 (가중치 3.0)
    'env_filter': {
        'keys': ['env2_attack', 'env2_decay', 'env2_sustain', 'env2_release', 'env2_lag'],
        'weight': 3.0,
    },
    # Mod 엔벨로프 (가중치 1.5)
    'env_mod': {
        'keys': ['env3_attack', 'env3_decay', 'env3_sustain', 'env3_release', 'env3_lag'],
        'weight': 1.5,
    },
    # FM (가중치 2.0)
    'fm': {
        'keys': ['fm_ratio', 'fm_depth'],
        'weight': 2.0,
    },
    # LFO (가중치 1.5)
    'lfo': {
        'keys': ['lfo1_rate', 'lfo1_depth', 'lfo1_waveform', 'lfo1_target',
                 'lfo2_rate', 'lfo2_depth', 'lfo2_waveform', 'lfo2_target',
                 'lfo3_rate', 'lfo3_depth', 'lfo3_waveform', 'lfo3_target',
                 'lfo4_rate', 'lfo4_depth', 'lfo4_waveform', 'lfo4_target'],
        'weight': 1.5,
    },
    # FX + 기타 (가중치 1.0)
    'fx': {
        'keys': ['drive', 'delay_wet', 'delay_time', 'delay_feedback',
                 'osc1_pan', 'osc2_pan', 'osc1_fine', 'osc2_fine',
                 'noise_color', 'filter_key_track', 'env3_target', 'env3_amount',
                 'master_volume'],
        'weight': 1.0,
    },
}

# 전체 파라미터 키 순서 (그룹 순서대로 flat)
PARAM_KEYS = []
PARAM_WEIGHTS = []
for group in PARAM_GROUPS.values():
    for k in group['keys']:
        if k not in PARAM_KEYS:
            PARAM_KEYS.append(k)
            PARAM_WEIGHTS.append(group['weight'])

PARAM_WEIGHTS = np.array(PARAM_WEIGHTS, dtype=np.float32)
PARAM_WEIGHTS /= PARAM_WEIGHTS.mean()  # 정규화 (평균=1)


def load_dataset(
    dataset_dir: str,
    clap_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """데이터셋 로드

    Returns:
        features: (N, 21) 또는 (N, 533) if CLAP
        params: (N, len(PARAM_KEYS))
    """
    ds = Path(dataset_dir)

    features = []
    with open(ds / 'features.jsonl') as f:
        for line in f:
            d = json.loads(line)
            features.append([float(d.get(k, 0.0)) for k in FEATURE_KEYS])
    features = np.array(features, dtype=np.float32)

    # CLAP 임베딩 추가
    if clap_path:
        clap = np.load(clap_path)
        if len(clap) == len(features):
            features = np.concatenate([features, clap], axis=1)
            logger.info(f'CLAP 임베딩 추가: {features.shape}')

    # params + spectral ground truth 보정
    params = []
    feat_lines = open(ds / 'features.jsonl').readlines() if (ds / 'features.jsonl').exists() else []
    feat_dicts = [json.loads(l) for l in feat_lines]

    with open(ds / 'params.jsonl') as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            row = [float(d.get(k, 0.0)) for k in PARAM_KEYS]

            # spectral NN ground truth로 WT 파라미터 보정
            if i < len(feat_dicts):
                fd = feat_dicts[i]
                if 'gt_table_idx' in fd:
                    # osc1_table → gt_table_idx, osc1_waveform → gt_frame_pos
                    for j, k in enumerate(PARAM_KEYS):
                        if k == 'osc1_table':
                            row[j] = float(fd['gt_table_idx'])
                        elif k == 'osc1_waveform' and d.get('osc1_mode', 0) > 0.3:
                            row[j] = float(fd['gt_frame_pos'])

            params.append(row)

    params = np.array(params, dtype=np.float32)
    return features, params


def _build_model(in_dim: int, out_dim: int, hidden_dim: int = 512):
    """모델 아키텍처"""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(0.1),

        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(0.1),

        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),

        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.GELU(),

        nn.Linear(hidden_dim // 2, out_dim),
        nn.Sigmoid(),
    )


def train(
    features: np.ndarray,
    params: np.ndarray,
    epochs: int = 300,
    hidden_dim: int = 512,
    lr: float = 0.001,
    val_split: float = 0.1,
    save_path: Optional[str] = None,
) -> dict:
    """그룹별 가중 MSE로 학습"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    # 정규화
    feat_mean = features.mean(0)
    feat_std = features.std(0)
    feat_std[feat_std < 1e-8] = 1.0
    X = (features - feat_mean) / feat_std
    Y = params

    # 파라미터 가중치 텐서
    weights = torch.tensor(PARAM_WEIGHTS, dtype=torch.float32).to(device)

    # Split
    n = len(X)
    n_val = int(n * val_split)
    idx = np.random.permutation(n)
    X_tr, X_va = X[idx[n_val:]], X[idx[:n_val]]
    Y_tr, Y_va = Y[idx[n_val:]], Y[idx[:n_val]]

    train_dl = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)),
                          batch_size=128, shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.tensor(X_va), torch.tensor(Y_va)),
                        batch_size=256)

    in_dim = features.shape[1]
    out_dim = params.shape[1]
    model = _build_model(in_dim, out_dim, hidden_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    def weighted_mse(pred, target):
        diff = (pred - target) ** 2  # (batch, params)
        return (diff * weights).mean()

    best_val = float('inf')
    best_group_losses = {}

    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = weighted_mse(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(xb)
        t_loss /= len(X_tr)

        model.eval()
        v_loss = 0
        group_losses = {g: 0.0 for g in PARAM_GROUPS}
        n_val_samples = 0

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                v_loss += weighted_mse(pred, yb).item() * len(xb)

                # 그룹별 손실 추적
                p_np = pred.cpu().numpy()
                y_np = yb.cpu().numpy()
                offset = 0
                for gname, ginfo in PARAM_GROUPS.items():
                    n_keys = len(ginfo['keys'])
                    g_pred = p_np[:, offset:offset+n_keys]
                    g_true = y_np[:, offset:offset+n_keys]
                    group_losses[gname] += np.mean((g_pred - g_true)**2) * len(xb)
                    offset += n_keys
                n_val_samples += len(xb)

        v_loss /= len(X_va)
        for g in group_losses:
            group_losses[g] /= n_val_samples

        scheduler.step()

        if v_loss < best_val:
            best_val = v_loss
            best_group_losses = group_losses.copy()
            if save_path:
                torch.save({
                    'model': model.state_dict(),
                    'feat_mean': feat_mean,
                    'feat_std': feat_std,
                    'feature_keys': FEATURE_KEYS,
                    'param_keys': PARAM_KEYS,
                    'param_weights': PARAM_WEIGHTS,
                    'in_dim': in_dim,
                    'out_dim': out_dim,
                    'hidden_dim': hidden_dim,
                }, save_path)

        if (epoch + 1) % 25 == 0:
            parts = ' | '.join(f'{g}={group_losses[g]:.4f}' for g in ['waveform', 'filter', 'env_amp'])
            print(f'  E{epoch+1:3d} train={t_loss:.5f} val={v_loss:.5f} | {parts}', flush=True)

    # 최종 그룹별 결과
    print(f'\n  Best val={best_val:.5f}', flush=True)
    print(f'  Group losses:', flush=True)
    for g, v in best_group_losses.items():
        w = PARAM_GROUPS[g]['weight']
        print(f'    {g:15s}: MSE={v:.5f} (weight={w:.1f})', flush=True)

    return {
        'val_loss': best_val,
        'group_losses': best_group_losses,
        'model_path': save_path,
        'n_train': len(X_tr),
        'n_val': len(X_va),
        'in_dim': in_dim,
        'out_dim': out_dim,
    }


def predict_patch(features: np.ndarray, model_path: str) -> dict:
    """Audio features (+ CLAP) → SynthPatch 파라미터"""
    import torch

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model = _build_model(ckpt['in_dim'], ckpt['out_dim'], ckpt['hidden_dim'])
    model.load_state_dict(ckpt['model'])
    model.eval()

    if features.ndim == 1:
        features = features.reshape(1, -1)
    X = (features - ckpt['feat_mean']) / ckpt['feat_std']
    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        pred = model(X).numpy()

    return {k: float(pred[0, i]) for i, k in enumerate(ckpt['param_keys'])}


def match_sound(
    target_audio: np.ndarray,
    sample_rate: int,
    model_path: str,
    note: int = 60,
    clap_model=None,
) -> tuple[dict, np.ndarray]:
    """타겟 오디오 → 분석 → 모델 → SynthPatch → 렌더링

    Returns: (patch_dict, rendered_audio)
    """
    from audioman.core.analysis import compute_audio_features
    from audioman.core.synth_engine import SynthPatch, render_patch

    features = compute_audio_features(target_audio, sample_rate)
    feat_vec = np.array([features[k] for k in FEATURE_KEYS], dtype=np.float32)

    # CLAP 임베딩 추가
    if clap_model is not None:
        import soundfile as sf
        import tempfile, os
        # 임시 WAV → CLAP
        tmp = tempfile.mktemp(suffix='.wav')
        sf.write(tmp, target_audio.T if target_audio.ndim == 2 else target_audio, sample_rate)
        clap_emb = clap_model.get_audio_embedding_from_filelist([tmp], use_tensor=False)[0]
        os.unlink(tmp)
        feat_vec = np.concatenate([feat_vec, clap_emb])

    patch_dict = predict_patch(feat_vec, model_path)
    patch = SynthPatch.from_dict(patch_dict)
    audio = render_patch(patch, note=note, sample_rate=sample_rate)

    return patch_dict, audio
