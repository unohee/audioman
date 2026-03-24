# Created: 2026-03-24
# Purpose: Audio Features → SynthPatch 파라미터 매칭 모델
# License: MIT
#
# Phase 2: Serum audio features → audioman synth params
# Phase 4: Transformer (CLAP → params) — 추후 확장

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 학습에 사용할 audio feature 키 (순서 고정)
FEATURE_KEYS = [
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
]

# SynthPatch 파라미터 키 (학습 대상, 순서 고정)
PARAM_KEYS = [
    'osc1_mode', 'osc1_waveform', 'osc1_volume', 'osc1_pan',
    'osc1_octave', 'osc1_semi', 'osc1_fine',
    'osc2_mode', 'osc2_waveform', 'osc2_volume', 'osc2_pan',
    'osc2_octave', 'osc2_semi', 'osc2_fine',
    'fm_ratio', 'fm_depth',
    'sub_volume', 'noise_volume', 'noise_color',
    'filter_cutoff', 'filter_resonance', 'filter_type',
    'filter_key_track', 'filter_env_amount', 'filter_lfo_amount',
    'env1_attack', 'env1_decay', 'env1_sustain', 'env1_release', 'env1_lag',
    'env2_attack', 'env2_decay', 'env2_sustain', 'env2_release', 'env2_lag',
    'env3_attack', 'env3_decay', 'env3_sustain', 'env3_release', 'env3_lag',
    'lfo1_rate', 'lfo1_depth', 'lfo1_waveform', 'lfo1_target',
    'lfo2_rate', 'lfo2_depth', 'lfo2_waveform', 'lfo2_target',
    'lfo3_rate', 'lfo3_depth', 'lfo3_waveform', 'lfo3_target',
    'lfo4_rate', 'lfo4_depth', 'lfo4_waveform', 'lfo4_target',
    'drive', 'delay_wet', 'delay_time', 'delay_feedback',
    'master_volume',
]


def load_dataset(dataset_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """데이터셋 로드 → (features, params) numpy 배열

    Returns:
        features: (N, 21) — audio features
        params: (N, 57) — synth parameters
    """
    ds = Path(dataset_dir)

    features = []
    with open(ds / 'features.jsonl') as f:
        for line in f:
            d = json.loads(line)
            features.append([float(d.get(k, 0.0)) for k in FEATURE_KEYS])

    params = []
    with open(ds / 'params.jsonl') as f:
        for line in f:
            d = json.loads(line)
            params.append([float(d.get(k, 0.0)) for k in PARAM_KEYS])

    return np.array(features, dtype=np.float32), np.array(params, dtype=np.float32)


def train_mlp(
    features: np.ndarray,
    params: np.ndarray,
    epochs: int = 200,
    hidden_dim: int = 256,
    lr: float = 0.001,
    val_split: float = 0.1,
    save_path: Optional[str] = None,
) -> dict:
    """Audio features → SynthPatch params MLP 학습

    Args:
        features: (N, 21)
        params: (N, 57)

    Returns:
        {'train_loss', 'val_loss', 'model_path'}
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'Training on {device}')

    # 정규화
    feat_mean = features.mean(0)
    feat_std = features.std(0)
    feat_std[feat_std < 1e-8] = 1.0

    X = (features - feat_mean) / feat_std
    Y = params  # 이미 0~1

    # Train/Val 분할
    n = len(X)
    n_val = int(n * val_split)
    indices = np.random.permutation(n)
    X_train, X_val = X[indices[n_val:]], X[indices[:n_val]]
    Y_train, Y_val = Y[indices[n_val:]], Y[indices[:n_val]]

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256)

    # MLP 모델
    in_dim = features.shape[1]
    out_dim = params.shape[1]

    model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, out_dim),
        nn.Sigmoid(),  # 출력을 0~1로 제한
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            if save_path:
                torch.save({
                    'model': model.state_dict(),
                    'feat_mean': feat_mean,
                    'feat_std': feat_std,
                    'feature_keys': FEATURE_KEYS,
                    'param_keys': PARAM_KEYS,
                    'in_dim': in_dim,
                    'out_dim': out_dim,
                    'hidden_dim': hidden_dim,
                }, save_path)

        if (epoch + 1) % 20 == 0:
            logger.info(f'Epoch {epoch+1}/{epochs}: train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}')
            print(f'  Epoch {epoch+1}/{epochs}: train={train_loss:.6f} val={val_loss:.6f}', flush=True)

    return {
        'train_loss': history['train_loss'][-1],
        'val_loss': best_val,
        'model_path': save_path,
        'epochs': epochs,
        'n_train': len(train_ds),
        'n_val': len(val_ds),
    }


def predict_patch(
    features: np.ndarray,
    model_path: str,
) -> dict:
    """Audio features → SynthPatch 파라미터 예측

    Args:
        features: (21,) 또는 (N, 21)

    Returns:
        dict — SynthPatch 파라미터
    """
    import torch
    import torch.nn as nn

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

    in_dim = checkpoint['in_dim']
    out_dim = checkpoint['out_dim']
    hidden_dim = checkpoint['hidden_dim']

    model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, out_dim),
        nn.Sigmoid(),
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 정규화
    feat_mean = checkpoint['feat_mean']
    feat_std = checkpoint['feat_std']

    if features.ndim == 1:
        features = features.reshape(1, -1)

    X = (features - feat_mean) / feat_std
    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        pred = model(X).numpy()

    # 첫 번째 결과를 SynthPatch dict로
    param_keys = checkpoint['param_keys']
    result = {k: float(pred[0, i]) for i, k in enumerate(param_keys)}
    return result


def match_sound(
    target_audio: np.ndarray,
    sample_rate: int,
    model_path: str,
    note: int = 60,
) -> tuple[dict, np.ndarray]:
    """타겟 오디오 → features 추출 → 모델 예측 → SynthPatch + 렌더링

    Returns:
        (patch_dict, rendered_audio)
    """
    from audioman.core.analysis import compute_audio_features
    from audioman.core.synth_engine import SynthPatch, render_patch

    # 타겟 분석
    features = compute_audio_features(target_audio, sample_rate)
    feat_vec = np.array([features[k] for k in FEATURE_KEYS], dtype=np.float32)

    # 예측
    patch_dict = predict_patch(feat_vec, model_path)
    patch = SynthPatch.from_dict(patch_dict)

    # 렌더링
    audio = render_patch(patch, note=note, sample_rate=sample_rate)

    return patch_dict, audio
