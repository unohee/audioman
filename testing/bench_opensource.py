#!/usr/bin/env python3
# Created: 2026-03-21
# Purpose: 오픈소스 denoise/dereverb vs RX 10 벤치마크
# Dependencies: pedalboard, noisereduce, pyrnnoise, torch, soundfile, numpy

"""
동일 오디오에 대해 다음을 비교:
1. iZotope RX 10 Spectral De-noise (VST3, 2-pass adaptive)
2. iZotope RX 10 De-reverb (VST3)
3. noisereduce (스펙트럴 게이팅, stationary + non-stationary)
4. pyrnnoise (RNNoise, GRU 기반)
5. torch spectral gating (기본 DSP 참조용)
"""

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

RESULTS_DIR = Path("/tmp/audioman_bench")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 테스트 파일들 — pilot-residue 실제 음성
TEST_FILES = [
    "/Volumes/Workroom_Studio/pilot-residue/youtube_flac/YOU1000000021_S0000027.flac",
    "/Volumes/Workroom_Studio/pilot-residue/youtube_flac/YOU1000000130_S0000060.flac",
    "/Volumes/Workroom_Studio/pilot-residue/youtube_flac/YOU1000000037_S0000367.flac",
    "/Volumes/Workroom_Studio/pilot-residue/youtube_flac/YOU1000000189_S0000106.flac",
    "/Volumes/Workroom_Studio/pilot-residue/youtube_flac/YOU1000000005_S0000020.flac",
]


def audio_stats(audio: np.ndarray, sr: int) -> dict:
    return {
        "rms": float(np.sqrt(np.mean(audio**2))),
        "peak": float(np.max(np.abs(audio))),
        "duration": len(audio) / sr,
        "sr": sr,
    }


def spectral_snr(original: np.ndarray, processed: np.ndarray, sr: int) -> float:
    """처리 전후 SNR 추정. 높을수록 원본 신호 보존이 좋고 부드러운 처리."""
    min_len = min(len(original), len(processed))
    diff = original[:min_len] - processed[:min_len]
    signal_power = np.mean(processed[:min_len]**2)
    diff_power = np.mean(diff**2)
    if diff_power < 1e-10:
        return 100.0
    return float(10 * np.log10(signal_power / diff_power))


# === 1. RX 10 Spectral De-noise (2-pass adaptive) ===
_rx_denoise_plugin = None

def bench_rx_denoise(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    global _rx_denoise_plugin
    from pedalboard import load_plugin
    if _rx_denoise_plugin is None:
        _rx_denoise_plugin = load_plugin("/Library/Audio/Plug-Ins/VST3/RX 10 Spectral De-noise.vst3")

    plugin = _rx_denoise_plugin
    plugin.adaptive_learning = True
    plugin.adaptive_learning_time = 5.0
    plugin.noise_reduction_db = 12.0
    plugin.artifact_control = 7.0
    plugin.output_noise_only = False

    audio_2d = audio.reshape(1, -1).astype(np.float32)
    _ = plugin.process(audio_2d, sr)  # Pass 1: 학습
    start = time.monotonic()
    result = plugin.process(audio_2d, sr)  # Pass 2: 처리
    elapsed = time.monotonic() - start
    return result[0], elapsed


# === 2. RX 10 De-reverb ===
_rx_dereverb_plugin = None

def bench_rx_dereverb(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    global _rx_dereverb_plugin
    from pedalboard import load_plugin
    if _rx_dereverb_plugin is None:
        _rx_dereverb_plugin = load_plugin("/Library/Audio/Plug-Ins/VST3/RX 10 De-reverb.vst3")

    plugin = _rx_dereverb_plugin
    plugin.reduction = 10.0
    plugin.tail_length = 1.5
    plugin.artifact_smoothing = 8.0

    audio_2d = audio.reshape(1, -1).astype(np.float32)
    start = time.monotonic()
    result = plugin.process(audio_2d, sr)
    elapsed = time.monotonic() - start
    return result[0], elapsed


# === 3. noisereduce (stationary) ===
def bench_noisereduce_stationary(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    import noisereduce as nr
    start = time.monotonic()
    result = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)
    elapsed = time.monotonic() - start
    return result, elapsed


# === 4. noisereduce (non-stationary) ===
def bench_noisereduce_nonstat(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    import noisereduce as nr
    start = time.monotonic()
    result = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.75)
    elapsed = time.monotonic() - start
    return result, elapsed


# === 5. pyrnnoise (RNNoise) ===
def bench_pyrnnoise(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    import torch
    import torchaudio
    from pyrnnoise import RNNoise

    # RNNoise는 48kHz mono 전용 — 리샘플 필요
    if sr != 48000:
        audio_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
        audio_48k = resampler(audio_t).squeeze().numpy()
        target_sr = 48000
    else:
        audio_48k = audio
        target_sr = sr

    denoiser = RNNoise(sample_rate=target_sr)

    start = time.monotonic()
    # denoise_chunk: yields (prob shape (1,1), frame shape (1, 480))
    # RNNoise 출력은 int16 범위 (-32768~32767) — float32로 정규화 필요
    frames = []
    for prob, denoised_frame in denoiser.denoise_chunk(audio_48k, partial=True):
        frames.append(denoised_frame[0])  # (1, 480) → (480,)
    result_48k = np.concatenate(frames) if frames else audio_48k
    # int16 → float32 정규화
    if np.max(np.abs(result_48k)) > 2.0:
        result_48k = result_48k / 32768.0
    # 원본 길이에 맞추기
    result_48k = result_48k[:len(audio_48k)].astype(np.float32)
    elapsed = time.monotonic() - start

    # 원래 SR로 리샘플
    if sr != 48000:
        result_t = torch.tensor(result_48k, dtype=torch.float32).unsqueeze(0)
        resampler_back = torchaudio.transforms.Resample(orig_freq=48000, new_freq=sr)
        result = resampler_back(result_t).squeeze().numpy()
    else:
        result = result_48k

    return result, elapsed


def run_benchmark():
    print("=" * 70)
    print("  오픈소스 vs RX 10 Denoise/Dereverb 벤치마크")
    print("  테스트 데이터: pilot-residue 실제 음성")
    print("=" * 70)

    all_results = []

    methods = [
        ("RX10-Denoise-2pass", bench_rx_denoise),
        ("RX10-Dereverb", bench_rx_dereverb),
        ("noisereduce-stat", bench_noisereduce_stationary),
        ("noisereduce-nonstat", bench_noisereduce_nonstat),
        ("pyrnnoise", bench_pyrnnoise),
    ]

    for fpath in TEST_FILES:
        if not Path(fpath).exists():
            print(f"  [SKIP] {fpath}")
            continue

        audio, sr = sf.read(fpath, dtype="float32")
        fname = Path(fpath).stem
        orig_stats = audio_stats(audio, sr)

        print(f"\n--- {fname} ({orig_stats['duration']:.1f}s, {sr}Hz, RMS={orig_stats['rms']:.4f}) ---")

        file_results = {"file": fname, "original": orig_stats, "methods": {}}

        for method_name, method_fn in methods:
            try:
                print(f"  {method_name:25s}", end=" ", flush=True)
                result_audio, elapsed = method_fn(audio.copy(), sr)

                min_len = min(len(audio), len(result_audio))
                snr = spectral_snr(audio[:min_len], result_audio[:min_len], sr)
                stats = audio_stats(result_audio[:min_len], sr)

                # 저장
                safe_name = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                out_path = RESULTS_DIR / f"{fname}_{safe_name}.wav"
                sf.write(str(out_path), result_audio[:min_len], sr)

                rms_reduction = (1 - stats["rms"] / orig_stats["rms"]) * 100

                print(f"RMS={stats['rms']:.4f} ({rms_reduction:+5.1f}%)  SNR={snr:5.1f}dB  t={elapsed:.3f}s")

                file_results["methods"][method_name] = {
                    "stats": stats,
                    "snr_db": round(snr, 2),
                    "rms_reduction_pct": round(rms_reduction, 2),
                    "time_seconds": round(elapsed, 4),
                    "output_path": str(out_path),
                }
            except Exception as e:
                print(f"FAIL: {e}")
                file_results["methods"][method_name] = {"error": str(e)}

        all_results.append(file_results)

    # 결과 요약
    print("\n" + "=" * 70)
    print("  결과 요약 (5파일 평균)")
    print("=" * 70)
    print(f"{'Method':<28} {'RMS Reduction':>13} {'SNR (dB)':>10} {'Time (s)':>10}")
    print("-" * 70)

    for method_name, _ in methods:
        rms_reds, snrs, times = [], [], []
        for fr in all_results:
            m = fr["methods"].get(method_name, {})
            if "error" not in m and m:
                rms_reds.append(m["rms_reduction_pct"])
                snrs.append(m["snr_db"])
                times.append(m["time_seconds"])

        if rms_reds:
            print(f"{method_name:<28} {np.mean(rms_reds):>+12.1f}% {np.mean(snrs):>9.1f}dB {np.mean(times):>9.3f}s")
        else:
            print(f"{method_name:<28} {'FAIL':>13}")

    print()
    print("참고:")
    print("  RMS Reduction: 높을수록 더 많은 노이즈 제거 (과도하면 신호 손상)")
    print("  SNR: 높을수록 원본 신호 보존 양호 (과도한 처리는 SNR 낮음)")
    print("  이상적: 적당한 RMS Reduction + 높은 SNR")

    # JSON 저장
    json_path = RESULTS_DIR / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n상세 결과: {json_path}")
    print(f"오디오 파일: {RESULTS_DIR}/")


if __name__ == "__main__":
    run_benchmark()
