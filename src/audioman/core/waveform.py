# Created: 2026-03-21
# Purpose: ASCII 웨이브폼 렌더러

import numpy as np

# 블록 문자 (해상도 8단계: 하단→상단)
BLOCKS_UP = " ▁▂▃▄▅▆▇█"
BLOCKS_DOWN = " ▔▔▀▀▀█████"  # 상단 미러용 (간소화)


def render_waveform(
    audio: np.ndarray,
    sample_rate: int,
    width: int = 80,
    height: int = 16,
    mode: str = "rms",
) -> str:
    """ASCII 웨이브폼 렌더링

    Args:
        audio: (channels, samples) 또는 (samples,)
        sample_rate: 샘플 레이트
        width: 가로 문자 수
        height: 세로 줄 수 (양쪽 합산, 짝수 권장)
        mode: "rms" (envelope) 또는 "peak" (피크)

    Returns:
        멀티라인 ASCII 문자열
    """
    # mono로 변환
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    n_samples = len(mono)
    samples_per_col = max(1, n_samples // width)
    half_h = height // 2

    # 컬럼별 RMS/peak 계산
    pos_values = []
    neg_values = []

    for i in range(width):
        start = i * samples_per_col
        end = min(start + samples_per_col, n_samples)
        chunk = mono[start:end]

        if len(chunk) == 0:
            pos_values.append(0.0)
            neg_values.append(0.0)
            continue

        if mode == "rms":
            pos_chunk = chunk[chunk >= 0]
            neg_chunk = chunk[chunk < 0]
            pos_val = float(np.sqrt(np.mean(pos_chunk**2))) if len(pos_chunk) > 0 else 0.0
            neg_val = float(np.sqrt(np.mean(neg_chunk**2))) if len(neg_chunk) > 0 else 0.0
        else:  # peak
            pos_val = float(np.max(chunk)) if np.any(chunk > 0) else 0.0
            neg_val = float(-np.min(chunk)) if np.any(chunk < 0) else 0.0

        pos_values.append(pos_val)
        neg_values.append(neg_val)

    # 정규화 (전체 max 기준)
    max_val = max(max(pos_values), max(neg_values), 1e-10)
    pos_norm = [v / max_val for v in pos_values]
    neg_norm = [v / max_val for v in neg_values]

    # 렌더링 (상단 = 양수, 하단 = 음수)
    lines = []

    # 상단 (양수 파형, 위에서 아래로)
    for row in range(half_h, 0, -1):
        threshold = row / half_h
        line = ""
        for col in range(width):
            val = pos_norm[col]
            if val >= threshold:
                # 이 줄을 완전히 채움
                line += "█"
            elif val >= threshold - (1.0 / half_h):
                # 부분 채움
                frac = (val - (threshold - 1.0 / half_h)) / (1.0 / half_h)
                idx = int(frac * (len(BLOCKS_UP) - 1))
                idx = max(0, min(len(BLOCKS_UP) - 1, idx))
                line += BLOCKS_UP[idx]
            else:
                line += " "
        lines.append(line)

    # 중앙선
    duration = n_samples / sample_rate
    center = "─" * width
    lines.append(center)

    # 하단 (음수 파형, 위에서 아래로 = 미러)
    for row in range(1, half_h + 1):
        threshold = row / half_h
        line = ""
        for col in range(width):
            val = neg_norm[col]
            if val >= threshold:
                line += "█"
            elif val >= threshold - (1.0 / half_h):
                frac = (val - (threshold - 1.0 / half_h)) / (1.0 / half_h)
                idx = int(frac * (len(BLOCKS_UP) - 1))
                idx = max(0, min(len(BLOCKS_UP) - 1, idx))
                line += BLOCKS_UP[idx]
            else:
                line += " "
        lines.append(line)

    # 시간 축 라벨
    time_label = _make_time_axis(width, duration)
    lines.append(time_label)

    # dB 라벨
    peak_db = 20 * np.log10(max_val) if max_val > 1e-10 else -100
    header = f"  peak: {max_val:.3f} ({peak_db:.1f}dB) | {duration:.2f}s | {sample_rate}Hz | mode: {mode}"

    return header + "\n" + "\n".join(lines)


def render_envelope(
    audio: np.ndarray,
    sample_rate: int,
    width: int = 80,
    height: int = 8,
) -> str:
    """단방향 RMS envelope (컴팩트 버전)"""
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    n_samples = len(mono)
    samples_per_col = max(1, n_samples // width)

    rms_values = []
    for i in range(width):
        start = i * samples_per_col
        end = min(start + samples_per_col, n_samples)
        chunk = mono[start:end]
        rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
        rms_values.append(rms)

    max_rms = max(rms_values) if rms_values else 1e-10
    if max_rms < 1e-10:
        max_rms = 1e-10
    norm = [v / max_rms for v in rms_values]

    # 아래서 위로 렌더링
    lines = []
    for row in range(height, 0, -1):
        threshold = row / height
        line = ""
        for col in range(width):
            val = norm[col]
            if val >= threshold:
                line += "█"
            elif val >= threshold - (1.0 / height):
                frac = (val - (threshold - 1.0 / height)) / (1.0 / height)
                idx = int(frac * (len(BLOCKS_UP) - 1))
                idx = max(0, min(len(BLOCKS_UP) - 1, idx))
                line += BLOCKS_UP[idx]
            else:
                line += " "
        lines.append(line)

    duration = n_samples / sample_rate
    lines.append(_make_time_axis(width, duration))

    peak_db = 20 * np.log10(max_rms) if max_rms > 1e-10 else -100
    header = f"  rms peak: {max_rms:.4f} ({peak_db:.1f}dB) | {duration:.2f}s"

    return header + "\n" + "\n".join(lines)


def render_spectral_envelope(
    spectral_centroid: list[float],
    spectral_entropy: list[float],
    sample_rate: int,
    duration: float,
    width: int = 80,
    height: int = 6,
) -> str:
    """spectral centroid와 entropy의 ASCII 시간축 플롯"""
    lines = []

    for label, values, unit in [
        ("centroid", spectral_centroid, "Hz"),
        ("entropy", spectral_entropy, "bits"),
    ]:
        if not values:
            continue

        arr = np.array(values)
        # width 컬럼으로 리샘플
        indices = np.linspace(0, len(arr) - 1, width).astype(int)
        resampled = arr[indices]

        max_val = float(np.max(resampled)) if len(resampled) > 0 else 1.0
        min_val = float(np.min(resampled))
        range_val = max_val - min_val if max_val > min_val else 1.0

        norm = [(v - min_val) / range_val for v in resampled]

        header = f"  {label}: {min_val:.1f}–{max_val:.1f} {unit}"
        lines.append(header)

        for row in range(height, 0, -1):
            threshold = row / height
            line = ""
            for col in range(width):
                val = norm[col]
                if val >= threshold:
                    line += "█"
                elif val >= threshold - (1.0 / height):
                    frac = (val - (threshold - 1.0 / height)) / (1.0 / height)
                    idx = int(frac * 8)
                    idx = max(0, min(8, idx))
                    line += BLOCKS_UP[idx]
                else:
                    line += " "
            lines.append(line)

        lines.append(_make_time_axis(width, duration))
        lines.append("")

    return "\n".join(lines)


def _make_time_axis(width: int, duration: float) -> str:
    """시간 축 라벨"""
    label = f"0s{'':>{width - 8}}{duration:.1f}s"
    return label
