# Created: 2026-03-21
# Purpose: 내장 DSP 함수 (fade, trim, cut, splice, normalize, gate, gain)

import numpy as np


def _length(audio: np.ndarray) -> int:
    return len(audio) if audio.ndim == 1 else audio.shape[1]


def _channels(audio: np.ndarray) -> int:
    return 1 if audio.ndim == 1 else audio.shape[0]


def _slice_time(audio: np.ndarray, start: int, end: int) -> np.ndarray:
    if audio.ndim == 1:
        return audio[start:end]
    return audio[:, start:end]


def _concat_time(parts: list[np.ndarray]) -> np.ndarray:
    """시간축 연결. 모든 part는 동일한 채널 수여야 한다."""
    parts = [p for p in parts if _length(p) > 0]
    if not parts:
        return parts[0] if parts else np.zeros(0, dtype=np.float32)
    axis = 0 if parts[0].ndim == 1 else 1
    return np.concatenate(parts, axis=axis)


def cut_region(
    audio: np.ndarray,
    start: int,
    end: int,
    crossfade_samples: int = 0,
) -> np.ndarray:
    """중간 구간 [start, end) 를 삭제하고 앞뒤를 이어붙인다.

    crossfade_samples > 0 이면 경계에서 동일 길이의 선형 crossfade를 적용해
    클릭/팝을 방지한다. (좌측 tail의 마지막 N 샘플과 우측 head의 첫 N 샘플을 섞음)
    """
    n = _length(audio)
    start = max(0, min(start, n))
    end = max(start, min(end, n))
    if start == end:
        return audio.copy()

    left = _slice_time(audio, 0, start)
    right = _slice_time(audio, end, n)

    if crossfade_samples <= 0:
        return _concat_time([left, right])

    cf = min(crossfade_samples, _length(left), _length(right))
    if cf == 0:
        return _concat_time([left, right])

    fade_out_curve = np.linspace(1.0, 0.0, cf, dtype=np.float32)
    fade_in_curve = np.linspace(0.0, 1.0, cf, dtype=np.float32)

    left_head = _slice_time(left, 0, _length(left) - cf)
    left_tail = _slice_time(left, _length(left) - cf, _length(left)).copy()
    right_head = _slice_time(right, 0, cf).copy()
    right_tail = _slice_time(right, cf, _length(right))

    if audio.ndim == 1:
        mixed = left_tail * fade_out_curve + right_head * fade_in_curve
    else:
        mixed = left_tail * fade_out_curve + right_head * fade_in_curve

    return _concat_time([left_head, mixed, right_tail])


def splice(
    base: np.ndarray,
    insert: np.ndarray,
    position: int,
    mode: str = "insert",
    crossfade_samples: int = 0,
) -> np.ndarray:
    """base에 insert 클립을 position 위치로 삽입하거나 덮어쓴다.

    mode:
        "insert"    — position 지점에 끼워 넣음. base 길이가 늘어남.
        "overwrite" — position부터 insert 길이만큼 base를 덮어씀. 길이 불변.
        "mix"       — position부터 insert를 base에 합산(섞음). 길이 불변.

    채널 수가 다르면 ValueError. 호출자가 사전 변환 책임.
    crossfade_samples는 insert 모드에서만 의미가 있고 양쪽 경계에 적용된다.
    """
    if _channels(base) != _channels(insert):
        raise ValueError(
            f"채널 수 불일치: base={_channels(base)}, insert={_channels(insert)}"
        )

    n_base = _length(base)
    n_ins = _length(insert)
    position = max(0, min(position, n_base))

    if mode == "insert":
        left = _slice_time(base, 0, position)
        right = _slice_time(base, position, n_base)

        if crossfade_samples <= 0:
            return _concat_time([left, insert, right])

        cf_left = min(crossfade_samples, _length(left), n_ins)
        cf_right = min(crossfade_samples, _length(right), n_ins - cf_left)

        ins_work = insert.copy()
        if cf_left > 0:
            curve = np.linspace(0.0, 1.0, cf_left, dtype=np.float32)
            left_tail = _slice_time(left, _length(left) - cf_left, _length(left)).copy()
            left_tail *= np.linspace(1.0, 0.0, cf_left, dtype=np.float32)
            if ins_work.ndim == 1:
                ins_work[:cf_left] = ins_work[:cf_left] * curve + left_tail
            else:
                ins_work[:, :cf_left] = ins_work[:, :cf_left] * curve + left_tail
            left = _slice_time(left, 0, _length(left) - cf_left)

        if cf_right > 0:
            curve = np.linspace(1.0, 0.0, cf_right, dtype=np.float32)
            right_head = _slice_time(right, 0, cf_right).copy()
            right_head *= np.linspace(0.0, 1.0, cf_right, dtype=np.float32)
            if ins_work.ndim == 1:
                ins_work[-cf_right:] = ins_work[-cf_right:] * curve + right_head
            else:
                ins_work[:, -cf_right:] = ins_work[:, -cf_right:] * curve + right_head
            right = _slice_time(right, cf_right, _length(right))

        return _concat_time([left, ins_work, right])

    if mode == "overwrite":
        out = base.copy()
        end = min(position + n_ins, n_base)
        write_len = end - position
        if write_len <= 0:
            return out
        if out.ndim == 1:
            out[position:end] = insert[:write_len]
        else:
            out[:, position:end] = insert[:, :write_len]
        return out

    if mode == "mix":
        out = base.copy()
        end = min(position + n_ins, n_base)
        write_len = end - position
        if write_len <= 0:
            return out
        if out.ndim == 1:
            out[position:end] = out[position:end] + insert[:write_len]
        else:
            out[:, position:end] = out[:, position:end] + insert[:, :write_len]
        return out

    raise ValueError(f"알 수 없는 splice mode: {mode!r} (insert/overwrite/mix)")


def concat(clips: list[np.ndarray], crossfade_samples: int = 0) -> np.ndarray:
    """여러 클립을 시간축으로 이어붙임. 모든 클립의 채널 수가 같아야 한다.

    crossfade_samples > 0 이면 인접 클립 경계마다 선형 crossfade를 적용한다.
    """
    if not clips:
        return np.zeros(0, dtype=np.float32)
    ch = _channels(clips[0])
    for i, c in enumerate(clips):
        if _channels(c) != ch:
            raise ValueError(f"clips[{i}] 채널 수 불일치: {_channels(c)} != {ch}")

    if crossfade_samples <= 0:
        return _concat_time(list(clips))

    out = clips[0]
    for nxt in clips[1:]:
        cf = min(crossfade_samples, _length(out), _length(nxt))
        if cf == 0:
            out = _concat_time([out, nxt])
            continue
        head = _slice_time(out, 0, _length(out) - cf)
        tail = _slice_time(out, _length(out) - cf, _length(out)).copy()
        nxt_head = _slice_time(nxt, 0, cf).copy()
        nxt_tail = _slice_time(nxt, cf, _length(nxt))
        tail *= np.linspace(1.0, 0.0, cf, dtype=np.float32)
        nxt_head *= np.linspace(0.0, 1.0, cf, dtype=np.float32)
        mixed = tail + nxt_head
        out = _concat_time([head, mixed, nxt_tail])
    return out


FADE_CURVES = ("linear", "cosine", "equal_power", "exponential", "logarithmic")


def _fade_curve(n: int, kind: str, direction: str) -> np.ndarray:
    """길이 n의 페이드 곡선. direction: 'in' (0→1) 또는 'out' (1→0).

    - linear: 진폭 선형
    - cosine: cosine equal-amplitude (S-curve, 가장 부드러움)
    - equal_power: sqrt(linear) — 합성 시 RMS 보존 (crossfade 표준)
    - exponential: 빠른 시작/느린 끝 (감쇠 자연)
    - logarithmic: 느린 시작/빠른 끝
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if kind not in FADE_CURVES:
        raise ValueError(f"알 수 없는 fade curve: {kind!r} (지원: {FADE_CURVES})")

    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if kind == "linear":
        c = x
    elif kind == "cosine":
        c = 0.5 * (1.0 - np.cos(np.pi * x)).astype(np.float32)
    elif kind == "equal_power":
        c = np.sqrt(x).astype(np.float32)
    elif kind == "exponential":
        # 60 dB dynamic range. x=0 → -60dB, x=1 → 0dB
        c = (10 ** ((x - 1.0) * 3.0)).astype(np.float32)
        c -= c[0]
        c /= c[-1] if c[-1] > 0 else 1.0
    elif kind == "logarithmic":
        c = (1.0 - 10 ** (-x * 3.0)).astype(np.float32)
        c -= c[0]
        c /= c[-1] if c[-1] > 0 else 1.0
    else:
        c = x

    return c if direction == "in" else c[::-1].copy()


def fade_in(audio: np.ndarray, samples: int, curve: str = "linear") -> np.ndarray:
    """fade in. samples: fade 길이 (샘플 수). curve: linear/cosine/equal_power/exponential/logarithmic"""
    out = audio.copy()
    if audio.ndim == 1:
        n = min(samples, len(out))
        out[:n] *= _fade_curve(n, curve, "in")
    else:
        n = min(samples, out.shape[1])
        out[:, :n] *= _fade_curve(n, curve, "in")
    return out


def fade_out(audio: np.ndarray, samples: int, curve: str = "linear") -> np.ndarray:
    """fade out. samples: fade 길이 (샘플 수). curve: linear/cosine/equal_power/exponential/logarithmic"""
    out = audio.copy()
    if audio.ndim == 1:
        n = min(samples, len(out))
        out[-n:] *= _fade_curve(n, curve, "out")
    else:
        n = min(samples, out.shape[1])
        out[:, -n:] *= _fade_curve(n, curve, "out")
    return out


def pad(
    audio: np.ndarray,
    head_samples: int = 0,
    tail_samples: int = 0,
) -> np.ndarray:
    """오디오 앞/뒤에 무음 패딩 추가. 마스터링 납품 표준 동작."""
    if head_samples < 0 or tail_samples < 0:
        raise ValueError("pad 길이는 음수일 수 없습니다.")
    if head_samples == 0 and tail_samples == 0:
        return audio.copy()

    if audio.ndim == 1:
        head = np.zeros(head_samples, dtype=audio.dtype)
        tail = np.zeros(tail_samples, dtype=audio.dtype)
        return np.concatenate([head, audio, tail])

    n_ch = audio.shape[0]
    head = np.zeros((n_ch, head_samples), dtype=audio.dtype)
    tail = np.zeros((n_ch, tail_samples), dtype=audio.dtype)
    return np.concatenate([head, audio, tail], axis=1)


def remove_dc(audio: np.ndarray) -> np.ndarray:
    """DC offset 제거. 채널별로 독립 mean을 빼낸다.

    마스터링 납품 전 표준 절차. 미세한 DC bias가 있으면 후속 처리 시
    intermodulation/headroom 손실이 발생한다.
    """
    if audio.ndim == 1:
        return (audio - np.mean(audio)).astype(audio.dtype)
    out = audio.copy()
    for ch in range(out.shape[0]):
        out[ch] = out[ch] - np.mean(out[ch])
    return out


def measure_dc_offset(audio: np.ndarray) -> list[float]:
    """채널별 DC offset (mean) 반환."""
    if audio.ndim == 1:
        return [float(np.mean(audio))]
    return [float(np.mean(audio[ch])) for ch in range(audio.shape[0])]


def trim(
    audio: np.ndarray,
    start: int = 0,
    end: int | None = None,
) -> np.ndarray:
    """샘플 단위 트리밍"""
    if audio.ndim == 1:
        return audio[start:end]
    return audio[:, start:end]


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    pad_samples: int = 0,
) -> np.ndarray:
    """앞뒤 silence 제거"""
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    threshold = 10 ** (threshold_db / 20.0)
    above = np.where(np.abs(mono) > threshold)[0]

    if len(above) == 0:
        return audio  # 전체가 silence면 원본 반환

    start = max(0, above[0] - pad_samples)
    end = min(len(mono), above[-1] + 1 + pad_samples)

    if audio.ndim == 1:
        return audio[start:end]
    return audio[:, start:end]


def normalize(
    audio: np.ndarray,
    peak_db: float | None = None,
    target_rms_db: float | None = None,
) -> np.ndarray:
    """피크 또는 RMS 기준 정규화"""
    out = audio.copy().astype(np.float32)

    if peak_db is not None:
        current_peak = np.max(np.abs(out))
        if current_peak < 1e-10:
            return out
        target_peak = 10 ** (peak_db / 20.0)
        out *= target_peak / current_peak

    elif target_rms_db is not None:
        current_rms = np.sqrt(np.mean(out**2))
        if current_rms < 1e-10:
            return out
        target_rms = 10 ** (target_rms_db / 20.0)
        out *= target_rms / current_rms

    return out


def gain(audio: np.ndarray, db: float) -> np.ndarray:
    """dB 단위 게인 적용"""
    return audio * (10 ** (db / 20.0))


def gate(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -50.0,
    attack_sec: float = 0.01,
    release_sec: float = 0.05,
    frame_size: int = 1024,
    hop_size: int = 512,
) -> np.ndarray:
    """RMS 기반 노이즈 게이트. 임계값 이하 구간을 silence로."""
    out = audio.copy()
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    threshold = 10 ** (threshold_db / 20.0)
    n_samples = len(mono)

    # 프레임별 RMS → envelope
    envelope = np.ones(n_samples, dtype=np.float32)
    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame_rms = np.sqrt(np.mean(mono[start : start + frame_size] ** 2))
        if frame_rms < threshold:
            envelope[start : start + hop_size] = 0.0

    # attack/release smoothing
    attack_samples = max(1, int(attack_sec * sample_rate))
    release_samples = max(1, int(release_sec * sample_rate))

    smoothed = np.copy(envelope)
    for i in range(1, n_samples):
        if smoothed[i] > smoothed[i - 1]:
            # attack (opening)
            alpha = 1.0 / attack_samples
            smoothed[i] = smoothed[i - 1] + alpha * (smoothed[i] - smoothed[i - 1])
        else:
            # release (closing)
            alpha = 1.0 / release_samples
            smoothed[i] = smoothed[i - 1] + alpha * (smoothed[i] - smoothed[i - 1])

    if out.ndim == 1:
        out *= smoothed
    else:
        out *= smoothed[np.newaxis, :]

    return out
