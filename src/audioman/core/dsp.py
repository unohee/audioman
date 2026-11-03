# Created: 2026-03-21
# Purpose: Built-in DSP functions (fade, trim, cut, splice, normalize, gate, gain)

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
    """Concatenate along the time axis. Every part must have the same channel count."""
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
    """Delete the middle region [start, end) and join the remaining sides.

    If crossfade_samples > 0, an equal-length linear crossfade is applied at the
    boundary to prevent clicks/pops. (The last N samples of the left tail are
    mixed with the first N samples of the right head.)
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
    """Insert the insert clip into base at position, or overwrite base with it.

    mode:
        "insert"    — splice in at position. base grows in length.
        "overwrite" — overwrite base from position for len(insert) samples. Length unchanged.
        "mix"       — add (mix) insert into base from position. Length unchanged.

    Raises ValueError on a channel-count mismatch; the caller is responsible for
    converting beforehand. crossfade_samples only applies to insert mode and is
    applied at both boundaries.
    """
    if _channels(base) != _channels(insert):
        raise ValueError(
            f"Channel count mismatch: base={_channels(base)}, insert={_channels(insert)}"
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

    raise ValueError(f"Unknown splice mode: {mode!r} (insert/overwrite/mix)")


def concat(clips: list[np.ndarray], crossfade_samples: int = 0) -> np.ndarray:
    """Concatenate several clips along the time axis. Every clip must have the
    same channel count.

    If crossfade_samples > 0, a linear crossfade is applied at every boundary
    between adjacent clips.
    """
    if not clips:
        return np.zeros(0, dtype=np.float32)
    ch = _channels(clips[0])
    for i, c in enumerate(clips):
        if _channels(c) != ch:
            raise ValueError(f"clips[{i}] channel count mismatch: {_channels(c)} != {ch}")

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
    """Fade curve of length n. direction: 'in' (0→1) or 'out' (1→0).

    - linear: linear amplitude
    - cosine: cosine equal-amplitude (S-curve, smoothest)
    - equal_power: sqrt(linear) — preserves RMS when summing (crossfade standard)
    - exponential: fast start/slow end (natural decay)
    - logarithmic: slow start/fast end
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if kind not in FADE_CURVES:
        raise ValueError(f"Unknown fade curve: {kind!r} (supported: {FADE_CURVES})")

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
    """Fade in. samples: fade length (in samples). curve: linear/cosine/equal_power/exponential/logarithmic"""
    if samples < 0:
        raise ValueError("fade samples cannot be negative")
    out = audio.copy()
    if samples == 0:
        return out
    if audio.ndim == 1:
        n = min(samples, len(out))
        out[:n] *= _fade_curve(n, curve, "in")
    else:
        n = min(samples, out.shape[1])
        out[:, :n] *= _fade_curve(n, curve, "in")
    return out


def fade_out(audio: np.ndarray, samples: int, curve: str = "linear") -> np.ndarray:
    """Fade out. samples: fade length (in samples). curve: linear/cosine/equal_power/exponential/logarithmic"""
    if samples < 0:
        raise ValueError("fade samples cannot be negative")
    out = audio.copy()
    if samples == 0:
        return out
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
    """Add silence padding before/after the audio. Standard mastering delivery practice."""
    if head_samples < 0 or tail_samples < 0:
        raise ValueError("pad length cannot be negative.")
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
    """Remove DC offset. Subtracts an independent mean per channel.

    Standard procedure before mastering delivery. A small DC bias causes
    intermodulation/headroom loss in later processing.
    """
    if audio.ndim == 1:
        return (audio - np.mean(audio)).astype(audio.dtype)
    out = audio.copy()
    for ch in range(out.shape[0]):
        out[ch] = out[ch] - np.mean(out[ch])
    return out


def measure_dc_offset(audio: np.ndarray) -> list[float]:
    """Return the per-channel DC offset (mean)."""
    if audio.ndim == 1:
        return [float(np.mean(audio))]
    return [float(np.mean(audio[ch])) for ch in range(audio.shape[0])]


def trim(
    audio: np.ndarray,
    start: int = 0,
    end: int | None = None,
) -> np.ndarray:
    """Trim in samples"""
    if audio.ndim == 1:
        return audio[start:end]
    return audio[:, start:end]


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -40.0,
    pad_samples: int = 0,
) -> np.ndarray:
    """Remove leading/trailing silence"""
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    threshold = 10 ** (threshold_db / 20.0)
    above = np.where(np.abs(mono) > threshold)[0]

    if len(above) == 0:
        return audio  # return the original if everything is silence

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
    """Normalize by peak or RMS"""
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
    """Apply gain in dB"""
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
    """RMS-based noise gate. Silences regions below the threshold."""
    out = audio.copy()
    if audio.ndim == 2:
        mono = audio.mean(axis=0)
    else:
        mono = audio

    threshold = 10 ** (threshold_db / 20.0)
    n_samples = len(mono)

    # per-frame RMS → envelope
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
