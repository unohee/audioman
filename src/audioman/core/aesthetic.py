# Created: 2026-04-29
# Purpose: Audio aesthetic issue screening with optional Essentia backend.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from audioman.core.analysis import long_term_spectrum, detect_hum as detect_hum_spectrum
from audioman.core.audio_file import read_audio
from audioman.core.qc import detect_clicks as detect_clicks_fallback


DEFAULT_ISSUES = (
    "click",
    "hum",
    "mouth_click",
    "sibilance",
    "breath",
    "background_noise",
    "rf_noise",
)

ISSUE_ALIASES = {
    "de_ess": "sibilance",
    "de-ess": "sibilance",
    "ess": "sibilance",
    "mouth-click": "mouth_click",
    "mouth_de_click": "mouth_click",
    "mouth-de-click": "mouth_click",
    "noise": "background_noise",
    "background-noise": "background_noise",
    "rf": "rf_noise",
    "rf-noise": "rf_noise",
}


@dataclass
class AestheticEvent:
    type: str
    start_sec: float
    end_sec: float
    confidence: float | None = None
    severity: str = "warn"
    backend: str = "fallback"
    detail: dict | None = None

    def to_dict(self) -> dict:
        out = {
            "type": self.type,
            "start_sec": round(self.start_sec, 6),
            "end_sec": round(self.end_sec, 6),
            "severity": self.severity,
            "backend": self.backend,
        }
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 4)
        if self.detail:
            out.update(self.detail)
        return out


def essentia_available() -> bool:
    try:
        import essentia.standard  # noqa: F401
    except Exception:
        return False
    return True


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    return mono.astype(np.float32, copy=False)


def _normalize_issue(issue: str) -> str:
    key = issue.strip().lower().replace(" ", "_")
    return ISSUE_ALIASES.get(key, key)


def _db(value: float, floor: float = 1e-12) -> float:
    return 20.0 * np.log10(max(float(value), floor))


def _power_db(value: float, floor: float = 1e-24) -> float:
    return 10.0 * np.log10(max(float(value), floor))


def _frame_bounds(n_samples: int, sample_rate: int, frame_ms: float, hop_ms: float) -> list[tuple[int, int]]:
    frame = max(16, int(round(frame_ms / 1000.0 * sample_rate)))
    hop = max(1, int(round(hop_ms / 1000.0 * sample_rate)))
    if n_samples < frame:
        return [(0, n_samples)] if n_samples else []
    return [(start, start + frame) for start in range(0, n_samples - frame + 1, hop)]


def _band_power(frame: np.ndarray, sample_rate: int, band: tuple[float, float]) -> float:
    if len(frame) < 2:
        return 0.0
    window = np.hanning(len(frame)).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
    freqs = np.fft.rfftfreq(len(frame), 1.0 / sample_rate)
    mask = (freqs >= band[0]) & (freqs < min(band[1], sample_rate / 2.0))
    if not np.any(mask):
        return 0.0
    return float(np.mean(spectrum[mask]))


def _spectral_flatness(frame: np.ndarray) -> float:
    if len(frame) < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) ** 2
    spectrum = spectrum[1:] + 1e-18
    if len(spectrum) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))


def _zero_crossing_rate(frame: np.ndarray) -> float:
    if len(frame) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.signbit(frame)))))


def _events_from_mask(
    mask: list[bool],
    bounds: list[tuple[int, int]],
    sample_rate: int,
    *,
    event_type: str,
    min_duration_sec: float,
    max_gap_sec: float,
    severity: str,
    backend: str,
    details: list[dict] | None = None,
) -> list[AestheticEvent]:
    if not mask:
        return []
    raw: list[AestheticEvent] = []
    start_idx: int | None = None
    for idx, active in enumerate(mask + [False]):
        if active and start_idx is None:
            start_idx = idx
        elif not active and start_idx is not None:
            end_idx = idx - 1
            start_sample = bounds[start_idx][0]
            end_sample = bounds[end_idx][1]
            duration = (end_sample - start_sample) / sample_rate
            if duration >= min_duration_sec:
                seg_details = details[start_idx : end_idx + 1] if details else []
                confidence = None
                detail = {"detector": f"heuristic.{event_type}"}
                if seg_details:
                    scores = [d.get("score") for d in seg_details if d.get("score") is not None]
                    confidence = float(np.clip(max(scores), 0.0, 1.0)) if scores else None
                    for key in ("rms_db", "ratio_db", "flatness", "zcr", "peak_frequency_hz"):
                        values = [d[key] for d in seg_details if key in d]
                        if values:
                            detail[key] = round(float(max(values)), 4)
                raw.append(
                    AestheticEvent(
                        type=event_type,
                        start_sec=start_sample / sample_rate,
                        end_sec=end_sample / sample_rate,
                        confidence=confidence,
                        severity=severity,
                        backend=backend,
                        detail=detail,
                    )
                )
            start_idx = None
    return _merge_events(raw, max_gap_sec=max_gap_sec, same_type=event_type)


def _merge_events(
    events: list[AestheticEvent],
    max_gap_sec: float,
    same_type: str | None = None,
) -> list[AestheticEvent]:
    if not events:
        return []
    events = sorted(events, key=lambda e: (e.type, e.start_sec, e.end_sec))
    merged: list[AestheticEvent] = [events[0]]
    for event in events[1:]:
        prev = merged[-1]
        type_matches = event.type == prev.type if same_type is None else event.type == same_type == prev.type
        if type_matches and event.start_sec - prev.end_sec <= max_gap_sec:
            prev.end_sec = max(prev.end_sec, event.end_sec)
            if event.confidence is not None:
                prev.confidence = max(prev.confidence or 0.0, event.confidence)
        else:
            merged.append(event)
    return merged


def detect_click_events(
    audio: np.ndarray,
    sample_rate: int,
    *,
    backend: str = "auto",
    detection_threshold: float = 30.0,
    frame_size: int = 512,
    hop_size: int = 256,
    fallback_sensitivity: float = 6.0,
) -> tuple[list[AestheticEvent], str]:
    if backend in ("auto", "essentia"):
        try:
            return _detect_click_events_essentia(
                audio,
                sample_rate,
                detection_threshold=detection_threshold,
                frame_size=frame_size,
                hop_size=hop_size,
            ), "essentia"
        except Exception:
            if backend == "essentia":
                raise

    result = detect_clicks_fallback(audio, sample_rate, sensitivity=fallback_sensitivity)
    events = [
        AestheticEvent(
            type="click",
            start_sec=float(loc),
            end_sec=float(loc) + 0.005,
            confidence=None,
            severity="fail",
            backend="fallback",
            detail={"detector": "qc.detect_clicks"},
        )
        for loc in result.get("locations_sec", [])
    ]
    return events, "fallback"


def _detect_click_events_essentia(
    audio: np.ndarray,
    sample_rate: int,
    *,
    detection_threshold: float,
    frame_size: int,
    hop_size: int,
) -> list[AestheticEvent]:
    from essentia.standard import ClickDetector, FrameGenerator

    mono = _to_mono_float32(audio)
    detector = ClickDetector(
        sampleRate=float(sample_rate),
        frameSize=frame_size,
        hopSize=hop_size,
        detectionThreshold=detection_threshold,
    )

    starts: list[float] = []
    ends: list[float] = []
    detector.reset()
    for frame in FrameGenerator(mono, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        frame_starts, frame_ends = detector(frame)
        starts.extend(float(v) for v in frame_starts)
        ends.extend(float(v) for v in frame_ends)

    events = []
    for start, end in zip(starts, ends, strict=False):
        end = max(end, start + 1.0 / sample_rate)
        events.append(
            AestheticEvent(
                type="click",
                start_sec=start,
                end_sec=end,
                confidence=None,
                severity="fail",
                backend="essentia",
                detail={"detector": "essentia.ClickDetector"},
            )
        )
    return _merge_events(events, max_gap_sec=0.003, same_type="click")


def detect_hum_events(
    audio: np.ndarray,
    sample_rate: int,
    *,
    backend: str = "auto",
    detection_threshold: float = 5.0,
    minimum_duration: float = 2.0,
    maximum_frequency: float = 400.0,
    number_harmonics: int = 4,
    fallback_snr_db: float = 10.0,
) -> tuple[list[AestheticEvent], str]:
    if backend in ("auto", "essentia"):
        try:
            return _detect_hum_events_essentia(
                audio,
                sample_rate,
                detection_threshold=detection_threshold,
                minimum_duration=minimum_duration,
                maximum_frequency=maximum_frequency,
                number_harmonics=number_harmonics,
            ), "essentia"
        except Exception:
            if backend == "essentia":
                raise

    duration = audio.shape[-1] / sample_rate
    freqs, power, _ = long_term_spectrum(audio, sample_rate, fft_size=16384, min_rms=0.0)
    hums = detect_hum_spectrum(freqs, power, snr_threshold_db=fallback_snr_db)
    events = []
    for hum in hums:
        if not hum.get("is_hum"):
            continue
        events.append(
            AestheticEvent(
                type="hum",
                start_sec=0.0,
                end_sec=duration,
                confidence=None,
                severity="warn",
                backend="fallback",
                detail={
                    "detector": "analysis.detect_hum",
                    "frequency_hz": hum["frequency_hz"],
                    "snr_db": hum["snr_db"],
                },
            )
        )
    return events, "fallback"


def _detect_hum_events_essentia(
    audio: np.ndarray,
    sample_rate: int,
    *,
    detection_threshold: float,
    minimum_duration: float,
    maximum_frequency: float,
    number_harmonics: int,
) -> list[AestheticEvent]:
    from essentia.standard import HumDetector

    mono = _to_mono_float32(audio)
    detector = HumDetector(
        sampleRate=float(sample_rate),
        detectionThreshold=detection_threshold,
        minimumDuration=minimum_duration,
        maximumFrequency=maximum_frequency,
        numberHarmonics=number_harmonics,
    )
    _, frequencies, saliences, starts, ends = detector(mono)

    events = []
    for frequency, salience, start, end in zip(frequencies, saliences, starts, ends, strict=False):
        events.append(
            AestheticEvent(
                type="hum",
                start_sec=float(start),
                end_sec=float(end),
                confidence=float(np.clip(salience, 0.0, 1.0)),
                severity="warn" if salience < 0.75 else "fail",
                backend="essentia",
                detail={
                    "detector": "essentia.HumDetector",
                    "frequency_hz": round(float(frequency), 3),
                    "salience": round(float(salience), 6),
                },
            )
        )
    return _merge_events(events, max_gap_sec=0.25, same_type="hum")


def detect_sibilance_events(
    audio: np.ndarray,
    sample_rate: int,
    *,
    ratio_threshold_db: float = 7.0,
    min_rms_db: float = -45.0,
    min_duration_sec: float = 0.025,
) -> tuple[list[AestheticEvent], str]:
    """Detect excessive de-ess candidates as high-band bursts.

    This is intentionally a screening detector, not a de-esser. It marks short
    regions where 4-10 kHz energy dominates the low-mid voice band.
    """
    mono = _to_mono_float32(audio)
    bounds = _frame_bounds(len(mono), sample_rate, frame_ms=20.0, hop_ms=10.0)
    mask: list[bool] = []
    details: list[dict] = []
    for start, end in bounds:
        frame = mono[start:end]
        rms_db = _db(float(np.sqrt(np.mean(frame**2))))
        low_mid = _band_power(frame, sample_rate, (700.0, 3500.0))
        sib = _band_power(frame, sample_rate, (4000.0, 10000.0))
        ratio_db = _power_db(sib / max(low_mid, 1e-24))
        active = rms_db >= min_rms_db and ratio_db >= ratio_threshold_db
        score = (ratio_db - ratio_threshold_db) / 12.0
        mask.append(active)
        details.append({"rms_db": rms_db, "ratio_db": ratio_db, "score": score})
    return (
        _events_from_mask(
            mask,
            bounds,
            sample_rate,
            event_type="sibilance",
            min_duration_sec=min_duration_sec,
            max_gap_sec=0.04,
            severity="warn",
            backend="heuristic",
            details=details,
        ),
        "heuristic",
    )


def detect_breath_events(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_rms_db: float = -55.0,
    max_rms_db: float = -18.0,
    min_duration_sec: float = 0.12,
) -> tuple[list[AestheticEvent], str]:
    """Detect breath-like airy broadband segments."""
    mono = _to_mono_float32(audio)
    bounds = _frame_bounds(len(mono), sample_rate, frame_ms=35.0, hop_ms=15.0)
    mask: list[bool] = []
    details: list[dict] = []
    for start, end in bounds:
        frame = mono[start:end]
        rms_db = _db(float(np.sqrt(np.mean(frame**2))))
        low = _band_power(frame, sample_rate, (80.0, 700.0))
        air = _band_power(frame, sample_rate, (1200.0, 8500.0))
        ratio_db = _power_db(air / max(low, 1e-24))
        flatness = _spectral_flatness(frame)
        zcr = _zero_crossing_rate(frame)
        active = min_rms_db <= rms_db <= max_rms_db and ratio_db >= 4.0 and zcr >= 0.04
        score = min((ratio_db - 4.0) / 14.0, zcr * 5.0)
        mask.append(active)
        details.append({"rms_db": rms_db, "ratio_db": ratio_db, "flatness": flatness, "zcr": zcr, "score": score})
    return (
        _events_from_mask(
            mask,
            bounds,
            sample_rate,
            event_type="breath",
            min_duration_sec=min_duration_sec,
            max_gap_sec=0.08,
            severity="warn",
            backend="heuristic",
            details=details,
        ),
        "heuristic",
    )


def detect_background_noise_events(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_rms_db: float = -70.0,
    max_rms_db: float = -28.0,
    min_duration_sec: float = 0.3,
) -> tuple[list[AestheticEvent], str]:
    """Detect sustained broadband floor/noise sections."""
    mono = _to_mono_float32(audio)
    bounds = _frame_bounds(len(mono), sample_rate, frame_ms=100.0, hop_ms=50.0)
    mask: list[bool] = []
    details: list[dict] = []
    for start, end in bounds:
        frame = mono[start:end]
        rms_db = _db(float(np.sqrt(np.mean(frame**2))))
        flatness = _spectral_flatness(frame)
        zcr = _zero_crossing_rate(frame)
        active = min_rms_db <= rms_db <= max_rms_db and zcr >= 0.03
        score = (rms_db - min_rms_db) / max(max_rms_db - min_rms_db, 1e-9)
        mask.append(active)
        details.append({"rms_db": rms_db, "flatness": flatness, "zcr": zcr, "score": score})
    return (
        _events_from_mask(
            mask,
            bounds,
            sample_rate,
            event_type="background_noise",
            min_duration_sec=min_duration_sec,
            max_gap_sec=0.15,
            severity="warn",
            backend="heuristic",
            details=details,
        ),
        "heuristic",
    )


def detect_rf_noise_events(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_frequency: float = 1500.0,
    max_frequency: float | None = None,
    snr_threshold_db: float = 18.0,
    min_tones: int = 2,
) -> tuple[list[AestheticEvent], str]:
    """Detect persistent narrowband high-frequency tones often heard as RF whine."""
    mono = _to_mono_float32(audio)
    duration = len(mono) / sample_rate
    if len(mono) < 2048:
        return [], "heuristic"
    fft_size = min(32768, 2 ** int(np.floor(np.log2(len(mono)))))
    if fft_size < 2048:
        return [], "heuristic"
    window = np.hanning(fft_size).astype(np.float32)
    frame = mono[:fft_size] if len(mono) == fft_size else mono[:fft_size]
    spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    if max_frequency is None:
        max_frequency = min(20000.0, sample_rate / 2.0 - 100.0)
    mask = (freqs >= min_frequency) & (freqs <= max_frequency)
    candidate_idx = np.where(mask)[0]
    tones: list[dict] = []
    for idx in candidate_idx:
        lo = max(1, idx - 30)
        hi = min(len(spectrum), idx + 31)
        neighborhood = np.concatenate([spectrum[lo:max(lo, idx - 3)], spectrum[min(hi, idx + 4):hi]])
        if len(neighborhood) == 0:
            continue
        floor = float(np.median(neighborhood))
        snr = _power_db(float(spectrum[idx]) / max(floor, 1e-24))
        if snr < snr_threshold_db:
            continue
        frequency = float(freqs[idx])
        if any(abs(frequency - tone["frequency_hz"]) < 80.0 for tone in tones):
            continue
        tones.append({"frequency_hz": round(frequency, 2), "snr_db": round(snr, 2)})
    tones = sorted(tones, key=lambda t: t["snr_db"], reverse=True)[:8]
    if len(tones) < min_tones:
        return [], "heuristic"
    confidence = float(np.clip((max(t["snr_db"] for t in tones) - snr_threshold_db) / 18.0, 0.0, 1.0))
    return [
        AestheticEvent(
            type="rf_noise",
            start_sec=0.0,
            end_sec=duration,
            confidence=confidence,
            severity="warn" if confidence < 0.8 else "fail",
            backend="heuristic",
            detail={"detector": "heuristic.rf_noise", "tones": tones},
        )
    ], "heuristic"


def detect_mouth_click_events(
    audio: np.ndarray,
    sample_rate: int,
    *,
    backend: str = "auto",
) -> tuple[list[AestheticEvent], str]:
    """Detect mouth-click candidates by reclassifying click events with mid/high skew."""
    click_events, used_backend = detect_click_events(
        audio,
        sample_rate,
        backend=backend,
        detection_threshold=24.0,
        fallback_sensitivity=4.5,
    )
    mono = _to_mono_float32(audio)
    events: list[AestheticEvent] = []
    pad = max(1, int(0.012 * sample_rate))
    for click in click_events:
        center = int(round(click.start_sec * sample_rate))
        start = max(0, center - pad)
        end = min(len(mono), center + pad)
        frame = mono[start:end]
        if len(frame) < 16:
            continue
        low = _band_power(frame, sample_rate, (80.0, 900.0))
        mouth = _band_power(frame, sample_rate, (1200.0, 7000.0))
        ratio_db = _power_db(mouth / max(low, 1e-24))
        rms_db = _db(float(np.sqrt(np.mean(frame**2))))
        if ratio_db < -25.0 or rms_db < -65.0:
            continue
        confidence = float(np.clip((ratio_db + 25.0) / 25.0, 0.0, 1.0))
        events.append(
            AestheticEvent(
                type="mouth_click",
                start_sec=click.start_sec,
                end_sec=max(click.end_sec, click.start_sec + 0.006),
                confidence=confidence,
                severity="warn" if confidence < 0.75 else "fail",
                backend=used_backend,
                detail={
                    "detector": "heuristic.mouth_click",
                    "source_detector": click.detail.get("detector") if click.detail else None,
                    "ratio_db": round(ratio_db, 3),
                    "rms_db": round(rms_db, 3),
                },
            )
        )
    return _merge_events(events, max_gap_sec=0.01, same_type="mouth_click"), used_backend


def screen_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    issues: Iterable[str] = DEFAULT_ISSUES,
    backend: str = "auto",
) -> dict:
    requested = tuple(dict.fromkeys(_normalize_issue(i) for i in issues if i.strip()))
    events: list[AestheticEvent] = []
    backends: dict[str, str] = {}
    unsupported: list[str] = []

    for issue in requested:
        if issue == "click":
            detected, used = detect_click_events(audio, sample_rate, backend=backend)
            events.extend(detected)
            backends[issue] = used
        elif issue == "hum":
            detected, used = detect_hum_events(audio, sample_rate, backend=backend)
            events.extend(detected)
            backends[issue] = used
        elif issue == "mouth_click":
            detected, used = detect_mouth_click_events(audio, sample_rate, backend=backend)
            events.extend(detected)
            backends[issue] = used
        elif issue == "sibilance":
            detected, used = detect_sibilance_events(audio, sample_rate)
            events.extend(detected)
            backends[issue] = used
        elif issue == "breath":
            detected, used = detect_breath_events(audio, sample_rate)
            events.extend(detected)
            backends[issue] = used
        elif issue == "background_noise":
            detected, used = detect_background_noise_events(audio, sample_rate)
            events.extend(detected)
            backends[issue] = used
        elif issue == "rf_noise":
            detected, used = detect_rf_noise_events(audio, sample_rate)
            events.extend(detected)
            backends[issue] = used
        else:
            unsupported.append(issue)

    event_dicts = [event.to_dict() for event in sorted(events, key=lambda e: (e.start_sec, e.type))]
    summary = {issue: 0 for issue in requested}
    for event in event_dicts:
        summary[event["type"]] = summary.get(event["type"], 0) + 1

    return {
        "sample_rate": sample_rate,
        "channels": 1 if audio.ndim == 1 else audio.shape[0],
        "duration": round(audio.shape[-1] / sample_rate, 6),
        "issues": list(requested),
        "backends": backends,
        "essentia_available": None if backend == "fallback" else essentia_available(),
        "unsupported_issues": unsupported,
        "events": event_dicts,
        "summary": summary,
    }


def screen_file(
    file_path: str | Path,
    *,
    issues: Iterable[str] = DEFAULT_ISSUES,
    backend: str = "auto",
) -> dict:
    audio, sr = read_audio(file_path)
    report = screen_audio(audio, sr, issues=issues, backend=backend)
    report["file"] = str(file_path)
    return report
