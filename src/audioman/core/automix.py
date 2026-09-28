# Created: 2026-04-07
# Purpose: Automix — hierarchical gain staging (K-20 calibration)

import logging
import re as _re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

from audioman.core.audio_file import read_audio

logger = logging.getLogger(__name__)


@dataclass
class BandDefinition:
    """Frequency band definition"""
    name: str
    low_hz: float
    high_hz: float

    @property
    def center_hz(self) -> float:
        """Band center frequency (geometric mean)"""
        return float(np.sqrt(self.low_hz * self.high_hz))


# Default 4-band definition
DEFAULT_BANDS = [
    BandDefinition("sub",  20,    200),
    BandDefinition("low",  200,   800),
    BandDefinition("mid",  800,   4000),
    BandDefinition("high", 4000,  20000),
]


@dataclass
class AutomixResult:
    """automix result"""
    gains_db: list[float]
    band_analysis: list[dict]
    target_profile: dict
    residual_error_db: float
    groups: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ────────────────────────────────────────────────────────
# Instrument group classification + hierarchical gain staging
# ────────────────────────────────────────────────────────

INSTRUMENT_GROUPS = {
    "drums": {
        "keywords": [
            "kick", "snare", "tom", "hihat", "hi-hat", "hat",
            "overhead", "oh", "cymbal", "ride", "crash", "clap",
            "room", "drum", "tambourine", "shaker", "perc",
        ],
        # relative level within a group (dB) — Kick is the 0 dB reference
        "relative_levels": {
            "kick": 0.0,
            "snare": -3.0,
            "tom": -5.0,
            "overhead": -8.0,
            "hihat": -10.0,
            "cymbal": -10.0,
            "ride": -10.0,
            "crash": -10.0,
            "clap": -6.0,
            "room": -12.0,
            "tambourine": -12.0,
            "shaker": -14.0,
            "perc": -10.0,
            "_default": -6.0,
        },
    },
    "bass": {
        "keywords": ["bass"],
        "relative_levels": {"_default": 0.0},
    },
    "guitars": {
        "keywords": ["gtr", "guitar", "elecgtr", "acougtr"],
        "relative_levels": {"_default": 0.0},
    },
    "keys": {
        "keywords": ["keys", "piano", "organ", "synth", "pad", "rhodes"],
        "relative_levels": {"_default": 0.0},
    },
    "vocals": {
        "keywords": ["vox", "vocal", "leadvox", "backing", "chorus", "bv"],
        "relative_levels": {
            "lead": 0.0,
            "leadvox": 0.0,
            "_default": -4.0,
        },
    },
}

# Relative level between groups — Drums is the 0 dB reference
GROUP_BALANCE_DB = {
    "drums":    0.0,
    "bass":    +2.0,
    "guitars":  0.0,
    "keys":    -6.0,
    "vocals":  +4.0,
    "other":   -8.0,
}


def classify_tracks(
    track_paths: list[str | Path],
) -> dict[str, list[int]]:
    """Auto-classify instrument groups from track filenames"""
    result: dict[str, list[int]] = {g: [] for g in INSTRUMENT_GROUPS}
    result["other"] = []

    for i, path in enumerate(track_paths):
        fname = Path(path).stem.lower().replace("-", "").replace(" ", "").replace("_", "")
        matched = False
        for group_name, group_def in INSTRUMENT_GROUPS.items():
            for kw in group_def["keywords"]:
                if kw in fname:
                    result[group_name].append(i)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            result["other"].append(i)

    return {g: indices for g, indices in result.items() if indices}


def _match_relative_keyword(fname: str, relative_levels: dict) -> float:
    """Match the most specific relative-level keyword in the filename"""
    fname_lower = fname.lower().replace("-", "").replace(" ", "").replace("_", "")
    best_match = "_default"
    best_len = 0
    for keyword in relative_levels:
        if keyword == "_default":
            continue
        if keyword in fname_lower and len(keyword) > best_len:
            best_match = keyword
            best_len = len(keyword)
    return relative_levels.get(best_match, relative_levels.get("_default", 0.0))


# ────────────────────────────────────────────────────────
# K-weighting (ITU-R BS.1770)
# ────────────────────────────────────────────────────────

def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def k_weight_magnitude(freqs: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """ITU-R BS.1770 K-weighting frequency response (magnitude)"""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    # ITU-R BS.1770 De Man calibration coefficients. The RBJ cookbook coefficients are
    # recomputed at every sample rate, so the 48 kHz response is not reused at other rates.
    from pyloudnorm import IIRfilter

    shelf = IIRfilter(
        3.99984385397, 0.7071752369554193, 1681.9744509555319,
        sample_rate, "high_shelf_DeMan",
    )
    high_pass = IIRfilter(
        0.0, 0.5003270373253953, 38.13547087613982,
        sample_rate, "high_pass_DeMan",
    )
    b1, a1 = shelf.b, shelf.a
    b2, a2 = high_pass.b, high_pass.a
    w = 2.0 * np.pi * np.minimum(freqs, sample_rate / 2 - 1) / sample_rate

    def biquad_response(b, a, w):
        ejw = np.exp(-1j * w)
        ejw2 = np.exp(-2j * w)
        return (b[0] + b[1] * ejw + b[2] * ejw2) / (a[0] + a[1] * ejw + a[2] * ejw2)

    k_mag = np.abs(biquad_response(b1, a1, w) * biquad_response(b2, a2, w))
    if len(freqs) > 0 and freqs[0] == 0:
        k_mag[0] = 0.0
    return k_mag.astype(np.float64)


K20_REF_LUFS = -20.0


# ────────────────────────────────────────────────────────
# Per-genre spectral profiles (based on analysis of 197 multitracks)
# K-weighted, normalized to -1dBFS peak
# ────────────────────────────────────────────────────────

GENRE_PROFILES = {
    "electronica": {
        "description": "Electronica / Dance / Experimental (N=53)",
        "bands": {"sub": -25.5, "low": -28.4, "mid": -29.3, "high": -33.1},
    },
    "pop": {
        "description": "Pop / Singer-Songwriter (N=61)",
        "bands": {"sub": -25.6, "low": -25.9, "mid": -27.3, "high": -35.3},
    },
    "rock": {
        "description": "Alt Rock / Blues / Indie / Funk / Reggae (N=83)",
        "bands": {"sub": -25.3, "low": -25.9, "mid": -27.6, "high": -36.2},
    },
    "default": {
        "description": "All genres average (N=197)",
        "bands": {"sub": -25.4, "low": -26.6, "mid": -28.0, "high": -35.1},
    },
    # Based on k-means clustering (k=8) of the YouTube library (47,016 tracks).
    # A -25 dB offset is applied to the peak=0 dB normalized cluster centroids to
    # match the scale of the existing profiles (original values and inter-band relative structure preserved).
    "yt_rock": {
        "description": "YouTube c5 — broadband rock/pop standard (N=13447, largest cluster)",
        "bands": {"sub": -25.0, "low": -32.6, "mid": -36.9, "high": -47.8},
    },
    "yt_bright_pop": {
        "description": "YouTube c2 — bright K-pop/country (N=7637)",
        "bands": {"sub": -26.1, "low": -27.1, "mid": -32.0, "high": -46.9},
    },
    "yt_hiphop": {
        "description": "YouTube c0 — bass-heavy hip-hop/trap (N=7664)",
        "bands": {"sub": -25.0, "low": -37.6, "mid": -45.3, "high": -54.4},
    },
    "yt_mid_scoop": {
        "description": "YouTube c1 — mid-scooped pop/indie (N=6107)",
        "bands": {"sub": -25.2, "low": -29.6, "mid": -41.1, "high": -54.1},
    },
    "yt_low_heavy_vocal": {
        "description": "YouTube c6 — sub-cut low-heavy vocal-forward (N=4580)",
        "bands": {"sub": -30.7, "low": -25.3, "mid": -36.1, "high": -50.6},
    },
    "yt_high_dr": {
        "description": "YouTube c7 — very high dynamic range (N=4523, DR=40.8dB)",
        "bands": {"sub": -25.1, "low": -31.6, "mid": -36.8, "high": -48.4},
    },
    "yt_ballad": {
        "description": "YouTube c3 — extreme sub-cut minimal ballad (N=1909)",
        "bands": {"sub": -44.0, "low": -26.0, "mid": -31.7, "high": -51.6},
    },
    "yt_dark_lofi": {
        "description": "YouTube c4 — dark lo-fi / indie ambient (N=1151, high -70dB)",
        "bands": {"sub": -27.3, "low": -30.7, "mid": -45.1, "high": -70.8},
    },
    # Based on k-means (k=6) of Archive/Collections (1,713 tracks, club/techno domain).
    # Completely different genre distribution from YouTube — club DJ sets, label compilations, vinyl rips.
    "archive_techno_standard": {
        "description": "Archive c4 — club standard techno/house (N=647, largest)",
        "bands": {"sub": -25.0, "low": -36.7, "mid": -42.6, "high": -51.8},
    },
    "archive_sub_kick_driven": {
        "description": "Archive c5 — sub-kick driven techno (N=363)",
        "bands": {"sub": -25.0, "low": -43.9, "mid": -47.5, "high": -57.3},
    },
    "archive_minimal_sub": {
        "description": "Archive c1 — ultra-minimal sub-kick (N=340)",
        "bands": {"sub": -25.0, "low": -42.3, "mid": -53.7, "high": -57.0},
    },
    "archive_groovy_low": {
        "description": "Archive c2 — low-heavy groovy house (N=160)",
        "bands": {"sub": -26.0, "low": -27.9, "mid": -37.2, "high": -49.3},
    },
    "archive_dub_techno": {
        "description": "Archive c3 — very dark dub-techno (N=152, high -53dB)",
        "bands": {"sub": -25.3, "low": -41.0, "mid": -56.8, "high": -78.8},
    },
    "archive_midrange_ambient": {
        "description": "Archive c0 — midrange-only ambient/drone (N=51, sub-cut)",
        "bands": {"sub": -42.5, "low": -26.1, "mid": -31.2, "high": -49.4},
    },
}


def genre_profile(
    genre: str = "default",
    bands: Optional[list[BandDefinition]] = None,
) -> list[float]:
    """Return a precomputed per-genre spectral profile

    Per-band average K-weighted RMS extracted from 197 multitrack automix results.
    Supported genres: electronica, pop, rock, default (overall average)
    """
    if bands is None:
        bands = DEFAULT_BANDS

    profile_data = GENRE_PROFILES.get(genre, GENRE_PROFILES["default"])
    band_values = profile_data["bands"]

    return [band_values.get(b.name, -30.0) for b in bands]


# ────────────────────────────────────────────────────────
# Per-band RMS measurement
# ────────────────────────────────────────────────────────

def compute_band_rms(
    audio: np.ndarray,
    sample_rate: int,
    bands: Optional[list[BandDefinition]] = None,
    k_weighted: bool = True,
) -> list[float]:
    """Frame-wise FFT → per-band K-weighted RMS (dBFS)"""
    if bands is None:
        bands = DEFAULT_BANDS

    mono = _to_mono(audio)
    n_samples = len(mono)
    frame_size = 4096
    hop_size = 2048
    n_bands = len(bands)

    window = np.hanning(frame_size).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    k_mag = k_weight_magnitude(freqs, sample_rate) if k_weighted else None

    band_masks = [(freqs >= b.low_hz) & (freqs < b.high_hz) for b in bands]
    band_power_accum = [[] for _ in range(n_bands)]

    for start in range(0, n_samples - frame_size + 1, hop_size):
        frame = mono[start:start + frame_size]
        spectrum = np.abs(np.fft.rfft(frame * window))
        spectrum *= 2.0 / frame_size
        if k_mag is not None:
            spectrum = spectrum * k_mag
        power = spectrum ** 2
        for b_idx, mask in enumerate(band_masks):
            bp = power[mask]
            if len(bp) > 0:
                band_power_accum[b_idx].append(float(np.sum(bp)))

    band_rms = []
    for b_idx in range(n_bands):
        powers = band_power_accum[b_idx]
        if not powers or max(powers) < 1e-20:
            band_rms.append(-120.0)
        else:
            rms_linear = np.sqrt(np.mean(powers))
            band_rms.append(float(20.0 * np.log10(max(rms_linear, 1e-10))))
    return band_rms


def compute_broadband_rms_db(audio: np.ndarray, sample_rate: int, k_weighted: bool = True) -> float:
    """K-weighted broadband RMS (dBFS)"""
    mono = _to_mono(audio)
    if k_weighted:
        frame_size = 4096
        hop_size = 2048
        window = np.hanning(frame_size).astype(np.float32)
        freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
        k_mag = k_weight_magnitude(freqs, sample_rate)
        powers = []
        for start in range(0, len(mono) - frame_size + 1, hop_size):
            frame = mono[start:start + frame_size]
            spectrum = np.abs(np.fft.rfft(frame * window)) * 2.0 / frame_size * k_mag
            powers.append(float(np.sum(spectrum ** 2)))
        if not powers:
            return -120.0
        return float(20.0 * np.log10(max(np.sqrt(np.mean(powers)), 1e-10)))
    else:
        rms = float(np.sqrt(np.mean(mono ** 2)))
        return float(20.0 * np.log10(max(rms, 1e-10)))


# ────────────────────────────────────────────────────────
# Target profiles
# ────────────────────────────────────────────────────────

def pink_noise_profile(
    bands: Optional[list[BandDefinition]] = None,
    ref_level_db: float = -20.0,
) -> list[float]:
    """Pink noise (-3 dB/octave) profile"""
    if bands is None:
        bands = DEFAULT_BANDS
    ref_center = bands[0].center_hz
    return [float(ref_level_db - 10.0 * np.log10(b.center_hz / ref_center)) for b in bands]


def reference_profile(
    ref_path: str | Path,
    bands: Optional[list[BandDefinition]] = None,
    k_weighted: bool = True,
) -> list[float]:
    """Extract the per-band RMS profile from a reference track"""
    if bands is None:
        bands = DEFAULT_BANDS
    audio, sr = read_audio(ref_path)
    return compute_band_rms(audio, sr, bands, k_weighted=k_weighted)


# ────────────────────────────────────────────────────────
# Hierarchical gain staging
# ────────────────────────────────────────────────────────

def compute_automix_gains(
    tracks_band_rms: list[list[float]],
    target_band_rms: list[float],
    max_gain_db: float = 12.0,
    min_gain_db: float = -24.0,
    track_paths: Optional[list[str | Path]] = None,
    group_balance: Optional[dict[str, float]] = None,
    track_rms_db: Optional[list[float]] = None,
) -> tuple[list[float], float, Optional[dict]]:
    """Hierarchical gain staging

    Two stages:
      1) In-group balance — practice-based relative levels per instrument
         Kick=0 dB, Snare=-3 dB, OH=-8 dB, Tambourine=-12 dB, etc.
      2) Between-group balance — based on the target spectrum
         Drums=0 dB, Vocals=-1 dB, Bass=-2 dB, Guitars=-4 dB

    Falls back to the flat method (broadband RMS equalization) when track_paths is None.
    """
    n_tracks = len(tracks_band_rms)
    n_bands = len(target_band_rms)

    if n_tracks == 0:
        return [], 0.0, None

    # dBFS → linear power
    A = np.zeros((n_bands, n_tracks), dtype=np.float64)
    for t in range(n_tracks):
        for b in range(n_bands):
            A[b, t] = 10.0 ** (tracks_band_rms[t][b] / 10.0)
    target_vec = np.array([10.0 ** (db / 10.0) for db in target_band_rms])

    gains_db_arr = np.zeros(n_tracks, dtype=np.float64)
    groups_info = None

    if track_paths is not None and track_rms_db is not None:
        groups = classify_tracks(track_paths)
        groups_info = {g: [int(i) for i in indices] for g, indices in groups.items()}
        if group_balance is None:
            group_balance = GROUP_BALANCE_DB

        # ── Step 1: in-group balance ──
        # use each group's loudest track as the reference (0 dB) and apply relative levels to the rest
        for group_name, indices in groups.items():
            group_def = INSTRUMENT_GROUPS.get(group_name, {})
            rel_levels = group_def.get("relative_levels", {"_default": 0.0})

            # current RMS per track within the group
            group_rms = [(idx, track_rms_db[idx]) for idx in indices]
            ref_rms = max(rms for _, rms in group_rms)

            for idx, current_rms in group_rms:
                fname = Path(track_paths[idx]).stem
                target_relative = _match_relative_keyword(fname, rel_levels)
                desired_rms = ref_rms + target_relative
                gains_db_arr[idx] = desired_rms - current_rms

        # ── Step 2: between-group balance ──
        # compute each group's summed RMS and match the between-group relative levels
        group_sum_rms_db = {}
        for group_name, indices in groups.items():
            # summed power after applying in-group gains
            group_power = 0.0
            for idx in indices:
                gain_linear = 10.0 ** (gains_db_arr[idx] / 20.0)
                rms_linear = 10.0 ** (track_rms_db[idx] / 20.0)
                group_power += (rms_linear * gain_linear) ** 2
            group_sum_rms_db[group_name] = 10.0 * np.log10(max(group_power, 1e-20))

        # reference group = drums (if present)
        ref_group = "drums" if "drums" in groups else max(
            groups, key=lambda g: group_sum_rms_db.get(g, -120)
        )
        ref_group_rms = group_sum_rms_db[ref_group]

        for group_name, indices in groups.items():
            if group_name == ref_group:
                continue
            target_offset = group_balance.get(group_name, group_balance.get("other", -8.0))
            ref_offset = group_balance.get(ref_group, 0.0)
            desired_diff = target_offset - ref_offset

            current_diff = group_sum_rms_db[group_name] - ref_group_rms
            correction = desired_diff - current_diff

            for idx in indices:
                gains_db_arr[idx] += correction

        # ── Step 3: match the overall level to the target ──
        gains_linear = 10.0 ** (gains_db_arr / 10.0)
        mix_power = np.zeros(n_bands)
        for t in range(n_tracks):
            mix_power += A[:, t] * gains_linear[t]
        mix_total = np.sum(mix_power)
        target_total = np.sum(target_vec)
        if mix_total > 1e-20:
            gains_db_arr += 10.0 * np.log10(target_total / mix_total)

    else:
        # ── Flat method ──
        target_total = np.sum(target_vec)
        for t in range(n_tracks):
            tp = np.sum(A[:, t])
            if tp > 1e-20:
                gains_db_arr[t] = 10.0 * np.log10(target_total / n_tracks / tp)

    # clip + round
    gains_db_arr = np.clip(gains_db_arr, min_gain_db, max_gain_db)
    gains_db = [round(float(g), 1) for g in gains_db_arr]

    # residual
    gl = 10.0 ** (gains_db_arr / 10.0)
    recon = np.zeros(n_bands)
    for t in range(n_tracks):
        recon += A[:, t] * gl[t]
    residual_db = float(10.0 * np.log10(max(np.mean((recon - target_vec) ** 2), 1e-20)))

    logger.info(f"Automix gains: {[f'{g:+.1f}dB' for g in gains_db]}, residual={residual_db:.1f}dB")
    return gains_db, residual_db, groups_info


# ────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────

def automix(
    track_paths: list[str | Path],
    target: str = "pink",
    reference_path: Optional[str | Path] = None,
    bands: Optional[list[BandDefinition]] = None,
    ref_level_db: float = K20_REF_LUFS,
    max_gain_db: float = 12.0,
    min_gain_db: float = -24.0,
    k_weighted: bool = True,
) -> AutomixResult:
    """Hierarchical automix — in-group/between-group balancing + K-20 calibration

    1) Auto-classify instrument groups from track names (drums/bass/guitars/keys/vocals/other)
    2) In-group: engineer-practice-based relative levels (Kick=0, Snare=-3, OH=-8, etc.)
    3) Between-group: Drums=0, Vocals=-1, Bass=-2, Guitars=-4 dB
    4) Match the overall level to the K-20 pink noise (or reference) target
    """
    if bands is None:
        bands = DEFAULT_BANDS

    # target profile
    if target == "reference" and reference_path:
        target_rms = reference_profile(reference_path, bands, k_weighted=k_weighted)
        target_info = {"type": "reference", "path": str(reference_path), "k_weighted": k_weighted}
    elif target in GENRE_PROFILES:
        target_rms = genre_profile(target, bands)
        target_info = {"type": "genre", "genre": target,
                       "description": GENRE_PROFILES[target]["description"],
                       "k_weighted": k_weighted}
    else:
        target_rms = pink_noise_profile(bands, ref_level_db)
        target_info = {"type": "pink_noise", "ref_level_db": ref_level_db, "k_weighted": k_weighted}
    target_info["bands"] = {b.name: round(rms, 1) for b, rms in zip(bands, target_rms)}

    # per-track analysis
    tracks_band_rms = []
    track_rms_db = []
    band_analysis = []

    for path in track_paths:
        audio, sr = read_audio(path)
        rms = compute_band_rms(audio, sr, bands, k_weighted=k_weighted)
        broadband = compute_broadband_rms_db(audio, sr, k_weighted=k_weighted)
        tracks_band_rms.append(rms)
        track_rms_db.append(broadband)
        band_analysis.append({
            "path": str(path),
            "bands": {b.name: round(r, 1) for b, r in zip(bands, rms)},
            "rms_db": round(broadband, 1),
        })

    # hierarchical gain computation
    gains_db, residual, groups = compute_automix_gains(
        tracks_band_rms, target_rms,
        max_gain_db=max_gain_db,
        min_gain_db=min_gain_db,
        track_paths=track_paths,
        track_rms_db=track_rms_db,
    )

    return AutomixResult(
        gains_db=gains_db,
        band_analysis=band_analysis,
        target_profile=target_info,
        residual_error_db=round(residual, 1),
        groups=groups,
    )
