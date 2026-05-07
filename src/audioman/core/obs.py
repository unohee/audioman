# Created: 2026-05-07
# Purpose: OBS 멀티트랙 영상 자동 진단 — 트랙 토폴로지 분류, voice/music/fullmix 분류,
#          spectrum_diagnostics + qc.evaluate를 묶어 처치 계획(dry-run) 생성.
# Dependencies: ffmpeg/ffprobe (외부 CLI), audioman.core.{analysis, qc, vad, audio_file, loudness}
#
# 사용 가이드: docs/obs-workflow.md
# CLI:        cli/obs.py (audioman obs probe / audioman obs dry-run)
# 진입점:      probe_topology() → classify_track() → diagnose_track() → recommend_treatment()
#             또는 dry_run_video()로 영상 1개 통합 실행

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

from audioman.core.analysis import spectrum_diagnostics
from audioman.core.audio_file import read_audio
from audioman.core.loudness import measure
from audioman.core.qc import (
    channel_imbalance_db,
    detect_clicks,
    detect_clipping,
    head_tail_silence,
    stereo_phase_correlation,
)
from audioman.core.vad import detect_speech

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 토폴로지 분류 (트랙 간 RMS 분포로 OBS 녹화 모드 식별)
# ---------------------------------------------------------------------------

Topology = Literal["multitrack", "single", "duplicated", "silent", "unknown"]
TrackKind = Literal["voice", "music", "fullmix", "silent", "unknown"]

_RMS_SILENCE_THRESHOLD = 1e-4
_RMS_DUPLICATE_TOLERANCE = 1e-3  # |rmsA - rmsB| < tol → 같은 신호로 간주


@dataclass
class TrackProbe:
    """트랙 한 개의 빠른 RMS 프로브 결과."""
    index: int
    rms: float
    is_silent: bool
    is_stereo: bool


@dataclass
class TopologyReport:
    topology: Topology
    n_streams: int
    track_probes: list[TrackProbe]
    active_indices: list[int]                    # 신호 있는 트랙
    unique_signal_groups: list[list[int]]        # 같은 RMS인 트랙끼리 그룹핑
    sample_rate: int
    duration_sec: float

    def to_dict(self) -> dict:
        return {
            "topology": self.topology,
            "n_streams": self.n_streams,
            "active_indices": self.active_indices,
            "unique_signal_groups": self.unique_signal_groups,
            "sample_rate": self.sample_rate,
            "duration_sec": round(self.duration_sec, 3),
            "tracks": [
                {
                    "index": t.index,
                    "rms": round(t.rms, 5),
                    "is_silent": t.is_silent,
                    "is_stereo": t.is_stereo,
                }
                for t in self.track_probes
            ],
        }


def _ensure_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe가 PATH에 없습니다.")


def _ffprobe_streams(video: Path) -> list[dict]:
    """video의 모든 audio 스트림 메타데이터."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries",
        "stream=index,codec_name,channels,sample_rate,duration,duration_ts",
        "-of", "json",
        str(video),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(r.stdout)
    return data.get("streams", [])


def _extract_track_to_wav(
    video: Path,
    audio_index: int,
    out_path: Path,
    *,
    duration_sec: float | None = None,
    start_sec: float = 0.0,
    sample_rate: int | None = None,
    mono: bool = False,
) -> bool:
    """ffmpeg로 video의 (0:a:audio_index) 트랙을 WAV로 추출. 성공 여부 반환."""
    cmd: list[str] = ["ffmpeg", "-y", "-loglevel", "error"]
    if start_sec > 0:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-i", str(video), "-map", f"0:a:{audio_index}"]
    if duration_sec is not None:
        cmd += ["-t", f"{duration_sec:.3f}"]
    if mono:
        cmd += ["-ac", "1"]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    cmd += ["-c:a", "pcm_s24le", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.debug("ffmpeg 추출 실패 idx=%d: %s", audio_index, r.stderr.strip()[:200])
        return False
    return out_path.exists() and out_path.stat().st_size > 1024


def probe_topology(
    video: str | Path,
    *,
    probe_seconds: float = 15.0,
    probe_sample_rate: int = 16000,
) -> TopologyReport:
    """ffprobe로 트랙 수 파악 → 각 트랙 앞 N초만 mono 16k로 추출 → RMS로 토폴로지 분류.

    분류 규칙:
      - n_active == 0       → silent
      - n_active == 1       → single
      - active 트랙 RMS가 모두 _RMS_DUPLICATE_TOLERANCE 안에 들어옴 → duplicated
      - 그 외                → multitrack
    """
    _ensure_tools()
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(f"파일 없음: {video}")

    streams = _ffprobe_streams(video)
    if not streams:
        return TopologyReport(
            topology="silent", n_streams=0, track_probes=[],
            active_indices=[], unique_signal_groups=[],
            sample_rate=0, duration_sec=0.0,
        )

    sr = int(streams[0].get("sample_rate", 0) or 0)
    duration_sec = 0.0
    try:
        # container duration 우선
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "csv=p=0", str(video)]
        duration_sec = float(subprocess.check_output(cmd, text=True).strip() or 0.0)
    except Exception:
        pass

    probes: list[TrackProbe] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for stream_pos, st in enumerate(streams):
            channels = int(st.get("channels", 1) or 1)
            wav = tmp_dir / f"probe_{stream_pos}.wav"
            ok = _extract_track_to_wav(
                video, stream_pos, wav,
                duration_sec=probe_seconds,
                sample_rate=probe_sample_rate,
                mono=True,
            )
            if not ok:
                probes.append(TrackProbe(stream_pos, 0.0, True, channels >= 2))
                continue
            data, _ = sf.read(str(wav), dtype="float32")
            if data.ndim == 2:
                data = data.mean(axis=1)
            rms = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
            probes.append(TrackProbe(
                index=stream_pos,
                rms=rms,
                is_silent=rms < _RMS_SILENCE_THRESHOLD,
                is_stereo=channels >= 2,
            ))

    active = [p for p in probes if not p.is_silent]
    active_indices = [p.index for p in active]

    # 같은 RMS인 트랙끼리 그룹핑
    groups: list[list[int]] = []
    used = set()
    for i, p in enumerate(active):
        if p.index in used:
            continue
        grp = [p.index]
        used.add(p.index)
        for q in active[i + 1:]:
            if q.index in used:
                continue
            if abs(p.rms - q.rms) < _RMS_DUPLICATE_TOLERANCE:
                grp.append(q.index)
                used.add(q.index)
        groups.append(grp)

    # 토폴로지 결정
    if not active:
        topology: Topology = "silent"
    elif len(active) == 1:
        topology = "single"
    elif len(groups) == 1 and len(active) == len(probes):
        # 모든 트랙이 활성 + 모두 같은 RMS → 복제 마스터 믹스
        topology = "duplicated"
    elif len(groups) == 1 and len(active) >= 2:
        # 활성 트랙이 모두 같은 RMS (silent 트랙은 따로)
        topology = "duplicated"
    else:
        topology = "multitrack"

    return TopologyReport(
        topology=topology,
        n_streams=len(streams),
        track_probes=probes,
        active_indices=active_indices,
        unique_signal_groups=groups,
        sample_rate=sr,
        duration_sec=duration_sec,
    )


# ---------------------------------------------------------------------------
# 트랙 분류 (voice / music / fullmix / silent)
# ---------------------------------------------------------------------------


@dataclass
class TrackClassification:
    kind: TrackKind
    speech_ratio: float           # speech_sec / duration_sec
    sub_band_pct: float
    low_band_pct: float
    presence_band_pct: float
    hf_slope_db: float | None
    confidence: float             # 0~1

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "speech_ratio": round(self.speech_ratio, 3),
            "sub_band_pct": round(self.sub_band_pct, 2),
            "low_band_pct": round(self.low_band_pct, 2),
            "presence_band_pct": round(self.presence_band_pct, 2),
            "hf_slope_db": self.hf_slope_db,
        }


def classify_track(
    audio: np.ndarray,
    sample_rate: int,
) -> TrackClassification:
    """VAD + spectrum_diagnostics로 트랙 종류 추정.

    경험 규칙 (예측치, dry-run용 — 의사결정 보조):
      - speech_ratio > 0.4  AND  sub < 10%  AND  presence > 1%  → voice
      - speech_ratio < 0.05 AND  sub > 15%                       → music
      - speech_ratio > 0.2  AND  sub > 15%                       → fullmix (음성+음악)
      - 그 외                                                       → fullmix (보수적)
      - rms ~ 0                                                   → silent
    """
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < _RMS_SILENCE_THRESHOLD:
        return TrackClassification(
            kind="silent", speech_ratio=0.0,
            sub_band_pct=0.0, low_band_pct=0.0, presence_band_pct=0.0,
            hf_slope_db=None, confidence=1.0,
        )

    speech = detect_speech(audio, sample_rate)
    n_samp = audio.shape[-1] if audio.ndim == 2 else len(audio)
    speech_sec = sum(s.duration_samples for s in speech) / sample_rate
    duration_sec = n_samp / sample_rate
    speech_ratio = speech_sec / max(duration_sec, 1e-9)

    diag = spectrum_diagnostics(audio, sample_rate, fft_size=8192, min_rms=0.005)
    bands = {b["band"]: b["percent"] for b in diag["band_energy"]}
    sub_pct = bands.get("sub", 0.0)
    low_pct = bands.get("low", 0.0)
    presence_pct = bands.get("presence", 0.0)
    hf_slope = diag["hf_slope"].get("slope_db")

    if speech_ratio > 0.4 and sub_pct < 10.0 and presence_pct > 1.0:
        kind: TrackKind = "voice"
        confidence = min(1.0, 0.5 + speech_ratio * 0.5)
    elif speech_ratio < 0.05 and sub_pct > 15.0:
        kind = "music"
        confidence = min(1.0, 0.5 + (sub_pct - 15.0) / 50.0)
    elif speech_ratio > 0.2 and sub_pct > 15.0:
        kind = "fullmix"
        confidence = 0.7
    elif speech_ratio > 0.2:
        # 음성 위주이지만 저역이 약하지 않음 — 보수적으로 fullmix
        kind = "voice" if presence_pct > 2.0 and sub_pct < 5.0 else "fullmix"
        confidence = 0.6
    else:
        kind = "fullmix"
        confidence = 0.4

    return TrackClassification(
        kind=kind,
        speech_ratio=speech_ratio,
        sub_band_pct=sub_pct,
        low_band_pct=low_pct,
        presence_band_pct=presence_pct,
        hf_slope_db=hf_slope,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# 트랙 진단 + 처치 추천
# ---------------------------------------------------------------------------


@dataclass
class Treatment:
    action: str                   # "denoise" / "dehum" / "declip" / "leveling" / "stem_separate" / ...
    plugin_short: str | None      # registry short name (없으면 None)
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    severity: str = "info"        # "info" / "warn" / "critical"

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "plugin": self.plugin_short,
            "params": self.params,
            "rationale": self.rationale,
            "severity": self.severity,
        }


def _measure_dc_offset(audio: np.ndarray) -> list[float]:
    if audio.ndim == 1:
        return [float(np.mean(audio))]
    return [float(np.mean(audio[ch])) for ch in range(audio.shape[0])]


def diagnose_track(
    audio: np.ndarray,
    sample_rate: int,
    classification: TrackClassification,
) -> dict:
    """트랙에 대해 spectrum_diagnostics + 핵심 QC를 모은 통합 진단."""
    diag = spectrum_diagnostics(audio, sample_rate, fft_size=16384, min_rms=0.005)
    loud = measure(audio, sample_rate).to_dict()
    clip = detect_clipping(audio)
    dc = _measure_dc_offset(audio)
    # click 검출은 비싸므로 60s 이하 발췌에만 (긴 파일은 요약만)
    n_samp = audio.shape[-1] if audio.ndim == 2 else len(audio)
    if n_samp / sample_rate <= 600:
        clicks = detect_clicks(audio, sample_rate)
    else:
        clicks = {"n_clicks": None, "skipped": "duration > 10min"}
    silences = head_tail_silence(audio, sample_rate)

    out = {
        "classification": classification.to_dict(),
        "loudness": loud,
        "spectrum": diag,
        "clipping": clip,
        "dc_offset_max": max(abs(x) for x in dc),
        "clicks": clicks,
        "head_tail_silence": silences,
    }
    if audio.ndim == 2 and audio.shape[0] == 2:
        out["phase"] = stereo_phase_correlation(audio, sample_rate=sample_rate)
        out["channel_imbalance"] = channel_imbalance_db(audio)
    return out


def recommend_treatment(track_diag: dict) -> list[Treatment]:
    """진단 결과 → 처치 계획 룰 엔진."""
    plan: list[Treatment] = []
    cls = track_diag["classification"]
    kind: TrackKind = cls["kind"]
    loud = track_diag["loudness"]
    spec = track_diag["spectrum"]
    clip = track_diag["clipping"]
    clicks = track_diag.get("clicks", {})
    dc_max = track_diag.get("dc_offset_max", 0.0)
    silences = track_diag.get("head_tail_silence", {})

    # 1. Hum (50/60Hz)
    hum_hits = [h for h in spec.get("hum_check", []) if h.get("is_hum")]
    if hum_hits:
        plan.append(Treatment(
            action="dehum",
            plugin_short="de-hum",
            params={"frequencies_hz": [h["frequency_hz"] for h in hum_hits]},
            rationale=f"전원 험 검출: {[(h['frequency_hz'], h['snr_db']) for h in hum_hits]}",
            severity="warn",
        ))

    # 2. Clipping → de-clip 우선
    if clip.get("n_samples", 0) > 0:
        sev = "critical" if clip["n_samples"] > 100 else "warn"
        plan.append(Treatment(
            action="declip",
            plugin_short="de-clip",
            params={"clipped_samples": clip["n_samples"]},
            rationale=f"클리핑 샘플 {clip['n_samples']}개",
            severity=sev,
        ))

    # 3. DC offset → HPF
    if dc_max > 0.001:
        plan.append(Treatment(
            action="dc_removal",
            plugin_short=None,
            params={"hpf_hz": 5.0},
            rationale=f"DC offset {dc_max:.4f} > 0.001",
            severity="warn",
        ))

    # 4. 트랙 종류별 핵심 처치
    if kind == "voice":
        # speech_ratio 높고 저역 적음 → voice de-noise
        plan.append(Treatment(
            action="denoise",
            plugin_short="voice-de-noise",
            params={},
            rationale=f"voice 트랙 (speech_ratio={cls['speech_ratio']}, presence={cls['presence_band_pct']}%)",
            severity="info",
        ))
        # voiceover.process는 leveling까지 묶여있음
        plan.append(Treatment(
            action="leveling",
            plugin_short=None,
            params={"target_lufs": -20.0, "max_true_peak_dbtp": -1.0},
            rationale="발화 단위 LUFS 평탄화",
            severity="info",
        ))

    elif kind == "music":
        # 음악 트랙: denoise 금지, EQ/loudness만
        if loud.get("integrated_lufs") is not None and loud["integrated_lufs"] > -10:
            plan.append(Treatment(
                action="loudness_check",
                plugin_short=None,
                params={"current_lufs": loud["integrated_lufs"]},
                rationale="음악 트랙 LUFS > -10, 헤드룸 부족",
                severity="warn",
            ))

    elif kind == "fullmix":
        # 풀믹스: voice de-noise를 그대로 적용하면 음악 손상 → stem 분리 권고
        plan.append(Treatment(
            action="stem_separate",
            plugin_short=None,
            params={
                "primary_tool": "demucs",
                "primary_model": "htdemucs",
                "primary_device": "mps",
                "expected_stems": ["vocals", "other"],
                "alternate_tool": "music-rebalance",   # RX 10 Music Rebalance도 가능
            },
            rationale="음성+음악 풀믹스 — Demucs(htdemucs/MPS) 우선, 대안 RX 10 Music Rebalance",
            severity="info",
        ))
        plan.append(Treatment(
            action="denoise",
            plugin_short="voice-de-noise",
            params={"apply_to": "vocals_stem_only"},
            rationale="stem 분리 후 vocals 트랙에 voice de-noise",
            severity="info",
        ))

    elif kind == "silent":
        plan.append(Treatment(
            action="skip",
            plugin_short=None,
            rationale="무음 트랙",
            severity="info",
        ))

    # 5. Clicks (transient artefact)
    n_clicks = clicks.get("n_clicks")
    if n_clicks is not None and n_clicks > 0:
        plan.append(Treatment(
            action="declick",
            plugin_short="de-click",
            params={"n_clicks": n_clicks},
            rationale=f"클릭 {n_clicks}개 검출",
            severity="warn" if n_clicks > 5 else "info",
        ))

    # 6. Phase (스테레오 모노 호환성)
    phase = track_diag.get("phase", {})
    neg_pct = phase.get("negative_correlation_pct", 0.0) if phase.get("applicable") else 0.0
    if neg_pct > 20.0:
        plan.append(Treatment(
            action="phase_warning",
            plugin_short=None,
            params={"negative_correlation_pct": neg_pct},
            rationale="모노 합산 시 cancellation 위험",
            severity="warn",
        ))

    # 7. Channel imbalance
    imb = track_diag.get("channel_imbalance", {})
    if imb.get("applicable") and imb.get("imbalance_db") is not None:
        if abs(imb["imbalance_db"]) > 1.5:
            plan.append(Treatment(
                action="channel_balance",
                plugin_short=None,
                params={"imbalance_db": imb["imbalance_db"]},
                rationale=f"L/R 불균형 {imb['imbalance_db']} dB",
                severity="warn",
            ))

    return plan


# ---------------------------------------------------------------------------
# 영상 한 개에 대한 dry-run 진단
# ---------------------------------------------------------------------------


@dataclass
class VideoDryRunReport:
    video_path: str
    topology: TopologyReport
    track_diagnostics: list[dict]    # 활성 트랙별 diagnose_track 결과
    treatments: list[dict]            # [{track_index, plan: [Treatment]}]
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "video": self.video_path,
            "topology": self.topology.to_dict(),
            "tracks": self.track_diagnostics,
            "treatments": self.treatments,
            "notes": self.notes,
        }


def dry_run_video(
    video: str | Path,
    *,
    analysis_seconds: float = 60.0,
    analysis_start_sec: float | None = None,
    classify_only_active: bool = True,
) -> VideoDryRunReport:
    """영상 1개를 dry-run 진단:
       1) probe_topology
       2) 활성 트랙 ([analysis_start_sec, +analysis_seconds] 구간) 추출 → classify + diagnose
       3) 처치 계획 생성

    실제 처리는 하지 않음. 결과는 JSON 직렬화 가능.
    """
    _ensure_tools()
    video = Path(video)
    notes: list[str] = []

    topo = probe_topology(video)
    notes.append(f"topology={topo.topology}, active={topo.active_indices}")

    track_diags: list[dict] = []
    treatments: list[dict] = []

    if topo.topology == "silent":
        notes.append("모든 트랙 무음 — 처리 불필요")
        return VideoDryRunReport(str(video), topo, [], [], notes)

    # 분석할 트랙 선정:
    #   - duplicated → 첫 번째 활성 트랙만 (나머지는 같은 신호)
    #   - multitrack → unique_signal_groups의 대표만 분석하고 나머진 결과 복사
    if topo.topology == "duplicated":
        analyze_indices = topo.active_indices[:1]
        notes.append("복제 믹스: 첫 활성 트랙만 분석")
        mirror_map: dict[int, int] = {idx: analyze_indices[0] for idx in topo.active_indices}
    elif topo.topology == "multitrack":
        analyze_indices = [grp[0] for grp in topo.unique_signal_groups]
        mirror_map = {}
        for grp in topo.unique_signal_groups:
            for idx in grp[1:]:
                mirror_map[idx] = grp[0]
        if mirror_map:
            notes.append(f"동일 신호 그룹 발견: {topo.unique_signal_groups}")
    else:
        analyze_indices = topo.active_indices
        mirror_map = {}

    # 분석 시작 시점: 영상 중간부 (말하는 구간일 확률 높음)
    if analysis_start_sec is None:
        if topo.duration_sec > analysis_seconds * 2:
            analysis_start_sec = max(0.0, topo.duration_sec / 2 - analysis_seconds / 2)
        else:
            analysis_start_sec = 0.0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for idx in analyze_indices:
            wav = tmp_dir / f"analyze_{idx}.wav"
            ok = _extract_track_to_wav(
                video, idx, wav,
                duration_sec=min(analysis_seconds, topo.duration_sec),
                start_sec=analysis_start_sec,
            )
            if not ok:
                notes.append(f"track {idx}: 추출 실패")
                continue
            audio, sr = read_audio(wav)
            cls = classify_track(audio, sr)
            diag = diagnose_track(audio, sr, cls)
            diag["track_index"] = idx
            diag["analysis_start_sec"] = round(analysis_start_sec, 2)
            diag["analysis_seconds"] = round(min(analysis_seconds, topo.duration_sec), 2)
            track_diags.append(diag)

            plan = recommend_treatment(diag)
            treatments.append({
                "track_index": idx,
                "kind": cls.kind,
                "plan": [t.to_dict() for t in plan],
                "mirrors": [],
            })

    # mirror된 트랙도 같은 처치를 받도록 treatments 확장
    if mirror_map:
        analyzed_by_idx = {tr["track_index"]: tr for tr in treatments}
        for mirror_idx, source_idx in mirror_map.items():
            if source_idx in analyzed_by_idx:
                analyzed_by_idx[source_idx]["mirrors"].append(mirror_idx)

    return VideoDryRunReport(
        video_path=str(video),
        topology=topo,
        track_diagnostics=track_diags,
        treatments=treatments,
        notes=notes,
    )
