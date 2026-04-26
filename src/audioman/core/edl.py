# Created: 2026-04-26
# Purpose: 비파괴 편집을 위한 EDL(Edit Decision List) 데이터 모델 + render 엔진
#
# EDL은 원본 오디오를 변경하지 않고 편집 의도를 ops 리스트로 누적한다.
# render 시점에 ops를 순차 적용해 최종 오디오를 생성한다.
# 시간 좌표는 *해당 op 직전 시점의 타임라인 기준*이다 (DAW history와 동일).

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from audioman.core import dsp
from audioman.core.audio_file import read_audio, write_audio


EDL_VERSION = 1

# 지원하는 op 타입과 필수 파라미터
OP_SCHEMA: dict[str, set[str]] = {
    "cut_region": {"start_sec", "end_sec"},
    "trim": {"start_sec", "end_sec"},
    "trim_silence": set(),
    "splice": {"clip", "position_sec", "mode"},
    "fade_in": {"duration_sec"},
    "fade_out": {"duration_sec"},
    "normalize": set(),
    "gain": {"db"},
    "gate": set(),
    "process": {"plugin"},
    "chain": {"steps"},
}


@dataclass
class EDL:
    """단일 입력 파일에 대한 비파괴 편집 의도."""
    source: str
    source_sha256: str
    sample_rate: int
    channels: int
    duration_sec: float
    ops: list[dict] = field(default_factory=list)
    version: int = EDL_VERSION
    created_at: str = ""
    modified_at: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_sec": self.duration_sec,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "ops": list(self.ops),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EDL":
        version = int(data.get("version", 1))
        if version > EDL_VERSION:
            raise ValueError(f"지원하지 않는 EDL version: {version} > {EDL_VERSION}")
        return cls(
            version=version,
            source=data["source"],
            source_sha256=data["source_sha256"],
            sample_rate=int(data["sample_rate"]),
            channels=int(data["channels"]),
            duration_sec=float(data["duration_sec"]),
            ops=list(data.get("ops", [])),
            created_at=data.get("created_at", ""),
            modified_at=data.get("modified_at", ""),
        )


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """입력 파일 무결성 해시. 큰 파일도 청크 단위로 처리."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_edl(source: str | Path) -> EDL:
    """입력 파일에 대한 새 EDL 생성. 오디오를 한 번만 읽어 메타데이터 추출."""
    source = Path(source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"파일 없음: {source}")
    audio, sr = read_audio(source)
    n_ch = 1 if audio.ndim == 1 else audio.shape[0]
    n_samples = audio.shape[-1]
    now = _now_iso()
    return EDL(
        source=str(source),
        source_sha256=file_sha256(source),
        sample_rate=int(sr),
        channels=int(n_ch),
        duration_sec=round(n_samples / sr, 6),
        ops=[],
        created_at=now,
        modified_at=now,
    )


def validate_op(op: dict) -> None:
    """op 형식 검증. 알 수 없는 type이거나 필수 키 누락 시 ValueError."""
    if not isinstance(op, dict):
        raise ValueError(f"op은 dict여야 합니다: {type(op)}")
    op_type = op.get("type")
    if op_type not in OP_SCHEMA:
        raise ValueError(
            f"알 수 없는 op type: {op_type!r} "
            f"(지원: {sorted(OP_SCHEMA.keys())})"
        )
    required = OP_SCHEMA[op_type]
    missing = required - set(op.keys())
    if missing:
        raise ValueError(f"op {op_type!r}에 필수 키 누락: {sorted(missing)}")


def add_op(edl: EDL, op: dict) -> EDL:
    """op을 EDL에 추가. 검증 후 modified_at 갱신."""
    validate_op(op)
    edl.ops.append(dict(op))
    edl.modified_at = _now_iso()
    return edl


def save_edl(edl: EDL, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(edl.to_dict(), indent=2, ensure_ascii=False))


def load_edl(path: str | Path) -> EDL:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EDL 파일 없음: {path}")
    return EDL.from_dict(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Render 엔진
# ---------------------------------------------------------------------------


def _sec_to_samples(sec: float, sr: int) -> int:
    return int(round(float(sec) * sr))


def _ms_to_samples(ms: float | None, sr: int) -> int:
    if ms is None:
        return 0
    return int(round(float(ms) / 1000.0 * sr))


def _apply_op(audio: np.ndarray, sr: int, op: dict) -> np.ndarray:
    """단일 op을 현재 오디오에 적용. 순수 함수 (입력 audio 미변경)."""
    t = op["type"]

    if t == "cut_region":
        start = _sec_to_samples(op["start_sec"], sr)
        end = _sec_to_samples(op["end_sec"], sr)
        cf = _ms_to_samples(op.get("crossfade_ms"), sr)
        return dsp.cut_region(audio, start=start, end=end, crossfade_samples=cf)

    if t == "trim":
        start = _sec_to_samples(op["start_sec"], sr)
        end = _sec_to_samples(op["end_sec"], sr)
        return dsp.trim(audio, start=start, end=end)

    if t == "trim_silence":
        return dsp.trim_silence(
            audio, sr,
            threshold_db=float(op.get("threshold_db", -40.0)),
            pad_samples=int(op.get("pad_samples", 0)),
        )

    if t == "splice":
        clip_path = op["clip"]
        clip_audio, clip_sr = read_audio(clip_path)
        if clip_sr != sr:
            raise ValueError(
                f"splice clip sample rate 불일치: edl={sr}Hz, clip={clip_sr}Hz "
                f"({clip_path})"
            )
        # 채널 자동 정렬
        in_ch = 1 if audio.ndim == 1 else audio.shape[0]
        clip_ch = 1 if clip_audio.ndim == 1 else clip_audio.shape[0]
        if in_ch != clip_ch:
            if in_ch == 2 and clip_ch == 1:
                src = clip_audio if clip_audio.ndim == 1 else clip_audio[0]
                clip_audio = np.stack([src, src], axis=0)
            elif in_ch == 1 and clip_ch == 2:
                clip_audio = clip_audio.mean(axis=0)
            else:
                raise ValueError(f"splice 채널 변환 불가: in={in_ch}, clip={clip_ch}")
        position = _sec_to_samples(op["position_sec"], sr)
        cf = _ms_to_samples(op.get("crossfade_ms"), sr)
        return dsp.splice(
            audio, clip_audio,
            position=position,
            mode=op["mode"],
            crossfade_samples=cf,
        )

    if t == "fade_in":
        n = _sec_to_samples(op["duration_sec"], sr)
        return dsp.fade_in(audio, n)

    if t == "fade_out":
        n = _sec_to_samples(op["duration_sec"], sr)
        return dsp.fade_out(audio, n)

    if t == "normalize":
        peak = op.get("peak_db")
        target_rms = op.get("target_rms_db")
        if peak is None and target_rms is None:
            peak = -1.0
        return dsp.normalize(audio, peak_db=peak, target_rms_db=target_rms)

    if t == "gain":
        return dsp.gain(audio, float(op["db"]))

    if t == "gate":
        return dsp.gate(
            audio, sr,
            threshold_db=float(op.get("threshold_db", -50.0)),
            attack_sec=float(op.get("attack_sec", 0.01)),
            release_sec=float(op.get("release_sec", 0.05)),
        )

    if t == "process":
        # VST3 plugin 호스팅 — pipeline 코드 재사용
        from audioman.core.registry import get_registry
        from audioman.plugins.vst3 import VST3PluginWrapper

        registry = get_registry()
        meta = registry.get(op["plugin"])
        if not meta:
            raise ValueError(f"플러그인을 찾을 수 없음: {op['plugin']!r}")
        wrapper = VST3PluginWrapper(meta.path)
        wrapper.load()
        params = op.get("params", {}) or {}
        if params:
            wrapper.set_parameters(params)
        passes = int(op.get("passes", 1))
        for _ in range(passes):
            audio = wrapper.process(audio, sr)
        return audio

    if t == "chain":
        from audioman.core.pipeline import PipelineStep
        from audioman.core.registry import get_registry
        from audioman.plugins.vst3 import VST3PluginWrapper

        registry = get_registry()
        for step in op["steps"]:
            meta = registry.get(step["plugin"])
            if not meta:
                raise ValueError(f"chain 플러그인 없음: {step['plugin']!r}")
            wrapper = VST3PluginWrapper(meta.path)
            wrapper.load()
            if step.get("params"):
                wrapper.set_parameters(step["params"])
            audio = wrapper.process(audio, sr)
        return audio

    raise ValueError(f"_apply_op: 알 수 없는 op type {t!r}")


@dataclass
class RenderResult:
    edl_path: str | None
    output_path: str
    n_ops: int
    input_duration_sec: float
    output_duration_sec: float
    elapsed_sec: float
    sample_rate: int
    channels: int

    def to_dict(self) -> dict:
        return {
            "edl_path": self.edl_path,
            "output_path": self.output_path,
            "n_ops": self.n_ops,
            "input_duration_sec": round(self.input_duration_sec, 4),
            "output_duration_sec": round(self.output_duration_sec, 4),
            "elapsed_sec": round(self.elapsed_sec, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


def render_edl(
    edl: EDL,
    output_path: str | Path,
    edl_path: str | Path | None = None,
    verify_source: bool = True,
) -> RenderResult:
    """EDL을 순차 적용해 최종 오디오를 출력 파일로 저장.

    verify_source=True면 source_sha256으로 입력 파일 변경을 감지한다.
    """
    start = time.monotonic()
    src = Path(edl.source)
    if not src.exists():
        raise FileNotFoundError(f"EDL source 파일 없음: {src}")
    if verify_source:
        actual = file_sha256(src)
        if actual != edl.source_sha256:
            raise ValueError(
                f"source 파일이 변경됨: expected={edl.source_sha256[:12]}, "
                f"actual={actual[:12]} ({src})"
            )

    audio, sr = read_audio(src)
    if sr != edl.sample_rate:
        raise ValueError(f"sample rate 불일치: edl={edl.sample_rate}, file={sr}")

    in_dur = audio.shape[-1] / sr

    for i, op in enumerate(edl.ops):
        try:
            audio = _apply_op(audio, sr, op)
        except Exception as e:
            raise RuntimeError(f"op #{i+1} ({op.get('type')}) 실패: {e}") from e

    write_audio(output_path, audio, sr)

    elapsed = time.monotonic() - start
    n_ch = 1 if audio.ndim == 1 else audio.shape[0]
    out_dur = audio.shape[-1] / sr

    return RenderResult(
        edl_path=str(edl_path) if edl_path else None,
        output_path=str(output_path),
        n_ops=len(edl.ops),
        input_duration_sec=in_dur,
        output_duration_sec=out_dur,
        elapsed_sec=elapsed,
        sample_rate=sr,
        channels=n_ch,
    )


# ---------------------------------------------------------------------------
# Workspace (.audioman/) 관리
# ---------------------------------------------------------------------------


WORKSPACE_DIRNAME = ".audioman"
EDL_FILENAME = "edit.json"
HISTORY_DIRNAME = "history"
REDO_DIRNAME = "redo"


def workspace_dir(source: str | Path) -> Path:
    """입력 파일이 있는 디렉터리에 .audioman/ 워크스페이스를 둔다."""
    src = Path(source).resolve()
    return src.parent / WORKSPACE_DIRNAME / src.stem


def edl_path(source: str | Path) -> Path:
    return workspace_dir(source) / EDL_FILENAME


def history_dir(source: str | Path) -> Path:
    return workspace_dir(source) / HISTORY_DIRNAME


def redo_dir(source: str | Path) -> Path:
    return workspace_dir(source) / REDO_DIRNAME


def _next_index(d: Path) -> int:
    if not d.exists():
        return 1
    indices = []
    for p in d.glob("*.json"):
        try:
            indices.append(int(p.stem))
        except ValueError:
            continue
    return max(indices, default=0) + 1


def _list_sorted(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(d.glob("*.json"))


def snapshot_history(edl: EDL, source: str | Path, clear_redo: bool = True) -> Path:
    """현재 EDL을 history/ 에 스냅샷.

    clear_redo=True면 새 op 추가 시점에 redo 큐를 비운다 (Pro Tools/REAPER와 동일).
    이는 "되돌렸다가 다른 길로 가면 옛 redo는 무효"라는 자연스러운 모델.
    """
    hist = history_dir(source)
    hist.mkdir(parents=True, exist_ok=True)
    idx = _next_index(hist)
    path = hist / f"{idx:04d}.json"
    save_edl(edl, path)

    if clear_redo:
        rd = redo_dir(source)
        if rd.exists():
            for p in rd.glob("*.json"):
                p.unlink()

    return path


def list_history(source: str | Path) -> list[Path]:
    return _list_sorted(history_dir(source))


def list_redo(source: str | Path) -> list[Path]:
    return _list_sorted(redo_dir(source))


def undo(source: str | Path) -> EDL | None:
    """가장 최근 history 스냅샷을 redo/로 옮기고 그 직전 상태를 active EDL로."""
    hist = list_history(source)
    if len(hist) < 2:
        return None
    rd = redo_dir(source)
    rd.mkdir(parents=True, exist_ok=True)
    # 가장 마지막 = 현재 상태 → redo로 이동
    current = hist[-1]
    target = hist[-2]
    redo_idx = _next_index(rd)
    current.rename(rd / f"{redo_idx:04d}.json")
    # 직전 상태를 active EDL로 복원
    edl = load_edl(target)
    save_edl(edl, edl_path(source))
    return edl


def redo(source: str | Path) -> EDL | None:
    """가장 최근 redo 스냅샷을 history/ 끝으로 되돌리고 active EDL로 복원."""
    rd_list = list_redo(source)
    if not rd_list:
        return None
    target = rd_list[-1]
    edl = load_edl(target)
    save_edl(edl, edl_path(source))
    # redo → history 로 이동 (다음 undo 대상이 됨)
    hist = history_dir(source)
    hist.mkdir(parents=True, exist_ok=True)
    new_idx = _next_index(hist)
    target.rename(hist / f"{new_idx:04d}.json")
    return edl
