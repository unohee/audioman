# Created: 2026-04-26
# Purpose: EDL (Edit Decision List) data model + render engine for non-destructive editing
#
# An EDL accumulates editing intent as a list of ops without touching the source audio.
# At render time the ops are applied in order to produce the final audio.
# Time coordinates are relative to *the timeline immediately before that op* (same as DAW history).

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audioman.core import dsp
from audioman.core.audio_file import read_audio, write_audio


EDL_VERSION = 1

# supported op types and their required parameters
OP_SCHEMA: dict[str, set[str]] = {
    "cut_region": {"start_sec", "end_sec"},
    "trim": {"start_sec", "end_sec"},
    "trim_silence": set(),
    "splice": {"clip", "position_sec", "mode"},
    "fade_in": {"duration_sec"},
    "fade_out": {"duration_sec"},
    "pad": set(),
    "remove_dc": set(),
    "loudness_normalize": set(),
    "normalize": set(),
    "gain": {"db"},
    "gate": set(),
    "process": {"plugin"},
    "chain": {"steps"},
}


@dataclass
class EDL:
    """Non-destructive editing intent for a single input file."""
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
            raise ValueError(f"unsupported EDL version: {version} > {EDL_VERSION}")
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
    """Input file integrity hash. Large files are processed in chunks."""
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
    """Create a new EDL for an input file. Reads the audio once to extract metadata."""
    source = Path(source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"file not found: {source}")
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
    """Validate the op format. Raises ValueError on an unknown type or missing required key."""
    if not isinstance(op, dict):
        raise ValueError(f"op must be a dict: {type(op)}")
    op_type = op.get("type")
    if op_type not in OP_SCHEMA:
        raise ValueError(
            f"unknown op type: {op_type!r} "
            f"(supported: {sorted(OP_SCHEMA.keys())})"
        )
    required = OP_SCHEMA[op_type]
    missing = required - set(op.keys())
    if missing:
        raise ValueError(f"op {op_type!r} is missing required keys: {sorted(missing)}")


def add_op(edl: EDL, op: dict) -> EDL:
    """Append an op to the EDL. Validates it, then refreshes modified_at."""
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
        raise FileNotFoundError(f"EDL file not found: {path}")
    return EDL.from_dict(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Render engine
# ---------------------------------------------------------------------------


def _sec_to_samples(sec: float, sr: int) -> int:
    return int(round(float(sec) * sr))


def _ms_to_samples(ms: float | None, sr: int) -> int:
    if ms is None:
        return 0
    return int(round(float(ms) / 1000.0 * sr))


def _apply_op(audio: np.ndarray, sr: int, op: dict) -> np.ndarray:
    """Apply a single op to the current audio. Pure function (input audio unchanged)."""
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
                f"splice clip sample rate mismatch: edl={sr}Hz, clip={clip_sr}Hz "
                f"({clip_path})"
            )
        # automatic channel alignment
        in_ch = 1 if audio.ndim == 1 else audio.shape[0]
        clip_ch = 1 if clip_audio.ndim == 1 else clip_audio.shape[0]
        if in_ch != clip_ch:
            if in_ch == 2 and clip_ch == 1:
                src = clip_audio if clip_audio.ndim == 1 else clip_audio[0]
                clip_audio = np.stack([src, src], axis=0)
            elif in_ch == 1 and clip_ch == 2:
                # keep the (1, samples) layout so dsp.splice sees matching ndim
                clip_audio = clip_audio.mean(axis=0, keepdims=True)
            else:
                raise ValueError(f"splice channel conversion not possible: in={in_ch}, clip={clip_ch}")
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
        return dsp.fade_in(audio, n, curve=op.get("curve", "linear"))

    if t == "fade_out":
        n = _sec_to_samples(op["duration_sec"], sr)
        return dsp.fade_out(audio, n, curve=op.get("curve", "linear"))

    if t == "pad":
        head = _ms_to_samples(op.get("head_ms"), sr)
        tail_ms = op.get("tail_ms")
        tail_sec = op.get("tail_sec")
        if tail_sec is not None:
            tail = _sec_to_samples(tail_sec, sr)
        else:
            tail = _ms_to_samples(tail_ms, sr)
        head_sec = op.get("head_sec")
        if head_sec is not None:
            head = _sec_to_samples(head_sec, sr)
        return dsp.pad(audio, head_samples=head, tail_samples=tail)

    if t == "remove_dc":
        return dsp.remove_dc(audio)

    if t == "loudness_normalize":
        from audioman.core import loudness as loudness_mod
        target_lufs = float(op.get("target_lufs", -14.0))
        max_tp = float(op.get("max_true_peak_dbtp", -1.0))
        out, _meta = loudness_mod.loudness_normalize(
            audio, sr,
            target_lufs=target_lufs,
            max_true_peak_dbtp=max_tp,
        )
        return out

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
        # VST3 plugin hosting — reuses the pipeline code
        from audioman.core.registry import get_registry
        from audioman.plugins.vst3 import VST3PluginWrapper

        registry = get_registry()
        meta = registry.get(op["plugin"])
        if not meta:
            raise ValueError(f"plugin not found: {op['plugin']!r}")
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
        from audioman.core.registry import get_registry
        from audioman.plugins.vst3 import VST3PluginWrapper

        registry = get_registry()
        for step in op["steps"]:
            meta = registry.get(step["plugin"])
            if not meta:
                raise ValueError(f"chain plugin not found: {step['plugin']!r}")
            wrapper = VST3PluginWrapper(meta.path)
            wrapper.load()
            if step.get("params"):
                wrapper.set_parameters(step["params"])
            audio = wrapper.process(audio, sr)
        return audio

    raise ValueError(f"_apply_op: unknown op type {t!r}")


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
    """Apply the EDL in order and write the final audio to the output file.

    With verify_source=True, source_sha256 is used to detect changes to the input file.
    """
    start = time.monotonic()
    src = Path(edl.source)
    if not src.exists():
        raise FileNotFoundError(f"EDL source file not found: {src}")
    if verify_source:
        actual = file_sha256(src)
        if actual != edl.source_sha256:
            raise ValueError(
                f"source file changed: expected={edl.source_sha256[:12]}, "
                f"actual={actual[:12]} ({src})"
            )

    audio, sr = read_audio(src)
    if sr != edl.sample_rate:
        raise ValueError(f"sample rate mismatch: edl={edl.sample_rate}, file={sr}")

    in_dur = audio.shape[-1] / sr

    for i, op in enumerate(edl.ops):
        try:
            audio = _apply_op(audio, sr, op)
        except Exception as e:
            raise RuntimeError(f"op #{i+1} ({op.get('type')}) failed: {e}") from e

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
# Workspace (.audioman/) management
# ---------------------------------------------------------------------------


WORKSPACE_DIRNAME = ".audioman"
EDL_FILENAME = "edit.json"
HISTORY_DIRNAME = "history"
REDO_DIRNAME = "redo"


def workspace_dir(source: str | Path) -> Path:
    """Put the .audioman/ workspace in the directory holding the input file."""
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
    """Snapshot the current EDL into history/.

    With clear_redo=True the redo queue is cleared when a new op is added
    (same as Pro Tools/REAPER). That is the natural model: "if you undo and take a
    different path, the old redos are void".
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
    """Move the most recent history snapshot to redo/ and make the previous state the active EDL."""
    hist = list_history(source)
    if len(hist) < 2:
        return None
    rd = redo_dir(source)
    rd.mkdir(parents=True, exist_ok=True)
    # the last one = current state -> move to redo
    current = hist[-1]
    target = hist[-2]
    redo_idx = _next_index(rd)
    current.rename(rd / f"{redo_idx:04d}.json")
    # restore the previous state as the active EDL
    edl = load_edl(target)
    save_edl(edl, edl_path(source))
    return edl


def redo(source: str | Path) -> EDL | None:
    """Move the most recent redo snapshot back to the end of history/ and restore it as the active EDL."""
    rd_list = list_redo(source)
    if not rd_list:
        return None
    target = rd_list[-1]
    edl = load_edl(target)
    save_edl(edl, edl_path(source))
    # move redo -> history (becomes the next undo target)
    hist = history_dir(source)
    hist.mkdir(parents=True, exist_ok=True)
    new_idx = _next_index(hist)
    target.rename(hist / f"{new_idx:04d}.json")
    return edl
