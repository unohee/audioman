# Created: 2026-04-05
# Purpose: YAML/JSON 세션 파일 로더 — 멀티트랙 설정 파싱

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from audioman.core.mixer import TrackConfig
from audioman.core.pipeline import PipelineStep, parse_chain_string

logger = logging.getLogger(__name__)

#: Output subtypes a session file is allowed to request. Anything else is rejected
#: before it reaches soundfile — these are the subtypes this project actually
#: writes (``core/audio_file.py`` default, ``core/qc.py`` bit-depth mapping,
#: ``core/plugin_analysis.py`` analysis dumps, ``core/mixer.py`` docstring).
ALLOWED_SUBTYPES = frozenset({"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"})


class SessionPathError(ValueError):
    """A session file referenced a path outside the session directory."""


def _resolve_within_base(raw_path: str, base_dir: Path, field_name: str) -> Path:
    """Resolve ``raw_path`` and require it to stay inside ``base_dir``.

    Relative paths are resolved against the session directory; absolute paths are
    accepted only when they already point inside it. ``..`` traversal and absolute
    paths that land elsewhere are rejected instead of being silently rewritten, so
    a session file cannot read or overwrite files it does not own.

    Raises:
        SessionPathError: the resolved path escapes ``base_dir``.
    """
    base = base_dir.resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(base):
        raise SessionPathError(
            f"{field_name} path escapes the session directory: {raw_path!r} "
            f"(session directory: {base})"
        )
    return resolved


def validate_subtype(value: Any) -> str:
    """Return the canonical subtype for ``value`` or raise ``ValueError``.

    The subtype from a session file reaches soundfile's encoder, so only the
    allowlisted values in ``ALLOWED_SUBTYPES`` are passed through. Case is
    normalized (soundfile is case-insensitive) but nothing else is rewritten.
    """
    allowed = ", ".join(sorted(ALLOWED_SUBTYPES))
    if not isinstance(value, str):
        raise ValueError(f"Invalid audio subtype: {value!r} (expected one of: {allowed})")
    canonical = value.strip().upper()
    if canonical not in ALLOWED_SUBTYPES:
        raise ValueError(f"Invalid audio subtype: {value!r} (expected one of: {allowed})")
    return canonical


@dataclass
class SessionConfig:
    """세션 파일 설정"""
    tracks: list[TrackConfig]
    output: str
    sample_rate: Optional[int] = None
    subtype: str = "PCM_24"
    master_chain: Optional[list[PipelineStep]] = None

    def to_dict(self) -> dict:
        d = {
            "output": self.output,
            "subtype": self.subtype,
            "tracks": [t.to_dict() for t in self.tracks],
        }
        if self.sample_rate:
            d["sample_rate"] = self.sample_rate
        if self.master_chain:
            d["master_chain"] = [s.to_dict() for s in self.master_chain]
        return d


def _parse_track(raw: dict, base_dir: Path) -> TrackConfig:
    """개별 트랙 딕셔너리를 TrackConfig로 변환"""
    path = raw.get("path", "")
    if not path:
        raise ValueError("트랙에 'path' 필드가 없습니다")

    # Relative path -> absolute path under the session directory (never escaping it)
    track_path = _resolve_within_base(path, base_dir, "track")

    chain = None
    chain_raw = raw.get("chain")
    if chain_raw:
        if isinstance(chain_raw, str):
            chain = parse_chain_string(chain_raw)
        elif isinstance(chain_raw, list):
            # 이미 구조화된 형태: [{"plugin": "denoise", "params": {...}}, ...]
            chain = []
            for step_raw in chain_raw:
                if isinstance(step_raw, str):
                    chain.extend(parse_chain_string(step_raw))
                elif isinstance(step_raw, dict):
                    chain.append(PipelineStep(
                        plugin_name=step_raw.get("plugin", step_raw.get("plugin_name", "")),
                        params=step_raw.get("params", {}),
                    ))

    return TrackConfig(
        path=str(track_path),
        gain_db=float(raw.get("gain_db", 0.0)),
        pan=float(raw.get("pan", 0.0)),
        mute=bool(raw.get("mute", False)),
        solo=bool(raw.get("solo", False)),
        chain=chain,
        offset_samples=int(raw.get("offset_samples", 0)),
    )


def load_session(path: str | Path) -> SessionConfig:
    """YAML 또는 JSON 세션 파일 로드 (확장자로 자동 판별)

    YAML 예시:
        output: mix.wav
        format: PCM_24
        tracks:
          - path: vocals.wav
            gain_db: -3.0
            pan: 0.0
            chain: "dereverb,denoise:threshold=-20"
          - path: guitar.wav
            gain_db: -6.0
            pan: -0.5
        master:
          chain: "limiter:threshold=-1"
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"세션 파일 없음: {path}")

    text = path.read_text(encoding="utf-8")
    base_dir = path.parent

    # 확장자로 포맷 판별
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "YAML 세션 파일을 사용하려면 pyyaml이 필요합니다: "
                "uv add pyyaml"
            )
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        # 확장자 불명 → YAML 시도 → JSON 폴백
        try:
            import yaml
            data = yaml.safe_load(text)
        except Exception:
            data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"세션 파일이 딕셔너리가 아닙니다: {type(data)}")

    # tracks 파싱
    raw_tracks = data.get("tracks", [])
    if not raw_tracks:
        raise ValueError("세션 파일에 'tracks' 항목이 없습니다")

    tracks = [_parse_track(t, base_dir) for t in raw_tracks]

    # 마스터 체인 파싱
    master_chain = None
    master_raw = data.get("master")
    if master_raw:
        chain_raw = master_raw.get("chain", "")
        if isinstance(chain_raw, str) and chain_raw:
            master_chain = parse_chain_string(chain_raw)
        elif isinstance(chain_raw, list):
            master_chain = []
            for step_raw in chain_raw:
                if isinstance(step_raw, str):
                    master_chain.extend(parse_chain_string(step_raw))
                elif isinstance(step_raw, dict):
                    master_chain.append(PipelineStep(
                        plugin_name=step_raw.get("plugin", ""),
                        params=step_raw.get("params", {}),
                    ))

    # 출력 경로
    output = data.get("output", "")
    if not output:
        raise ValueError("세션 파일에 'output' 항목이 없습니다")

    output_path = _resolve_within_base(output, base_dir, "output")

    return SessionConfig(
        tracks=tracks,
        output=str(output_path),
        sample_rate=data.get("sample_rate"),
        subtype=validate_subtype(data.get("format", data.get("subtype", "PCM_24"))),
        master_chain=master_chain,
    )
