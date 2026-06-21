# Created: 2026-05-11
# Purpose: Audio fault 통합 스키마 — LLM이 audioman 분석 결과를
#          단일 finding[] 배열로 소비할 수 있게 한다.
# JSONSchema 발행: src/audioman/schemas/finding.v1.json
#
# Category 정의:
#   - signal   : 시간영역 결함 (clipping, DC offset, silence, channel imbalance)
#   - spectral : 주파수영역/지각 결함 (hum, hiss, harshness, LUFS, click density)
#   - plugin   : 플러그인 처리 전/후 결함 (true-peak, gain stage, NaN/inf)
#   - container: 컨테이너/인코딩 결함 (codec, channel layout, mapping_family 등)

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


SCHEMA_URI = "audioman://schema/finding.v1.json"


class Category(str, Enum):
    SIGNAL = "signal"
    SPECTRAL = "spectral"
    PLUGIN = "plugin"
    CONTAINER = "container"


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warn": 1, "critical": 2}[self.value]


# 기계 판독용 안정 코드. 새 코드 추가는 가능하나 기존 코드 제거/변경 금지.
class Code(str, Enum):
    # signal
    CLIP_SAMPLE_PEAK_EXCEEDED = "CLIP_SAMPLE_PEAK_EXCEEDED"
    DC_OFFSET_DETECTED = "DC_OFFSET_DETECTED"
    CHANNEL_IMBALANCE = "CHANNEL_IMBALANCE"
    SILENCE_LEADING = "SILENCE_LEADING"
    SILENCE_TRAILING = "SILENCE_TRAILING"
    SILENCE_INNER = "SILENCE_INNER"
    SAMPLE_DROPOUT = "SAMPLE_DROPOUT"
    # spectral
    MAINS_HUM = "MAINS_HUM"
    HF_NOISE_FLOOR = "HF_NOISE_FLOOR"
    CLICK_DENSITY = "CLICK_DENSITY"
    LUFS_OUT_OF_RANGE = "LUFS_OUT_OF_RANGE"
    HARSHNESS_HIGH = "HARSHNESS_HIGH"
    # plugin
    TRUE_PEAK_EXCEEDED = "TRUE_PEAK_EXCEEDED"
    GAIN_STAGE_LEAK = "GAIN_STAGE_LEAK"
    NONFINITE_SAMPLES = "NONFINITE_SAMPLES"
    # container
    OPUS_MAPPING_FAMILY_MISSING = "OPUS_MAPPING_FAMILY_MISSING"
    SAMPLE_RATE_MISMATCH = "SAMPLE_RATE_MISMATCH"
    UNSUPPORTED_CODEC = "UNSUPPORTED_CODEC"
    CHANNEL_LAYOUT_LOSS = "CHANNEL_LAYOUT_LOSS"


@dataclass
class Where:
    """Finding이 가리키는 위치. 모든 필드는 옵션."""
    file: Optional[str] = None
    track: Optional[int] = None
    start_sample: Optional[int] = None
    end_sample: Optional[int] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    channel: Optional[int] = None
    frequency_hz: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class FixHint:
    """이 finding을 해결하기 위한 audioman/외부 명령 힌트."""
    kind: str  # "ffmpeg-plan" | "audioman-fx" | "audioman-process" | "manual"
    args: list[str] = field(default_factory=list)
    note: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "args": list(self.args)}
        if self.note is not None:
            d["note"] = self.note
        return d


@dataclass
class Finding:
    code: Code
    category: Category
    severity: Severity
    hint: str = ""
    where: Where = field(default_factory=Where)
    measurement: dict[str, Any] = field(default_factory=dict)
    fix_hint: Optional[FixHint] = None
    id: Optional[str] = None  # 자동 생성 가능

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id or self._auto_id(),
            "code": self.code.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "where": self.where.to_dict(),
            "measurement": dict(self.measurement),
            "hint": self.hint,
        }
        if self.fix_hint is not None:
            d["fix_hint"] = self.fix_hint.to_dict()
        return d

    def _auto_id(self) -> str:
        # category-code의 hash가 아닌, 안정적인 슬러그
        slug = self.code.value.lower().replace("_", "-")
        return slug


def filter_findings(
    findings: list[Finding],
    *,
    categories: Optional[set[str]] = None,
    min_severity: Severity = Severity.INFO,
) -> list[Finding]:
    """카테고리/심각도 필터."""
    result = []
    for f in findings:
        if categories is not None and f.category.value not in categories:
            continue
        if f.severity.rank < min_severity.rank:
            continue
        result.append(f)
    return result


def envelope(
    findings: list[Finding],
    *,
    file: Optional[str] = None,
    audioman_version: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """findings[]를 LLM-friendly JSON envelope으로 감싼다."""
    from audioman import __version__

    out: dict[str, Any] = {
        "$schema": SCHEMA_URI,
        "audioman_version": audioman_version or __version__,
        "findings": [f.to_dict() for f in findings],
        "summary": {
            "total": len(findings),
            "by_severity": {
                "info": sum(1 for f in findings if f.severity is Severity.INFO),
                "warn": sum(1 for f in findings if f.severity is Severity.WARN),
                "critical": sum(1 for f in findings if f.severity is Severity.CRITICAL),
            },
            "by_category": {
                "signal": sum(1 for f in findings if f.category is Category.SIGNAL),
                "spectral": sum(1 for f in findings if f.category is Category.SPECTRAL),
                "plugin": sum(1 for f in findings if f.category is Category.PLUGIN),
                "container": sum(1 for f in findings if f.category is Category.CONTAINER),
            },
        },
    }
    if file is not None:
        out["file"] = file
    if extra:
        out.update(extra)
    return out
