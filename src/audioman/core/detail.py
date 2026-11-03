# Created: 2026-11-04
# Purpose: Token-budgeted `--json` output: detail levels, array spill files, byte ceilings.
# Dependencies: stdlib only
#
# An LLM reads `--json` output through a context window. A five-minute file analysed
# with `--frames` used to print ~2 MB, which no agent can use. This module shapes a
# payload for a chosen level of detail:
#
#   full      the payload as computed, unchanged.
#   standard  numeric arrays longer than 64 values, and finding/event lists longer
#             than 50 items, are written to a JSON file and replaced by a reference
#             that carries length and min/max/mean (arrays) or the top items (lists).
#             The printed JSON stays under 32 KiB.
#   summary   the same with tighter limits (8 values, top 5 items) and under 4 KiB.
#
# Nothing is lost at any level: everything removed from stdout is in a spill file
# whose absolute path is in the reference (or in `detail.truncated` for trimmed
# lists), and `detail.spilled` lists the key path of every value that moved.

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DETAIL_LEVELS = ("summary", "standard", "full")
DEFAULT_DETAIL = "standard"


@dataclass(frozen=True)
class DetailPolicy:
    inline_values: int  # longest numeric array kept inline
    top_k: int  # items kept inline from a list of records (findings/events ranked first)
    inline_chars: int  # longest string kept inline (ASCII renderings)
    byte_ceiling: int | None  # bound on the printed JSON (indent=2)


POLICIES = {
    "summary": DetailPolicy(inline_values=8, top_k=5, inline_chars=512, byte_ceiling=4096),
    "standard": DetailPolicy(inline_values=64, top_k=50, inline_chars=8192, byte_ceiling=32768),
    "full": DetailPolicy(inline_values=-1, top_k=-1, inline_chars=-1, byte_ceiling=None),
}

# Progressively tighter limits tried when a level's first pass is over its ceiling.
_TIGHTENING = ((8, 10, 512), (0, 5, 128), (0, 1, 0), (0, 0, 0))

# Lists of records ranked by importance before truncation (most severe first).
RANKED_LISTS = ("findings", "events", "checks", "gates")
_SEVERITY_RANK = {"critical": 0, "fail": 0, "error": 0, "warn": 1, "warning": 1, "info": 2, "pass": 3}

# Envelope metadata that must never be spilled or reordered.
_PROTECTED = {"$schema", "audioman_version", "command", "digest", "detail"}
# Sections that are shaped inside (long lists trimmed) but never replaced by a
# reference as a whole: the report.v2 block must always validate in place.
_INLINE_SECTIONS = {"report"}
# Lists that are never trimmed: a report's gates are the basis of its verdict.
_NEVER_TRIMMED = {"report.gates"}
# A reference costs ~150-250 bytes (absolute path + length/stats); moving anything
# smaller would grow the output, so values at or under this size always stay inline.
_MIN_SPILL_BYTES = 256


def printed_size(payload: Any) -> int:
    """Byte length of the payload as `print_json` prints it."""
    return len(json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8"))


def _is_numeric_array(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def _is_record_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(isinstance(v, dict) for v in value)


def _stats(values: list[float]) -> dict[str, float | None]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": round(min(finite), 6),
        "max": round(max(finite), 6),
        "mean": round(sum(finite) / len(finite), 6),
    }


def _rank_key(item: dict) -> tuple:
    severity = str(item.get("severity", item.get("status", "info"))).lower()
    where = item.get("where") if isinstance(item.get("where"), dict) else {}
    start = item.get("start_sec", where.get("sec_start", 0.0))
    try:
        start = float(start)
    except (TypeError, ValueError):
        start = 0.0
    return (_SEVERITY_RANK.get(severity, 4), -float(item.get("confidence", 0.0) or 0.0), start)


class _Spiller:
    def __init__(self, spill_dir: Path, label: str) -> None:
        self.dir = spill_dir
        self.label = "".join(c if c.isalnum() or c in "-_." else "_" for c in label) or "payload"
        self.files: list[dict[str, Any]] = []

    def write(self, key_path: str, value: Any) -> str:
        body = json.dumps({"key": key_path, "value": value}, ensure_ascii=False, default=str)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
        safe_key = key_path.replace(".", "-").replace("[", "-").replace("]", "")
        path = self.dir / f"{self.label}-{safe_key}-{digest}.json"
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        ref = str(path.resolve())
        self.files.append({"key": key_path, "$ref": ref, "length": len(value)})
        return ref


def _shape(node: Any, path: str, policy: DetailPolicy, spiller: _Spiller, truncated: dict) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            child = f"{path}.{key}" if path else key
            if not path and key in _PROTECTED:
                out[key] = value
                continue
            out[key] = _shape_value(key, value, child, policy, spiller, truncated)
        return out
    return _shape_value(None, node, path, policy, spiller, truncated)


def _worth_spilling(value) -> bool:
    return printed_size(value) > _MIN_SPILL_BYTES


def _shape_value(key, value, path, policy, spiller, truncated):
    # Inside the report block only list trimming is allowed: replacing a value by a
    # reference would break its schema.
    in_report = path == "report" or path.startswith("report.")
    if _is_numeric_array(value):
        if len(value) > policy.inline_values and not in_report and _worth_spilling(value):
            return {"$ref": spiller.write(path, value), "length": len(value), **_stats(value)}
        return value
    if _is_record_list(value) and len(value) > policy.top_k and path not in _NEVER_TRIMMED:
        kept = sorted(value, key=_rank_key) if key in RANKED_LISTS else value
        truncated[path] = {"total": len(value), "shown": policy.top_k, "$ref": spiller.write(path, value)}
        return [_shape(v, f"{path}[{i}]", policy, spiller, truncated) for i, v in enumerate(kept[: policy.top_k])]
    if isinstance(value, str) and len(value) > policy.inline_chars and not in_report and _worth_spilling(value):
        return {"$ref": spiller.write(path, value), "length": len(value)}
    if isinstance(value, (dict, list)):
        if isinstance(value, dict):
            return _shape(value, path, policy, spiller, truncated)
        return [_shape(v, f"{path}[{i}]", policy, spiller, truncated) for i, v in enumerate(value)]
    return value


def apply_detail(
    payload: dict[str, Any],
    level: str = DEFAULT_DETAIL,
    *,
    spill_dir: Path,
    label: str,
) -> dict[str, Any]:
    """Shape `payload` for `level`; return a new dict carrying a `detail` block.

    The byte ceiling is a guarantee, not a hope: if the first pass is still too
    large, limits are tightened (fewer inline values, fewer items) until it fits or
    nothing spillable is left inline.
    """
    if level not in POLICIES:
        raise ValueError(f"unknown detail level {level!r}; expected one of {DETAIL_LEVELS}")
    policy = POLICIES[level]
    if policy.byte_ceiling is None:
        return {
            **payload,
            "detail": {"level": level, "byte_ceiling": None, "within_ceiling": True, "spilled": [], "truncated": {}},
        }

    attempts = [policy] + [
        DetailPolicy(
            min(inline, policy.inline_values),
            min(top, policy.top_k),
            min(chars, policy.inline_chars),
            policy.byte_ceiling,
        )
        for inline, top, chars in _TIGHTENING
        if inline < policy.inline_values or top < policy.top_k or chars < policy.inline_chars
    ]

    shaped: dict[str, Any] = {}
    for attempt in attempts:
        spiller = _Spiller(spill_dir, label)
        truncated: dict[str, Any] = {}
        shaped = _shape(payload, "", attempt, spiller, truncated)
        shaped["detail"] = {
            "level": level,
            "byte_ceiling": policy.byte_ceiling,
            "within_ceiling": True,
            "spilled": [f["key"] for f in spiller.files],
            "truncated": truncated,
        }
        if printed_size(shaped) <= policy.byte_ceiling:
            return shaped

    # Many small references can outweigh what they replaced. Last resort: move whole
    # top-level sections, largest first, into one spill file each. Try this from the
    # most inline shaping to the tightest, so whatever still fits keeps its detail;
    # sections that must stay inline (the report) are only trimmed by the tighter passes.
    for attempt in attempts:
        shaped = _shape_with_sections(payload, attempt, policy, level, spill_dir, label)
        if printed_size(shaped) <= policy.byte_ceiling:
            return shaped
        # The index of moved values can itself be the largest part: move it to one file.
        indexed = _move_index(shaped, spill_dir, label)
        if printed_size(indexed) <= policy.byte_ceiling:
            return indexed
    # Only scalar metadata is left inline; report the miss instead of dropping data.
    shaped["detail"]["within_ceiling"] = False
    return shaped


def _move_index(shaped: dict, spill_dir: Path, label: str) -> dict:
    """Replace detail.spilled/truncated by a reference to one file holding both."""
    detail = shaped["detail"]
    if not detail["spilled"] and not detail["truncated"]:
        return shaped
    ref = _Spiller(spill_dir, label).write("detail-index", {"spilled": detail["spilled"],
                                                            "truncated": detail["truncated"]})
    return {**shaped, "detail": {**detail, "spilled": [], "truncated": {}, "index_ref": ref}}


def _shape_with_sections(payload, attempt, policy, level, spill_dir, label):
    spiller = _Spiller(spill_dir, label)
    truncated: dict[str, Any] = {}
    shaped = _shape(payload, "", attempt, spiller, truncated)
    shaped["detail"] = {
        "level": level,
        "byte_ceiling": policy.byte_ceiling,
        "within_ceiling": True,
        "spilled": [f["key"] for f in spiller.files],
        "truncated": truncated,
    }
    movable = sorted(
        (k for k in shaped if k not in _PROTECTED | _INLINE_SECTIONS and isinstance(shaped[k], (dict, list, str))),
        key=lambda k: printed_size(shaped[k]),
        reverse=True,
    )
    for key in movable:
        if printed_size(shaped) <= policy.byte_ceiling:
            break
        if not _worth_spilling(shaped[key]):
            continue
        original = payload[key]
        shaped[key] = {"$ref": spiller.write(key, original), "section": True}
        if isinstance(original, (list, str)):
            shaped[key]["length"] = len(original)
        # spill files written for parts of this section are superseded by the section file
        spiller.files[:] = [f for f in spiller.files if not f["key"].startswith(f"{key}.") and not f["key"].startswith(f"{key}[")]
        for t in [t for t in truncated if t == key or t.startswith(f"{key}.") or t.startswith(f"{key}[")]:
            del truncated[t]
    shaped["detail"]["spilled"] = [f["key"] for f in spiller.files]
    return shaped
