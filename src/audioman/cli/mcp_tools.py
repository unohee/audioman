# Created: 2026-11-04
# Purpose: Tools exposed by `audioman mcp`: analyze, describe, diff, uat_compare.
#
# Each tool returns three things, sized for a context window:
#   digest      one short paragraph (the text content of the MCP result)
#   result_ref  absolute path of a JSON file holding the complete, unshaped result
#   summary     the result shaped at the requested --detail level (structuredContent),
#               whose own spilled arrays are further file references
# so an agent reads the digest first and opens files only when it needs the numbers.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from audioman.core.detail import DETAIL_LEVELS

_PATH = {"type": "string", "description": "Absolute or working-directory-relative audio file path"}
_DETAIL = {"type": "string", "enum": list(DETAIL_LEVELS), "default": "summary",
           "description": "Size of the inline summary: summary (<= 4 KiB), standard (<= 32 KiB) or full"}


class ToolError(Exception):
    """A tool could not run on its input (bad path, unreadable file, rate mismatch)."""


def _read(path: str):
    from audioman.core.audio_file import read_audio

    p = Path(path).expanduser()
    if not p.is_file():
        raise ToolError(f"file not found: {path}")
    try:
        return read_audio(p)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ToolError(f"cannot read {path}: {exc}") from exc


def _results_dir() -> Path:
    from audioman.cli._detail import default_spill_dir

    return default_spill_dir().parent / "mcp"


def _store(tool: str, payload: dict) -> str:
    """Write the complete result once; the name is content-addressed."""
    body = json.dumps(payload, ensure_ascii=False, default=str)
    path = _results_dir() / f"{tool}-{hashlib.sha256(body.encode()).hexdigest()[:16]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path.resolve())


def _shape(command: str, payload: dict, detail: str, label: str) -> dict:
    from audioman.cli._detail import shape_json

    return shape_json(command, payload, SimpleNamespace(detail=detail, spill_dir=None), label=label)


def _finish(tool: str, command: str, payload: dict, detail: str, label: str) -> dict:
    shaped = _shape(command, payload, detail, label)
    result_ref = _store(tool, {**payload, "digest": shaped["digest"], "report": shaped["report"]})
    return {"digest": shaped["digest"], "result_ref": result_ref, "summary": shaped}


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def tool_analyze(args: dict) -> dict:
    from audioman.cli.analyze import _analyze_file
    from audioman.core.findings import json_envelope, schema_uri

    path = Path(args["path"]).expanduser()
    _read(str(path))  # validates before the heavier pipeline
    result = _analyze_file(path, 2048, 512, -40.0, False, spectrum=bool(args.get("spectrum", True)),
                           texture=bool(args.get("texture", False)),
                           segment_sec=float(args.get("segment_sec", 1.0)))
    payload = json_envelope("analyze", result, schema=schema_uri("analyze"))
    return _finish("analyze", "analyze", payload, args.get("detail", "summary"), path.stem)


def tool_describe(args: dict) -> dict:
    from audioman.core.digest import digest_texture
    from audioman.core.texture import describe

    audio, sr = _read(args["path"])
    texture = describe(audio, sr, segment_sec=float(args.get("segment_sec", 1.0)))
    payload = {"file": str(Path(args["path"]).expanduser().resolve()), **texture}
    digest = digest_texture(payload)
    ref = _store("describe", {**payload, "digest": digest})
    from audioman.core.detail import apply_detail
    from audioman.cli._detail import default_spill_dir

    summary = apply_detail({"digest": digest, **payload}, args.get("detail", "summary"),
                           spill_dir=default_spill_dir(), label=f"describe-{Path(args['path']).stem}")
    return {"digest": digest, "result_ref": ref, "summary": summary}


def tool_diff(args: dict) -> dict:
    from audioman.core.diff import diff_audio
    from audioman.core.findings import json_envelope, schema_uri

    a, sr_a = _read(args["a"])
    b, sr_b = _read(args["b"])
    if sr_a != sr_b:
        raise ToolError(f"sample rates differ ({sr_a} vs {sr_b} Hz); resample one file first")
    result = diff_audio(a, b, sr_a, segment_sec=float(args.get("segment_sec", 1.0)),
                        do_align=bool(args.get("align", True)))
    payload = {"a": str(Path(args["a"]).expanduser().resolve()), "b": str(Path(args["b"]).expanduser().resolve()),
               **result}
    if args.get("png"):
        from audioman.core.diff_figure import render

        figure = _results_dir() / f"diff-{Path(args['a']).stem}-vs-{Path(args['b']).stem}.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        payload["figure"] = str(render(result, figure).resolve())
    envelope = json_envelope("diff", payload, schema=schema_uri("diff"))
    out = _finish("diff", "diff", envelope, args.get("detail", "summary"),
                  f"{Path(args['a']).stem}-vs-{Path(args['b']).stem}")
    if "figure" in payload:
        out["figure"] = payload["figure"]
    return out


def tool_uat_compare(args: dict) -> dict:
    from audioman.core.digest import digest_report
    from audioman.core.uat_gates import TakeSet, compare_takes

    source, sr = _read(args["source"])
    takes = {}
    for key in ("offline", "realtime", "bypassed"):
        if args.get(key):
            audio, rate = _read(args[key])
            if rate != sr:
                raise ToolError(f"{key} is {rate} Hz but source is {sr} Hz")
            takes[key] = audio
    report = compare_takes(
        TakeSet(source, takes["offline"], takes["realtime"], sr, takes.get("bypassed"), args.get("render_seconds")),
        expect_processing=not bool(args.get("transparent", False)),
    ).to_dict()
    report["subject"]["file"] = str(Path(args["source"]).expanduser().resolve())
    digest = digest_report(report)
    ref = _store("uat_compare", {"digest": digest, "inputs": args, "report": report})
    return {"digest": digest, "result_ref": ref, "summary": {"digest": digest, "report": report}}


TOOLS: dict[str, dict[str, Any]] = {
    "analyze": {
        "title": "Analyze an audio file",
        "description": "Levels, silence, spectrum diagnostics and findings for one file, with a report.v2 verdict. "
                       "Returns a digest, the full result as a JSON file reference, and a size-bounded summary.",
        "inputSchema": {"type": "object", "required": ["path"], "additionalProperties": False, "properties": {
            "path": _PATH, "spectrum": {"type": "boolean", "default": True},
            "texture": {"type": "boolean", "default": False, "description": "Include texture descriptors"},
            "segment_sec": {"type": "number", "default": 1.0}, "detail": _DETAIL}},
        "run": tool_analyze,
    },
    "describe": {
        "title": "Describe the texture of an audio file",
        "description": "Texture descriptors (brightness, noisiness, change, dynamics, stereo image) for the whole "
                       "file and each segment, each with a repeat floor: changes smaller than the floor are noise "
                       "of the measurement.",
        "inputSchema": {"type": "object", "required": ["path"], "additionalProperties": False, "properties": {
            "path": _PATH, "segment_sec": {"type": "number", "default": 1.0}, "detail": _DETAIL}},
        "run": tool_describe,
    },
    "diff": {
        "title": "What changed between two renders",
        "description": "Aligns B to A and reports texture descriptor deltas (B - A) for the whole file, each "
                       "segment and each band; only changes beyond twice the combined floor are significant.",
        "inputSchema": {"type": "object", "required": ["a", "b"], "additionalProperties": False, "properties": {
            "a": {**_PATH, "description": "Reference (before)"}, "b": {**_PATH, "description": "Compared (after)"},
            "segment_sec": {"type": "number", "default": 1.0}, "align": {"type": "boolean", "default": True},
            "png": {"type": "boolean", "default": False, "description": "Also write a figure and return its path"},
            "detail": _DETAIL}},
        "run": tool_diff,
    },
    "uat_compare": {
        "title": "Real-time take vs offline render acceptance gates",
        "description": "Judges a host's real-time recording against its offline render and the dry source "
                       "(optionally a bypassed take and the render time): dropouts, unprocessed blocks, delay "
                       "compensation, bypass. Returns PASS/WARN/FAIL gates.",
        "inputSchema": {"type": "object", "required": ["source", "offline", "realtime"], "additionalProperties": False,
                        "properties": {"source": _PATH, "offline": _PATH, "realtime": _PATH, "bypassed": _PATH,
                                       "render_seconds": {"type": "number"},
                                       "transparent": {"type": "boolean", "default": False}}},
        "run": tool_uat_compare,
    },
}


def validate_arguments(name: str, arguments: Any) -> str | None:
    """Minimal input-schema check: object, required keys, no unknown keys, primitive types."""
    if not isinstance(arguments, dict):
        return "arguments must be an object"
    schema = TOOLS[name]["inputSchema"]
    missing = [k for k in schema["required"] if k not in arguments]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    unknown = [k for k in arguments if k not in schema["properties"]]
    if unknown:
        return f"unknown argument(s): {', '.join(unknown)}"
    checks: dict[str, Callable[[Any], bool]] = {
        "string": lambda v: isinstance(v, str),
        "boolean": lambda v: isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    }
    for key, value in arguments.items():
        prop = schema["properties"][key]
        if not checks[prop["type"]](value):
            return f"argument {key!r} must be a {prop['type']}"
        if "enum" in prop and value not in prop["enum"]:
            return f"argument {key!r} must be one of {prop['enum']}"
    return None
