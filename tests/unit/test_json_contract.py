# tests/unit/test_json_contract.py
# Purpose: enforce the audioman --json contract (Linear AUD-1852).
#
# Every `--json` payload must carry `$schema` + `audioman_version` + `command`,
# every emitted `$schema` URI must resolve to a published file under
# src/audioman/schemas/, and findings-shaped payloads must validate against
# their published JSONSchema.

from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audioman import __version__
from audioman.cli import app
from audioman.core.findings import SCHEMA_URI_PREFIX, json_envelope, schema_uri
from audioman.core import findings as findings_mod
from audioman.core.findings import Code, Category, Severity, envelope


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = REPO_ROOT / "src" / "audioman" / "schemas"

# Keywords the repo's schemas are allowed to use, and the subset validator below
# only understands these. Kept deliberately small and explicit.
_ALLOWED_KEYWORDS = {
    "$schema", "$id", "title", "description", "type", "required", "properties",
    "items", "enum", "const", "additionalProperties", "anyOf", "$ref",
}


# ---------------------------------------------------------------------------
# Payload validation (jsonschema is not a dependency of this project, so this
# module ships a focused structural validator instead of importing it)
# ---------------------------------------------------------------------------


class SchemaError(AssertionError):
    pass


def _resolve_ref(ref: str) -> dict:
    if not ref.startswith(SCHEMA_URI_PREFIX):
        raise SchemaError(f"only audioman:// refs are supported, got {ref!r}")
    path = SCHEMAS_DIR / ref[len(SCHEMA_URI_PREFIX):]
    if not path.is_file():
        raise SchemaError(f"unresolvable $ref: {ref} (missing {path})")
    return json.loads(path.read_text(encoding="utf-8"))


def _type_matches(value, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise SchemaError(f"unknown type {expected!r}")


def validate_against(instance, schema: dict, *, where: str = "$") -> None:
    """Validate `instance` against a repository schema (supported subset).

    Supports: type (string or list), const, enum, required, properties, items,
    anyOf, additionalProperties and audioman:// $ref.
    """
    unknown = set(schema) - _ALLOWED_KEYWORDS
    if unknown:
        raise SchemaError(f"{where}: schema uses unsupported keywords {sorted(unknown)}")

    if "$ref" in schema:
        validate_against(instance, _resolve_ref(schema["$ref"]), where=where)

    if "const" in schema and instance != schema["const"]:
        raise SchemaError(f"{where}: expected const {schema['const']!r}, got {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaError(f"{where}: {instance!r} not in enum {schema['enum']!r}")

    if "type" in schema:
        expected = schema["type"]
        expected_list = expected if isinstance(expected, list) else [expected]
        if not any(_type_matches(instance, t) for t in expected_list):
            raise SchemaError(
                f"{where}: expected type {expected!r}, got {type(instance).__name__}"
            )

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                raise SchemaError(f"{where}: missing required key {key!r}")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in props:
                validate_against(value, props[key], where=f"{where}.{key}")
            elif extra is False:
                raise SchemaError(f"{where}: unexpected key {key!r}")
            elif isinstance(extra, dict):
                validate_against(value, extra, where=f"{where}.{key}")

    if isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            validate_against(item, schema["items"], where=f"{where}[{index}]")

    if "anyOf" in schema:
        errors = []
        for candidate in schema["anyOf"]:
            try:
                validate_against(instance, candidate, where=where)
                return
            except SchemaError as exc:
                errors.append(str(exc))
        raise SchemaError(f"{where}: no anyOf branch matched: {errors}")


def load_schema(name: str) -> dict:
    path = SCHEMAS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI harness: invoke in-process so tests stay fast and deterministic
# ---------------------------------------------------------------------------


@pytest.fixture
def audio_wav(tmp_path) -> Path:
    path = tmp_path / "tone.wav"
    sr = 8000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False, dtype=np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), tone, sr)
    return path


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Isolate cache/preset dirs so no host state leaks into the run."""
    cache = tmp_path / "cache"
    presets = tmp_path / "presets"
    monkeypatch.setenv("AUDIOMAN_CACHE_DIR", str(cache))
    monkeypatch.setenv("AUDIOMAN_PRESET_DIR", str(presets))
    monkeypatch.setenv("AUDIOMAN_PLAIN", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    from audioman.config import settings as settings_module
    settings_module.reset_settings()
    import audioman.core.registry as registry_module
    registry_module._registry = None
    yield
    settings_module.reset_settings()
    registry_module._registry = None


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    """Run the CLI in-process; return (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        app.main(argv)
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr
    return code, out.getvalue(), err.getvalue()


def run_json(argv: list[str]) -> tuple[int, dict]:
    code, out, _err = run_cli(["--json", "--plain", *argv])
    assert out.strip(), f"no stdout for {argv} (exit {code})"
    return code, json.loads(out)


# ---------------------------------------------------------------------------
# 1. Every command emits the envelope
# ---------------------------------------------------------------------------

# Commands that need no plugin and no GUI. Each entry is (argv, expected command).
RUNNABLE_JSON_COMMANDS = [
    (["scan"], "scan"),
    (["list"], "list"),
    (["preset", "list"], "preset list"),
    (["schemas", "list"], "schemas"),
    (["changelog"], "changelog"),
    (["master", "list-profiles"], "master list-profiles"),
]


@pytest.mark.parametrize("argv,command", RUNNABLE_JSON_COMMANDS, ids=lambda v: str(v))
def test_command_payload_has_envelope(cli_env, argv, command):
    code, payload = run_json(argv)
    assert code == 0
    assert payload["$schema"].startswith(SCHEMA_URI_PREFIX)
    assert payload["audioman_version"] == __version__
    assert payload["command"] == command


@pytest.mark.parametrize("argv,command", [
    (["observe", "{wav}"], "observe"),
    (["screen", "{wav}"], "screen"),
    (["master", "qc", "{wav}"], "master qc"),
    (["edl", "status", "-s", "{wav}"], "edl status"),
    (["vo", "analyze", "{wav}"], "vo analyze"),
])
def test_file_commands_payload_has_envelope(cli_env, audio_wav, argv, command):
    argv = [str(audio_wav) if a == "{wav}" else a for a in argv]
    code, payload = run_json(argv)
    assert code == 0
    assert payload["$schema"].startswith(SCHEMA_URI_PREFIX)
    assert payload["audioman_version"] == __version__
    assert payload["command"] == command


@pytest.mark.parametrize("argv,command", [
    (["dump", "--all", "--filter", "__no_such_plugin__"], "dump"),
])
def test_dump_rejects_unknown_filter_without_faking_a_payload(cli_env, argv, command):
    """dump always emits machine-readable output; when the filter matches nothing
    it must fail loudly on stderr instead of printing an empty/fake JSON record."""
    code, out, err = run_cli(["--plain", *argv])
    assert code == 1
    assert "error" in err
    assert not out.lstrip().startswith("{")


# ---------------------------------------------------------------------------
# 2. Every emitted $schema URI resolves to a published file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv,_command", RUNNABLE_JSON_COMMANDS, ids=lambda v: str(v))
def test_emitted_schema_uri_resolves(cli_env, argv, _command):
    _code, payload = run_json(argv)
    uri = payload["$schema"]
    assert uri.startswith(SCHEMA_URI_PREFIX)
    published = SCHEMAS_DIR / uri[len(SCHEMA_URI_PREFIX):]
    assert published.is_file(), f"{uri} has no published file ({published})"


def test_findings_shaped_schema_uri_resolves(cli_env, audio_wav):
    _code, payload = run_json(["observe", str(audio_wav)])
    uri = payload["$schema"]
    assert (SCHEMAS_DIR / uri[len(SCHEMA_URI_PREFIX):]).is_file()


# ---------------------------------------------------------------------------
# 3. No dangling URIs anywhere in src/
# ---------------------------------------------------------------------------

_URI_RE = re.compile(r"audioman://schema/[A-Za-z0-9._-]+\.json")
_SCHEMA_URI_CALL_RE = re.compile(r"schema_uri\(\s*[\"']([A-Za-z0-9._-]+)[\"']\s*\)")


def _referenced_schema_uris() -> dict[str, list[str]]:
    """Every schema URI reachable from src/: literal URIs plus schema_uri("name")."""
    referenced: dict[str, list[str]] = {}
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(REPO_ROOT))
        uris = set(_URI_RE.findall(text))
        uris |= {f"{SCHEMA_URI_PREFIX}{name}.v1.json"
                 for name in _SCHEMA_URI_CALL_RE.findall(text)}
        for uri in uris:
            referenced.setdefault(uri, []).append(rel)
    return referenced


def test_no_dangling_schema_uri_in_src():
    """Every schema URI reachable from src/ must have a published file.

    Covers both forms: hand-written literals (analyze.py, changelog_cmd.py,
    findings.SCHEMA_URI) and URIs built via schema_uri("name"). This is the guard
    that keeps dangling URIs from returning as commands are added.
    """
    src = REPO_ROOT / "src"
    assert src.is_dir()

    referenced = _referenced_schema_uris()
    assert referenced, "no schema URIs found in src/ — the scanner is broken"

    missing = {
        uri: where for uri, where in referenced.items()
        if not (SCHEMAS_DIR / uri[len(SCHEMA_URI_PREFIX):]).is_file()
    }
    assert missing == {}, f"dangling schema URIs: {missing}"


def test_every_schema_ref_target_is_published():
    """No published schema may $ref a URI that does not resolve to a file."""
    for path in sorted(SCHEMAS_DIR.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        for uri in set(_URI_RE.findall(text)):
            target = SCHEMAS_DIR / uri[len(SCHEMA_URI_PREFIX):]
            assert target.is_file(), f"{path.name} references missing schema {uri}"


def test_every_published_schema_matches_its_filename():
    """$id must equal audioman://schema/<filename>, so schemas list is truthful."""
    for path in sorted(SCHEMAS_DIR.glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema.get("$id") == f"{SCHEMA_URI_PREFIX}{path.name}", path.name


def test_published_schemas_use_only_supported_keywords():
    """The published schemas must stay inside the subset this test can validate."""
    for path in sorted(SCHEMAS_DIR.glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert set(schema) <= _ALLOWED_KEYWORDS, (path.name, set(schema) - _ALLOWED_KEYWORDS)


def test_schemas_list_reports_only_published_files(cli_env):
    """`schemas list --json` must advertise exactly the files on disk."""
    _code, payload = run_json(["schemas", "list"])
    listed = {entry["id"] for entry in payload["schemas"]}
    on_disk = {f"{SCHEMA_URI_PREFIX}{p.name}" for p in SCHEMAS_DIR.glob("*.json")}
    assert listed == on_disk


# ---------------------------------------------------------------------------
# 4. Findings-shaped payloads validate against their published JSONSchema
# ---------------------------------------------------------------------------


def _make_findings() -> list:
    return [
        findings_mod.Finding(
            code=Code.CLIP_SAMPLE_PEAK_EXCEEDED,
            category=Category.SIGNAL,
            severity=Severity.CRITICAL,
            hint="Clipping detected",
            measurement={"samples_clipped": 5},
        ),
        findings_mod.Finding(
            code=Code.SILENCE_LEADING,
            category=Category.SIGNAL,
            severity=Severity.INFO,
            hint="Leading silence",
        ),
    ]


def test_envelope_helper_output_validates_against_published_schemas():
    """The shared helper's output must satisfy its own schema, for both
    findings-shaped commands (observe, stream triage).

    Each payload mirrors the real emit site: observe always passes `file`, and
    stream triage always adds the `stream` block.
    """
    observe_payload = envelope(
        _make_findings(),
        schema=schema_uri("observe"),
        command="observe",
        file="x.wav",
        extra={
            "sample_rate": 48000,
            "channels": 2,
            "duration_sec": 0.5,
            "total_samples": 24000,
            "filter": {"categories": ["signal"], "min_severity": "info"},
        },
    )
    validate_against(observe_payload, load_schema("observe.v1"))
    assert observe_payload["$schema"] == "audioman://schema/observe.v1.json"
    assert observe_payload["audioman_version"] == __version__
    assert observe_payload["command"] == "observe"

    summary = {
        "mode": "streamed", "block_size": 512, "reset_per_block": False,
        "sample_rate": 48000, "blocks": 4, "audio_seconds": 0.5,
        "total_process_sec": 0.01, "realtime_factor": 0.02,
    }
    triage_payload = envelope(
        _make_findings(),
        schema=schema_uri("stream"),
        command="stream triage",
        file="x.wav",
        extra={"stream": {
            "block_size": 512,
            "reset_per_block": False,
            "reset_first": True,
            "offline_summary": summary,
            "streamed_summary": summary,
        }},
    )
    validate_against(triage_payload, load_schema("stream.v1"))
    assert triage_payload["$schema"] == "audioman://schema/stream.v1.json"
    assert triage_payload["command"] == "stream triage"


def test_observe_payload_validates_against_observe_schema(cli_env, audio_wav):
    _code, payload = run_json(["observe", str(audio_wav)])
    validate_against(payload, load_schema("observe.v1"))
    assert payload["$schema"] == "audioman://schema/observe.v1.json"


def test_stream_triage_payload_validates_against_stream_schema(cli_env):
    """stream triage used to advertise finding.v1.json while adding a `stream`
    key, so it validated against nothing. It must now validate."""
    code, payload = run_json([
        "stream", "triage", "sine", "-p", "builtin:reverb",
        "--block-size", "512", "--duration", "0.3",
    ])
    assert code == 0
    validate_against(payload, load_schema("stream.v1"))
    assert payload["$schema"] == "audioman://schema/stream.v1.json"
    assert payload["command"] == "stream triage"
    assert payload["stream"]["block_size"] == 512


def test_finding_objects_validate_against_finding_schema():
    for finding in _make_findings():
        validate_against(finding.to_dict(), load_schema("finding.v1"))


def test_summary_block_matches_findings_count():
    payload = envelope(_make_findings())
    assert payload["summary"]["total"] == len(payload["findings"])
    assert sum(payload["summary"]["by_severity"].values()) == len(payload["findings"])


# ---------------------------------------------------------------------------
# 5. The generic helper cannot be bypassed by a body that carries the metadata
# ---------------------------------------------------------------------------


def test_envelope_body_cannot_override_metadata():
    payload = json_envelope(
        "scan",
        {"$schema": "audioman://schema/bogus.v1.json", "command": "bogus", "count": 1},
        schema=schema_uri("scan"),
    )
    assert payload["$schema"] == schema_uri("scan")
    assert payload["command"] == "scan"
    assert payload["audioman_version"] == __version__
    assert payload["count"] == 1


def test_envelope_keeps_body_key_order_after_metadata():
    payload = json_envelope("list", {"count": 0, "plugins": []}, schema=schema_uri("list"))
    assert list(payload)[:3] == ["$schema", "audioman_version", "command"]
    assert list(payload)[3:] == ["count", "plugins"]


# ---------------------------------------------------------------------------
# 6. Surfaces without --json support must say so in --help
# ---------------------------------------------------------------------------


def _help_text(argv: list[str]) -> str:
    code, out, _err = run_cli([*argv, "--help"])
    assert code == 0
    return out


def test_fader_test_documents_gui_only_contract():
    """fader-test is an interactive GUI; it emits no JSON payload by design."""
    text = _help_text(["fader-test"])
    assert "GUI" in text
    # The help must state the deliberate absence, not merely omit a flag.
    assert "no JSON output mode" in text


def test_stream_play_exposes_no_json_payload():
    """stream play drives a real audio device; its result is audible, not JSON.

    --json is a *global* flag, so it always parses; the contract is that play
    never consults it and its own help states the deliberate absence.
    """
    parser = app.build_parser()
    parsed = parser.parse_args(["stream", "play", "sine", "-p", "builtin:reverb"])
    assert parsed.func.__name__ == "run_play"
    text = _help_text(["stream", "play"])
    assert "machine-readable" in text


def test_dump_documents_implicit_json():
    """dump always emits machine-readable output; --json is accepted but implied."""
    text = _help_text(["dump"])
    assert "--json is implied" in text
    assert "not needed" in text


def test_visualize_has_no_json_payload():
    """visualize writes SVL/PNG artifacts; it deliberately emits no JSON document.

    --json is global so it parses, but nothing in this command's help advertises
    a JSON payload.
    """
    text = _help_text(["visualize"])
    assert "json" not in text.lower()
