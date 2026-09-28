# Created: 2026-09-28
# Purpose: In-process harness for the CLI command coverage tests (AUD-1851).
#
# Every CLI test here drives the real argument parser and the real run(args)
# entry point inside the current process, so `--cov=audioman` attributes the
# executed statements to the command modules instead of a subprocess.
#
# This host has no registrable VST3 plugin, so anything that would resolve a
# real plugin goes through the fake registry / fake wrapper below.

from __future__ import annotations

import io
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import soundfile as sf

from audioman.cli import app
from audioman.plugins.parameter import ParameterInfo, PluginMeta


@dataclass
class CliResult:
    """Captured outcome of an in-process CLI invocation."""

    code: int
    out: str
    err: str


@contextmanager
def capture_streams():
    """Swap ``sys.stdout`` / ``sys.stderr`` for in-memory buffers, then restore.

    ``contextlib.redirect_stdout`` cannot be used together with rich: a rich
    ``Console`` created before the redirect keeps writing to its original
    stream, whereas assigning ``sys.stdout`` is picked up because rich resolves
    ``sys.stdout`` at write time.

    ``pytest``'s ``monkeypatch`` cannot be used for the same job: with capture
    enabled, pytest re-binds ``sys.stdout`` to its own replacement after fixture
    setup, silently discarding the monkeypatched object.
    """
    out, err = io.StringIO(), io.StringIO()
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        yield out, err
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr


def run_cli(argv: list[str]) -> CliResult:
    """Run ``audioman <argv>`` in-process; return (exit code, stdout, stderr).

    ``app.main`` also mutates ``os.environ["AUDIOMAN_PLAIN"]`` as a side effect
    of ``--plain``; the previous value is restored so the flag cannot leak into
    unrelated tests.
    """
    had_plain = "AUDIOMAN_PLAIN" in os.environ
    plain_before = os.environ.get("AUDIOMAN_PLAIN")
    with capture_streams() as (out, err):
        try:
            app.main(argv)
            code = 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    if had_plain:
        os.environ["AUDIOMAN_PLAIN"] = plain_before  # type: ignore[assignment]
    else:
        os.environ.pop("AUDIOMAN_PLAIN", None)
    return CliResult(code=code, out=out.getvalue(), err=err.getvalue())


def run_command(argv: list[str]) -> CliResult:
    """Run a command through the real root parser.

    The root parser (``app.build_parser``) defines the global ``--json`` /
    ``--plain`` flags and the per-command defaults, so tests must go through it
    rather than a hand-built parser. Equivalent to ``run_cli``; the separate
    name marks call sites that assert on command behaviour.
    """
    return run_cli(argv)


# ---------------------------------------------------------------------------
# Audio fixtures written with numpy + soundfile (no network, no host state)
# ---------------------------------------------------------------------------


def write_wav(
    path: Path,
    *,
    sample_rate: int = 44100,
    duration: float = 1.0,
    freq: float = 440.0,
    amp: float = 0.5,
    channels: int = 2,
    offset: float = 0.0,
    subtype: str = "PCM_16",
) -> Path:
    """Write a sine tone WAV. ``offset`` adds a DC bias."""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = (amp * np.sin(2 * np.pi * freq * t) + offset).astype(np.float32)
    data = np.stack([mono] * channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, subtype=subtype)
    return path


def write_pattern_wav(
    path: Path,
    *,
    sample_rate: int = 44100,
    segment_sec: float = 0.2,
    silence_segments: int = 2,
    freq: float = 440.0,
    amp: float = 0.5,
    channels: int = 2,
) -> Path:
    """Tone with alternating silent/tone segments (silence detection input)."""
    n = int(sample_rate * segment_sec)
    t = np.arange(n, dtype=np.float32) / sample_rate
    tone = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    parts: list[np.ndarray] = []
    for _ in range(silence_segments):
        parts.append(np.zeros(n, dtype=np.float32))
        parts.append(tone)
    mono = np.concatenate(parts)
    data = np.stack([mono] * channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")
    return path


def write_silence_wav(path: Path, *, sample_rate: int = 44100, duration: float = 1.0,
                      channels: int = 2) -> Path:
    n = int(sample_rate * duration)
    data = np.zeros((n, channels), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")
    return path


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read back a written WAV as (channels, samples) float32."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T, sr


# ---------------------------------------------------------------------------
# Registry / wrapper stubs
# ---------------------------------------------------------------------------


def make_meta(
    short_name: str = "fake-denoise",
    name: str = "Fake De-noise",
    path: str = "/nonexistent/fake.vst3",
    fmt: str = "vst3",
    vendor: str = "FakeVendor",
    version: str = "1.0",
    aliases: Optional[list[str]] = None,
    param_count: int = 0,
) -> PluginMeta:
    return PluginMeta(
        name=name,
        short_name=short_name,
        path=path,
        format=fmt,
        vendor=vendor,
        version=version,
        aliases=list(aliases or []),
        param_count=param_count,
    )


class StubRegistry:
    """Registry double: records the calls the CLI makes and returns canned metas."""

    def __init__(self, plugins: Iterable[PluginMeta] = ()) -> None:
        self.plugins = list(plugins)
        self.scan_calls: list[dict] = []
        self.list_calls: list[dict] = []
        self.get_calls: list[str] = []

    def scan(self, extra_paths=None, refresh: bool = False) -> list[PluginMeta]:
        self.scan_calls.append({"extra_paths": extra_paths, "refresh": refresh})
        return list(self.plugins)

    def list(self, fmt=None, vendor=None) -> list[PluginMeta]:
        self.list_calls.append({"fmt": fmt, "vendor": vendor})
        results = list(self.plugins)
        if fmt:
            results = [p for p in results if p.format == fmt]
        if vendor:
            results = [p for p in results if vendor.lower() in p.vendor.lower()]
        return results

    def get(self, name: str) -> Optional[PluginMeta]:
        self.get_calls.append(name)
        for p in self.plugins:
            if name == p.short_name or name in p.aliases:
                return p
        return None


def stub_plugin_params() -> list[ParameterInfo]:
    """One parameter of each kind the human-readable range column renders."""
    return [
        ParameterInfo(name="threshold", label="Threshold", min_value=-60.0, max_value=0.0,
                      current_value=-20.0, type="float"),
        ParameterInfo(name="mode", label="Mode", current_value="fast", type="enum"),
        ParameterInfo(name="bypass", label="Bypass", current_value=False, type="bool"),
        ParameterInfo(name="unbounded", label="Unbounded", current_value=None, type="float"),
    ]


class FakePlugin:
    """Minimal pedalboard-plugin double: attribute container + parameters map."""

    identifier = "com.fake.plugin"
    version = "9.9"

    def __init__(self, **values: Any) -> None:
        self.parameters = {name: object() for name in (*values, *getattr(self, "extra_params", ()))}
        self._values = values

    def __getattr__(self, item: str) -> Any:
        values = self.__dict__.get("_values", {})
        if item in values:
            return values[item]
        raise AttributeError(item)


class FakeWrapper:
    """VST3PluginWrapper double. ``plugin_factory`` may raise to simulate failure."""

    instances: list["FakeWrapper"] = []

    def __init__(self, path, plugin_factory: Callable[[], Any] = FakePlugin) -> None:
        self.path = path
        self._plugin = None
        self._factory = plugin_factory
        self.loaded = False
        self.applied: list[dict] = []
        FakeWrapper.instances.append(self)

    def load(self) -> None:
        self._plugin = self._factory()
        self.loaded = True

    def set_parameters(self, params: dict) -> None:
        self.applied.append(dict(params))

    def process(self, audio, sample_rate, reset: bool = True):
        return audio


def wrapper_factory(plugin_factory: Callable[[], Any] = FakePlugin) -> Callable[..., FakeWrapper]:
    """Build a ``VST3PluginWrapper(path)`` replacement with a fixed fake plugin."""

    def _factory(path) -> FakeWrapper:
        return FakeWrapper(path, plugin_factory)

    return _factory


class FakePool:
    """Deterministic stand-in for ``multiprocessing.Pool``.

    Runs the workers in-process (in order) instead of forking, so the parallel
    batch branches stay fast and reproducible and keep the monkeypatched module
    state visible to the worker function.
    """

    def __init__(self, processes: int = 1) -> None:
        self.processes = processes

    def __enter__(self) -> "FakePool":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def imap_unordered(self, func: Callable, jobs: Iterable) -> Iterable:
        return (func(job) for job in jobs)
