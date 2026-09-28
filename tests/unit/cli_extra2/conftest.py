# tests/unit/cli_extra2/conftest.py
# Purpose: harness for the second batch of CLI coverage tests (AUD-1851):
#          bounce / vo / commit / fader-test, plus the residual-line proofs.
#
# Everything runs in-process through the real root parser (`app.main(argv)`) so
# `--cov=audioman` attributes the executed statements to the command modules.
# Host state is redirected into tmp_path and every module-level singleton that
# the CLI mutates is reset around each test.
#
# This host has no registrable VST3 plugin (AUD-1857), so plugin resolution goes
# through the stub registry / recording wrapper below.

from __future__ import annotations

import importlib
import io
import json
import sys
import types
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pytest
import soundfile as sf

from audioman.plugins.parameter import PluginMeta


class CliResult:
    """Exit status plus the captured streams of one in-process invocation."""

    def __init__(self, code: int, stdout: str, stderr: str):
        self.code = code
        self.stdout = stdout
        self.stderr = stderr

    @property
    def payload(self) -> dict:
        return json.loads(self.stdout)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<CliResult code={self.code} out={self.stdout[:200]!r} err={self.stderr[:200]!r}>"


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Isolate cache/preset/home and reset the settings + registry singletons.

    ``COLUMNS`` is pinned so the rich console cannot wrap a long tmp_path in the
    middle of an assertion target; line wrapping is incidental to every claim
    these tests make.
    """
    monkeypatch.setenv("AUDIOMAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUDIOMAN_PRESET_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("AUDIOMAN_PLAIN", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "400")

    from audioman.config import settings as settings_module
    import audioman.core.registry as registry_module

    settings_module.reset_settings()
    registry_module._registry = None
    yield
    settings_module.reset_settings()
    registry_module._registry = None


@pytest.fixture
def run_cli(cli_env):
    """Run `audioman <argv>` in-process; returns (exit code, stdout, stderr)."""

    def _run(argv, *, json_mode: bool = False) -> CliResult:
        from audioman.cli import app

        full = ["--plain"]
        if json_mode:
            full.append("--json")
        full.extend(str(a) for a in argv)

        out, err = io.StringIO(), io.StringIO()
        real_stdout, real_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            app.main(full)
            code = 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        finally:
            sys.stdout, sys.stderr = real_stdout, real_stderr
        return CliResult(code, out.getvalue(), err.getvalue())

    return _run


def run_command(func, args) -> CliResult:
    """Invoke a command function directly with a prepared namespace.

    Used for branches the argparse surface cannot produce on its own (for
    example a session-supplied default for a flag the parser marks required).
    The real command function still runs; only the namespace is hand-built.
    """
    out, err = io.StringIO(), io.StringIO()
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        func(args)
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr
    return CliResult(code, out.getvalue(), err.getvalue())


@pytest.fixture
def silent_error(monkeypatch):
    """Replace `print_error` in a CLI module with a recorder that does not exit.

    `print_error` calls `sys.exit(1)`, which would mask the explicit `return`
    guards that follow it. Neutralising the exit is what makes those `return`
    statements observable (and proves they, not the exit, stop the command).
    """

    def _install(module_name: str) -> list[str]:
        module = importlib.import_module(module_name)
        seen: list[str] = []
        monkeypatch.setattr(module, "print_error", lambda message: seen.append(message))
        return seen

    return _install


def write_wav(
    path: Path,
    *,
    sample_rate: int = 8000,
    duration: float = 0.2,
    frequency: float = 440.0,
    amplitude: float = 0.4,
    channels: int = 2,
    subtype: str = "PCM_16",
) -> Path:
    """Deterministic sine fixture (no network, no host state)."""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    data = mono if channels == 1 else np.stack([mono] * channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, subtype=subtype)
    return path


def make_meta(
    short_name: str = "fake-denoiser",
    name: str = "Fake De-noise",
    path: str = "/nonexistent/fake.vst3",
    aliases: Optional[list[str]] = None,
) -> PluginMeta:
    return PluginMeta(
        name=name,
        short_name=short_name,
        path=path,
        format="vst3",
        vendor="FakeVendor",
        version="1.0",
        aliases=list(aliases or []),
        param_count=0,
    )


class StubRegistry:
    """Registry double: resolves the canned metas, records the lookups."""

    def __init__(self, plugins: Iterable[PluginMeta] = ()) -> None:
        self.plugins = list(plugins)
        self.get_calls: list[str] = []

    def get(self, name: str) -> Optional[PluginMeta]:
        self.get_calls.append(name)
        for plugin in self.plugins:
            if name == plugin.short_name or name in plugin.aliases:
                return plugin
        return None

    def list(self, fmt=None, vendor=None):  # pragma: no cover - not exercised here
        return list(self.plugins)

    def scan(self, extra_paths=None, refresh: bool = False):  # pragma: no cover
        return list(self.plugins)


def install_registry(monkeypatch, modules: Iterable[str], metas: Iterable[PluginMeta]) -> StubRegistry:
    """Bind `get_registry` in every listed module to one stub registry.

    These modules did `from audioman.core.registry import get_registry` at import
    time, so patching the registry module alone would not reach them.
    """
    stub = StubRegistry(metas)
    monkeypatch.setattr("audioman.core.registry.get_registry", lambda: stub)
    for name in modules:
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "get_registry", lambda: stub)
    return stub


class RecordingWrapper:
    """`VST3PluginWrapper` double: delay line + gain + optional tail.

    Deterministic so latency compensation and tail trimming can be asserted on
    exact sample indices.
    """

    def __init__(self, path, *, delay: int = 0, tail: int = 0, gain: float = 1.0,
                 reported: Optional[int] = None) -> None:
        self.path = path
        self.delay = delay
        self.tail = tail
        self.gain = gain
        self.name = "Fake Plugin"
        self.loaded = False
        self.applied: list[dict] = []
        self.processed = 0
        plugin = types.SimpleNamespace()
        if reported is not None:
            plugin.latency_samples = reported
        self._plugin = plugin

    def load(self) -> None:
        self.loaded = True

    def reset(self) -> None:
        pass

    def set_parameters(self, params: dict) -> None:
        self.applied.append(dict(params))

    def process(self, audio: np.ndarray, sample_rate: int, reset: bool = True) -> np.ndarray:
        self.processed += 1
        out = np.array(audio, dtype=np.float32, copy=True)
        if self.gain != 1.0:
            out *= np.float32(self.gain)
        if self.delay:
            shifted = np.zeros_like(out)
            shifted[..., self.delay:] = out[..., : out.shape[-1] - self.delay]
            out = shifted
        if self.tail:
            pad = np.zeros(out.shape[:-1] + (self.tail,), dtype=out.dtype)
            out = np.concatenate([out, pad], axis=-1)
        return out


def install_wrapper(monkeypatch, modules: Iterable[str], *, instances: Optional[list] = None,
                    **kwargs: Any) -> list[RecordingWrapper]:
    """Replace `VST3PluginWrapper` in the listed modules; returns created wrappers."""
    created: list = instances if instances is not None else []

    def _factory(path) -> RecordingWrapper:
        wrapper = RecordingWrapper(path, **kwargs)
        created.append(wrapper)
        return wrapper

    for name in modules:
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "VST3PluginWrapper", _factory)
    return created
