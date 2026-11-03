# tests/unit/cli_commands/conftest.py
# Purpose: isolation for the CLI command coverage tests (AUD-1851).
#
# The CLI writes to the app cache/preset directories and mutates module-level
# singletons (registry, plain-mode consoles), so every test starts from a clean,
# tmp_path-bound state and never touches the developer's home directory.

from __future__ import annotations

from pathlib import Path

import pytest

from audioman.cli import output as output_module


# Every module that imported ``get_registry`` by value. Patching the function in
# ``audioman.core.registry`` alone would not affect them, because the name was
# already bound into each module's namespace at import time.
_REGISTRY_CONSUMERS = (
    "audioman.cli.scan",
    "audioman.cli.list_cmd",
    "audioman.cli.info",
    "audioman.cli.dump",
    "audioman.core.engine",
    "audioman.core.pipeline",
)


def _install_registry(monkeypatch, stub) -> None:
    import importlib

    import audioman.core.registry as registry_module

    monkeypatch.setattr(registry_module, "get_registry", lambda: stub)
    for name in _REGISTRY_CONSUMERS:
        module = importlib.import_module(name)
        if hasattr(module, "get_registry"):
            monkeypatch.setattr(module, "get_registry", lambda: stub)


@pytest.fixture(autouse=True)
def isolated_cli_env(tmp_path: Path, monkeypatch):
    """Redirect cache/preset/plain state at tmp_path and reset singletons."""
    monkeypatch.setenv("AUDIOMAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUDIOMAN_PRESET_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AUDIOMAN_PLAIN", raising=False)

    from audioman.config import settings as settings_module
    import audioman.core.registry as registry_module

    settings_module.reset_settings()
    registry_module._registry = None
    output_module.set_plain(False)

    yield

    settings_module.reset_settings()
    registry_module._registry = None
    output_module.set_plain(False)


@pytest.fixture
def empty_registry(monkeypatch):
    """Install an empty registry (this host has zero registrable plugins)."""
    from harness import StubRegistry

    stub = StubRegistry()
    _install_registry(monkeypatch, stub)
    return stub


@pytest.fixture
def fake_registry(monkeypatch):
    """Install a registry holding two fake plugins and return it."""
    from harness import StubRegistry, make_meta

    stub = StubRegistry([
        make_meta(
            short_name="fake-denoise",
            name="Fake De-noise",
            path="/nonexistent/Fake De-noise.vst3",
            fmt="vst3",
            vendor="FakeVendor",
            aliases=["denoise", "deno"],
        ),
        make_meta(
            short_name="fake-comp",
            name="Fake Comp",
            path="/nonexistent/Fake Comp.component",
            fmt="au",
            vendor="OtherVendor",
        ),
    ])
    _install_registry(monkeypatch, stub)
    return stub
