# tests/unit/cli_workflow/conftest.py
# Purpose: shared harness for the editing/mastering workflow CLI commands
#          (stream, edl, master, mixdown, doctor, eq-profile, visualize).
#
# Every test drives the real command functions in-process (`app.main(argv)`) so
# the assertions cover the CLI's own wiring: argparse handling, payload shape,
# files actually written, exit status and error messages. No host state leaks:
# HOME, the registry cache and the preset dir are redirected into tmp_path.

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Isolate cache/preset/home so no host state leaks into a CLI run."""
    monkeypatch.setenv("AUDIOMAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUDIOMAN_PRESET_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("AUDIOMAN_PLAIN", "1")
    monkeypatch.setenv("HOME", str(tmp_path))

    from audioman.config import settings as settings_module
    import audioman.core.registry as registry_module

    settings_module.reset_settings()
    registry_module._registry = None
    yield
    settings_module.reset_settings()
    registry_module._registry = None


class CliResult:
    """Exit status plus captured streams of one in-process CLI invocation."""

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
def run_cli(cli_env):
    """Run `audioman <argv>` in-process; returns (exit_code, stdout, stderr)."""

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


@pytest.fixture
def real_cli_console_binding(monkeypatch):
    """`--plain` 실행 시 실제 CLI가 갖는 콘솔 바인딩을 재현한다.

    `cli/*.py`는 `from audioman.cli.output import output_console`으로 이름을
    import 시점에 값으로 묶는다. 실제 프로세스에서는 `AUDIOMAN_PLAIN`이 import
    전에 `_early_plain_detect`로 설정되므로 그 이름이 **plain 콘솔**을 가리키고,
    태그가 붙은 문자열을 `console.print`에 직접 넘기면 태그가 그대로 찍힌다.

    in-process 테스트는 이미 rich 콘솔이 묶인 뒤 `app.main(["--plain", ...])`을
    부르므로 그 상태가 되지 않는다 — 즉 이 픽스처 없이는 plain 마크업 누수를
    탐지할 수 없다 (실제 버그를 통과시킨다).
    """
    from audioman.cli import output as output_module

    def _bind_plain(*modules):
        output_module.set_plain(True)
        for module in modules:
            monkeypatch.setattr(module, "output_console", output_module.output_console)

    return _bind_plain


def write_wav(
    path: Path,
    *,
    sample_rate: int = 8000,
    duration: float = 1.0,
    frequency: float = 440.0,
    amplitude: float = 0.3,
    channels: int = 2,
    subtype: str = "PCM_16",
    dc_offset: float = 0.0,
) -> Path:
    """Deterministic sine fixture written with numpy + soundfile."""
    n = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)
    mono = (amplitude * np.sin(2 * np.pi * frequency * t) + dc_offset).astype(np.float32)
    data = mono if channels == 1 else np.stack([mono] * channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, subtype=subtype)
    return path
