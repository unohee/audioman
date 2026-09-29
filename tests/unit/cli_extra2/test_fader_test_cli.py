# tests/unit/cli_extra2/test_fader_test_cli.py
# Purpose: cover `audioman fader-test` — the argument surface, the input
#          validation, the PyQt6-missing error path and the GUI launch.
#
# `core.multitrack_player` imports `sounddevice`, which this host cannot load
# (no PortAudio), so a stub module is installed before the import — exactly what
# a machine without an audio device would need. Qt runs offscreen, and
# `QApplication.exec` is replaced so the GUI branch returns instead of blocking
# on an event loop no test can close.
#
# QT_QPA_PLATFORM is forced offscreen for the whole module: creating a
# QApplication without a display aborts the process, so a missing guard would
# kill the test session instead of failing one test.

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# PortAudio is absent on this host, so `import sounddevice` raises OSError. The
# stub is installed only long enough to import `core.multitrack_player` (which
# keeps a binding to it); it is then removed so the rest of the session sees the
# normal world. `OutputStream` stays None because nothing here opens a device.
_sd_stub = types.ModuleType("sounddevice")
_sd_stub.OutputStream = None
_had_sd = "sounddevice" in sys.modules
if not _had_sd:
    sys.modules["sounddevice"] = _sd_stub

from audioman.cli import fader_test  # noqa: E402
from audioman.cli.fader_test import run  # noqa: E402
from audioman.core import multitrack_player  # noqa: E402  (must precede stub removal)


def _qt_native_libs_loadable() -> bool:
    """Whether PyQt6's native Qt libraries can actually be dlopened.

    PyQt6 ships its own Qt build but still dlopens system libEGL/libGL. On a bare
    container image (including the GitHub runner's default) those are absent, so
    `import PyQt6.QtWidgets` raises `libEGL.so.1: cannot open shared object file`.
    Tests that need Qt are skipped there: the library is missing, not the behaviour
    under test. The CI coverage job installs libegl1/libgl1, so the coverage gate
    still measures these paths.
    """
    try:
        from PyQt6.QtWidgets import QApplication  # noqa: F401
    except ImportError:
        return False
    return True


QT_NATIVE_OK = _qt_native_libs_loadable()
needs_qt_native = pytest.mark.skipif(
    not QT_NATIVE_OK,
    reason="PyQt6 native libs unavailable (libEGL/libGL) — install libegl1 libgl1",
)

if not _had_sd:
    del sys.modules["sounddevice"]


@pytest.fixture(autouse=True)
def offscreen_qt(monkeypatch):
    """Never let an accidental QApplication() reach a real display."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def stems(tmp_path):
    """Two one-second stems, so the player has a deterministic duration."""
    directory = tmp_path / "stems"
    directory.mkdir()
    for name, freq in (("kick", 110.0), ("bass", 220.0)):
        n = 8000
        t = np.arange(n, dtype=np.float32) / 8000
        mono = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(str(directory / f"{name}.wav"), np.stack([mono, mono], axis=1),
                 8000, subtype="PCM_16")
    return directory


class TestArgumentSurface:
    def test_block_size_defaults_to_1024(self):
        from audioman.cli.app import build_parser

        args = build_parser().parse_args(["fader-test", "stems"])
        assert args.block_size == 1024
        assert args.input == "stems"
        assert args.load is None
        assert args.func is run

    def test_load_and_block_size_are_accepted(self):
        from audioman.cli.app import build_parser

        args = build_parser().parse_args(
            ["fader-test", "stems", "--load", "gains.json", "--block-size", "256"],
        )
        assert args.load == "gains.json"
        assert args.block_size == 256

    def test_block_size_must_be_an_integer(self, run_cli, tmp_path):
        result = run_cli(["fader-test", str(tmp_path), "--block-size", "big"])
        assert result.code == 2
        assert "invalid int value" in result.stderr


class TestNonGuiPaths:
    def test_missing_directory_is_reported(self, run_cli, tmp_path):
        result = run_cli(["fader-test", str(tmp_path / "nope")])
        assert result.code == 1
        assert "Not a directory" in result.stderr

    @needs_qt_native
    def test_missing_portaudio_is_reported_not_raised(self, run_cli, tmp_path, monkeypatch):
        """The audio backend needs the PortAudio system library.

        `import sounddevice` raises OSError when it is absent, which is an OS-level
        gap rather than a Python dependency problem. It used to surface as an
        uncaught traceback; it must now name the library and the package to install.
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sounddevice" or name.startswith("sounddevice."):
                raise OSError("PortAudio library not found")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "audioman.core.multitrack_player", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        result = run_cli(["fader-test", str(tmp_path)])

        assert result.code == 1
        assert "PortAudio" in result.stderr
        assert "libportaudio2" in result.stderr
        assert "Traceback" not in result.stderr

    @needs_qt_native
    def test_missing_portaudio_returns_before_launching_a_window(
        self, run_cli, tmp_path, monkeypatch, silent_error,
    ):
        """Neutralise print_error's exit, so the explicit `return` is what stops it.

        Without the guard the command would carry on to build a QApplication and a
        player for a backend it just reported as unavailable.
        """
        import builtins

        seen = silent_error("audioman.cli.fader_test")
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sounddevice" or name.startswith("sounddevice."):
                raise OSError("PortAudio library not found")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "audioman.core.multitrack_player", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        run_cli(["fader-test", str(tmp_path)])

        assert seen and "PortAudio" in seen[0]
        # Falling through would print this after the player is constructed.
        assert "loading stems" not in "".join(seen)

    def test_missing_directory_returns_before_importing_qt(self, run_cli, tmp_path, silent_error):
        """The guard must return explicitly, not lean on print_error's exit.

        Without the `return` the command would proceed to build a QApplication
        (and a player) for a path it just rejected.
        """
        seen = silent_error("audioman.cli.fader_test")
        result = run_cli(["fader-test", str(tmp_path / "nope")])

        assert result.code == 0, result.stderr
        assert seen and "Not a directory" in seen[0]
        assert "loading stems" not in result.stdout

    def test_a_file_instead_of_a_directory_is_rejected(self, run_cli, tmp_path):
        a_file = tmp_path / "song.wav"
        a_file.write_bytes(b"RIFF")
        result = run_cli(["fader-test", str(a_file)])

        assert result.code == 1
        assert "Not a directory" in result.stderr

    def test_pyqt6_missing_is_reported(self, run_cli, stems, monkeypatch):
        """A machine without PyQt6 must get an actionable message, not a traceback."""
        monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", None)

        result = run_cli(["fader-test", str(stems)])

        assert result.code == 1
        assert result.stderr.splitlines()[0] == "error: PyQt6 not installed — run: uv add PyQt6"
        assert "loading stems" not in result.stdout

    def test_pyqt6_native_library_failure_names_the_real_cause(
        self, run_cli, stems, monkeypatch,
    ):
        """A dlopen failure must not be reported as a missing package.

        PyQt6 is present but its native Qt libraries are not, which is an OS-level
        gap (`libEGL.so.1: cannot open shared object file`). Saying "PyQt6 not
        installed" sends the reader to the wrong fix, so the message must name the
        library and the package to install.
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "PyQt6.QtWidgets":
                raise ImportError(
                    "libEGL.so.1: cannot open shared object file: "
                    "No such file or directory"
                )
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        result = run_cli(["fader-test", str(stems)])

        assert result.code == 1
        assert "libEGL" in result.stderr
        assert "libegl1" in result.stderr
        assert "loading stems" not in result.stdout

    def test_pyqt6_missing_returns_before_reading_the_stems(
        self, run_cli, stems, monkeypatch, silent_error,
    ):
        """The import failure must stop the run, not fall through into the GUI."""
        seen = silent_error("audioman.cli.fader_test")
        monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", None)

        result = run_cli(["fader-test", str(stems)])

        assert result.code == 0, result.stderr
        assert seen and "PyQt6 not installed" in seen[0]
        assert "loaded" not in result.stderr

    @needs_qt_native
    def test_empty_stem_directory_load_failure_is_reported(self, run_cli, tmp_path):
        """A directory with no wav files surfaces as a load error, not a crash."""
        empty = tmp_path / "empty"
        empty.mkdir()

        result = run_cli(["fader-test", str(empty)])

        assert result.code == 1
        assert "Load failed" in result.stderr
        assert "No wav files" in result.stderr
        assert "loaded" not in result.stderr

    @needs_qt_native
    def test_load_failure_returns_before_launching_a_window(
        self, run_cli, tmp_path, monkeypatch, silent_error,
    ):
        """Without the `return`, a failed load would still open the mixer window."""
        from audioman.cli import _fader_test_ui

        seen = silent_error("audioman.cli.fader_test")
        empty = tmp_path / "empty"
        empty.mkdir()

        launched = []
        monkeypatch.setattr(
            _fader_test_ui, "FaderTestWindow",
            lambda *a, **k: launched.append(a) or pytest.fail("window must not be built"),
        )

        result = run_cli(["fader-test", str(empty)])

        assert result.code == 0, result.stderr
        assert seen and "Load failed" in seen[0]
        assert launched == []

    @needs_qt_native
    def test_unreadable_stem_surfaces_as_a_load_error(self, run_cli, tmp_path):
        directory = tmp_path / "broken"
        directory.mkdir()
        (directory / "broken.wav").write_bytes(b"not audio")

        result = run_cli(["fader-test", str(directory)])

        assert result.code == 1
        assert "Load failed" in result.stderr


@needs_qt_native
class TestGuiLaunch:
    """The GUI branch: Qt runs offscreen and `exec` returns immediately."""

    @pytest.fixture
    def no_event_loop(self, monkeypatch):
        """Make `QApplication.exec` return 0 instead of spinning forever."""
        from PyQt6.QtWidgets import QApplication

        monkeypatch.setattr(QApplication, "exec", lambda self: 0)
        return QApplication

    def test_launch_loads_stems_and_reports_them(self, run_cli, stems, no_event_loop):
        result = run_cli(["fader-test", str(stems)])

        assert result.code == 0, result.stderr
        assert f"loading stems from {stems.resolve()} ..." in result.stdout
        assert "loaded 2 tracks" in result.stderr
        assert "(1.0s @ 8000Hz)" in result.stderr

    def test_launch_shows_a_window_bound_to_the_stems(self, run_cli, stems, no_event_loop, monkeypatch):
        from audioman.cli import _fader_test_ui

        captured = {}
        original = _fader_test_ui.FaderTestWindow

        class _Spy(original):
            def __init__(self, player, source_dir=None):
                captured["tracks"] = len(player.tracks)
                captured["source_dir"] = source_dir
                super().__init__(player, source_dir=source_dir)

            def show(self):
                captured["shown"] = True
                super().show()

        monkeypatch.setattr(_fader_test_ui, "FaderTestWindow", _Spy)

        result = run_cli(["fader-test", str(stems)])

        assert result.code == 0, result.stderr
        assert captured["tracks"] == 2
        assert captured["source_dir"] == stems.resolve()
        assert captured.get("shown") is True

    def test_block_size_flag_reaches_the_player(self, run_cli, stems, no_event_loop, monkeypatch):
        from audioman.core import multitrack_player

        seen = {}
        original = multitrack_player.MultitrackPlayer.from_directory

        def _spy(cls, directory, block_size=1024, target_sr=None):
            seen["block_size"] = block_size
            return original(directory, block_size=block_size, target_sr=target_sr)

        monkeypatch.setattr(multitrack_player.MultitrackPlayer, "from_directory",
                            classmethod(_spy))

        result = run_cli(["fader-test", str(stems), "--block-size", "256"])

        assert result.code == 0, result.stderr
        assert seen["block_size"] == 256

    def test_load_applies_gains_from_the_ground_truth_file(self, run_cli, stems, no_event_loop, tmp_path):
        gains_file = tmp_path / "gains.json"
        gains_file.write_text(json.dumps({"gains": {"kick": -6.0, "bass": 3.0}}), encoding="utf-8")

        result = run_cli(["fader-test", str(stems), "--load", str(gains_file)])

        assert result.code == 0, result.stderr
        assert f"loaded gains for 2 tracks from {gains_file}" in result.stderr

    def test_load_accepts_the_bare_mapping_form(self, run_cli, stems, no_event_loop, tmp_path):
        """Older exports stored the gains mapping at the top level."""
        gains_file = tmp_path / "bare.json"
        gains_file.write_text(json.dumps({"kick": -6.0}), encoding="utf-8")

        result = run_cli(["fader-test", str(stems), "--load", str(gains_file)])

        assert result.code == 0, result.stderr
        assert "loaded gains for 1 tracks" in result.stderr

    def test_load_reports_zero_matches_for_unknown_track_names(
        self, run_cli, stems, no_event_loop, tmp_path,
    ):
        gains_file = tmp_path / "other.json"
        gains_file.write_text(json.dumps({"gains": {"snare": -6.0}}), encoding="utf-8")

        result = run_cli(["fader-test", str(stems), "--load", str(gains_file)])

        assert result.code == 0, result.stderr
        assert "loaded gains for 0 tracks" in result.stderr

    def test_load_of_a_missing_file_raises_uncaught(self, run_cli, stems, no_event_loop, tmp_path):
        """KNOWN GAP: `--load` is not guarded, so a missing file aborts with OSError.

        Documented rather than asserted-away: the GUI has already loaded the
        stems by this point, and `print_error` is never reached.
        """
        missing = tmp_path / "absent.json"

        with pytest.raises(FileNotFoundError):
            run_cli(["fader-test", str(stems), "--load", str(missing)])

    def test_exec_exit_code_is_propagated(self, run_cli, stems, monkeypatch):
        from PyQt6.QtWidgets import QApplication

        monkeypatch.setattr(QApplication, "exec", lambda self: 7)

        result = run_cli(["fader-test", str(stems)])

        assert result.code == 7
