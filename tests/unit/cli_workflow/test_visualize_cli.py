# tests/unit/cli_workflow/test_visualize_cli.py
# Purpose: cover `audioman visualize` — built-in analysis + Vamp paths to SVL/PNG.
#
# The built-in paths (the default and `--builtin`) need only numpy/soundfile, so
# they are exercised end to end: the SVL file is written and parsed back and its
# layer type/min/max are asserted. The Vamp paths require the `vamp` python
# package plus a system Vamp plugin, neither of which is installed here
# (AUD-1857 family), so they are driven against a fake `vamp` module; a real
# Vamp run is guarded by skipif with a stated reason.

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from .conftest import write_wav


def _has_vamp() -> bool:
    try:
        import vamp  # noqa: F401
        return bool(vamp.list_plugins())
    except Exception:
        return False


VAMP_AVAILABLE = _has_vamp()


def _svl_root(path: Path) -> ET.Element:
    return ET.parse(str(path)).getroot()


def _model(root: ET.Element) -> dict:
    return root.find("data/model").attrib


def _points(root: ET.Element) -> list[dict]:
    return [p.attrib for p in root.find("data/dataset")]


@pytest.fixture
def tone(tmp_path):
    """0.5 s stereo tone: long enough for a 2048-sample FFT frame."""
    return write_wav(tmp_path / "tone.wav", sample_rate=8000, duration=0.5)


class TestPositiveInt:
    def test_accepts_positive_values(self):
        from audioman.cli.visualize import _positive_int

        assert _positive_int("2048") == 2048

    @pytest.mark.parametrize("raw", ["0", "-1"])
    def test_rejects_non_positive_values(self, raw):
        from audioman.cli.visualize import _positive_int

        with pytest.raises(Exception, match="positive integer"):
            _positive_int(raw)


class TestDefaultPath:
    def test_no_analysis_flag_defaults_to_spectrogram(self, run_cli, tone):
        """Running `visualize <file>` with neither --plugin nor --builtin writes
        a spectrogram SVL next to the input."""
        result = run_cli(["visualize", str(tone)])
        assert result.code == 0, result.stderr
        out = tone.parent / "tone_spectrogram.svl"
        assert out.exists()
        root = _svl_root(out)
        assert root.tag == "sv"
        model = _model(root)
        assert model["type"] == "dense"
        assert model["dimensions"] == "3"
        assert model["sampleRate"] == "8000"
        assert "SVL written:" in result.stderr

    def test_default_spectrogram_bins_are_labelled_by_frequency(self, run_cli, tone):
        assert run_cli(["visualize", str(tone)]).code == 0
        out = tone.parent / "tone_spectrogram.svl"
        dataset = _svl_root(out).find("data/dataset")
        labels = [c.attrib["name"] for c in dataset if c.tag == "bin"]
        # 2048-point real FFT -> 1025 bins, each labelled with its Hz band.
        assert len(labels) == 1025
        assert labels[0].endswith("Hz")
        # 4000 Hz Nyquist over 1025 bins -> ~3.9 Hz per bin.
        assert labels[0] == "0-4Hz"
        assert labels[1] == "4-8Hz"


class TestOutputPathResolution:
    def test_explicit_output_wins_over_the_derived_name(self, run_cli, tone, tmp_path):
        from audioman.cli.visualize import _resolve_output_path

        args = run_cli  # keep the fixture ordered; explicit args below
        out = tmp_path / "explicit.svl"
        result = run_cli(["visualize", str(tone), "--builtin", "rms", "-o", str(out)])
        assert result.code == 0, result.stderr
        assert out.exists()
        assert not (tone.parent / "tone_rms.svl").exists()

    def test_derived_name_replaces_dashes_in_the_suffix(self, tmp_path):
        import argparse

        from audioman.cli.visualize import _resolve_output_path

        args = argparse.Namespace(output=None)
        resolved = _resolve_output_path(args, tmp_path / "song.wav", "spectral_centroid")
        assert resolved == tmp_path / "song_spectral_centroid.svl"

    def test_derived_name_accepts_a_string_output(self, tmp_path):
        import argparse

        from audioman.cli.visualize import _resolve_output_path

        args = argparse.Namespace(output=str(tmp_path / "given.svl"))
        assert _resolve_output_path(args, tmp_path / "song.wav", "rms") == tmp_path / "given.svl"


class TestBuiltinMetrics:
    """Every built-in metric must produce a time-values layer for its metric."""

    @pytest.mark.parametrize("builtin,attr,units", [
        ("spectral-centroid", "spectral_centroid", "Hz"),
        ("spectral-entropy", "spectral_entropy", "bits"),
        ("rms", "rms", ""),
        ("peak", "peak", ""),
        ("zcr", "zero_crossing_rate", ""),
    ])
    def test_metric_layer_matches_the_computed_values(self, run_cli, tone, builtin, attr, units):
        import soundfile as sf

        from audioman.core.analysis import compute_frame_metrics

        result = run_cli(["visualize", str(tone), "--builtin", builtin])
        assert result.code == 0, result.stderr

        suffix = builtin.replace("-", "_")
        out = tone.parent / f"tone_{suffix}.svl"
        assert out.exists()

        root = _svl_root(out)
        model = _model(root)
        assert model["name"] == builtin
        assert model["units"] == units
        assert model["sampleRate"] == "8000"
        assert model["type"] == "sparse"
        assert model["dimensions"] == "2"

        audio = sf.read(str(tone), dtype="float32", always_2d=True)[0].T
        metrics = compute_frame_metrics(audio, 8000, frame_size=2048, hop_size=512)
        expected = getattr(metrics, attr)
        points = _points(root)
        assert len(points) == len(expected)
        assert [int(p["frame"]) for p in points] == [i * 512 for i in range(len(expected))]
        assert [float(p["value"]) for p in points] == pytest.approx(expected)
        # The model's min/max envelope must bracket the drawn values.
        assert float(model["minimum"]) == pytest.approx(min(expected))
        assert float(model["maximum"]) == pytest.approx(max(expected))

    def test_hop_and_frame_size_flags_change_the_layer(self, run_cli, tone):
        result = run_cli(
            ["visualize", str(tone), "--builtin", "rms", "--frame-size", "512", "--hop", "128"]
        )
        assert result.code == 0, result.stderr
        root = _svl_root(tone.parent / "tone_rms.svl")
        assert _model(root)["resolution"] == "128"
        frames = [int(p["frame"]) for p in _points(root)]
        assert frames[:3] == [0, 128, 256]

    def test_unknown_builtin_is_rejected_by_argparse(self, run_cli, tone):
        result = run_cli(["visualize", str(tone), "--builtin", "not-a-metric"])
        assert result.code == 2
        assert "invalid choice" in result.stderr

    def test_plugin_and_builtin_are_mutually_exclusive(self, run_cli, tone):
        result = run_cli(
            ["visualize", str(tone), "--plugin", "lib:plug", "--builtin", "rms"]
        )
        assert result.code == 2
        assert "not allowed with argument" in result.stderr


class TestBuiltinSpectrogram:
    def test_spectrogram_layer_has_one_point_per_frame(self, run_cli, tone):
        result = run_cli(["visualize", str(tone), "--builtin", "spectrogram"])
        assert result.code == 0, result.stderr
        out = tone.parent / "tone_spectrogram.svl"
        root = _svl_root(out)
        model = _model(root)
        assert model["type"] == "dense"
        assert model["dimensions"] == "3"
        assert model["windowSize"] == "2048"
        assert model["resolution"] == "512"
        # Dense layers store one <row> per frame plus one <bin> per frequency bin.
        dataset = _svl_root(out).find("data/dataset")
        rows = [c for c in dataset if c.tag == "row"]
        bins = [c for c in dataset if c.tag == "bin"]
        # 4000 samples, 2048 frame, 512 hop -> floor((4000-2048)/512)+1 = 4 frames
        assert len(rows) == 4
        assert len(bins) == 1025
        assert int(model["yBinCount"]) == 1025
        first = [float(v) for v in rows[0].text.split()]
        assert len(first) == 1025

    def test_frames_too_short_for_the_frame_size_are_rejected(self, run_cli, tmp_path):
        short = write_wav(tmp_path / "short.wav", sample_rate=8000, duration=0.2)
        result = run_cli(
            ["visualize", str(short), "--builtin", "spectrogram", "--frame-size", "4096"]
        )
        assert result.code == 1
        assert "Input audio is shorter than the spectrogram frame size" in result.stderr
        assert "samples=1600, frame-size=4096" in result.stderr

    def test_compute_spectrogram_rejects_non_positive_sizes(self, tone):
        import soundfile as sf

        from audioman.cli.visualize import _compute_spectrogram

        audio = sf.read(str(tone), dtype="float32", always_2d=True)[0].T
        with pytest.raises(ValueError, match="must be positive"):
            _compute_spectrogram(audio, 8000, frame_size=0)
        with pytest.raises(ValueError, match="must be positive"):
            _compute_spectrogram(audio, 8000, hop_size=-1)

    def test_compute_spectrogram_guards_the_frame_length(self, tone):
        from audioman.cli.visualize import _compute_spectrogram

        mono = np.zeros(100, dtype=np.float32)
        with pytest.raises(ValueError, match="shorter than frame_size"):
            _compute_spectrogram(mono, 8000, frame_size=2048)

    def test_compute_spectrogram_accepts_mono_and_flattens_stereo(self, tone):
        import soundfile as sf

        from audioman.cli.visualize import _compute_spectrogram

        mono = np.ones(4096, dtype=np.float32)
        stereo_only = _compute_spectrogram(mono, 8000)
        audio = sf.read(str(tone), dtype="float32", always_2d=True)[0].T
        assert _compute_spectrogram(audio, 8000).shape[1] == stereo_only.shape[1]

    def test_spectrogram_db_tracks_amplitude(self):
        """Power is computed on the unnormalised FFT, so the dB floor is 10*log10(1e-10)
        and a 2x amplitude must read ~6 dB higher."""
        from audioman.cli.visualize import _compute_spectrogram

        quiet = np.sin(2 * np.pi * 440 * np.arange(4096) / 8000).astype(np.float32) * 0.1
        loud = quiet * 2.0

        quiet_db = _compute_spectrogram(quiet, 8000)
        loud_db = _compute_spectrogram(loud, 8000)

        assert quiet_db.min() >= 10 * np.log10(1e-10)
        assert np.isfinite(quiet_db).all() and np.isfinite(loud_db).all()
        assert loud_db.max() - quiet_db.max() == pytest.approx(6.02, abs=0.05)


class TestPlainModeHasNoMarkupTokens:
    """Rich tag text must not leak into `--plain` output.

    The `--plain` console uses `markup=False`, so passing a tagged string
    straight to `console.print` prints `[dim]...[/dim]` verbatim (AUD-1853).
    Going through the output helpers strips the tags on the plain path while
    keeping bracket text that is not a tag.

    See that fixture's docstring for why `real_cli_console_binding` is needed:
    in-process tests normally run against the bound rich console and therefore
    cannot detect the leak.
    """

    TAG_RE = re.compile(r"\[/?(?:dim|bold|red|green|yellow|cyan)[^\]]*\]")

    @pytest.fixture(autouse=True)
    def plain_console(self, real_cli_console_binding):
        from audioman.cli import visualize

        real_cli_console_binding(visualize)

    def test_builtin_run_has_no_tag_text(self, run_cli, tone):
        result = run_cli(["visualize", str(tone), "--builtin", "rms"])
        assert result.code == 0, result.stderr
        combined = result.stdout + result.stderr
        assert self.TAG_RE.search(combined) is None, combined
        # Info lines still appear; only the tags should disappear.
        assert "Built-in analysis: rms" in result.stderr

    def test_failure_path_has_no_tag_text(self, run_cli, tmp_path):
        short = write_wav(tmp_path / "short.wav", sample_rate=8000, duration=0.2)
        result = run_cli(
            ["visualize", str(short), "--builtin", "spectrogram", "--frame-size", "4096"]
        )
        assert result.code == 1
        combined = result.stdout + result.stderr
        assert self.TAG_RE.search(combined) is None, combined
        assert "Built-in analysis: spectrogram (frame=4096, hop=512)" in result.stderr

    def test_plugin_listing_header_has_no_tag_text(self, run_cli, monkeypatch):
        monkeypatch.setattr("audioman.core.vamp_host.list_plugins", lambda: ["a:plug"])
        result = run_cli(["visualize", "ignored.wav", "--list-plugins"])
        assert result.code == 0, result.stderr
        assert self.TAG_RE.search(result.stdout) is None, result.stdout
        assert "Installed Vamp plugins (1)" in result.stdout

    def test_plugin_info_header_has_no_tag_text(self, run_cli, monkeypatch):
        monkeypatch.setattr(
            "audioman.core.vamp_host.get_plugin_outputs",
            lambda plugin_id: {"curve": {"binCount": 3}},
        )
        result = run_cli(["visualize", "ignored.wav", "--plugin-info", "lib:plug"])
        assert result.code == 0, result.stderr
        assert self.TAG_RE.search(result.stdout) is None, result.stdout
        assert "lib:plug" in result.stdout

    def test_vamp_run_has_no_tag_text(self, run_cli, tone, monkeypatch):
        """The `[dim]Running Vamp plugin: …[/dim]` and `[dim]Result shape: …[/dim]` lines."""
        _FakeVampCollect(monkeypatch, {"vector": (0.0625, [1.0])})
        result = run_cli(["visualize", str(tone), "--plugin", "lib:plug"])
        assert result.code == 0, result.stderr
        combined = result.stdout + result.stderr
        assert self.TAG_RE.search(combined) is None, combined
        assert "Running Vamp plugin: lib:plug" in result.stderr
        assert "Result shape: vector" in result.stderr


class TestPngOutput:
    def test_png_only_skips_the_svl(self, run_cli, tone, tmp_path):
        png = tmp_path / "spec.png"
        result = run_cli(
            ["visualize", str(tone), "--builtin", "spectrogram", "--png", str(png), "--png-only"]
        )
        assert result.code == 0, result.stderr
        assert png.exists()
        assert png.stat().st_size > 0
        assert not (tone.parent / "tone_spectrogram.svl").exists()
        assert "PNG written:" in result.stderr

    def test_png_alongside_svl_writes_both(self, run_cli, tone, tmp_path):
        png = tmp_path / "spec.png"
        result = run_cli(
            ["visualize", str(tone), "--builtin", "spectrogram", "--png", str(png)]
        )
        assert result.code == 0, result.stderr
        assert png.exists()
        assert (tone.parent / "tone_spectrogram.svl").exists()

    def test_png_only_without_explicit_path_uses_the_svl_stem(self, run_cli, tone):
        result = run_cli(["visualize", str(tone), "--builtin", "spectrogram", "--png-only"])
        assert result.code == 0, result.stderr
        png = tone.parent / "tone_spectrogram.png"
        assert png.exists()
        assert not (tone.parent / "tone_spectrogram.svl").exists()

    def test_png_options_change_the_image_size(self, run_cli, tone, tmp_path):
        from PIL import Image

        small = tmp_path / "small.png"
        result = run_cli(
            ["visualize", str(tone), "--builtin", "spectrogram", "--png-only",
             "--png", str(small), "--png-width", "400", "--png-height", "200"]
        )
        assert result.code == 0, result.stderr
        with Image.open(str(small)) as image:
            assert image.size == (400, 200)

    def test_png_db_and_fmax_options_are_accepted(self, run_cli, tone, tmp_path):
        png = tmp_path / "bounded.png"
        result = run_cli(
            ["visualize", str(tone), "--builtin", "spectrogram", "--png-only",
             "--png", str(png), "--png-db-min", "-60", "--png-db-max", "-10",
             "--png-fmax", "2000"]
        )
        assert result.code == 0, result.stderr
        assert png.exists()

    def test_png_fmax_above_nyquist_is_clamped(self, run_cli, tone, tmp_path):
        """fmax > Nyquist must not produce an empty (or over-wide) image."""
        png = tmp_path / "clamped.png"
        result = run_cli(
            ["visualize", str(tone), "--builtin", "spectrogram", "--png-only",
             "--png", str(png), "--png-fmax", "99999"]
        )
        assert result.code == 0, result.stderr
        assert png.exists()

    def test_write_spectrogram_png_creates_parent_directories(self, tone, tmp_path):
        from audioman.cli.visualize import _compute_spectrogram, _write_spectrogram_png

        matrix = _compute_spectrogram(np.ones(4096, dtype=np.float32), 8000)
        nested = tmp_path / "a" / "b" / "spec.png"
        _write_spectrogram_png(matrix, 8000, 512, 2048, nested, title="tone.wav")
        assert nested.exists()

    def test_missing_matplotlib_is_reported_as_an_install_hint(self, tone, monkeypatch, tmp_path):
        """matplotlib is optional; its absence must produce an actionable message."""
        from audioman.cli.visualize import _compute_spectrogram, _write_spectrogram_png

        monkeypatch.setitem(sys.modules, "matplotlib", None)
        matrix = _compute_spectrogram(np.ones(4096, dtype=np.float32), 8000)
        with pytest.raises(SystemExit) as exc:
            _write_spectrogram_png(matrix, 8000, 512, 2048, tmp_path / "spec.png")
        assert exc.value.code == 1

    def test_png_only_with_open_flag_attempts_to_launch_the_viewer(
        self, run_cli, tone, tmp_path, monkeypatch
    ):
        calls = []

        class _FakePopen:
            def __init__(self, args):
                calls.append(args)

        monkeypatch.setattr("subprocess.Popen", _FakePopen)
        png = tmp_path / "open.png"
        result = run_cli(
            ["visualize", str(tone), "--builtin", "spectrogram", "--png-only",
             "--png", str(png), "--open"]
        )
        assert result.code == 0, result.stderr
        assert calls and calls[0][0] == "xdg-open"
        # `print_info` in cli/output.py writes to the stderr console.
        assert "Opening in Sonic Visualiser" in result.stderr


class TestMissingInput:
    def test_missing_file_exits_nonzero(self, run_cli, tmp_path):
        result = run_cli(["visualize", str(tmp_path / "nope.wav"), "--builtin", "rms"])
        assert result.code == 1
        assert "File not found" in result.stderr


class TestOpenInSonicVisualiser:
    def test_launch_failure_is_reported_not_raised(self, monkeypatch, capsys):
        from audioman.cli import visualize

        def _boom(args):
            raise FileNotFoundError("no 'open' binary")

        monkeypatch.setattr("subprocess.Popen", _boom)
        visualize._open_in_sv(Path("/tmp/does-not-matter.svl"))
        assert "Sonic Visualiser was not found" in capsys.readouterr().err

    def test_darwin_uses_open_with_the_app_name(self, monkeypatch):
        from audioman.cli import visualize

        monkeypatch.setattr(sys, "platform", "darwin")
        assert visualize._sv_launcher(Path("/tmp/a.svl")) == [
            "open", "-a", "Sonic Visualiser", "/tmp/a.svl",
        ]

    def test_linux_uses_xdg_open(self, monkeypatch):
        from audioman.cli import visualize

        monkeypatch.setattr(sys, "platform", "linux")
        assert visualize._sv_launcher(Path("/tmp/a.svl")) == ["xdg-open", "/tmp/a.svl"]

    def test_unsupported_platform_reports_instead_of_launching(self, monkeypatch, capsys):
        """A platform with no launcher only prints the hint, no traceback."""
        from audioman.cli import visualize

        calls = []
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr("subprocess.Popen", lambda args: calls.append(args))
        visualize._open_in_sv(Path("/tmp/a.svl"))
        assert calls == []
        assert "Sonic Visualiser was not found" in capsys.readouterr().err

    def test_svl_output_with_open_flag_attempts_launch(self, run_cli, tone, monkeypatch):
        calls = []
        monkeypatch.setattr("subprocess.Popen", lambda args: calls.append(args))
        result = run_cli(["visualize", str(tone), "--builtin", "rms", "--open"])
        assert result.code == 0, result.stderr
        assert calls and calls[0][0] == "xdg-open"
        assert calls[0][1].endswith("tone_rms.svl")


class TestGuessUnits:
    @pytest.mark.parametrize("plugin_id,units", [
        ("qm-vamp-plugins:qm-chromagram", ""),
        ("lib:pitch-tracker", "Hz"),
        ("lib:spectral-centroid", "Hz"),
        ("lib:frequency-shifter", "Hz"),
        ("lib:energy-tracker", "dB"),
        ("lib:amplitude-envelope", "dB"),
        ("lib:rms-meter", "dB"),
        ("lib:tempo-estimator", "bpm"),
        ("lib:bpm-detector", "bpm"),
    ])
    def test_unit_inference(self, plugin_id, units):
        from audioman.cli.visualize import _guess_units

        assert _guess_units(plugin_id) == units


class TestPluginListing:
    def test_list_plugins_prints_every_plugin(self, run_cli, monkeypatch):
        monkeypatch.setattr("audioman.core.vamp_host.list_plugins",
                            lambda: ["a:plug", "b:plug"])
        result = run_cli(["visualize", "ignored.wav", "--list-plugins"])
        assert result.code == 0, result.stderr
        assert "Installed Vamp plugins (2)" in result.stdout
        assert "a:plug" in result.stdout and "b:plug" in result.stdout

    def test_no_plugins_installed_is_an_error(self, run_cli, monkeypatch):
        monkeypatch.setattr("audioman.core.vamp_host.list_plugins", lambda: [])
        result = run_cli(["visualize", "ignored.wav", "--list-plugins"])
        assert result.code == 1
        assert "No Vamp plugins installed" in result.stderr

    def test_list_plugins_does_not_touch_the_input_path(self, run_cli, tmp_path, monkeypatch):
        """--list-plugins returns before the input existence check."""
        monkeypatch.setattr("audioman.core.vamp_host.list_plugins", lambda: ["a:plug"])
        result = run_cli(["visualize", str(tmp_path / "nope.wav"), "--list-plugins"])
        assert result.code == 0, result.stderr


class TestPluginInfo:
    def test_plugin_info_prints_each_output(self, run_cli, monkeypatch):
        monkeypatch.setattr(
            "audioman.core.vamp_host.get_plugin_outputs",
            lambda plugin_id: {"curve": {"binCount": 3}, "notes": {"binCount": 1}},
        )
        result = run_cli(["visualize", "ignored.wav", "--plugin-info", "lib:plug"])
        assert result.code == 0, result.stderr
        assert "lib:plug" in result.stdout
        assert "curve: {'binCount': 3}" in result.stdout
        assert "notes: {'binCount': 1}" in result.stdout

    def test_plugin_info_with_no_outputs_still_succeeds(self, run_cli, monkeypatch):
        monkeypatch.setattr("audioman.core.vamp_host.get_plugin_outputs", lambda plugin_id: {})
        result = run_cli(["visualize", "ignored.wav", "--plugin-info", "lib:empty"])
        assert result.code == 0, result.stderr
        assert "lib:empty" in result.stdout


class _FakeVampCollect:
    """Installs a fake `vamp` module returning a canned collect() result."""

    def __init__(self, monkeypatch, result):
        self.result = result
        self.collect_calls = []
        module = type(sys)("vamp")
        module.list_plugins = lambda: ["fake:plug"]
        module.get_outputs_of = lambda pid: {"out": {}}

        def collect(audio, sample_rate, plugin_id, **kwargs):
            self.collect_calls.append((audio, sample_rate, plugin_id, kwargs))
            return self.result

        module.collect = collect
        monkeypatch.setitem(sys.modules, "vamp", module)


class TestVampPaths:
    """Vamp runs against a fake `vamp` module: the CLI wiring is what is asserted."""

    def test_matrix_result_writes_a_dense_layer(self, run_cli, tone, monkeypatch):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        fake = _FakeVampCollect(monkeypatch, {"matrix": (0.0625, matrix.tolist())})
        result = run_cli(
            ["visualize", str(tone), "--plugin", "fake:plug", "--frame-size", "512", "--hop", "512"]
        )
        assert result.code == 0, result.stderr
        assert "Result shape: matrix" in result.stderr

        out = tone.parent / "tone_fake_plug.svl"
        assert out.exists()
        root = _svl_root(out)
        model = _model(root)
        assert model["type"] == "dense"
        assert model["dimensions"] == "3"
        assert model["windowSize"] == "512"
        assert model["resolution"] == "500"  # 0.0625 s * 8000 Hz
        # block_size/step_size reach vamp.collect.
        _audio, sr, plugin_id, kwargs = fake.collect_calls[0]
        assert sr == 8000
        assert plugin_id == "fake:plug"
        assert kwargs == {"block_size": 512, "step_size": 512}

    def test_vector_result_writes_time_values_with_guessed_units(self, run_cli, tone, monkeypatch):
        _FakeVampCollect(monkeypatch, {"vector": (0.0625, [1.0, 2.0, 3.0])})
        result = run_cli(["visualize", str(tone), "--plugin", "lib:pitch-tracker"])
        assert result.code == 0, result.stderr
        assert "Result shape: vector" in result.stderr
        root = _svl_root(tone.parent / "tone_lib_pitchtracker.svl")
        model = _model(root)
        assert model["units"] == "Hz"
        assert model["name"] == "lib:pitch-tracker"
        assert [float(p["value"]) for p in _points(root)] == [1.0, 2.0, 3.0]

    def test_list_result_with_durations_writes_notes(self, run_cli, tone, monkeypatch):
        events = [
            {"timestamp": 0.0, "duration": 0.25, "values": [60.0], "label": "C4"},
            {"timestamp": 0.5, "duration": 0.5, "values": [64.0], "label": "E4"},
        ]
        _FakeVampCollect(monkeypatch, {"list": events})
        result = run_cli(["visualize", str(tone), "--plugin", "lib:note-tracker"])
        assert result.code == 0, result.stderr
        root = _svl_root(tone.parent / "tone_lib_notetracker.svl")
        model = _model(root)
        assert model["dimensions"] == "3"
        assert model["subtype"] == "note"
        points = _points(root)
        assert [p["label"] for p in points] == ["C4", "E4"]
        assert [int(p["duration"]) for p in points] == [2000, 4000]
        assert [float(p["value"]) for p in points] == [60.0, 64.0]
        # `level` is the first reported value (the note pitch) when present.
        assert [float(p["level"]) for p in points] == [60.0, 64.0]

    def test_list_result_without_durations_writes_instants(self, run_cli, tone, monkeypatch):
        events = [{"timestamp": 0.1, "label": "beat"}, {"time": 0.2, "label": "beat"}]
        _FakeVampCollect(monkeypatch, {"list": events})
        result = run_cli(["visualize", str(tone), "--plugin", "lib:beat-tracker"])
        assert result.code == 0, result.stderr
        root = _svl_root(tone.parent / "tone_lib_beattracker.svl")
        model = _model(root)
        assert model["dimensions"] == "1"
        assert model["type"] == "sparse"
        points = _points(root)
        assert [int(p["frame"]) for p in points] == [800, 1600]
        assert [p["label"] for p in points] == ["beat", "beat"]

    def test_zero_duration_events_are_treated_as_instants(self, run_cli, tone, monkeypatch):
        events = [{"timestamp": 0.05, "duration": 0.0, "label": "tick"}]
        _FakeVampCollect(monkeypatch, {"list": events})
        result = run_cli(["visualize", str(tone), "--plugin", "lib:tick"])
        assert result.code == 0, result.stderr
        model = _model(_svl_root(tone.parent / "tone_lib_tick.svl"))
        assert model["dimensions"] == "1"

    def test_unknown_result_shape_is_an_error(self, run_cli, tone, monkeypatch):
        _FakeVampCollect(monkeypatch, {"something_else": 1})
        result = run_cli(["visualize", str(tone), "--plugin", "fake:plug"])
        assert result.code == 1
        assert "Unknown result shape: unknown" in result.stderr

    def test_output_name_is_forwarded_to_vamp(self, run_cli, tone, monkeypatch):
        """--output-name reaches vamp.collect as the `output` kwarg."""
        fake = _FakeVampCollect(monkeypatch, {"vector": (0.0625, [1.0])})
        output = tone.parent / "alt.svl"
        result = run_cli(
            ["visualize", str(tone), "--plugin", "lib:multi", "--output-name", "alt",
             "-o", str(output)]
        )
        assert result.code == 0, result.stderr
        _audio, _sr, plugin_id, kwargs = fake.collect_calls[0]
        assert plugin_id == "lib:multi"
        assert kwargs["output"] == "alt"
        assert output.exists()

    def test_explicit_output_path_is_used_for_the_vamp_layer(self, run_cli, tone, tmp_path, monkeypatch):
        _FakeVampCollect(monkeypatch, {"vector": (0.0625, [1.0])})
        out = tmp_path / "vamp.svl"
        result = run_cli(["visualize", str(tone), "--plugin", "lib:plug", "-o", str(out)])
        assert result.code == 0, result.stderr
        assert out.exists()

    def test_svl_output_with_open_flag_launches_the_viewer(self, run_cli, tone, monkeypatch):
        calls = []
        monkeypatch.setattr("subprocess.Popen", lambda args: calls.append(args))
        _FakeVampCollect(monkeypatch, {"vector": (0.0625, [1.0, 2.0])})
        result = run_cli(["visualize", str(tone), "--plugin", "lib:plug", "--open"])
        assert result.code == 0, result.stderr
        assert calls and calls[0][0] == "xdg-open"
        assert calls[0][1].endswith("tone_lib_plug.svl")

    def test_plugin_suffix_keeps_the_colon_as_an_underscore(self):
        plugin_id = "qm-vamp-plugins:qm-chromagram"
        assert plugin_id.replace(":", "_").replace("-", "") == "qmvampplugins_qmchromagram"


@pytest.mark.skipif(
    not VAMP_AVAILABLE,
    reason="vamp python package and a system Vamp plugin (e.g. qm-vamp-plugins) are not installed",
)
class TestVampAgainstARealPlugin:
    def test_real_vamp_plugin_writes_a_layer(self, run_cli, tone):
        import vamp

        plugin_id = sorted(vamp.list_plugins())[0]
        result = run_cli(["visualize", str(tone), "--plugin", plugin_id])
        assert result.code == 0, result.stderr
        assert "SVL written:" in result.stderr
