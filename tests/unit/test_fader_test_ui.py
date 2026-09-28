# tests/unit/test_fader_test_ui.py — headless tests for the PyQt6 fader-test UI.
#
# Technique: offscreen Qt (QT_QPA_PLATFORM=offscreen is set before the module is
# imported) plus a fake `sounddevice` module, so the window can be built and
# driven without a display or an audio device. Real mouse input is not
# simulated; instead the widgets are driven through the Qt signals/slots the UI
# itself connects, which is what the model and the JSON export depend on.
#
# Everything below targets model state and the exported gains JSON — the file
# `cli/fader_test.py` advertises as the machine-readable artifact.

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# The offscreen platform has to be selected before QtWidgets is imported, so it
# is set here and in the session-scoped fixture below.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# PyQt6 links libEGL/libGL, which bare container images (including the GitHub
# runner) do not ship. Without this guard the whole module fails to *collect*,
# which reads as a broken gate rather than a missing system library. Skipping is
# correct here: the library, not the code, is unavailable.
try:  # pragma: no cover - exercised only where libEGL is absent (CI image)
    from PyQt6.QtCore import Qt  # noqa: E402
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402
except ImportError as exc:  # pragma: no cover - see above
    pytest.skip(
        f"PyQt6 cannot load its native Qt libraries ({exc}); "
        f"install libegl1 (Debian/Ubuntu) to run the GUI tests.",
        allow_module_level=True,
    )

# --- import-time stub for the audio backend ---------------------------------
_sd_stub = types.ModuleType("sounddevice")
_sd_stub.OutputStream = None
_had_stub = "sounddevice" in sys.modules
if not _had_stub:
    sys.modules["sounddevice"] = _sd_stub

from audioman.core import multitrack_player as mtp  # noqa: E402
from audioman.core.multitrack_player import MultitrackPlayer, TrackState  # noqa: E402
from audioman.cli import _fader_test_ui as ui  # noqa: E402
from audioman.cli._fader_test_ui import (  # noqa: E402
    SLIDER_MAX_DB,
    SLIDER_MIN_DB,
    SLIDER_RESOLUTION,
    FaderTestWindow,
    TrackStrip,
    _fmt_time,
    db_to_slider,
    slider_to_db,
)

if not _had_stub:
    del sys.modules["sounddevice"]

SR = 100


class FakeOutputStream:
    """Records transport calls so no PortAudio device is touched."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = 0
        self.closed = False

    def start(self):
        self.started = True
        self.stopped = 0

    def stop(self):
        self.stopped += 1
        self.started = False

    def close(self):
        self.closed = True


@pytest.fixture(scope="session")
def qt_app():
    # Qt reads the platform plugin when the QApplication is created, so the
    # selection is re-asserted here for the first test that needs it.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    assert app.platformName() == "offscreen", "tests must not need a display"
    return app


@pytest.fixture(autouse=True)
def fake_audio_backend(monkeypatch):
    monkeypatch.setattr(
        mtp, "sd", types.SimpleNamespace(OutputStream=FakeOutputStream), raising=False
    )


@pytest.fixture
def track_data():
    return np.tile(np.linspace(-1.0, 1.0, 50, dtype=np.float32), (2, 10))


@pytest.fixture
def player(track_data):
    return MultitrackPlayer(
        tracks=[
            TrackState(name="drums", path="/stems/drums.wav", audio=track_data, sample_rate=SR),
            TrackState(name="bass", path="/stems/bass.wav", audio=track_data * 0.5, sample_rate=SR),
        ],
        sample_rate=SR,
    )


@pytest.fixture
def window(qt_app, player, tmp_path):
    win = FaderTestWindow(player, source_dir=tmp_path / "stems")
    yield win
    win.close()


def _patch_dialog(monkeypatch, name: str, result):
    """Patch a QFileDialog static method for the duration of one test."""
    calls = []

    def _fake(*args, **kwargs):
        calls.append(args)
        return result

    monkeypatch.setattr(QFileDialog, name, staticmethod(_fake))
    return calls


class TestSliderScale:
    def test_endpoints_map_to_the_configured_db_range(self):
        assert db_to_slider(SLIDER_MIN_DB) == 0
        assert db_to_slider(SLIDER_MAX_DB) == int(round((SLIDER_MAX_DB - SLIDER_MIN_DB) * SLIDER_RESOLUTION))
        assert db_to_slider(0.0) == int(round(-SLIDER_MIN_DB * SLIDER_RESOLUTION))

    def test_slider_to_db_inverts_db_to_slider(self):
        for db in (-60.0, -24.0, -6.0, 0.0, 3.5, 12.0):
            assert slider_to_db(db_to_slider(db)) == pytest.approx(db, abs=1e-9)

    def test_resolution_is_tenth_of_a_db(self):
        assert SLIDER_RESOLUTION == 10
        assert slider_to_db(0) == pytest.approx(-60.0)
        assert slider_to_db(1) - slider_to_db(0) == pytest.approx(0.1)

    def test_out_of_range_db_values_clamp_on_the_slider(self, window):
        strip = window.strips[0]
        strip.set_gain_external(-500.0)
        assert strip.fader.value() == db_to_slider(SLIDER_MIN_DB)
        strip.set_gain_external(500.0)
        assert strip.fader.value() == db_to_slider(SLIDER_MAX_DB)


class TestFormatTime:
    def test_formats_minutes_and_seconds(self):
        assert _fmt_time(0.0) == "00:00.00"
        assert _fmt_time(75.5) == "01:15.50"
        assert _fmt_time(1234.5) == "20:34.50"

    def test_seconds_roll_over_within_the_minute_field(self):
        # the seconds field is formatted to two decimals, so 59.999 rounds up
        # to 60.00 instead of carrying into the minute field
        assert _fmt_time(59.999) == "00:60.00"

    def test_sub_second_precision_is_two_decimals(self):
        assert _fmt_time(61.239) == "01:01.24"

    def test_large_values_keep_counting_minutes(self):
        assert _fmt_time(3600.0) == "60:00.00"


class TestWindowConstruction:
    def test_window_title_reports_track_count_and_duration(self, window, player):
        assert len(player.tracks) == 2
        assert window.windowTitle() == (
            f"audioman fader-test — 2 tracks ({player.duration_sec:.0f}s)"
        )

    def test_one_strip_per_track_in_order(self, window):
        assert len(window.strips) == 2
        assert [s.track.name for s in window.strips] == ["drums", "bass"]
        assert [s.track_index for s in window.strips] == [0, 1]

    def test_strips_bound_to_the_same_player(self, window, player):
        assert all(s.player is player for s in window.strips)

    def test_position_slider_range_matches_duration(self, window, player):
        assert window.position_slider.minimum() == 0
        assert window.position_slider.maximum() == int(player.duration_sec * 10)

    def test_initial_widget_state(self, window):
        assert window.play_btn.text() == "▶ Play"
        assert window.stop_btn.text() == "■ Stop"
        assert window.time_label.text() == "00:00 / 00:00"
        assert window.master_peak_label.text() == "0.0 dB"
        assert window.status_label.text() == "ready"
        assert window.save_btn.text() == "💾 Save Gains JSON"
        assert window.load_btn.text() == "📂 Load Gains JSON"
        assert window.reset_btn.text() == "⟳ Reset All to 0 dB"
        assert window.unmute_btn.text() == "🔊 Clear Mute/Solo"

    def test_dark_palette_is_applied(self, window):
        base = window.palette().color(window.palette().ColorRole.Base)
        assert base.getRgb()[:3] == (25, 25, 25)

    def test_timer_runs_at_twenty_hertz(self, window):
        assert window.timer.isActive() is True
        assert window.timer.interval() == 50

    def test_resize_accounts_for_track_count(self, qt_app, player):
        win = FaderTestWindow(player, source_dir=None)
        try:
            # 2 tracks: 80 * 2 + 100 = 260, well under the 1500 px cap
            assert win.width() == 80 * len(player.tracks) + 100
            assert win.height() == 380
        finally:
            win.close()

    def test_many_tracks_hit_the_width_cap(self, qt_app, track_data):
        tracks = [
            TrackState(name=f"t{i}", path=f"/s{i}.wav", audio=track_data, sample_rate=SR)
            for i in range(30)
        ]
        win = FaderTestWindow(MultitrackPlayer(tracks=tracks, sample_rate=SR), source_dir=None)
        try:
            assert win.width() == 1500  # capped, not 80 * 30 + 100
            assert len(win.strips) == 30
        finally:
            win.close()

    def test_source_dir_is_stored(self, window, tmp_path):
        assert window.source_dir == tmp_path / "stems"


class TestTrackStrip:
    def test_long_names_are_elided(self, qt_app, track_data):
        player = MultitrackPlayer(
            tracks=[
                TrackState(
                    name="a_very_long_track_name",
                    path="/x.wav",
                    audio=track_data,
                    sample_rate=SR,
                )
            ],
            sample_rate=SR,
        )
        strip = TrackStrip(player, 0)
        try:
            labels = [w.text() for w in strip.findChildren(ui.QLabel)]
            # names longer than 14 chars are cut to 13 plus an ellipsis
            assert "a_very_long_t…" in labels
            assert "a_very_long_track_name" not in labels
        finally:
            strip.deleteLater()

    def test_short_names_are_not_elided(self, qt_app, track_data):
        player = MultitrackPlayer(
            tracks=[TrackState(name="drums", path="/x.wav", audio=track_data, sample_rate=SR)],
            sample_rate=SR,
        )
        strip = TrackStrip(player, 0)
        try:
            labels = [w.text() for w in strip.findChildren(ui.QLabel)]
            assert "drums" in labels
        finally:
            strip.deleteLater()

    def test_fader_starts_at_unity(self, window):
        strip = window.strips[0]
        assert strip.fader.value() == db_to_slider(0.0)
        assert strip.db_label.text() == "0.0 dB"
        assert strip.mute_btn.isChecked() is False
        assert strip.solo_btn.isChecked() is False

    def test_fader_bounds_match_the_slider_scale(self, window):
        strip = window.strips[0]
        assert strip.fader.minimum() == db_to_slider(SLIDER_MIN_DB)
        assert strip.fader.maximum() == db_to_slider(SLIDER_MAX_DB)
        assert strip.fader.orientation() == Qt.Orientation.Vertical

    def test_fader_move_updates_player_gain_and_readout(self, window, player):
        strip = window.strips[0]

        strip.fader.setValue(db_to_slider(-6.0))

        assert player.tracks[0].gain_db == pytest.approx(-6.0)
        assert strip.db_label.text() == "-6.0 dB"

    def test_positive_gain_readout_carries_a_plus_sign(self, window, player):
        strip = window.strips[0]

        strip.fader.setValue(db_to_slider(3.5))

        assert player.tracks[0].gain_db == pytest.approx(3.5)
        assert strip.db_label.text() == "+3.5 dB"

    def test_context_menu_reset_returns_the_fader_to_unity(self, window, player):
        strip = window.strips[0]
        strip.fader.setValue(db_to_slider(-12.0))
        assert player.tracks[0].gain_db == pytest.approx(-12.0)

        # the right-click handler is a lambda wired to customContextMenuRequested
        strip.fader.customContextMenuRequested.emit(strip.fader.rect().center())

        assert strip.fader.value() == db_to_slider(0.0)
        assert player.tracks[0].gain_db == pytest.approx(0.0)
        assert strip.db_label.text() == "+0.0 dB"

    def test_mute_button_updates_the_player(self, window, player):
        strip = window.strips[1]

        strip.mute_btn.setChecked(True)
        assert player.tracks[1].muted is True

        strip.mute_btn.setChecked(False)
        assert player.tracks[1].muted is False

    def test_solo_button_updates_the_player(self, window, player):
        strip = window.strips[1]

        strip.solo_btn.setChecked(True)
        assert player.tracks[1].soloed is True

        strip.solo_btn.setChecked(False)
        assert player.tracks[1].soloed is False

    def test_muting_clears_solo_on_the_same_strip(self, window, player):
        strip = window.strips[0]
        strip.solo_btn.setChecked(True)

        strip.mute_btn.setChecked(True)

        assert player.tracks[0].muted is True
        assert player.tracks[0].soloed is False
        assert strip.solo_btn.isChecked() is False

    def test_soloing_clears_mute_on_the_same_strip(self, window, player):
        strip = window.strips[0]
        strip.mute_btn.setChecked(True)

        strip.solo_btn.setChecked(True)

        assert player.tracks[0].soloed is True
        assert player.tracks[0].muted is False
        assert strip.mute_btn.isChecked() is False

    def test_reset_clears_gain_mute_and_solo(self, window, player, track_data):
        strip = window.strips[0]
        strip.fader.setValue(db_to_slider(-20.0))
        strip.mute_btn.setChecked(True)
        window.strips[1].solo_btn.setChecked(True)

        strip.reset()

        assert strip.fader.value() == db_to_slider(0.0)
        assert strip.mute_btn.isChecked() is False
        assert strip.solo_btn.isChecked() is False
        assert player.tracks[0].gain_db == pytest.approx(0.0)
        assert player.tracks[0].muted is False
        # resetting one strip does not touch the other
        assert window.strips[1].solo_btn.isChecked() is True

    def test_set_gain_external_propagates_through_the_signal(self, window, player):
        strip = window.strips[0]

        strip.set_gain_external(-4.5)

        assert strip.fader.value() == db_to_slider(-4.5)
        assert player.tracks[0].gain_db == pytest.approx(-4.5)
        assert strip.db_label.text() == "-4.5 dB"


class TestStripMeter:
    def test_silence_shows_an_empty_meter(self, window, player):
        strip = window.strips[0]
        player.tracks[0].rms = 0.0

        strip.update_meter()

        assert strip.meter.value() == 0

    def test_sub_threshold_rms_shows_an_empty_meter(self, window, player):
        strip = window.strips[0]
        player.tracks[0].rms = 1e-7  # below the 1e-6 gate

        strip.update_meter()

        assert strip.meter.value() == 0

    def test_full_scale_rms_fills_the_meter(self, window, player):
        strip = window.strips[0]
        player.tracks[0].rms = 1.0
        player.tracks[0].peak = 0.5

        strip.update_meter()

        assert strip.meter.value() == 100

    def test_half_scale_rms_is_mapped_through_the_db_scale(self, window, player):
        strip = window.strips[0]
        player.tracks[0].rms = 0.5  # -6.02 dBFS
        player.tracks[0].peak = 0.5

        strip.update_meter()

        # (-6.02 + 60) / 60 * 100 = 89.96 → 89
        assert strip.meter.value() == 89

    def test_meter_never_leaves_the_0_100_range(self, window, player):
        strip = window.strips[0]
        player.tracks[0].rms = 4.0  # hotter than full scale
        player.tracks[0].peak = 0.5

        strip.update_meter()

        assert strip.meter.value() == 100

    def test_chunk_colour_tracks_the_peak_level(self, window, player):
        strip = window.strips[0]
        player.tracks[0].rms = 0.5

        player.tracks[0].peak = 0.5
        strip.update_meter()
        assert "#3c3" in strip.meter.styleSheet()

        player.tracks[0].peak = 0.95  # warning band
        strip.update_meter()
        assert "#cc3" in strip.meter.styleSheet()

        player.tracks[0].peak = 1.0  # clipping
        strip.update_meter()
        assert "#c33" in strip.meter.styleSheet()

    def test_meter_range_is_fixed(self, window):
        strip = window.strips[0]
        assert strip.meter.minimum() == 0
        assert strip.meter.maximum() == 100
        assert strip.meter.orientation() == Qt.Orientation.Vertical


class TestTransportHandlers:
    def test_toggle_play_starts_the_player_and_flips_the_label(self, window, player):
        window._toggle_play()

        assert player.is_playing is True
        assert window.play_btn.text() == "⏸ Pause"

        window._toggle_play()

        assert player.is_playing is False
        assert window.play_btn.text() == "▶ Play"

    def test_play_button_click_reaches_the_handler(self, window, player):
        window.play_btn.click()

        assert player.is_playing is True
        assert window.play_btn.text() == "⏸ Pause"

    def test_stop_pauses_and_rewinds_the_ui(self, window, player):
        window._toggle_play()
        player.seek(0.5)
        window.position_slider.setValue(5)

        window._stop()

        assert player.is_playing is False
        assert player.get_position_sec() == 0.0
        assert window.play_btn.text() == "▶ Play"
        assert window.position_slider.value() == 0

    def test_stop_button_click_reaches_the_handler(self, window, player):
        window.play_btn.click()
        window.stop_btn.click()

        assert player.is_playing is False
        assert window.position_slider.value() == 0

    def test_seek_start_sets_the_seeking_flag(self, window):
        window._on_seek_start()
        assert window._seeking is True

    def test_seek_releases_the_flag_and_moves_the_player(self, window, player):
        window._on_seek_start()
        window.position_slider.setValue(4)

        window._on_seek()

        assert window._seeking is False
        assert player.get_position_sec() == pytest.approx(0.4)

    def test_slider_signals_are_wired(self, window, player):
        window.position_slider.sliderPressed.emit()
        assert window._seeking is True

        window.position_slider.setValue(7)
        window.position_slider.sliderReleased.emit()

        assert window._seeking is False
        assert player.get_position_sec() == pytest.approx(0.7)


class TestTick:
    def test_tick_updates_every_strip_meter(self, window, player):
        player.tracks[0].rms = 1.0
        player.tracks[0].peak = 0.5
        player.tracks[1].rms = 0.0

        window._tick()

        assert window.strips[0].meter.value() == 100
        assert window.strips[1].meter.value() == 0

    def test_master_meter_tracks_master_rms(self, window, player):
        player.master_rms = 0.5

        window._tick()

        assert window.master_meter.value() == 89

    def test_master_meter_is_empty_below_the_gate(self, window, player):
        player.master_rms = 1e-7

        window._tick()

        assert window.master_meter.value() == 0

    def test_peak_label_shows_the_master_peak_in_db(self, window, player):
        player.master_peak = 0.5

        window._tick()

        assert window.master_peak_label.text() == "-6.0 dB"
        assert "color: #ddd" in window.master_peak_label.styleSheet()

    def test_near_full_scale_peak_turns_the_label_yellow(self, window, player):
        player.master_peak = 0.95

        window._tick()

        assert window.master_peak_label.text() == "-0.4 dB"
        assert "yellow" in window.master_peak_label.styleSheet()

    def test_peak_at_full_scale_turns_the_label_red(self, window, player):
        player.master_peak = 1.05
        player.clipping_count = 2

        window._tick()

        assert window.master_peak_label.text() == "+0.4 dB CLIP×2"
        assert "red" in window.master_peak_label.styleSheet()

    def test_clip_counter_appears_in_the_label(self, window, player):
        player.master_peak = 1.0
        player.clipping_count = 7

        window._tick()

        assert window.master_peak_label.text().endswith("CLIP×7")

    def test_silent_master_reports_the_floor_value(self, window, player):
        player.master_peak = 0.0

        window._tick()

        assert window.master_peak_label.text() == "-60.0 dB"
        assert "CLIP" not in window.master_peak_label.text()

    def test_position_slider_follows_playback_while_playing(self, window, player):
        window._toggle_play()
        player.seek(0.35)

        window._tick()

        assert window.position_slider.value() == 3

    def test_position_slider_is_frozen_while_seeking(self, window, player):
        window._toggle_play()
        player.seek(0.35)
        window._on_seek_start()
        window.position_slider.setValue(9)

        window._tick()

        assert window.position_slider.value() == 9

    def test_position_slider_is_frozen_while_paused(self, window, player):
        player.seek(0.35)
        window.position_slider.setValue(7)

        window._tick()

        assert window.position_slider.value() == 7

    def test_time_label_shows_position_and_total(self, window, player):
        player.seek(0.35)

        window._tick()

        assert window.time_label.text() == (
            f"{_fmt_time(0.35)} / {_fmt_time(player.duration_sec)}"
        )

    def test_tick_runs_from_the_timer(self, window, player):
        player.master_rms = 0.5

        window.timer.timeout.emit()

        assert window.master_meter.value() == 89


class TestSaveGains:
    def test_default_path_points_at_the_audioman_folder(self, window, monkeypatch, tmp_path):
        calls = _patch_dialog(monkeypatch, "getSaveFileName", ("", ""))

        window._save_gains()

        assert len(calls) == 1
        assert calls[0][1] == "Save gains JSON"
        default_path = Path(calls[0][2])
        # the folder sits next to the source dir, not inside it
        assert default_path.parent == tmp_path / ".audioman" / "fader_test"
        assert default_path.parent.is_dir()
        assert default_path.name.startswith("gains_")
        assert default_path.suffix == ".json"
        assert calls[0][3] == "JSON files (*.json)"

    def test_filename_is_a_sortable_timestamp(self, window, monkeypatch):
        calls = _patch_dialog(monkeypatch, "getSaveFileName", ("", ""))

        window._save_gains()

        stamp = Path(calls[0][2]).stem.removeprefix("gains_")
        assert len(stamp) == len("20260928_120000")
        assert stamp[8] == "_"
        assert stamp.replace("_", "").isdigit()

    def test_cancelled_dialog_writes_nothing(self, window, monkeypatch, player):
        _patch_dialog(monkeypatch, "getSaveFileName", ("", ""))
        player.set_gain_db(0, -5.0)

        window._save_gains()

        assert window.status_label.text() == "ready"

    def test_default_path_without_source_dir(self, qt_app, player, monkeypatch):
        win = FaderTestWindow(player, source_dir=None)
        try:
            calls = _patch_dialog(monkeypatch, "getSaveFileName", ("", ""))

            win._save_gains()

            assert Path(calls[0][2]).name.startswith("gains_")
            assert Path(calls[0][2]).parent == Path(".")
        finally:
            win.close()

    def test_exported_json_contains_the_mixer_state(self, window, player, monkeypatch, tmp_path):
        player.set_gain_db(0, -6.0)
        player.set_gain_db(1, 2.5)
        player.set_muted(1, True)
        player.set_soloed(0, True)
        player.set_master_gain_db(-1.5)
        out = tmp_path / "gains_out.json"
        _patch_dialog(monkeypatch, "getSaveFileName", (str(out), "JSON files (*.json)"))

        window._save_gains()

        data = json.loads(out.read_text())
        assert data["version"] == 1
        assert data["n_tracks"] == 2
        assert data["sample_rate"] == SR
        assert data["duration_sec"] == pytest.approx(player.duration_sec)
        assert data["master_gain_db"] == -1.5
        assert data["source_dir"] == str(window.source_dir)
        assert data["gains"] == {"drums": -6.0, "bass": 2.5}
        assert [t["name"] for t in data["state"]["tracks"]] == ["drums", "bass"]
        assert data["state"]["tracks"][0]["soloed"] is True
        assert data["state"]["tracks"][1]["muted"] is True
        assert data["state"]["tracks"][1]["path"] == "/stems/bass.wav"

    def test_exported_json_defines_the_mix_for_a_fresh_player(
        self, qt_app, monkeypatch, tmp_path
    ):
        """End-to-end contract: the saved JSON rebalances a newly loaded session.

        `cli/fader_test.py --load` and `fader-compare` both consume this file, so
        the test writes real stems, mixes them through the UI, then replays the
        gains onto a player built from the same directory.
        """
        import soundfile as sf

        stems = tmp_path / "stems"
        stems.mkdir()
        rng = np.random.default_rng(5)
        for name in ("drums", "bass"):
            sf.write(
                str(stems / f"{name}.wav"),
                (0.3 * rng.standard_normal(SR * 2)).astype(np.float32),
                SR,
                subtype="FLOAT",
            )

        loaded = MultitrackPlayer.from_directory(stems)
        assert [t.name for t in loaded.tracks] == ["bass", "drums"]  # alphabetical
        win = FaderTestWindow(loaded, source_dir=stems)
        try:
            by_name = {s.track.name: s for s in win.strips}
            by_name["drums"].fader.setValue(db_to_slider(-3.25))
            by_name["bass"].fader.setValue(db_to_slider(1.5))
            # the slider quantises to 0.1 dB and export_gains rounds to 2 dp
            drums_db = round(slider_to_db(db_to_slider(-3.25)), 2)
            assert loaded.tracks[1].gain_db == pytest.approx(drums_db)
            out = tmp_path / "round_trip.json"
            _patch_dialog(monkeypatch, "getSaveFileName", (str(out), ""))

            win._save_gains()
        finally:
            win.close()

        # a second, independent load of the same directory — what `--load` does
        replay = MultitrackPlayer.from_directory(stems)
        assert replay.export_gains() == {"bass": 0.0, "drums": 0.0}
        data = json.loads(out.read_text())
        assert replay.import_gains(data["gains"]) == 2
        assert replay.export_gains() == {"bass": 1.5, "drums": drums_db}
        # the JSON carries the full state too, not only the gains map
        assert data["state"]["master_gain_db"] == 0.0
        assert [t["gain_db"] for t in data["state"]["tracks"]] == [1.5, drums_db]

    def test_exported_at_is_an_iso_timestamp(self, window, monkeypatch, tmp_path):
        out = tmp_path / "stamp.json"
        _patch_dialog(monkeypatch, "getSaveFileName", (str(out), ""))

        window._save_gains()

        data = json.loads(out.read_text())
        # timespec="seconds" → "YYYY-MM-DDTHH:MM:SS"
        assert len(data["exported_at"]) == 19
        assert data["exported_at"][10] == "T"

    def test_status_label_reports_the_saved_filename(self, window, monkeypatch, tmp_path):
        out = tmp_path / "my mix gains.json"
        _patch_dialog(monkeypatch, "getSaveFileName", (str(out), ""))

        window._save_gains()

        assert window.status_label.text() == "saved: my mix gains.json"

    def test_json_is_pretty_printed(self, window, monkeypatch, tmp_path):
        out = tmp_path / "pretty.json"
        _patch_dialog(monkeypatch, "getSaveFileName", (str(out), ""))

        window._save_gains()

        text = out.read_text()
        assert text.startswith("{\n  \"version\": 1")
        assert "\n" in text

    def test_save_button_click_reaches_the_handler(self, window, monkeypatch):
        calls = _patch_dialog(monkeypatch, "getSaveFileName", ("", ""))

        window.save_btn.click()

        assert len(calls) == 1


class TestLoadGains:
    def test_load_applies_gains_through_the_fader_signals(self, window, player, monkeypatch, tmp_path):
        payload = tmp_path / "gains.json"
        payload.write_text(json.dumps({"gains": {"drums": -9.0, "bass": 3.0}}))
        _patch_dialog(monkeypatch, "getOpenFileName", (str(payload), "JSON files (*.json)"))

        window._load_gains()

        assert player.tracks[0].gain_db == pytest.approx(-9.0)
        assert player.tracks[1].gain_db == pytest.approx(3.0)
        assert window.strips[0].fader.value() == db_to_slider(-9.0)
        assert window.strips[1].db_label.text() == "+3.0 dB"
        assert window.status_label.text() == "loaded 2 gains from gains.json"

    def test_bare_gain_dict_is_accepted(self, window, player, monkeypatch, tmp_path):
        payload = tmp_path / "bare.json"
        payload.write_text(json.dumps({"drums": -2.0}))
        _patch_dialog(monkeypatch, "getOpenFileName", (str(payload), ""))

        window._load_gains()

        assert player.tracks[0].gain_db == pytest.approx(-2.0)
        assert player.tracks[1].gain_db == pytest.approx(0.0)
        assert window.status_label.text() == "loaded 1 gains from bare.json"

    def test_unknown_track_names_are_counted_out(self, window, player, monkeypatch, tmp_path):
        payload = tmp_path / "unknown.json"
        payload.write_text(json.dumps({"gains": {"ghost": -1.0}}))
        _patch_dialog(monkeypatch, "getOpenFileName", (str(payload), ""))

        window._load_gains()

        assert player.tracks[0].gain_db == pytest.approx(0.0)
        assert window.status_label.text() == "loaded 0 gains from unknown.json"

    def test_cancelled_dialog_changes_nothing(self, window, player, monkeypatch):
        _patch_dialog(monkeypatch, "getOpenFileName", ("", ""))

        window._load_gains()

        assert player.tracks[0].gain_db == pytest.approx(0.0)
        assert window.status_label.text() == "ready"

    def test_malformed_json_warns_and_leaves_state_alone(self, window, player, monkeypatch, tmp_path):
        warnings = []
        payload = tmp_path / "broken.json"
        payload.write_text("{not json")
        _patch_dialog(monkeypatch, "getOpenFileName", (str(payload), ""))
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            staticmethod(lambda parent, title, text: warnings.append((title, text))),
        )

        window._load_gains()

        assert len(warnings) == 1
        assert warnings[0][0] == "Load failed"
        assert "broken" not in warnings[0][1]  # the parser message, not a traceback
        assert player.tracks[0].gain_db == pytest.approx(0.0)
        assert window.status_label.text() == "ready"

    def test_non_dict_gains_warns(self, window, player, monkeypatch, tmp_path):
        warnings = []
        payload = tmp_path / "list.json"
        payload.write_text(json.dumps({"gains": [1, 2, 3]}))
        _patch_dialog(monkeypatch, "getOpenFileName", (str(payload), ""))
        monkeypatch.setattr(
            QMessageBox,
            "warning",
            staticmethod(lambda parent, title, text: warnings.append((title, text))),
        )

        window._load_gains()

        assert warnings == [("Load failed", "JSON에 'gains' dict가 없습니다.")]
        assert player.tracks[0].gain_db == pytest.approx(0.0)

    def test_load_round_trips_what_save_wrote(self, window, player, monkeypatch, tmp_path):
        out = tmp_path / "round_trip.json"
        player.set_gain_db(0, -7.5)
        player.set_gain_db(1, 4.5)
        _patch_dialog(monkeypatch, "getSaveFileName", (str(out), ""))
        window._save_gains()
        player.set_gain_db(0, 0.0)
        player.set_gain_db(1, 0.0)
        _patch_dialog(monkeypatch, "getOpenFileName", (str(out), ""))

        window._load_gains()

        assert player.tracks[0].gain_db == pytest.approx(-7.5)
        assert player.tracks[1].gain_db == pytest.approx(4.5)

    def test_load_button_click_reaches_the_handler(self, window, monkeypatch, tmp_path):
        payload = tmp_path / "click.json"
        payload.write_text(json.dumps({"gains": {"drums": -1.5}}))
        calls = _patch_dialog(monkeypatch, "getOpenFileName", (str(payload), ""))

        window.load_btn.click()

        assert len(calls) == 1
        assert window.status_label.text() == "loaded 1 gains from click.json"


class TestResetAndClear:
    def test_reset_all_zeroes_every_strip(self, window, player):
        for strip in window.strips:
            strip.fader.setValue(db_to_slider(-15.0))
            strip.mute_btn.setChecked(True)
            strip.solo_btn.setChecked(True)
        player.clipping_count = 5

        window._reset_all()

        assert [s.fader.value() for s in window.strips] == [db_to_slider(0.0)] * 2
        assert all(s.mute_btn.isChecked() is False for s in window.strips)
        assert all(s.solo_btn.isChecked() is False for s in window.strips)
        assert [t.gain_db for t in player.tracks] == [0.0, 0.0]
        assert player.clipping_count == 0
        assert window.status_label.text() == "reset all to 0 dB"

    def test_reset_button_click_reaches_the_handler(self, window, player):
        window.strips[0].fader.setValue(db_to_slider(-30.0))

        window.reset_btn.click()

        assert player.tracks[0].gain_db == pytest.approx(0.0)
        assert window.status_label.text() == "reset all to 0 dB"

    def test_clear_mute_solo_releases_every_strip(self, window, player):
        window.strips[0].mute_btn.setChecked(True)
        window.strips[1].solo_btn.setChecked(True)

        window._clear_mute_solo()

        assert player.tracks[0].muted is False
        assert player.tracks[1].soloed is False
        assert all(s.mute_btn.isChecked() is False for s in window.strips)
        assert all(s.solo_btn.isChecked() is False for s in window.strips)
        assert window.status_label.text() == "cleared mute/solo"

    def test_clear_mute_solo_keeps_gains(self, window, player):
        window.strips[0].fader.setValue(db_to_slider(-8.0))
        window.strips[0].mute_btn.setChecked(True)

        window._clear_mute_solo()

        assert player.tracks[0].gain_db == pytest.approx(-8.0)

    def test_unmute_button_click_reaches_the_handler(self, window, player):
        window.strips[0].mute_btn.setChecked(True)

        window.unmute_btn.click()

        assert player.tracks[0].muted is False
        assert window.status_label.text() == "cleared mute/solo"


class TestCloseEvent:
    def test_close_stops_the_timer_and_the_player(self, qt_app, player):
        win = FaderTestWindow(player, source_dir=None)
        win._toggle_play()
        assert player.is_playing is True

        win.close()

        assert win.timer.isActive() is False
        assert player.is_playing is False
        assert player._stream is None

    def test_close_on_an_idle_window_is_safe(self, qt_app, player):
        win = FaderTestWindow(player, source_dir=None)

        win.close()

        assert win.timer.isActive() is False
        assert player.is_playing is False
