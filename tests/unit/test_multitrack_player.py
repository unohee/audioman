# tests/unit/test_multitrack_player.py — device-independent logic of the
# multitrack playback engine (transport, gain math, mixing callback).
#
# Technique: fake `sounddevice` module + recording fake output stream.
# PortAudio is not installed on the CI host, so `import sounddevice` raises
# OSError. A stub module is installed into sys.modules only long enough to
# import `audioman.core.multitrack_player` (the module keeps a reference to it),
# then removed again so the rest of the test session sees the normal
# "no PortAudio" world. Tests that touch the transport swap the player's module
# level `sd` handle for a recorder, so no device is ever opened and the
# PortAudio-thread callback is driven synchronously from the test thread.

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest
import soundfile as sf

# --- import-time stub: satisfy the module level `import sounddevice as sd` -----
_stub = types.ModuleType("sounddevice")
_stub.OutputStream = None  # replaced per test; never used here
_had_stub = "sounddevice" in sys.modules
if not _had_stub:
    sys.modules["sounddevice"] = _stub

from audioman.core import multitrack_player as mtp  # noqa: E402
from audioman.core.multitrack_player import (  # noqa: E402
    MultitrackPlayer,
    TrackState,
    _load_track,
)

if not _had_stub:
    del sys.modules["sounddevice"]


SR = 100  # small sample rate keeps the mixing assertions exact and fast


class FakeOutputStream:
    """Records transport calls; `render()` drives the callback synchronously."""

    instances: list["FakeOutputStream"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs.get("callback")
        self.started = False
        self.stopped = 0
        self.closed = False
        FakeOutputStream.instances.append(self)

    def start(self):
        self.started = True
        self.stopped = 0

    def stop(self):
        self.stopped += 1
        self.started = False

    def close(self):
        self.closed = True

    def render(self, frames, status=None):
        out = np.zeros((frames, 2), dtype=np.float32)
        self.callback(out, frames, None, status)
        return out


@pytest.fixture
def fake_sd(monkeypatch):
    """Swap the module's sounddevice handle for a recorder."""
    FakeOutputStream.instances.clear()
    fake = types.SimpleNamespace(OutputStream=FakeOutputStream)
    monkeypatch.setattr(mtp, "sd", fake)
    return fake


def _track(name: str, data: np.ndarray, path: str | None = None) -> TrackState:
    return TrackState(
        name=name, path=path or f"/stems/{name}.wav", audio=data,
        sample_rate=SR,
    )


@pytest.fixture
def stereo_tracks():
    ramp = np.linspace(-1.0, 1.0, SR).astype(np.float32)
    return [
        _track("drums", np.stack([ramp, ramp])),
        _track("bass", np.stack([ramp * 0.5, ramp * 0.5])),
    ]


@pytest.fixture
def player(stereo_tracks):
    return MultitrackPlayer(tracks=stereo_tracks, sample_rate=SR)


class TestTrackState:
    def test_defaults(self):
        data = np.zeros((2, 10), dtype=np.float32)
        t = TrackState(name="vocals", path="/x/vocals.wav", audio=data, sample_rate=44100)
        assert t.gain_db == 0.0
        assert t.muted is False
        assert t.soloed is False
        assert t.rms == 0.0
        assert t.peak == 0.0
        assert t.audio is data


class TestLoadTrack:
    def test_loads_as_float32_channels_first(self, tmp_path):
        # write float64 data; the loader must force float32 output
        sr = 8000
        data = np.linspace(-0.5, 0.5, sr, dtype=np.float64)
        path = tmp_path / "stem.wav"
        sf.write(str(path), np.stack([data, data]).T, sr, subtype="FLOAT")

        audio, file_sr = _load_track(path)

        assert file_sr == sr
        assert audio.dtype == np.float32
        assert audio.shape == (2, sr)
        np.testing.assert_allclose(audio[0], data, atol=1e-6)

    def test_matching_target_sr_is_accepted(self, tmp_path):
        sr = 8000
        path = tmp_path / "ok.wav"
        sf.write(str(path), np.zeros(sr, dtype=np.float32), sr, subtype="FLOAT")

        audio, file_sr = _load_track(path, target_sr=sr)
        assert file_sr == sr
        assert audio.shape == (1, sr)

    def test_sample_rate_mismatch_raises_with_filenames(self, tmp_path):
        path = tmp_path / "wrong.wav"
        sf.write(str(path), np.zeros(64, dtype=np.float32), 8000, subtype="FLOAT")

        with pytest.raises(ValueError) as exc:
            _load_track(path, target_sr=44100)

        msg = str(exc.value)
        assert "wrong.wav" in msg
        assert "8000" in msg
        assert "44100" in msg


class TestConstruction:
    def test_requires_at_least_one_track(self):
        with pytest.raises(ValueError):
            MultitrackPlayer(tracks=[], sample_rate=SR)

    def test_total_samples_and_duration(self, player):
        assert player.total_samples == SR
        assert player.duration_sec == pytest.approx(1.0)
        assert player.master_gain_db == 0.0
        assert player.master_rms == 0.0
        assert player.master_peak == 0.0
        assert player.clipping_count == 0
        assert player.is_playing is False

    def test_mismatched_lengths_warn_uses_longest(self, caplog):
        long = _track("long", np.zeros((2, 500), dtype=np.float32))
        short = _track("short", np.zeros((2, 100), dtype=np.float32))

        with caplog.at_level(logging.WARNING, logger="audioman.core.multitrack_player"):
            p = MultitrackPlayer(tracks=[long, short], sample_rate=SR)

        assert p.total_samples == 500
        assert "트랙 길이가 다름" in caplog.text


class TestFromDirectory:
    def test_loads_wavs_sorted_alphabetically(self, tmp_path):
        sr = 8000
        for name in ("kick", "bass"):
            sf.write(str(tmp_path / f"{name}.wav"), np.zeros(64, dtype=np.float32), sr)
        # a non-wav file must be ignored
        (tmp_path / "notes.txt").write_text("ignore me")

        p = MultitrackPlayer.from_directory(tmp_path)

        assert [t.name for t in p.tracks] == ["bass", "kick"]
        assert p.sample_rate == sr
        assert p.tracks[0].path == str(tmp_path / "bass.wav")
        assert p.tracks[0].sample_rate == sr

    def test_block_size_argument_is_kept(self, tmp_path):
        sr = 8000
        sf.write(str(tmp_path / "a.wav"), np.zeros(64, dtype=np.float32), sr)
        p = MultitrackPlayer.from_directory(tmp_path, block_size=256)
        assert p.block_size == 256

    def test_directory_without_wavs_raises(self, tmp_path):
        with pytest.raises(ValueError) as exc:
            MultitrackPlayer.from_directory(tmp_path)
        assert "wav 파일 없음" in str(exc.value)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(ValueError) as exc:
            MultitrackPlayer.from_directory(tmp_path / "nope")
        assert "디렉터리 아님" in str(exc.value)

    def test_target_sr_validated_against_every_file(self, tmp_path):
        sf.write(str(tmp_path / "a.wav"), np.zeros(64, dtype=np.float32), 8000)
        with pytest.raises(ValueError):
            MultitrackPlayer.from_directory(tmp_path, target_sr=44100)

    def test_target_sr_accepted_when_files_match(self, tmp_path):
        sf.write(str(tmp_path / "a.wav"), np.zeros(64, dtype=np.float32), 8000)
        sf.write(str(tmp_path / "b.wav"), np.zeros(64, dtype=np.float32), 8000)
        p = MultitrackPlayer.from_directory(tmp_path, target_sr=8000)
        assert p.sample_rate == 8000
        assert len(p.tracks) == 2


class TestTransport:
    def test_play_opens_stream_once_with_expected_arguments(self, player, fake_sd):
        player.play()

        assert len(FakeOutputStream.instances) == 1
        stream = FakeOutputStream.instances[0]
        assert stream.started is True
        assert stream.kwargs["samplerate"] == SR
        assert stream.kwargs["channels"] == 2
        assert stream.kwargs["dtype"] == "float32"
        assert stream.kwargs["blocksize"] == player.block_size
        assert stream.kwargs["callback"] == player._audio_callback
        assert player.is_playing is True

        # second play() is a no-op (does not open a second stream)
        player.play()
        assert len(FakeOutputStream.instances) == 1
        assert player.is_playing is True

    def test_pause_stops_stream_and_keeps_position(self, player, fake_sd):
        player.play()
        player.seek(0.25)
        player.pause()

        stream = FakeOutputStream.instances[0]
        assert player.is_playing is False
        assert stream.stopped == 1
        assert stream.closed is False
        assert player.get_position_sec() == pytest.approx(0.25)

    def test_pause_before_play_is_safe(self, player):
        player.pause()
        assert player.is_playing is False
        assert player._stream is None

    def test_stop_closes_stream_and_rewinds(self, player, fake_sd):
        player.play()
        player.seek(0.5)
        player.stop()

        stream = FakeOutputStream.instances[0]
        assert stream.stopped == 1
        assert stream.closed is True
        assert player._stream is None
        assert player.is_playing is False
        assert player.get_position_sec() == 0.0

    def test_stop_before_play_is_safe(self, player):
        player.stop()
        assert player._stream is None

    def test_play_after_stop_opens_a_fresh_stream(self, player, fake_sd):
        player.play()
        player.stop()
        player.play()

        assert len(FakeOutputStream.instances) == 2
        assert FakeOutputStream.instances[1].started is True

    def test_seek_clamps_to_valid_range(self, player):
        player.seek(-5.0)
        assert player.get_position_sec() == 0.0

        player.seek(999.0)
        assert player.get_position_sec() == pytest.approx(player.duration_sec)

        player.seek(0.4)
        assert player.get_position_sec() == pytest.approx(0.4)


class TestTrackControls:
    def test_set_gain_db_stores_float(self, player):
        player.set_gain_db(1, -3.456)
        assert player.tracks[1].gain_db == pytest.approx(-3.456)
        assert isinstance(player.tracks[1].gain_db, float)

    def test_set_muted_and_soloed(self, player):
        player.set_muted(0, True)
        player.set_soloed(1, True)
        assert player.tracks[0].muted is True
        assert player.tracks[1].soloed is True

    def test_set_master_gain_db(self, player):
        player.set_master_gain_db(-6.0)
        assert player.master_gain_db == pytest.approx(-6.0)

    def test_export_gains_rounds_to_two_decimals(self, player):
        player.set_gain_db(0, -3.456)
        player.set_gain_db(1, 2.0)
        assert player.export_gains() == {"drums": -3.46, "bass": 2.0}

    def test_export_state_contains_full_mixer_state(self, player):
        player.set_gain_db(0, -3.456)
        player.set_muted(1, True)
        player.set_soloed(0, True)
        player.set_master_gain_db(1.239)

        state = player.export_state()

        assert state["master_gain_db"] == 1.24
        assert [t["name"] for t in state["tracks"]] == ["drums", "bass"]
        assert state["tracks"][0] == {
            "name": "drums",
            "path": "/stems/drums.wav",
            "gain_db": -3.46,
            "muted": False,
            "soloed": True,
        }
        assert state["tracks"][1]["muted"] is True
        assert state["tracks"][1]["gain_db"] == 0.0

    def test_import_gains_matches_by_name_and_counts(self, player):
        matched = player.import_gains({"bass": -7.5, "ghost": 3.0})

        assert matched == 1
        assert player.tracks[1].gain_db == pytest.approx(-7.5)
        assert player.tracks[0].gain_db == 0.0

    def test_import_gains_then_export_round_trips(self, player):
        player.import_gains({"drums": -1.25, "bass": 0.75})
        assert player.export_gains() == {"drums": -1.25, "bass": 0.75}


class TestAudioCallback:
    def test_status_is_logged_at_debug(self, player, fake_sd, caplog):
        player.play()
        stream = FakeOutputStream.instances[0]
        player.pause()

        with caplog.at_level(logging.DEBUG, logger="audioman.core.multitrack_player"):
            stream.render(10, status="input overflow")

        assert "input overflow" in caplog.text

    def test_silent_when_not_playing(self, player, fake_sd):
        player.play()
        stream = FakeOutputStream.instances[0]
        stream.render(5)  # prime nothing; just make sure playing path works
        player.pause()

        out = stream.render(8)
        assert out.shape == (8, 2)
        assert np.all(out == 0.0)

    def test_at_end_of_stream_emits_silence_and_stops(self, player, fake_sd):
        player.play()
        stream = FakeOutputStream.instances[0]
        player.seek(player.duration_sec)

        out = stream.render(16)

        assert np.all(out == 0.0)
        assert player.is_playing is False
        assert player.get_position_sec() == pytest.approx(player.duration_sec)

    def test_gain_db_reaches_callback(self, player, fake_sd):
        player.play()
        stream = FakeOutputStream.instances[0]
        player.set_gain_db(0, -6.0)
        player.set_gain_db(1, -6.0)
        player.set_muted(1, True)  # keep only the first track in the mix

        out = stream.render(10)

        expected = player.tracks[0].audio[:, :10].T * np.float32(10 ** (-6.0 / 20.0))
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_master_gain_scales_the_mix(self, player, fake_sd):
        player.play()
        stream = FakeOutputStream.instances[0]
        player.set_muted(1, True)
        player.set_master_gain_db(-6.0)

        out = stream.render(10)

        expected = player.tracks[0].audio[:, :10].T * np.float32(10 ** (-6.0 / 20.0))
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_muted_track_is_excluded_from_mix(self, player, fake_sd):
        player.play()
        stream = FakeOutputStream.instances[0]
        player.set_muted(0, True)

        out = stream.render(10)

        expected = player.tracks[1].audio[:, :10].T  # unity gain
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_solo_mutes_every_other_track(self, player, fake_sd):
        player.play()
        stream = FakeOutputStream.instances[0]
        player.set_soloed(1, True)

        out = stream.render(10)

        expected = player.tracks[1].audio[:, :10].T
        np.testing.assert_allclose(out, expected, atol=1e-6)

        # solo wins over mute on the soloed track: muting the soloed track
        # changes nothing while it stays soloed
        player.set_muted(1, True)
        player.seek(0.0)
        out2 = stream.render(10)
        np.testing.assert_allclose(out2, expected, atol=1e-6)

    def test_mono_track_is_broadcast_to_both_channels(self, fake_sd):
        mono = _track("mono", np.linspace(-1.0, 1.0, SR, dtype=np.float32)[None, :])
        p = MultitrackPlayer(tracks=[mono], sample_rate=SR)
        p.play()
        stream = FakeOutputStream.instances[0]

        out = stream.render(32)

        np.testing.assert_allclose(out[:, 0], mono.audio[0, :32], atol=1e-7)
        np.testing.assert_allclose(out[:, 1], mono.audio[0, :32], atol=1e-7)

    def test_track_shorter_than_position_is_skipped(self, fake_sd):
        long = _track("long", np.ones((2, 100), dtype=np.float32))
        short = _track("short", np.ones((2, 40), dtype=np.float32))
        p = MultitrackPlayer(tracks=[long, short], sample_rate=SR)
        p.play()
        stream = FakeOutputStream.instances[0]
        p.seek(0.6)  # past the end of `short`

        out = stream.render(10)

        assert np.all(out == 1.0)  # only `long` contributes
        assert short.rms == 0.0
        assert long.rms == pytest.approx(1.0)

    def test_position_advances_by_frames(self, player, fake_sd):
        player.play()
        stream = FakeOutputStream.instances[0]

        stream.render(10)
        assert player.get_position_sec() == pytest.approx(0.10)
        stream.render(25)
        assert player.get_position_sec() == pytest.approx(0.35)

    def test_tail_is_zero_filled_and_playback_stops(self, player, fake_sd):
        player.play()
        stream = FakeOutputStream.instances[0]
        player.seek(0.95)

        out = stream.render(20)

        assert out.shape == (20, 2)
        assert np.all(out[5:] == 0.0)  # only 5 samples remained
        assert np.any(out[:5] != 0.0)
        assert player.is_playing is False
        assert player.get_position_sec() == pytest.approx(player.duration_sec)

    def test_track_meters_report_segment_levels(self, fake_sd):
        ones = _track("ones", np.ones((2, SR), dtype=np.float32))
        p = MultitrackPlayer(tracks=[ones], sample_rate=SR)
        p.play()
        stream = FakeOutputStream.instances[0]
        p.set_gain_db(0, -6.0)  # keeps the master bus below full scale

        stream.render(20)

        lin = np.float32(10 ** (-6.0 / 20.0))
        # per-track meters are pre-fader (raw segment level)
        assert ones.rms == pytest.approx(1.0)
        assert ones.peak == pytest.approx(1.0)
        # master meters are post-fader
        assert p.master_rms == pytest.approx(float(lin), abs=1e-6)
        assert p.master_peak == pytest.approx(float(lin), abs=1e-6)
        assert p.clipping_count == 0

    def test_master_meter_and_clipping_counter(self, fake_sd):
        ones = _track("ones", np.ones((2, 60), dtype=np.float32))
        p = MultitrackPlayer(tracks=[ones], sample_rate=SR)
        p.play()
        stream = FakeOutputStream.instances[0]
        p.set_master_gain_db(6.0)  # unity input * 1.995 → over full scale

        out = stream.render(10)

        np.testing.assert_allclose(out, np.float32(10 ** (6.0 / 20.0)), atol=1e-6)
        assert p.master_peak > 1.0
        assert p.master_rms > 1.0
        # every sample of both channels is >= 1.0
        assert p.clipping_count == 20

    def test_outdata_is_overwritten_not_accumulated(self, player, fake_sd):
        player.play()
        stream = FakeOutputStream.instances[0]
        dirty = np.full((4, 2), 9.0, dtype=np.float32)

        stream.callback(dirty, 4, None, None)

        assert not np.any(dirty == 9.0)
