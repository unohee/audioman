# tests/unit/test_audio_file.py

import numpy as np
import pytest
from audioman.core.audio_file import read_audio, write_audio, get_audio_stats


class TestReadAudio:
    def test_read_returns_channels_samples(self, test_wav, sample_rate):
        audio, sr = read_audio(test_wav)
        assert sr == sample_rate
        assert audio.ndim == 2
        assert audio.shape[0] == 2  # stereo
        assert audio.shape[1] == sample_rate  # 1초

    def test_read_dtype_float32(self, test_wav):
        audio, _ = read_audio(test_wav)
        assert audio.dtype == np.float32

    def test_read_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_audio(tmp_path / "nonexistent.wav")


class TestWriteAudio:
    def test_write_and_read_roundtrip(self, tmp_path, test_audio, sample_rate):
        path = tmp_path / "out.wav"
        write_audio(path, test_audio, sample_rate)
        assert path.exists()

        audio, sr = read_audio(path)
        assert sr == sample_rate
        assert audio.shape == test_audio.shape
        # PCM_24 양자화 오차 허용
        np.testing.assert_allclose(audio, test_audio, atol=1e-4)

    def test_write_creates_parent_dirs(self, tmp_path, test_audio, sample_rate):
        path = tmp_path / "sub" / "dir" / "out.wav"
        write_audio(path, test_audio, sample_rate)
        assert path.exists()


class TestGetAudioStats:
    def test_stats_correct(self, test_audio, sample_rate):
        stats = get_audio_stats(test_audio, sample_rate)
        assert stats.sample_rate == sample_rate
        assert stats.channels == 2
        assert stats.frames == sample_rate
        assert abs(stats.duration - 1.0) < 0.01
        assert 0.4 < stats.peak < 0.6  # 0.5 amplitude
        assert stats.rms > 0

    def test_stats_silent(self, silent_audio, sample_rate):
        stats = get_audio_stats(silent_audio, sample_rate)
        assert stats.peak == 0.0
        assert stats.rms == 0.0
