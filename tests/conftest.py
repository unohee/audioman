# tests/conftest.py — 공통 fixture

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path


@pytest.fixture
def sample_rate():
    return 44100


@pytest.fixture
def test_audio(sample_rate):
    """1초 440Hz 사인파 스테레오 (channels, samples)"""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    return np.stack([mono, mono])  # (2, 44100)


@pytest.fixture
def test_audio_mono(test_audio):
    """모노 버전"""
    return test_audio[0]


@pytest.fixture
def silent_audio(sample_rate):
    """무음 스테레오"""
    return np.zeros((2, sample_rate), dtype=np.float32)


@pytest.fixture
def test_wav(tmp_path, test_audio, sample_rate):
    """임시 WAV 파일 경로"""
    path = tmp_path / "test.wav"
    sf.write(str(path), test_audio.T, sample_rate, subtype="PCM_24")
    return path


@pytest.fixture
def test_wav_silent(tmp_path, silent_audio, sample_rate):
    """무음 WAV 파일"""
    path = tmp_path / "silent.wav"
    sf.write(str(path), silent_audio.T, sample_rate, subtype="PCM_24")
    return path


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Pin the rich console width for every test.

    macOS tmp paths (``/private/var/folders/...``) are long enough that an 80-column
    console wraps them mid-path, which breaks substring assertions on human output.
    Line wrapping is incidental to every claim the suite makes.
    """
    monkeypatch.setenv("COLUMNS", "400")
