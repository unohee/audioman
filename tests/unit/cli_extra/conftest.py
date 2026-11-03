# tests/unit/cli_extra/conftest.py
# Purpose: shared fixtures for the third CLI test slice (AUD-1851):
#          obs, observe, screen, fader-compare, changelog, schemas.
#
# Tests drive the real parsers and the real `run(args)` entry points in-process,
# so `--cov=audioman` attributes the executed statements to the command modules.
# Host state is isolated (HOME / cache / preset dir into tmp_path) and the
# synthetic media fixtures are deterministic: WAVs come from numpy, and the
# multitrack video is built with the local ffmpeg the OBS paths require.

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


SR = 48000


@pytest.fixture(autouse=True)
def isolated_cli_env(tmp_path: Path, monkeypatch):
    """Redirect cache/preset/home into tmp_path and reset the singletons."""
    monkeypatch.setenv("AUDIOMAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AUDIOMAN_PRESET_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AUDIOMAN_PLAIN", raising=False)

    from audioman.config import settings as settings_module
    import audioman.core.registry as registry_module
    from audioman.cli import output as output_module

    settings_module.reset_settings()
    registry_module._registry = None
    output_module.set_plain(False)

    yield

    settings_module.reset_settings()
    registry_module._registry = None
    output_module.set_plain(False)


class CliResult:
    """Exit status plus the captured streams of one in-process CLI invocation."""

    def __init__(self, code: int, stdout: str, stderr: str):
        self.code = code
        self.stdout = stdout
        self.stderr = stderr

    @property
    def payload(self) -> dict:
        return json.loads(self.stdout)

    def __repr__(self) -> str:
        return f"<CliResult code={self.code} out={self.stdout[:200]!r} err={self.stderr[:200]!r}>"


@pytest.fixture
def run_cli():
    """Run `audioman <argv>` in-process; return (exit code, stdout, stderr).

    ``--plain`` is prepended so rich resolves its console against the swapped
    ``sys.stdout`` and the assertions read stable, markup-free text.
    """

    def _run(argv, *, plain: bool = True, json_mode: bool = False) -> CliResult:
        from audioman.cli import app

        full = ["--plain"] if plain else []
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


# ---------------------------------------------------------------------------
# audio fixtures
# ---------------------------------------------------------------------------


def write_tone(
    path: Path,
    *,
    sample_rate: int = SR,
    duration: float = 1.0,
    frequency: float = 440.0,
    amplitude: float = 0.3,
    channels: int = 1,
) -> Path:
    """Mono/stereo sine whose level stays clear of the clip and silence guards."""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    data = mono if channels == 1 else np.stack([mono] * channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")
    return path


def write_click_wav(path: Path, *, sample_rate: int = 44100, duration: float = 1.0) -> Path:
    """Impulse train over digital silence: the fallback click detector fires."""
    n = int(sample_rate * duration)
    mono = np.zeros(n, dtype=np.float32)
    for pos in (int(0.1 * sample_rate), int(0.4 * sample_rate), int(0.7 * sample_rate)):
        mono[pos:pos + 24] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mono, sample_rate, subtype="PCM_16")
    return path


def write_hum_wav(path: Path, *, sample_rate: int = 44100, duration: float = 1.0,
                  frequency: float = 60.0) -> Path:
    """Sustained mains tone — `screen --issues hum` reports it with the fallback."""
    n = int(sample_rate * duration)
    t = np.arange(n, dtype=np.float32) / sample_rate
    mono = (0.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mono, sample_rate, subtype="PCM_16")
    return path


def write_corrupt_wav(path: Path) -> Path:
    """Bytes that soundfile refuses to open (undecodable input)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not an audio payload")
    return path


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_multitrack_video(
    out_path: Path,
    tracks: list[np.ndarray],
    *,
    sample_rate: int = SR,
    video_duration: float = 1.5,
) -> Path | None:
    """Mux N mono tracks plus a colour video track into one container.

    Returns None when ffmpeg is missing or refuses the encode, so callers can
    skip instead of asserting on an environment gap.
    """
    if not _ffmpeg_available():
        return None

    wav_paths = []
    for i, audio in enumerate(tracks):
        wav = out_path.parent / f"_src_track{i}.wav"
        data = audio if audio.ndim == 1 else audio.T
        sf.write(str(wav), data, sample_rate, subtype="PCM_24")
        wav_paths.append(wav)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=size=64x64:rate=30:duration={video_duration}",
    ]
    for wav in wav_paths:
        cmd += ["-i", str(wav)]
    cmd += ["-map", "0:v"]
    for i in range(len(wav_paths)):
        cmd += ["-map", f"{i + 1}:a"]
    cmd += ["-c:v", "libx264", "-c:a", "aac", "-shortest", str(out_path)]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not out_path.exists():
        return None
    return out_path


def voice_track(sample_rate: int = SR, duration: float = 1.0) -> np.ndarray:
    """Speech-like band + modulation (probe RMS is well above the silence floor)."""
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    sig = 0.3 * np.sin(2 * np.pi * 300 * t) * (1.0 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return sig.astype(np.float32)


def music_track(sample_rate: int = SR, duration: float = 1.0) -> np.ndarray:
    """Sub-heavy signal: a distinct RMS from `voice_track` for grouping."""
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    return (0.5 * np.sin(2 * np.pi * 60 * t)).astype(np.float32)


def write_stems(directory: Path, names=("kick.wav", "snare.wav", "bass.wav"),
                *, sample_rate: int = 44100) -> list[Path]:
    """Named stems for `fader-compare`: automix needs real decodable material."""
    directory.mkdir(parents=True, exist_ok=True)
    freqs = {"kick": 60.0, "snare": 200.0, "bass": 80.0}
    paths = []
    for name in names:
        stem = Path(name).stem
        freq = freqs.get(stem, 150.0)
        t = np.arange(sample_rate, dtype=np.float32) / sample_rate
        sig = (0.6 * np.sin(2 * np.pi * freq * t) * (1.0 + 0.3 * np.sin(2 * np.pi * 4 * t)))
        data = np.stack([sig, sig], axis=1).astype(np.float32)
        path = directory / name
        sf.write(str(path), data, sample_rate, subtype="PCM_16")
        paths.append(path)
    return paths


def write_ground_truth(path: Path, source_dir: Path, gains: dict) -> Path:
    """The fader-test export shape consumed by `fader-compare`."""
    payload = {
        "version": 1,
        "exported_at": "2026-09-28T00:00:00",
        "source_dir": str(source_dir),
        "n_tracks": len(gains),
        "gains": gains,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_json_stream(text: str) -> list:
    """Decode a run of concatenated JSON values (indented, newline-separated).

    Batch commands emit one pretty-printed envelope per input file straight to
    stdout, so the stream cannot be split on newlines. ``raw_decode`` consumes
    one value at a time and reports where the next one starts.
    """
    decoder = json.JSONDecoder()
    values: list = []
    pos = 0
    length = len(text)
    while pos < length:
        while pos < length and text[pos] in " \t\r\n":
            pos += 1
        if pos >= length:
            break
        value, pos = decoder.raw_decode(text, pos)
        values.append(value)
    return values
