# Created: 2026-04-27
# Purpose: 멀티트랙 동기 재생 엔진. fader-test UI의 audio backend.
#
# 모든 stem을 RAM에 float32로 미리 로드 → sounddevice OutputStream callback에서
# gain 적용 후 stereo mix. sample-accurate sync, low-latency.
#
# Thread model:
#   - Main thread (Qt): fader gain 변경, transport 명령
#   - Audio thread (PortAudio callback): mix + output
#   - 두 thread 사이 데이터 전달은 numpy array의 atomic assignment에 의존
#     (numpy float64/float32 element assign은 GIL 하에서 atomic)

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """단일 트랙의 재생 상태."""
    name: str
    path: str
    audio: np.ndarray  # shape (channels, samples), float32
    sample_rate: int
    gain_db: float = 0.0
    muted: bool = False
    soloed: bool = False
    rms: float = 0.0  # 마지막 callback의 RMS (meter용)
    peak: float = 0.0


def _load_track(path: Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """단일 wav를 float32 (channels, samples) 형태로 로드."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data.T  # (samples, ch) → (ch, samples)
    if target_sr is not None and sr != target_sr:
        raise ValueError(
            f"Sample rate mismatch: {path.name} is {sr}Hz, expected {target_sr}Hz"
        )
    return audio, sr


class MultitrackPlayer:
    """멀티트랙 stem 폴더를 동기 재생하는 엔진.

    사용법:
        player = MultitrackPlayer.from_directory(Path("/path/to/stems"))
        player.set_gain_db(track_index=0, db=-3.5)
        player.play()
        ... (재생 중) ...
        player.pause()
        gains = player.export_gains()  # {track_name: gain_db}
    """

    def __init__(
        self,
        tracks: list[TrackState],
        sample_rate: int,
        block_size: int = 1024,
        master_gain_db: float = 0.0,
    ):
        if not tracks:
            raise ValueError("최소 1개의 트랙이 필요합니다.")
        self.tracks = tracks
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.master_gain_db = master_gain_db

        # 모든 트랙 길이 일치 확인 (DAW stem export 가정)
        lengths = {t.audio.shape[1] for t in tracks}
        if len(lengths) > 1:
            logger.warning(f"트랙 길이가 다름: {lengths}. 가장 긴 길이로 재생.")
        self.total_samples = max(t.audio.shape[1] for t in tracks)
        self.duration_sec = self.total_samples / sample_rate

        # 재생 상태 (lock 보호)
        self._lock = threading.RLock()
        self._position = 0  # 현재 sample 위치
        self._playing = False
        self._stream: sd.OutputStream | None = None

        # Master meter (read-only by UI)
        self.master_rms = 0.0
        self.master_peak = 0.0
        self.clipping_count = 0  # 누적

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        block_size: int = 1024,
        target_sr: int | None = None,
    ) -> "MultitrackPlayer":
        """디렉터리의 모든 .wav를 트랙으로 로드. 알파벳 순서."""
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"디렉터리 아님: {directory}")
        wav_paths = sorted(directory.glob("*.wav"))
        if not wav_paths:
            raise ValueError(f"wav 파일 없음: {directory}")

        tracks = []
        sr = target_sr
        for p in wav_paths:
            audio, file_sr = _load_track(p, target_sr=sr)
            if sr is None:
                sr = file_sr
            tracks.append(TrackState(
                name=p.stem.strip(),
                path=str(p),
                audio=audio,
                sample_rate=file_sr,
            ))
            logger.info(f"loaded: {p.name} ({audio.shape[1]/file_sr:.1f}s, {audio.shape[0]}ch)")

        return cls(tracks=tracks, sample_rate=sr, block_size=block_size)

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def play(self) -> None:
        with self._lock:
            if self._playing:
                return
            if self._stream is None:
                self._stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=2,
                    dtype="float32",
                    blocksize=self.block_size,
                    callback=self._audio_callback,
                )
            self._playing = True
            self._stream.start()

    def pause(self) -> None:
        with self._lock:
            self._playing = False
            if self._stream is not None:
                self._stream.stop()

    def stop(self) -> None:
        with self._lock:
            self._playing = False
            self._position = 0
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

    def seek(self, position_sec: float) -> None:
        with self._lock:
            new_pos = int(max(0, min(position_sec * self.sample_rate, self.total_samples)))
            self._position = new_pos

    def get_position_sec(self) -> float:
        with self._lock:
            return self._position / self.sample_rate

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    # ------------------------------------------------------------------
    # Per-track control (UI-callable, audio-thread-safe)
    # ------------------------------------------------------------------

    def set_gain_db(self, track_index: int, db: float) -> None:
        # numpy/python float assignment는 GIL 하에 atomic이라 lock 불필요
        self.tracks[track_index].gain_db = float(db)

    def set_muted(self, track_index: int, muted: bool) -> None:
        self.tracks[track_index].muted = bool(muted)

    def set_soloed(self, track_index: int, soloed: bool) -> None:
        self.tracks[track_index].soloed = bool(soloed)

    def set_master_gain_db(self, db: float) -> None:
        self.master_gain_db = float(db)

    def export_gains(self) -> dict[str, float]:
        """현재 모든 트랙의 gain dB를 dict로."""
        return {t.name: round(t.gain_db, 2) for t in self.tracks}

    def export_state(self) -> dict:
        """완전한 상태 (gain + mute + solo + master)."""
        return {
            "master_gain_db": round(self.master_gain_db, 2),
            "tracks": [
                {
                    "name": t.name,
                    "path": t.path,
                    "gain_db": round(t.gain_db, 2),
                    "muted": t.muted,
                    "soloed": t.soloed,
                }
                for t in self.tracks
            ],
        }

    def import_gains(self, gains: dict[str, float]) -> int:
        """이름 매칭으로 gain 일괄 적용. 매칭된 트랙 수 반환."""
        matched = 0
        for t in self.tracks:
            if t.name in gains:
                t.gain_db = float(gains[t.name])
                matched += 1
        return matched

    # ------------------------------------------------------------------
    # Audio callback (PortAudio thread)
    # ------------------------------------------------------------------

    def _audio_callback(self, outdata, frames, time_info, status):
        if status:
            logger.debug(f"PortAudio status: {status}")

        with self._lock:
            playing = self._playing
            pos = self._position

        if not playing:
            outdata.fill(0.0)
            return

        # 어떤 트랙이 들리는가? (solo가 있으면 solo만, 없으면 unmuted 전부)
        any_solo = any(t.soloed for t in self.tracks)

        # mix buffer (frames, 2)
        mix = np.zeros((frames, 2), dtype=np.float32)
        end = min(pos + frames, self.total_samples)
        n_to_copy = end - pos

        if n_to_copy <= 0:
            outdata.fill(0.0)
            with self._lock:
                self._playing = False
            return

        master_lin = np.float32(10 ** (self.master_gain_db / 20.0))

        for t in self.tracks:
            if any_solo:
                if not t.soloed:
                    continue
            elif t.muted:
                continue

            lin = np.float32(10 ** (t.gain_db / 20.0))
            track_audio = t.audio
            ch = track_audio.shape[0]

            # 현재 위치에서 frames만큼 잘라내 (n_to_copy)
            tend = min(pos + frames, track_audio.shape[1])
            tn = tend - pos
            if tn <= 0:
                continue

            seg = track_audio[:, pos:tend]  # (ch, tn)
            if ch == 1:
                # mono → stereo broadcast
                mix[:tn, 0] += seg[0] * lin
                mix[:tn, 1] += seg[0] * lin
            else:
                mix[:tn, 0] += seg[0] * lin
                mix[:tn, 1] += seg[1] * lin

            # 트랙별 RMS/peak (UI meter용)
            t.rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
            t.peak = float(np.max(np.abs(seg))) if seg.size else 0.0

        mix *= master_lin

        # Master meter
        self.master_rms = float(np.sqrt(np.mean(mix ** 2)))
        self.master_peak = float(np.max(np.abs(mix)))
        if self.master_peak >= 1.0:
            self.clipping_count += int(np.sum(np.abs(mix) >= 1.0))

        # 끝까지 안 채웠으면 0으로
        if n_to_copy < frames:
            mix[n_to_copy:].fill(0.0)
            with self._lock:
                self._position = self.total_samples
                self._playing = False
        else:
            with self._lock:
                self._position = pos + frames

        outdata[:] = mix
