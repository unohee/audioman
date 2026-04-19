# tests/unit/test_batch.py — 배치 유틸리티 테스트

import pytest
from pathlib import Path

from audioman.core.batch import collect_audio_files, resolve_output_path, AUDIO_EXTENSIONS


class TestCollectAudioFiles:
    """오디오 파일 수집"""

    def test_single_file(self, tmp_path):
        wav = tmp_path / "song.wav"
        wav.touch()
        result = collect_audio_files(wav)
        assert result == [wav]

    def test_directory_flat(self, tmp_path):
        for name in ["a.wav", "b.flac", "c.mp3"]:
            (tmp_path / name).touch()
        # 비오디오 파일
        (tmp_path / "readme.txt").touch()

        result = collect_audio_files(tmp_path)
        assert len(result) == 3
        extensions = {f.suffix for f in result}
        assert extensions == {".wav", ".flac", ".mp3"}

    def test_directory_recursive(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "top.wav").touch()
        (tmp_path / "sub" / "nested.wav").touch()

        flat = collect_audio_files(tmp_path, recursive=False)
        assert len(flat) == 1

        recursive = collect_audio_files(tmp_path, recursive=True)
        assert len(recursive) == 2

    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            collect_audio_files(tmp_path / "nope")

    def test_empty_directory(self, tmp_path):
        result = collect_audio_files(tmp_path)
        assert result == []

    def test_sorted_output(self, tmp_path):
        for name in ["z.wav", "a.wav", "m.wav"]:
            (tmp_path / name).touch()
        result = collect_audio_files(tmp_path)
        names = [f.name for f in result]
        assert names == sorted(names)

    def test_all_extensions(self, tmp_path):
        """지원하는 모든 확장자 수집"""
        for ext in AUDIO_EXTENSIONS:
            (tmp_path / f"test{ext}").touch()
        result = collect_audio_files(tmp_path)
        assert len(result) == len(AUDIO_EXTENSIONS)


class TestResolveOutputPath:
    def test_basic(self, tmp_path):
        input_path = tmp_path / "input" / "song.wav"
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        result = resolve_output_path(input_path, input_dir, output_dir)
        assert result == output_dir / "song.wav"

    def test_with_suffix(self, tmp_path):
        input_path = tmp_path / "song.wav"
        result = resolve_output_path(
            input_path, input_path, tmp_path / "out", suffix="_processed",
        )
        assert result.name == "song_processed.wav"

    def test_custom_extension(self, tmp_path):
        input_path = tmp_path / "song.wav"
        result = resolve_output_path(
            input_path, input_path, tmp_path / "out", ext=".flac",
        )
        assert result.suffix == ".flac"

    def test_preserves_subdirectory(self, tmp_path):
        input_dir = tmp_path / "input"
        sub = input_dir / "vocals"
        sub.mkdir(parents=True)
        input_path = sub / "take1.wav"
        input_path.touch()

        output_dir = tmp_path / "output"
        result = resolve_output_path(input_path, input_dir, output_dir)
        assert result == output_dir / "vocals" / "take1.wav"

    def test_creates_parent_dirs(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        input_path = input_dir / "deep" / "nested" / "file.wav"

        output_dir = tmp_path / "output"
        result = resolve_output_path(input_path, input_dir, output_dir)
        # parent가 생성됨
        assert result.parent.exists()
