# tests/unit/test_session.py — 세션 파일 파싱 테스트

import json
import pytest
from pathlib import Path

from audioman.core.session import load_session


class TestLoadSession:
    """YAML/JSON 세션 파일 파싱"""

    def test_yaml_session(self, tmp_path):
        """YAML 세션 파일 파싱"""
        session_file = tmp_path / "session.yaml"
        # 더미 트랙 파일 생성
        (tmp_path / "vocals.wav").touch()
        (tmp_path / "guitar.wav").touch()

        session_file.write_text("""
output: mix.wav
format: PCM_24
sample_rate: 48000
tracks:
  - path: vocals.wav
    gain_db: -3.0
    pan: 0.0
    chain: "denoise:threshold=-20"
  - path: guitar.wav
    gain_db: -6.0
    pan: -0.5
master:
  chain: "limiter:threshold=-1"
""")
        config = load_session(session_file)

        assert len(config.tracks) == 2
        assert config.tracks[0].gain_db == -3.0
        assert config.tracks[0].pan == 0.0
        assert config.tracks[0].chain is not None
        assert len(config.tracks[0].chain) == 1
        assert config.tracks[0].chain[0].plugin_name == "denoise"

        assert config.tracks[1].gain_db == -6.0
        assert config.tracks[1].pan == -0.5
        assert config.tracks[1].chain is None

        assert config.master_chain is not None
        assert len(config.master_chain) == 1
        assert config.master_chain[0].plugin_name == "limiter"

        assert config.sample_rate == 48000
        assert config.subtype == "PCM_24"
        # 상대 경로가 세션 파일 기준 절대 경로로 변환됨
        assert Path(config.output).is_absolute()

    def test_json_session(self, tmp_path):
        """JSON 세션 파일 파싱"""
        session_file = tmp_path / "session.json"
        (tmp_path / "track1.wav").touch()

        data = {
            "output": "out.wav",
            "tracks": [
                {"path": "track1.wav", "gain_db": 0.0, "pan": 0.0}
            ],
        }
        session_file.write_text(json.dumps(data))

        config = load_session(session_file)
        assert len(config.tracks) == 1
        assert config.subtype == "PCM_24"  # 기본값

    def test_missing_tracks(self, tmp_path):
        """tracks 없으면 에러"""
        session_file = tmp_path / "bad.yaml"
        session_file.write_text("output: out.wav\n")

        with pytest.raises(ValueError, match="tracks"):
            load_session(session_file)

    def test_missing_output(self, tmp_path):
        """output 없으면 에러"""
        session_file = tmp_path / "bad.yaml"
        (tmp_path / "t.wav").touch()
        session_file.write_text("tracks:\n  - path: t.wav\n")

        with pytest.raises(ValueError, match="output"):
            load_session(session_file)

    def test_relative_paths_resolved(self, tmp_path):
        """상대 경로 → 세션 파일 디렉토리 기준 절대 경로"""
        subdir = tmp_path / "project"
        subdir.mkdir()
        (subdir / "vocal.wav").touch()
        session = subdir / "mix.yaml"
        session.write_text("""
output: result.wav
tracks:
  - path: vocal.wav
""")

        config = load_session(session)
        assert Path(config.tracks[0].path) == subdir / "vocal.wav"
        assert Path(config.output) == subdir / "result.wav"
