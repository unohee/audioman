# tests/unit/test_session_extended.py — session.py 미커버 영역 추가 테스트

import json
import pytest
from pathlib import Path

from audioman.core.session import load_session, _parse_track, SessionConfig
from audioman.core.pipeline import PipelineStep
from audioman.core.mixer import TrackConfig


class TestParseTrack:
    """_parse_track 개별 테스트"""

    def test_basic(self, tmp_path):
        (tmp_path / "vocal.wav").touch()
        track = _parse_track({"path": "vocal.wav"}, tmp_path)
        assert Path(track.path) == tmp_path / "vocal.wav"
        assert track.gain_db == 0.0
        assert track.pan == 0.0

    def test_absolute_path(self, tmp_path):
        abs_path = str(tmp_path / "abs.wav")
        track = _parse_track({"path": abs_path}, tmp_path)
        assert track.path == abs_path

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="path"):
            _parse_track({}, tmp_path)

    def test_all_fields(self, tmp_path):
        (tmp_path / "t.wav").touch()
        track = _parse_track({
            "path": "t.wav",
            "gain_db": -6.0,
            "pan": 0.5,
            "mute": True,
            "solo": True,
            "offset_samples": 1024,
        }, tmp_path)
        assert track.gain_db == -6.0
        assert track.pan == 0.5
        assert track.mute is True
        assert track.solo is True
        assert track.offset_samples == 1024

    def test_chain_as_string(self, tmp_path):
        (tmp_path / "t.wav").touch()
        track = _parse_track({
            "path": "t.wav",
            "chain": "denoise:threshold=-20",
        }, tmp_path)
        assert track.chain is not None
        assert len(track.chain) == 1
        assert track.chain[0].plugin_name == "denoise"

    def test_chain_as_list_of_dicts(self, tmp_path):
        (tmp_path / "t.wav").touch()
        track = _parse_track({
            "path": "t.wav",
            "chain": [
                {"plugin": "denoise", "params": {"threshold": -20}},
                {"plugin_name": "declick", "params": {}},
            ],
        }, tmp_path)
        assert len(track.chain) == 2
        assert track.chain[0].plugin_name == "denoise"
        assert track.chain[1].plugin_name == "declick"

    def test_chain_as_list_of_strings(self, tmp_path):
        (tmp_path / "t.wav").touch()
        track = _parse_track({
            "path": "t.wav",
            "chain": ["denoise:threshold=-20", "declick"],
        }, tmp_path)
        assert len(track.chain) == 2

    def test_no_chain(self, tmp_path):
        (tmp_path / "t.wav").touch()
        track = _parse_track({"path": "t.wav"}, tmp_path)
        assert track.chain is None


class TestSessionConfigToDict:
    """SessionConfig.to_dict() 테스트"""

    def test_basic(self):
        config = SessionConfig(
            tracks=[TrackConfig(path="/tmp/t.wav")],
            output="/tmp/out.wav",
        )
        d = config.to_dict()
        assert d["output"] == "/tmp/out.wav"
        assert len(d["tracks"]) == 1
        assert "sample_rate" not in d
        assert "master_chain" not in d

    def test_with_sample_rate(self):
        config = SessionConfig(
            tracks=[TrackConfig(path="/tmp/t.wav")],
            output="/tmp/out.wav",
            sample_rate=48000,
        )
        d = config.to_dict()
        assert d["sample_rate"] == 48000

    def test_with_master_chain(self):
        config = SessionConfig(
            tracks=[TrackConfig(path="/tmp/t.wav")],
            output="/tmp/out.wav",
            master_chain=[PipelineStep(plugin_name="limiter", params={"threshold": -1.0})],
        )
        d = config.to_dict()
        assert "master_chain" in d
        assert d["master_chain"][0]["plugin"] == "limiter"


class TestLoadSessionEdgeCases:
    """load_session 추가 edge case"""

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(tmp_path / "nonexistent.yaml")

    def test_not_dict_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="딕셔너리"):
            load_session(f)

    def test_json_session_with_master(self, tmp_path):
        (tmp_path / "t.wav").touch()
        data = {
            "output": "out.wav",
            "tracks": [{"path": "t.wav"}],
            "master": {"chain": "limiter:threshold=-1"},
        }
        f = tmp_path / "s.json"
        f.write_text(json.dumps(data))
        config = load_session(f)
        assert config.master_chain is not None

    def test_master_chain_as_list(self, tmp_path):
        (tmp_path / "t.wav").touch()
        f = tmp_path / "s.yaml"
        f.write_text("""
output: out.wav
tracks:
  - path: t.wav
master:
  chain:
    - "limiter:threshold=-1"
    - "eq:gain=2"
""")
        config = load_session(f)
        assert len(config.master_chain) == 2

    def test_master_chain_as_list_of_dicts(self, tmp_path):
        (tmp_path / "t.wav").touch()
        data = {
            "output": "out.wav",
            "tracks": [{"path": "t.wav"}],
            "master": {
                "chain": [
                    {"plugin": "limiter", "params": {"threshold": -1}},
                ],
            },
        }
        f = tmp_path / "s.json"
        f.write_text(json.dumps(data))
        config = load_session(f)
        assert config.master_chain[0].plugin_name == "limiter"
