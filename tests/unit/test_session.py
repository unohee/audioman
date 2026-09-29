# tests/unit/test_session.py - session file parsing tests

import json
import pytest
from pathlib import Path

from audioman.core.session import load_session


class TestLoadSession:
    """YAML/JSON session file parsing."""

    def test_yaml_session(self, tmp_path):
        """YAML session file parsing."""
        session_file = tmp_path / "session.yaml"
        # Create dummy track files
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
        # The relative path was turned into an absolute path under the session dir
        assert Path(config.output).is_absolute()

    def test_json_session(self, tmp_path):
        """JSON session file parsing."""
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
        assert config.subtype == "PCM_24"  # default

    def test_missing_tracks(self, tmp_path):
        """Missing tracks is an error."""
        session_file = tmp_path / "bad.yaml"
        session_file.write_text("output: out.wav\n")

        with pytest.raises(ValueError, match="tracks"):
            load_session(session_file)

    def test_missing_output(self, tmp_path):
        """Missing output is an error."""
        session_file = tmp_path / "bad.yaml"
        (tmp_path / "t.wav").touch()
        session_file.write_text("tracks:\n  - path: t.wav\n")

        with pytest.raises(ValueError, match="output"):
            load_session(session_file)

    def test_relative_paths_resolved(self, tmp_path):
        """A relative path resolves against the session file directory."""
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


class TestUnknownExtensionFallback:
    """Extension-less session files try YAML then fall back to JSON."""

    def test_yaml_parseable_extensionless(self, tmp_path):
        from audioman.core.session import load_session
        track = tmp_path / "a.wav"
        track.write_bytes(b"x")
        f = tmp_path / "session.conf"
        f.write_text("output: out.wav\ntracks:\n  - path: a.wav\n")
        session = load_session(f)
        assert session.output.endswith("out.wav")  # resolved against the session dir
        assert len(session.tracks) == 1

    def test_json_fallback_when_yaml_fails(self, tmp_path, monkeypatch):
        import builtins
        from audioman.core import session as session_mod

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("yaml disabled for the test")
            return real_import(name, *args, **kwargs)

        track = tmp_path / "a.wav"
        track.write_bytes(b"x")
        f = tmp_path / "session.conf"
        f.write_text(json.dumps({"output": "out.wav", "tracks": [{"path": "a.wav"}]}))

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        session = session_mod.load_session(f)
        assert session.output.endswith("out.wav")
        assert len(session.tracks) == 1

    def test_yaml_extension_without_pyyaml_raises(self, tmp_path, monkeypatch):
        import builtins
        from audioman.core import session as session_mod

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("yaml disabled for the test")
            return real_import(name, *args, **kwargs)

        f = tmp_path / "session.yaml"
        f.write_text("output: out.wav\ntracks: []\n")
        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ImportError, match="pyyaml"):
            session_mod.load_session(f)
