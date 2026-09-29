# tests/unit/test_session_security.py — session file path containment + subtype allowlist

import json
from pathlib import Path

import pytest

from audioman.core.session import (
    ALLOWED_SUBTYPES,
    SessionPathError,
    _parse_track,
    load_session,
    validate_subtype,
)


def _write_session(path, tracks, output="out.wav", extra=None):
    data = {"output": output, "tracks": tracks}
    if extra:
        data.update(extra)
    path.write_text(json.dumps(data))
    return path


class TestTrackPathContainment:
    """Track paths must stay inside the session directory"""

    def test_relative_escape_rejected(self, tmp_path):
        session_dir = tmp_path / "project"
        session_dir.mkdir()
        session = _write_session(
            session_dir / "s.json", [{"path": "../../../etc/passwd"}]
        )
        with pytest.raises(SessionPathError, match="escapes the session directory"):
            load_session(session)

    def test_absolute_outside_rejected(self, tmp_path):
        session = _write_session(tmp_path / "s.json", [{"path": "/etc/passwd"}])
        with pytest.raises(SessionPathError, match="escapes the session directory"):
            load_session(session)

    def test_sibling_escape_rejected(self, tmp_path):
        session_dir = tmp_path / "project"
        session_dir.mkdir()
        session = _write_session(
            session_dir / "s.json", [{"path": "../outside.wav"}]
        )
        with pytest.raises(SessionPathError):
            load_session(session)

    def test_inner_relative_path_accepted(self, tmp_path):
        session_dir = tmp_path / "project"
        (session_dir / "stems").mkdir(parents=True)
        (session_dir / "stems" / "vocal.wav").touch()
        session = _write_session(
            session_dir / "s.json", [{"path": "stems/vocal.wav"}]
        )
        config = load_session(session)
        assert config.tracks[0].path == str(session_dir / "stems" / "vocal.wav")

    def test_inner_dotdot_that_stays_inside_accepted(self, tmp_path):
        session_dir = tmp_path / "project"
        (session_dir / "stems").mkdir(parents=True)
        (session_dir / "vocal.wav").touch()
        session = _write_session(
            session_dir / "s.json", [{"path": "stems/../vocal.wav"}]
        )
        config = load_session(session)
        assert config.tracks[0].path == str(session_dir / "vocal.wav")

    def test_absolute_inside_accepted(self, tmp_path):
        session_dir = tmp_path / "project"
        session_dir.mkdir()
        target = session_dir / "vocal.wav"
        target.touch()
        session = _write_session(session_dir / "s.json", [{"path": str(target)}])
        config = load_session(session)
        assert config.tracks[0].path == str(target)

    def test_symlink_escape_rejected(self, tmp_path):
        session_dir = tmp_path / "project"
        session_dir.mkdir()
        secret = tmp_path / "secret.wav"
        secret.touch()
        link = session_dir / "link.wav"
        try:
            link.symlink_to(secret)
        except OSError:
            pytest.skip("symlinks unavailable")
        session = _write_session(session_dir / "s.json", [{"path": "link.wav"}])
        with pytest.raises(SessionPathError):
            load_session(session)

    def test_parse_track_direct_escape(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        with pytest.raises(SessionPathError):
            _parse_track({"path": "../../etc/passwd"}, base)

    def test_error_is_value_error(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        with pytest.raises(ValueError):
            _parse_track({"path": "/etc/passwd"}, base)

    def test_relative_base_dir_still_accepted(self, tmp_path, monkeypatch):
        """A relative session path (CLI usage) must not be blocked"""
        (tmp_path / "vocal.wav").touch()
        (tmp_path / "s.json").write_text(
            json.dumps({"output": "out.wav", "tracks": [{"path": "vocal.wav"}]})
        )
        monkeypatch.chdir(tmp_path)
        config = load_session("s.json")
        assert config.tracks[0].path == str(tmp_path / "vocal.wav")
        assert config.output == str(tmp_path / "out.wav")

    def test_relative_path_input_object(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        (base / "t.wav").touch()
        track = _parse_track({"path": Path("t.wav")}, base)
        assert track.path == str(base / "t.wav")


class TestOutputPathContainment:
    """The output path may only be written inside the session directory"""

    def test_relative_escape_rejected(self, tmp_path):
        session_dir = tmp_path / "project"
        session_dir.mkdir()
        (session_dir / "vocal.wav").touch()
        session = _write_session(
            session_dir / "s.json",
            [{"path": "vocal.wav"}],
            output="../../../../tmp/escaped.wav",
        )
        with pytest.raises(SessionPathError, match="escapes the session directory"):
            load_session(session)

    def test_absolute_outside_rejected(self, tmp_path):
        (tmp_path / "vocal.wav").touch()
        session = _write_session(
            tmp_path / "s.json", [{"path": "vocal.wav"}], output="/etc/passwd"
        )
        with pytest.raises(SessionPathError):
            load_session(session)

    def test_inner_output_accepted(self, tmp_path):
        session_dir = tmp_path / "project"
        (session_dir / "renders").mkdir(parents=True)
        (session_dir / "vocal.wav").touch()
        session = _write_session(
            session_dir / "s.json",
            [{"path": "vocal.wav"}],
            output="renders/mix.wav",
        )
        config = load_session(session)
        assert config.output == str(session_dir / "renders" / "mix.wav")


class TestSubtypeAllowlist:
    """Session subtypes outside the allowlist are not passed through"""

    @pytest.mark.parametrize("subtype", sorted(ALLOWED_SUBTYPES))
    def test_allowed_subtypes_accepted(self, tmp_path, subtype):
        (tmp_path / "vocal.wav").touch()
        session = _write_session(
            tmp_path / "s.json", [{"path": "vocal.wav"}], extra={"format": subtype}
        )
        config = load_session(session)
        assert config.subtype == subtype

    def test_arbitrary_subtype_rejected(self, tmp_path):
        (tmp_path / "vocal.wav").touch()
        session = _write_session(
            tmp_path / "s.json",
            [{"path": "vocal.wav"}],
            extra={"format": "PCM_24; rm -rf /"},
        )
        with pytest.raises(ValueError, match="Invalid audio subtype"):
            load_session(session)

    def test_unknown_soundfile_subtype_rejected(self, tmp_path):
        """Subtypes soundfile knows but this project never writes are rejected"""
        (tmp_path / "vocal.wav").touch()
        session = _write_session(
            tmp_path / "s.json", [{"path": "vocal.wav"}], extra={"format": "MPEG_LAYER_III"}
        )
        with pytest.raises(ValueError, match="Invalid audio subtype"):
            load_session(session)

    def test_non_string_subtype_rejected(self, tmp_path):
        (tmp_path / "vocal.wav").touch()
        session = _write_session(
            tmp_path / "s.json", [{"path": "vocal.wav"}], extra={"format": 24.0}
        )
        with pytest.raises(ValueError, match="Invalid audio subtype"):
            load_session(session)

    def test_subtype_key_fallback(self, tmp_path):
        (tmp_path / "vocal.wav").touch()
        session = _write_session(
            tmp_path / "s.json", [{"path": "vocal.wav"}], extra={"subtype": "PCM_16"}
        )
        config = load_session(session)
        assert config.subtype == "PCM_16"

    def test_default_subtype_when_absent(self, tmp_path):
        (tmp_path / "vocal.wav").touch()
        session = _write_session(tmp_path / "s.json", [{"path": "vocal.wav"}])
        assert load_session(session).subtype == "PCM_24"

    def test_case_normalized(self):
        assert validate_subtype("pcm_16") == "PCM_16"
        assert validate_subtype("float") == "FLOAT"

    def test_validate_subtype_rejects_garbage(self):
        with pytest.raises(ValueError, match="Invalid audio subtype"):
            validate_subtype("nonsense")

    def test_explicit_null_format_rejected(self, tmp_path):
        """`format:` with no value must not fall through to the default"""
        (tmp_path / "vocal.wav").touch()
        session = _write_session(
            tmp_path / "s.json", [{"path": "vocal.wav"}], extra={"format": None}
        )
        with pytest.raises(ValueError, match="Invalid audio subtype"):
            load_session(session)

    def test_whitespace_padded_subtype_accepted(self, tmp_path):
        (tmp_path / "vocal.wav").touch()
        session = _write_session(
            tmp_path / "s.json", [{"path": "vocal.wav"}], extra={"format": " PCM_24 "}
        )
        assert load_session(session).subtype == "PCM_24"
