# Created: 2026-05-11
# Purpose: changelog 파서 + --since 필터 회귀.

import json
import os
import subprocess
import sys

from audioman.cli.changelog_cmd import filter_since, parse_changelog


SAMPLE = """\
# Changelog

## [Unreleased]
### Added
- foo

## [0.2.0] - 2026-05-10
### Added
- new feature X
- another item

### Changed
- behavior Y

## [0.1.0] - 2026-03-26
### Added
- initial release
"""


class TestParser:
    def test_parses_versions(self):
        entries = parse_changelog(SAMPLE)
        versions = [e["version"] for e in entries]
        assert versions == ["Unreleased", "0.2.0", "0.1.0"]

    def test_parses_dates(self):
        entries = parse_changelog(SAMPLE)
        assert entries[1]["date"] == "2026-05-10"
        assert entries[2]["date"] == "2026-03-26"
        assert entries[0]["date"] is None

    def test_parses_section_bullets(self):
        entries = parse_changelog(SAMPLE)
        v020 = entries[1]
        assert "new feature X" in v020["sections"]["added"]
        assert "behavior Y" in v020["sections"]["changed"]


class TestSinceFilter:
    def test_since_filters_older(self):
        entries = parse_changelog(SAMPLE)
        filtered = filter_since(entries, "0.1.0")
        versions = [e["version"] for e in filtered]
        assert "0.2.0" in versions
        assert "Unreleased" in versions
        assert "0.1.0" not in versions


class TestChangelogCommand:
    def test_json_envelope(self):
        env = os.environ.copy()
        result = subprocess.run(
            [sys.executable, "-m", "audioman", "--json", "changelog"],
            env=env,
            capture_output=True,
            text=True,
            cwd="/Users/unohee/dev/audioman",
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["$schema"] == "audioman://schema/changelog.v1.json"
        assert payload["command"] == "changelog"
        assert isinstance(payload["entries"], list)
        assert len(payload["entries"]) >= 1
