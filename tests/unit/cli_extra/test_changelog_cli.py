# tests/unit/cli_extra/test_changelog_cli.py
# Purpose: cover `audioman changelog` end to end (AUD-1851) — the --since
#          filter, the JSON envelope, the missing-file error path and the
#          parser's section handling.
#
# The command targets a real file in the repo by default, so the tests pin
# `--path` at a fixture CHANGELOG to keep the assertions independent of the
# project's own release history. The default lookup order is covered separately
# by driving the command with no --path from a repo-root cwd.

from __future__ import annotations

import json

import pytest

from audioman.cli.changelog_cmd import _version_tuple, filter_since, parse_changelog


SAMPLE = """\
# Changelog

All notable changes to this project are documented here.

## [Unreleased]
### Added
- a pending feature

## [0.2.0] - 2026-05-10
### Added
- new feature X
- another item
  continued on a wrapped line

### Changed
- behavior Y

### Removed

## [0.1.5] - 2026-04-01
### Fixed
- patch level bug

## [0.1.0] - 2026-03-26
### Added
- initial release
"""


@pytest.fixture
def changelog(tmp_path):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(SAMPLE, encoding="utf-8")
    return path


class TestParser:
    def test_versions_are_read_in_file_order(self):
        entries = parse_changelog(SAMPLE)
        assert [e["version"] for e in entries] == ["Unreleased", "0.2.0", "0.1.5", "0.1.0"]

    def test_dates_are_optional(self):
        entries = parse_changelog(SAMPLE)
        by_version = {e["version"]: e["date"] for e in entries}
        assert by_version["Unreleased"] is None
        assert by_version["0.2.0"] == "2026-05-10"
        assert by_version["0.1.0"] == "2026-03-26"

    def test_sections_are_lowercased_and_bucketed(self):
        v020 = next(e for e in parse_changelog(SAMPLE) if e["version"] == "0.2.0")
        assert set(v020["sections"]) == {"added", "changed", "removed"}
        assert v020["sections"]["added"] == [
            "new feature X",
            "another item continued on a wrapped line",
        ]
        assert v020["sections"]["changed"] == ["behavior Y"]

    def test_section_without_bullets_stays_empty(self):
        v020 = next(e for e in parse_changelog(SAMPLE) if e["version"] == "0.2.0")
        assert v020["sections"]["removed"] == []

    def test_text_before_the_first_header_is_ignored(self):
        """The preamble must not be attached to the first version."""
        entries = parse_changelog(SAMPLE)
        assert "All notable changes" not in json.dumps(entries)

    def test_bullet_text_is_stripped(self):
        v010 = next(e for e in parse_changelog(SAMPLE) if e["version"] == "0.1.0")
        assert v010["sections"]["added"] == ["initial release"]

    def test_empty_document_yields_no_entries(self):
        assert parse_changelog("") == []
        assert parse_changelog("# Changelog\n\nnothing here\n") == []

    def test_header_without_brackets_is_not_a_version(self):
        """A plain `## Something` line must not create an entry."""
        assert parse_changelog("## Not a version\n- orphan bullet\n") == []

    def test_bullets_before_any_section_are_dropped(self):
        text = "## [1.0.0] - 2026-01-01\n- floating bullet\n### Added\n- real\n"
        entry = parse_changelog(text)[0]
        assert entry["sections"] == {"added": ["real"]}

    def test_continuation_line_without_a_preceding_bullet_is_dropped(self):
        text = "## [1.0.0] - 2026-01-01\n### Added\n  dangling continuation\n- real\n"
        entry = parse_changelog(text)[0]
        assert entry["sections"]["added"] == ["real"]


class TestVersionTuple:
    def test_unreleased_sorts_above_every_release(self):
        assert _version_tuple("Unreleased") > _version_tuple("99.99.99")
        assert _version_tuple("unreleased") > _version_tuple("0.2.0")

    def test_numeric_parts_compare_numerically(self):
        assert _version_tuple("0.10.0") > _version_tuple("0.9.0")

    def test_prerelease_suffix_is_split_off(self):
        assert _version_tuple("1.0.0-rc1") == (1, 0, 0, "rc1")

    def test_plus_suffix_is_split_off(self):
        assert _version_tuple("1.0.0+build") == (1, 0, 0, "build")


class TestSinceFilter:
    def test_keeps_only_newer_versions(self):
        entries = parse_changelog(SAMPLE)
        kept = [e["version"] for e in filter_since(entries, "0.1.5")]
        assert kept == ["Unreleased", "0.2.0"]

    def test_boundary_version_is_excluded(self):
        entries = parse_changelog(SAMPLE)
        assert "0.2.0" not in [e["version"] for e in filter_since(entries, "0.2.0")]

    def test_since_unreleased_yields_nothing(self):
        assert filter_since(parse_changelog(SAMPLE), "Unreleased") == []

    def test_since_below_every_version_keeps_all(self):
        entries = parse_changelog(SAMPLE)
        assert filter_since(entries, "0.0.1") == entries


class TestJsonOutput:
    def test_envelope_carries_source_and_entries(self, run_cli, changelog):
        result = run_cli(["--json", "changelog", "--path", str(changelog)])
        assert result.code == 0, result.stderr

        payload = result.payload
        assert payload["$schema"] == "audioman://schema/changelog.v1.json"
        assert payload["command"] == "changelog"
        assert payload["source"] == str(changelog)
        assert [e["version"] for e in payload["entries"]] == [
            "Unreleased", "0.2.0", "0.1.5", "0.1.0",
        ]

    def test_since_filters_the_entries_list(self, run_cli, changelog):
        payload = run_cli([
            "--json", "changelog", "--path", str(changelog), "--since", "0.1.5",
        ]).payload
        assert [e["version"] for e in payload["entries"]] == ["Unreleased", "0.2.0"]

    def test_entry_sections_survive_serialisation(self, run_cli, changelog):
        payload = run_cli(["--json", "changelog", "--path", str(changelog)]).payload
        v020 = next(e for e in payload["entries"] if e["version"] == "0.2.0")
        assert v020["date"] == "2026-05-10"
        assert v020["sections"]["added"][1].endswith("wrapped line")


class TestPlainTextOutput:
    def test_versions_and_sections_are_printed(self, run_cli, changelog):
        result = run_cli(["changelog", "--path", str(changelog)])
        assert result.code == 0, result.stderr

        out = result.stdout
        assert "## [Unreleased]" in out
        assert "## [0.2.0] - 2026-05-10" in out
        assert "### added" in out
        assert "- new feature X" in out

    def test_undated_version_prints_without_a_dash(self, run_cli, changelog):
        out = run_cli(["changelog", "--path", str(changelog)]).stdout
        assert "## [Unreleased]\n###" in out or "## [Unreleased]\n" in out
        assert "## [Unreleased] -" not in out

    def test_empty_sections_are_skipped(self, run_cli, changelog):
        out = run_cli(["changelog", "--path", str(changelog)]).stdout
        assert "### removed" not in out

    def test_since_filter_applies_to_the_text_output(self, run_cli, changelog):
        out = run_cli([
            "changelog", "--path", str(changelog), "--since", "0.2.0",
        ]).stdout
        assert "## [Unreleased]" in out
        assert "## [0.2.0]" not in out
        assert "## [0.1.5]" not in out

    def test_since_above_everything_prints_only_separators(self, run_cli, changelog):
        result = run_cli(["changelog", "--path", str(changelog), "--since", "Unreleased"])
        assert result.code == 0
        assert result.stdout.strip() == ""


class TestErrors:
    def test_missing_path_exits_nonzero(self, run_cli, tmp_path):
        result = run_cli(["changelog", "--path", str(tmp_path / "nope.md")])
        assert result.code == 1
        assert "CHANGELOG.md not found" in result.stderr
        assert result.stdout == ""

    def test_missing_path_in_json_mode_still_emits_an_envelope(self, run_cli, tmp_path):
        result = run_cli(["--json", "changelog", "--path", str(tmp_path / "nope.md")])
        assert result.code == 1

        payload = result.payload
        assert payload["$schema"] == "audioman://schema/changelog.v1.json"
        assert payload["command"] == "changelog"
        assert payload["error"] == "CHANGELOG.md not found"
        assert payload["entries"] == []

    def test_directory_path_is_rejected(self, run_cli, tmp_path):
        result = run_cli(["changelog", "--path", str(tmp_path)])
        assert result.code == 1
        assert "CHANGELOG.md not found" in result.stderr


class TestDefaultPathLookup:
    def test_found_from_the_repository_root(self, run_cli):
        """With no --path the command locates the repo CHANGELOG.md."""
        result = run_cli(["--json", "changelog"])
        assert result.code == 0, result.stderr

        payload = result.payload
        assert payload["entries"]
        assert payload["source"].endswith("CHANGELOG.md")

    def test_repo_changelog_parses_into_versions_with_dates(self, run_cli):
        payload = run_cli(["--json", "changelog"]).payload
        dated = [e for e in payload["entries"] if e["date"]]
        assert dated
        assert all(e["sections"] for e in payload["entries"])
