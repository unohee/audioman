# tests/unit/cli_extra/test_schemas_cli.py
# Purpose: cover `audioman schemas {list,show}` end to end (AUD-1851) —
#          the plain/JSON listings, `show` for every published schema, the
#          unknown-name error and the path-traversal rejection.
#
# The published schemas ship inside the package, so the tests enumerate that
# directory and require every file to be reachable through the CLI: a schema
# added without a matching `show` path would be a contract hole.

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


SCHEMA_DIR = Path(__file__).resolve().parents[3] / "src" / "audioman" / "schemas"
ALL_SCHEMAS = sorted(SCHEMA_DIR.glob("*.json"))
SCHEMA_NAMES = [p.stem for p in ALL_SCHEMAS]


def test_fixture_directory_is_populated():
    """Guard: an empty glob would silently turn the parametrized tests into no-ops."""
    assert len(SCHEMA_NAMES) >= 20


class TestList:
    def test_plain_listing_prints_a_row_per_schema(self, run_cli):
        result = run_cli(["schemas", "list"])
        assert result.code == 0, result.stderr

        rows = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(rows) == len(ALL_SCHEMAS)
        assert rows == sorted(rows, key=lambda r: r.split("\t")[0])

    def test_row_columns_are_name_id_and_title(self, run_cli):
        result = run_cli(["schemas", "list"])
        row = next(line for line in result.stdout.splitlines() if line.startswith("observe.v1"))

        name, schema_id, title = row.split("\t")
        assert name == "observe.v1"
        assert schema_id == "audioman://schema/observe.v1.json"
        assert "observe" in title.lower()

    def test_json_envelope_wraps_the_schema_index(self, run_cli):
        result = run_cli(["--json", "schemas", "list"])
        assert result.code == 0, result.stderr

        payload = result.payload
        assert payload["$schema"] == "audioman://schema/schemas.v1.json"
        assert payload["command"] == "schemas"
        assert len(payload["schemas"]) == len(ALL_SCHEMAS)

    def test_json_entries_carry_name_id_title_and_path(self, run_cli):
        payload = run_cli(["--json", "schemas", "list"]).payload
        entry = next(s for s in payload["schemas"] if s["name"] == "finding.v1")

        assert set(entry) == {"name", "id", "title", "path"}
        assert entry["id"] == "audioman://schema/finding.v1.json"
        assert entry["title"]
        assert entry["path"].endswith("schemas/finding.v1.json")

    def test_bare_schemas_behaves_like_list(self, run_cli):
        """`_run_default` forwards to the list action when no subcommand is given."""
        bare = run_cli(["schemas"])
        listed = run_cli(["schemas", "list"])
        assert bare.code == 0, bare.stderr
        assert bare.stdout == listed.stdout


class TestShow:
    @pytest.mark.parametrize("name", SCHEMA_NAMES)
    def test_every_published_schema_is_retrievable(self, run_cli, name):
        result = run_cli(["schemas", "show", name])
        assert result.code == 0, result.stderr

        payload = json.loads(result.stdout)
        assert payload["$id"] == f"audioman://schema/{name}.json"
        assert payload["title"]

    @pytest.mark.parametrize("name", SCHEMA_NAMES)
    def test_shown_body_matches_the_file_on_disk(self, run_cli, name):
        result = run_cli(["schemas", "show", name])
        on_disk = SCHEMA_DIR / f"{name}.json"
        assert json.loads(result.stdout) == json.loads(on_disk.read_text(encoding="utf-8"))

    def test_extension_may_be_omitted(self, run_cli):
        without = run_cli(["schemas", "show", "finding.v1"])
        with_ext = run_cli(["schemas", "show", "finding.v1.json"])
        assert without.code == 0, without.stderr
        assert without.stdout == with_ext.stdout

    def test_bare_name_falls_back_to_a_prefix_match(self, run_cli):
        """`finding` resolves to `finding.v1.json` when that exact file is absent."""
        result = run_cli(["schemas", "show", "finding"])
        assert result.code == 0, result.stderr

        payload = json.loads(result.stdout)
        assert payload["$id"] == "audioman://schema/finding.v1.json"

    def test_output_is_valid_json_regardless_of_the_json_flag(self, run_cli):
        plain = run_cli(["schemas", "show", "observe.v1"])
        json_mode = run_cli(["--json", "schemas", "show", "observe.v1"])
        assert plain.stdout == json_mode.stdout
        assert json.loads(plain.stdout)["$id"] == "audioman://schema/observe.v1.json"

    def test_output_ends_with_exactly_one_newline(self, run_cli):
        result = run_cli(["schemas", "show", "observe.v1"])
        assert result.stdout.endswith("\n")
        assert not result.stdout.endswith("\n\n")

    def test_hyphenated_names_are_accepted(self, run_cli):
        result = run_cli(["schemas", "show", "fader-compare.v1"])
        assert result.code == 0, result.stderr
        assert json.loads(result.stdout)["$id"] == "audioman://schema/fader-compare.v1.json"


class TestShowErrors:
    def test_unknown_name_exits_one(self, run_cli):
        result = run_cli(["schemas", "show", "no-such-schema"])
        assert result.code == 1
        assert "schema not found: no-such-schema" in result.stderr
        assert result.stdout == ""

    def test_unknown_name_with_extension_exits_one(self, run_cli):
        result = run_cli(["schemas", "show", "no-such-schema.json"])
        assert result.code == 1
        assert "schema not found" in result.stderr

    @pytest.mark.parametrize("name", [
        "../secrets",
        "../../etc/passwd",
        "..%2Fpasswd",
        "sub/dir",
        "/etc/passwd",
        "a\\b",
        ".hidden",
        "name with space",
        "",
    ])
    def test_path_traversal_and_illegal_names_are_rejected(self, run_cli, name):
        result = run_cli(["schemas", "show", name])
        assert result.code == 2
        assert "invalid schema name" in result.stderr
        assert result.stdout == ""

    def test_dash_prefixed_name_is_rejected_once_it_reaches_validation(self, run_cli):
        """argparse owns bare `-x`; after `--` the regex validator rejects it."""
        bare = run_cli(["schemas", "show", "-leading-dash"])
        assert bare.code == 2
        assert "required: name" in bare.stderr

        explicit = run_cli(["schemas", "show", "--", "-leading-dash"])
        assert explicit.code == 2
        assert "invalid schema name" in explicit.stderr

    def test_absolute_path_outside_the_schema_dir_is_rejected(self, run_cli, tmp_path):
        secret = tmp_path / "secret.json"
        secret.write_text('{"$id": "leaked"}', encoding="utf-8")

        result = run_cli(["schemas", "show", str(secret)])
        assert result.code == 2
        assert "invalid schema name" in result.stderr
        assert "leaked" not in result.stdout

    def test_traversal_cannot_reach_a_sibling_json(self, run_cli):
        """`../schemas/../schemas/observe.v1.json` must not bypass containment."""
        result = run_cli(["schemas", "show", "../audioman/schemas/observe.v1.json"])
        assert result.code == 2
        assert "invalid schema name" in result.stderr

    def test_missing_name_argument_exits_two(self, run_cli):
        result = run_cli(["schemas", "show"])
        assert result.code == 2
        assert "name" in result.stderr


class TestContractAlignment:
    def test_schema_ids_follow_the_published_uri_pattern(self, run_cli):
        payload = run_cli(["--json", "schemas", "list"]).payload
        pattern = re.compile(r"^audioman://schema/[A-Za-z0-9._-]+\.json$")
        for entry in payload["schemas"]:
            assert pattern.match(entry["id"]), entry

    def test_schema_file_names_match_their_ids(self, run_cli):
        payload = run_cli(["--json", "schemas", "list"]).payload
        for entry in payload["schemas"]:
            assert entry["id"].rsplit("/", 1)[-1] == f"{entry['name']}.json"

    def test_listed_paths_exist_on_disk(self, run_cli):
        payload = run_cli(["--json", "schemas", "list"]).payload
        for entry in payload["schemas"]:
            assert Path(entry["path"]).is_file()
