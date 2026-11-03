"""`audioman mcp`: MCP over stdio, driven by an in-process client (and once by a real subprocess)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal as ss

from audioman.cli import mcp_tools
from audioman.cli.mcp import LATEST, SUPPORTED_VERSIONS, McpServer

SR = 48000


class Client:
    """Minimal MCP client: speaks newline-delimited JSON-RPC to a server in this process."""

    def __init__(self, server: McpServer | None = None):
        self.server = server or McpServer()
        self._id = 0

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            msg["params"] = params
        response = self.server.handle_line(json.dumps(msg))
        assert response["id"] == self._id and response["jsonrpc"] == "2.0"
        return response

    def notify(self, method: str) -> None:
        assert self.server.handle_line(json.dumps({"jsonrpc": "2.0", "method": method})) is None

    def initialize(self, version: str = LATEST) -> dict:
        result = self.request("initialize", {"protocolVersion": version, "capabilities": {},
                                             "clientInfo": {"name": "test", "version": "0"}})["result"]
        self.notify("notifications/initialized")
        return result

    def call(self, name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    from audioman.config import settings as settings_module

    monkeypatch.setenv("AUDIOMAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_module.reset_settings()
    yield
    settings_module.reset_settings()


def _pink(seed: int, n: int = SR * 3) -> np.ndarray:
    b, a = [0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400]
    x = ss.lfilter(b, a, np.random.default_rng(seed).standard_normal(n))
    return 0.5 * x / np.max(np.abs(x))


@pytest.fixture
def wavs(tmp_path) -> dict[str, str]:
    x = np.stack([_pink(0), _pink(1)])
    dark = ss.sosfilt(ss.butter(6, 2000, fs=SR, output="sos"), x, axis=-1)
    saturated = np.tanh(3 * x) / 3  # memoryless: processed without delay
    files = {"a": x, "b": dark, "same": x.copy(), "sat": saturated, "silent": np.zeros_like(x)}
    out = {}
    for name, audio in files.items():
        path = tmp_path / f"{name}.wav"
        sf.write(path, audio.T.astype(np.float32), SR, subtype="FLOAT")
        out[name] = str(path)
    low = tmp_path / "low.wav"
    sf.write(low, x.T.astype(np.float32), 44100, subtype="FLOAT")
    out["low"] = str(low)
    return out


def _texts(result: dict) -> list[str]:
    return [c["text"] for c in result["content"] if c["type"] == "text"]


def _links(result: dict) -> dict[str, str]:
    return {c["name"]: c["uri"] for c in result["content"] if c["type"] == "resource_link"}


def _open_link(uri: str):
    assert uri.startswith("file://")
    path = Path(uri[len("file://"):])
    assert path.is_file()
    return path


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def test_initialize_negotiates_the_protocol_version():
    for version in SUPPORTED_VERSIONS:
        assert Client().initialize(version)["protocolVersion"] == version
    result = Client().initialize("1999-01-01")
    assert result["protocolVersion"] == LATEST
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert result["serverInfo"]["name"] == "audioman" and "digest" in result["instructions"]


def test_ping_and_tools_list():
    client = Client()
    client.initialize()
    assert client.request("ping")["result"] == {}
    tools = {t["name"]: t for t in client.request("tools/list")["result"]["tools"]}
    assert set(tools) == {"analyze", "describe", "diff", "uat_compare"}
    for tool in tools.values():
        assert tool["inputSchema"]["type"] == "object" and tool["description"] and tool["title"]


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


def test_analyze_returns_digest_file_reference_and_bounded_summary(wavs):
    client = Client()
    client.initialize()
    result = client.call("analyze", {"path": wavs["a"], "texture": True})["result"]
    assert result["isError"] is False
    digest = _texts(result)[0]
    assert digest.startswith("0:03.0, stereo, 48 kHz") or digest.startswith("3.0 s, stereo, 48 kHz")
    full = json.loads(_open_link(_links(result)["result"]).read_text())
    assert full["texture"]["descriptors"]["spectral_centroid"]["value"] is not None  # complete data on disk
    structured = result["structuredContent"]
    assert structured["result_ref"] == _links(result)["result"][len("file://"):]
    assert structured["report"]["command"] == "analyze"
    assert len(json.dumps(structured, indent=2).encode()) <= 4096 + 512  # summary level (+ result_ref)


def test_describe_reports_floors(wavs):
    client = Client()
    client.initialize()
    result = client.call("describe", {"path": wavs["a"], "segment_sec": 1.5, "detail": "standard"})["result"]
    assert "spectral_centroid" in _texts(result)[0] and "(+/-" in _texts(result)[0]
    full = json.loads(_open_link(_links(result)["result"]).read_text())
    assert full["segment_sec"] == 1.5 and full["descriptors"]["spectral_centroid"]["floor"] is not None
    silent = _texts(client.call("describe", {"path": wavs["silent"]})["result"])[0]
    assert "spectral_centroid" not in silent and silent.endswith("onset_density 0 1/s.")
    from audioman.core.digest import digest_texture

    assert digest_texture({"descriptors": {}}).endswith("No defined descriptors (silent input).")


def test_diff_reports_significant_change_and_a_figure(wavs):
    client = Client()
    client.initialize()
    result = client.call("diff", {"a": wavs["a"], "b": wavs["b"], "png": True})["result"]
    assert "significant descriptor change" in _texts(result)[0]
    links = _links(result)
    assert _open_link(links["figure"]).suffix == ".png"
    assert json.loads(_open_link(links["result"]).read_text())["changes"]
    same = client.call("diff", {"a": wavs["a"], "b": wavs["same"]})["result"]
    assert _texts(same)[0].endswith("no significant change.")


def test_uat_compare_gates(wavs, tmp_path):
    client = Client()
    client.initialize()
    a = wavs["a"]
    ok = client.call("uat_compare", {"source": a, "offline": wavs["sat"], "realtime": wavs["sat"],
                                      "bypassed": a, "render_seconds": 0.5})["result"]
    assert _texts(ok)[0].startswith("uat compare: PASS")
    assert ok["structuredContent"]["report"]["verdict"] == "PASS"
    # a minimum-phase filter delays the signal: the PDC gate notices
    delayed = client.call("uat_compare", {"source": a, "offline": wavs["b"], "realtime": wavs["b"]})["result"]
    assert "delay_compensated FAIL" in _texts(delayed)[0]
    dropped = sf.read(wavs["sat"], always_2d=True)[0]
    dropped[SR: SR + 2048] = 0.0
    bad_path = tmp_path / "dropped.wav"
    sf.write(bad_path, dropped, SR, subtype="FLOAT")
    bad = client.call("uat_compare", {"source": a, "offline": wavs["sat"], "realtime": str(bad_path)})["result"]
    assert bad["isError"] is False  # a failing gate is a result, not a tool error
    assert "no_dropout_blocks FAIL" in _texts(bad)[0]


def test_older_protocol_versions_get_text_only_results(wavs):
    client = Client()
    client.initialize("2024-11-05")
    result = client.call("diff", {"a": wavs["a"], "b": wavs["b"], "png": True})["result"]
    assert {c["type"] for c in result["content"]} == {"text"}
    assert "structuredContent" not in result
    refs = _texts(result)[1]
    assert refs.startswith("result: /") and "figure: /" in refs
    assert "changes" in json.loads(_texts(result)[2])


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,code",
    [
        ("{not json", -32700),
        (json.dumps({"jsonrpc": "1.0", "id": 1, "method": "ping"}), -32600),
        (json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]), -32600),
        (json.dumps({"jsonrpc": "2.0", "id": 1, "method": "resources/list"}), -32601),
        (json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": [1]}), -32602),
        (json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "nope"}}), -32602),
    ],
)
def test_protocol_errors(line, code):
    response = McpServer().handle_line(line)
    assert response["error"]["code"] == code


@pytest.mark.parametrize(
    "name,arguments,fragment",
    [
        ("analyze", {}, "missing required argument(s): path"),
        ("analyze", {"path": "x", "colour": True}, "unknown argument(s): colour"),
        ("analyze", {"path": 3}, "'path' must be a string"),
        ("analyze", {"path": "x", "texture": "yes"}, "'texture' must be a boolean"),
        ("describe", {"path": "x", "segment_sec": True}, "'segment_sec' must be a number"),
        ("diff", {"a": "x", "b": "y", "detail": "huge"}, "must be one of"),
        ("analyze", [1], "arguments must be an object"),
    ],
)
def test_invalid_arguments_are_protocol_errors(name, arguments, fragment):
    client = Client()
    client.initialize()
    response = client.call(name, arguments)
    assert response["error"]["code"] == -32602 and fragment in response["error"]["message"]


def test_tool_failures_are_results_with_is_error(wavs, monkeypatch):
    client = Client()
    client.initialize()
    missing = client.call("analyze", {"path": "/no/such.wav"})["result"]
    assert missing["isError"] and "file not found" in _texts(missing)[0]
    rate = client.call("diff", {"a": wavs["a"], "b": wavs["low"]})["result"]
    assert rate["isError"] and "sample rates differ" in _texts(rate)[0]
    uat_rate = client.call("uat_compare", {"source": wavs["a"], "offline": wavs["low"], "realtime": wavs["a"]})
    assert uat_rate["result"]["isError"] and "44100 Hz" in _texts(uat_rate["result"])[0]

    junk = Path(wavs["a"]).with_name("junk.wav")
    junk.write_bytes(b"not audio")
    unreadable = client.call("describe", {"path": str(junk)})["result"]
    assert unreadable["isError"] and "cannot read" in _texts(unreadable)[0]

    def boom(arguments):
        raise ValueError("internal")

    monkeypatch.setitem(mcp_tools.TOOLS["describe"], "run", boom)
    crashed = client.call("describe", {"path": wavs["a"]})["result"]
    assert crashed["isError"] and _texts(crashed)[0] == "ValueError: internal"


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------


def test_serve_keeps_stdout_protocol_only(wavs, monkeypatch):
    def noisy(arguments):
        print("a tool printing to stdout")
        return {"digest": "d", "result_ref": "/tmp/r.json", "summary": {}}

    monkeypatch.setitem(mcp_tools.TOOLS["describe"], "run", noisy)
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": LATEST}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "",
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "describe", "arguments": {"path": wavs["a"]}}}),
    ]
    out = io.StringIO()
    McpServer().serve(io.StringIO("\n".join(lines) + "\n"), out)
    responses = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in responses] == [1, 2]  # one line per request, nothing else
    assert responses[1]["result"]["content"][0]["text"] == "d"


def test_cli_entry_runs_the_server(monkeypatch):
    from audioman.cli import app

    request = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"}) + "\n"
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(request))
    monkeypatch.setattr(sys, "stdout", out)
    app.main(["mcp"])
    assert json.loads(out.getvalue()) == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_real_subprocess_round_trip(wavs, tmp_path):
    """The server as an agent host runs it: a child process on real stdio."""
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": LATEST}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "diff", "arguments": {"a": wavs["a"], "b": wavs["same"]}}},
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "audioman", "mcp"],
        input="".join(json.dumps(m) + "\n" for m in messages),
        capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "AUDIOMAN_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert proc.returncode == 0, proc.stderr
    responses = [json.loads(line) for line in proc.stdout.splitlines()]
    assert [r["id"] for r in responses] == [1, 2, 3]
    assert len(responses[1]["result"]["tools"]) == 4
    assert responses[2]["result"]["content"][0]["text"].endswith("no significant change.")
