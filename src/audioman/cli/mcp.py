# Created: 2026-11-04
# Purpose: `audioman mcp` — a Model Context Protocol server over stdio.
#
# Implements the subset of MCP an agent needs to call tools: JSON-RPC 2.0 messages,
# one per line on stdin/stdout (the stdio transport), and the methods initialize,
# notifications/initialized, ping, tools/list and tools/call. It is written against
# the specification directly instead of the `mcp` SDK, whose web-server dependencies
# (starlette, uvicorn, pydantic, ...) an audio CLI has no other use for.
#
# stdout carries protocol messages only. Anything a tool prints is redirected to
# stderr while it runs, so stray output can never corrupt the stream.

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import Any, TextIO

from audioman.cli.mcp_tools import TOOLS, ToolError, validate_arguments

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST = SUPPORTED_VERSIONS[0]
# resource_link content and structuredContent arrived in 2025-06-18
_RICH_RESULTS = {"2025-06-18"}

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603

INSTRUCTIONS = (
    "audioman measures audio and returns evidence, not verdicts on taste. Each tool returns a short digest "
    "(read it first), a result_ref file with the complete JSON, and a size-bounded summary. Texture values "
    "carry floors: differences below a floor are measurement noise. Paths are local files."
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("mcp", help="Run a Model Context Protocol server on stdio (for LLM agents)")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    McpServer().serve(sys.stdin, sys.stdout)


class McpServer:
    def __init__(self) -> None:
        self.protocol_version: str | None = None

    # -- transport -----------------------------------------------------------

    def serve(self, reader: TextIO, writer: TextIO) -> None:
        """Read one JSON-RPC message per line until EOF; write one response per request."""
        for line in reader:
            line = line.strip()
            if not line:
                continue
            response = self.handle_line(line)
            if response is not None:
                writer.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
                writer.flush()

    def handle_line(self, line: str) -> dict | None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            return _error(None, PARSE_ERROR, f"parse error: {exc}")
        return self.handle(message)

    # -- dispatch ------------------------------------------------------------

    def handle(self, message: Any) -> dict | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return _error(message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST,
                          "expected a JSON-RPC 2.0 request object")
        method, params = message["method"], message.get("params") or {}
        is_request = "id" in message
        if not is_request:
            return None  # notifications (initialized, cancelled, ...) need no answer
        rid = message["id"]
        handler = {
            "initialize": self._initialize,
            "ping": lambda p: {},
            "tools/list": self._tools_list,
            "tools/call": self._tools_call,
        }.get(method)
        if handler is None:
            return _error(rid, METHOD_NOT_FOUND, f"method not found: {method}")
        if not isinstance(params, dict):
            return _error(rid, INVALID_PARAMS, "params must be an object")
        try:
            return {"jsonrpc": "2.0", "id": rid, "result": handler(params)}
        except _ParamsError as exc:
            return _error(rid, INVALID_PARAMS, str(exc))

    def _initialize(self, params: dict) -> dict:
        requested = params.get("protocolVersion")
        self.protocol_version = requested if requested in SUPPORTED_VERSIONS else LATEST
        from audioman import __version__

        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "audioman", "title": "audioman", "version": __version__},
            "instructions": INSTRUCTIONS,
        }

    def _tools_list(self, params: dict) -> dict:
        return {"tools": [
            {"name": name, "title": spec["title"], "description": spec["description"],
             "inputSchema": spec["inputSchema"]}
            for name, spec in TOOLS.items()
        ]}

    def _tools_call(self, params: dict) -> dict:
        name = params.get("name")
        if name not in TOOLS:
            raise _ParamsError(f"unknown tool: {name}")
        arguments = params.get("arguments") or {}
        problem = validate_arguments(name, arguments)
        if problem:
            raise _ParamsError(problem)
        try:
            # tools never write protocol output; anything they print goes to stderr
            with contextlib.redirect_stdout(sys.stderr):
                result = TOOLS[name]["run"](arguments)
        except ToolError as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        except Exception as exc:  # a failing tool is reported to the model, not a protocol error
            return {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}
        return self._tool_result(result)

    def _tool_result(self, result: dict) -> dict:
        refs = [("result", result["result_ref"], "application/json")]
        if result.get("figure"):
            refs.append(("figure", result["figure"], "image/png"))
        content: list[dict] = [{"type": "text", "text": result["digest"]}]
        if self.protocol_version in _RICH_RESULTS:
            content += [{"type": "resource_link", "uri": f"file://{path}", "name": label, "mimeType": mime}
                        for label, path, mime in refs]
            return {"content": content, "structuredContent": {**result["summary"], "result_ref": result["result_ref"]},
                    "isError": False}
        # older clients: references and the summary travel as text
        content.append({"type": "text", "text": "\n".join(f"{label}: {path}" for label, path, _ in refs)})
        content.append({"type": "text", "text": json.dumps(result["summary"], ensure_ascii=False, default=str)})
        return {"content": content, "isError": False}


class _ParamsError(Exception):
    pass


def _error(rid: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}
