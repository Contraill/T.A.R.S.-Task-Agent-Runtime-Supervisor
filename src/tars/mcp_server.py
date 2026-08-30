from __future__ import annotations

import sys

from .mcp import ControlledMCPServer
from .state_store import health
from .tool_core import ToolResult


def _state_health(arguments):
    report = health()
    return ToolResult("state.health", "succeeded" if report["ok"] else "failed", report)


def build_server():
    return ControlledMCPServer().register(
        "state_health", "Inspect canonical T.A.R.S. state-store integrity and counts.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        _state_health, effect="read")


def main():
    build_server().serve_stdio(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
