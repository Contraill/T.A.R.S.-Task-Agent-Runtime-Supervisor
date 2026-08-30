from tars.cli import build_parser
from tars.tool_registry import ToolDescriptor, ToolRegistry


def test_native_tool_precedes_trusted_mcp_and_terminal_fallback():
    registry = ToolRegistry((ToolDescriptor("fs.read", "filesystem.read", "read"),))
    selected = registry.select(
        "filesystem.read", mcp_tools=({"name": "remote.read", "capability": "filesystem.read",
                                       "trusted": True},),
    )
    assert selected == {"kind": "native", "tool": "fs.read"}
    assert registry.select(
        "search.custom", mcp_tools=({"name": "mcp.search", "capability": "search.custom",
                                     "trusted": True},),
    ) == {"kind": "mcp", "tool": "mcp.search"}
    assert registry.select("unknown") == {"kind": "terminal", "tool": "terminal.run"}


def test_untrusted_mcp_is_not_selected_over_terminal():
    registry = ToolRegistry(())
    selected = registry.select(
        "remote.mutate", mcp_tools=({"name": "unsafe", "capability": "remote.mutate",
                                     "trusted": False},),
    )
    assert selected["kind"] == "terminal"


def test_tool_and_evidence_cli_parsing():
    parser = build_parser()
    assert parser.parse_args(["tool", "list"]).tool_command == "list"
    evidence = parser.parse_args(["evidence", "--task", "task-one"])
    assert evidence.task == "task-one"
