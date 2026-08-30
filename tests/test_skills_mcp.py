import json
from types import SimpleNamespace

import pytest

from tars import mcp, prompt_compiler, skills, state_store
from tars.tool_core import ToolResult


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    return tmp_path


def write_skill(root, name, *, description="A useful skill", version="1.0.0",
                body="Follow this procedure.", resources=None):
    folder = root / name
    folder.mkdir(parents=True)
    resource_line = f"resources: {json.dumps(resources)}\n" if resources else ""
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\nversion: {version}\n"
        f"{resource_line}---\n{body}\n")
    return folder


def test_skill_progressive_disclosure_and_scope_override(isolated, monkeypatch):
    global_root = isolated / "global"
    project = isolated / "project"
    write_skill(global_root, "review", body="global private instructions")
    write_skill(project / ".tars" / "skills", "review", version="2.0.0",
                body="project private instructions")
    registry = skills.SkillRegistry(global_root=global_root)
    discovered = registry.discover(project_path=project)
    assert discovered[0].scope == "project" and discovered[0].version == "2.0.0"
    summaries = skills.prompt_summaries(registry, project_path=project)
    assert "project private instructions" not in summaries[0]
    loaded = registry.load("review", project_path=project)
    assert loaded.instructions == "project private instructions"


def test_prompt_compiler_discloses_skill_body_only_when_selected(isolated, monkeypatch):
    root = isolated / "skills"
    write_skill(root, "review", body="private procedure")
    registry = skills.SkillRegistry(global_root=root)
    monkeypatch.setattr(prompt_compiler, "resolve_role_id", lambda value: "general")
    monkeypatch.setattr(prompt_compiler, "get_role", lambda value: SimpleNamespace(
        display_name="General", description="", capabilities=()))
    monkeypatch.setattr(prompt_compiler, "load_identity", lambda *args, **kwargs: SimpleNamespace(
        identity="identity", soul="soul", role_overlay="", sources=()))
    compiler = prompt_compiler.PromptCompiler()
    summary = compiler.compile(role_name="general", skill_registry=registry)
    assert "private procedure" not in summary.messages[0]["content"]
    selected = compiler.compile(role_name="general", skill_registry=registry,
                                selected_skills=("review",))
    assert "private procedure" in selected.messages[0]["content"]
    assert "never authorization" in selected.messages[0]["content"]


def test_skill_doctor_rejects_invalid_and_symlink_escape(isolated):
    root = isolated / "skills"
    write_skill(root, "broken", description="")
    outside = write_skill(isolated / "outside", "escape")
    root.mkdir(exist_ok=True)
    (root / "escape").symlink_to(outside, target_is_directory=True)
    report = skills.SkillRegistry(global_root=root).doctor()
    assert not report["ok"] and {item["name"] for item in report["invalid"]} == {"broken", "escape"}


class FakeTransport:
    def __init__(self, *, secret_values=()):
        self.calls = []
        self.secret_values = secret_values

    def request(self, payload):
        self.calls.append(payload)
        if payload["method"] == "initialize":
            result = {"protocolVersion": mcp.PROTOCOL_VERSION}
        elif payload["method"] == "tools/list":
            result = {"tools": [
                {"name": "echo", "description": "Echo", "inputSchema": {"type": "object"}},
                {"name": "hidden", "inputSchema": {"type": "object"}},
            ]}
        else:
            result = {"content": [{"type": "text", "text": "token-secret"}], "isError": False}
        return {"jsonrpc": "2.0", "id": payload["id"], "result": result}

    def close(self):
        pass


def register_fixture_server():
    return mcp.register(
        "fixture", "stdio", {"argv": ["fixture-server"]},
        tool_filter={"include": ["echo"]}, effect_policy={"echo": "read"})


def test_mcp_discovery_filter_policy_audit_and_secret_redaction(isolated):
    server = register_fixture_server()
    transport = FakeTransport(secret_values=("token-secret",))
    client = mcp.MCPClient(server, transport=transport)
    tools = client.discover_tools()
    assert [tool["remote_name"] for tool in tools] == ["echo"]
    assert tools[0]["name"].startswith("mcp.fixture.")
    assert tools[0]["trusted"] is False
    assert "inputSchema" not in client.tool_summaries()[0]
    assert client.tool_schema("echo")["inputSchema"]["type"] == "object"
    result = client.call_tool("echo", {"text": "hello"})
    assert result.succeeded and result.data["content"][0]["text"] == "[REDACTED]"
    conn = state_store.connect()
    try:
        action = conn.execute("SELECT * FROM action_journal ORDER BY created_at DESC").fetchone()
    finally:
        conn.close()
    assert action["tool"] == "mcp.fixture.echo" and action["state"] == "succeeded"
    assert transport.calls[-1]["method"] == "tools/call"


def test_unclassified_mcp_tool_is_denied_without_execution(isolated):
    server = mcp.register("fixture", "stdio", {"argv": ["fixture"]},
                          tool_filter={"include": ["echo"]})
    transport = FakeTransport()
    client = mcp.MCPClient(server, transport=transport)
    with pytest.raises(PermissionError):
        client.call_tool("echo", {})
    assert not any(call["method"] == "tools/call" for call in transport.calls)


def test_mcp_registry_enable_and_secret_reference_validation(isolated):
    server = register_fixture_server()
    assert mcp.set_enabled(server.name, False).enabled is False
    with pytest.raises(RuntimeError, match="disabled"):
        mcp.MCPClient("fixture", transport=FakeTransport())
    with pytest.raises(ValueError, match="secret references"):
        mcp.register("bad", "stdio", {"argv": ["x"], "env": {"TOKEN": "plaintext"}})
    with pytest.raises(ValueError, match="credentials"):
        mcp.register("bad-argv", "stdio", {"argv": ["x", "--token=plaintext"]})
    with pytest.raises(ValueError, match="loopback"):
        mcp.register("private-http", "streamable-http", {"url": "http://127.0.0.1:9000/mcp"})


def test_controlled_mcp_server_applies_scopeguard_and_requires_real_result(isolated):
    server = mcp.ControlledMCPServer().register(
        "health", "Health", {"type": "object"},
        lambda arguments: ToolResult("system.info", "succeeded", {"ok": True}),
        effect="read")
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert listed["result"]["tools"][0]["name"] == "health"
    called = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {"name": "health", "arguments": {}}})
    assert called["result"]["isError"] is False
    unsafe = mcp.ControlledMCPServer().register(
        "unsafe", "Unsafe", {"type": "object"},
        lambda arguments: ToolResult("unsafe", "succeeded", {}), effect="elevated")
    denied = unsafe.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                            "params": {"name": "unsafe", "arguments": {}}})
    assert "error" in denied
