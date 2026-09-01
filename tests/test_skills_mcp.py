import json
from types import SimpleNamespace

import pytest

from tars import mcp, policy, prompt_compiler, skills, state_store
from tars.approvals import ApprovalBroker
from tars.tool_core import ToolResult, ToolRuntime


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
    def __init__(self, *, secret_values=(), tools=None):
        self.calls = []
        self.secret_values = secret_values
        self.tools = tools or [
            {"name": "echo", "description": "Echo", "inputSchema": {"type": "object"}},
            {"name": "hidden", "inputSchema": {"type": "object"}},
        ]

    def request(self, payload):
        self.calls.append(payload)
        if payload["method"] == "initialize":
            result = {"protocolVersion": mcp.PROTOCOL_VERSION}
        elif payload["method"] == "tools/list":
            result = {"tools": self.tools}
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


def test_mcp_path_authority_is_derived_from_sent_arguments(isolated):
    root = isolated / "allowed"
    root.mkdir()
    inside = root / "document.txt"
    outside = isolated / "outside.txt"
    outside.write_text("private")
    symlink = root / "link"
    symlink.symlink_to(outside)
    schema = {
        "type": "object",
        "properties": {
            "request": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        "required": ["request"],
        "additionalProperties": False,
    }
    server = mcp.register(
        "files", "stdio", {"argv": ["fixture"]},
        tool_filter={"include": ["write_file"]},
        effect_policy={"write_file": {"scopes": [{
            "effect": "write", "target": "/request/path", "target_kind": "path",
            "allowed_paths": [str(root)],
        }]}},
    )
    transport = FakeTransport(tools=[{
        "name": "write_file", "description": "write", "inputSchema": schema,
    }])
    policy.add_rule("write", "allow", target=str(root), target_kind="path")
    client = mcp.MCPClient(server, transport=transport)

    with pytest.raises(PermissionError, match="outside authorized filesystem scope"):
        client.call_tool("write_file", {
            "request": {"path": str(outside), "content": "same-size"},
        })
    with pytest.raises(PermissionError, match="outside authorized filesystem scope"):
        client.call_tool("write_file", {
            "request": {"path": str(symlink), "content": "same-size"},
        })
    assert not any(call["method"] == "tools/call" for call in transport.calls)

    result = client.call_tool("write_file", {
        "request": {"path": str(inside), "content": "same-size"},
    })
    assert result.succeeded
    sent = transport.calls[-1]["params"]
    assert sent == {"name": "write_file", "arguments": {
        "request": {"path": str(inside), "content": "same-size"},
    }}
    conn = state_store.connect()
    try:
        action = conn.execute(
            "SELECT target,normalized_arguments_json FROM action_journal "
            "WHERE tool='mcp.files.write_file' AND state='succeeded'"
        ).fetchone()
    finally:
        conn.close()
    arguments = json.loads(action["normalized_arguments_json"])
    assert action["target"] == str(inside.resolve())
    assert arguments["remote_arguments"] == sent["arguments"]
    assert len(arguments["server_identity_sha256"]) == 64
    assert len(arguments["tool_contract_sha256"]) == 64


def test_mcp_call_has_no_parallel_caller_claimed_target(isolated):
    server = register_fixture_server()
    transport = FakeTransport()
    client = mcp.MCPClient(server, transport=transport)
    with pytest.raises(TypeError, match="unexpected keyword argument 'target'"):
        client.call_tool("echo", {"path": "/outside"}, target="/allowed")
    assert transport.calls == []


def test_mcp_compound_nested_and_array_targets_are_all_authorized(isolated):
    root = isolated / "workspace"
    root.mkdir()
    first = root / "a.txt"
    second = root / "sub" / "b.txt"
    server = mcp.register(
        "compound", "stdio", {"argv": ["fixture"]},
        tool_filter={"include": ["fetch_many"]},
        effect_policy={"fetch_many": {"scopes": [
            {"name": "outputs", "effect": "write", "target": "/outputs",
             "target_kind": "path", "allowed_paths": [str(root)]},
            {"name": "source", "effect": "network", "target": "/request/url",
             "target_kind": "network", "allowed_hosts": ["https://api.example.com:8443"]},
        ]}},
    )
    transport = FakeTransport(tools=[{
        "name": "fetch_many", "inputSchema": {
            "type": "object",
            "properties": {
                "outputs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "request": {"type": "object", "properties": {
                    "url": {"type": "string"}}, "required": ["url"]},
            },
            "required": ["outputs", "request"],
        },
    }])
    client = mcp.MCPClient(server, transport=transport)
    prepared = client.prepare_call("fetch_many", {
        "outputs": [str(first), str(second)],
        "request": {"url": "https://api.example.com:8443/items?id=1"},
    })
    assert [key for key, _ in prepared.requests] == ["outputs:0", "outputs:1", "source"]
    decisions = [policy.ScopeGuard().evaluate(request) for _, request in prepared.requests]
    assert [item.target for item in decisions] == [
        str(first.resolve()), str(second.resolve()),
        "https://api.example.com:8443/items?id=1",
    ]
    assert [item.effect for item in decisions] == ["write", "write", "network"]
    policy.add_rule("write", "allow", target=str(root), target_kind="path")
    policy.add_rule("network", "allow", target="api.example.com")
    result = client.call_tool("fetch_many", {
        "outputs": [str(first), str(second)],
        "request": {"url": "https://api.example.com:8443/items?id=1"},
    })
    assert result.succeeded and len(result.action_ids) == 3
    assert sum(call["method"] == "tools/call" for call in transport.calls) == 1
    for changed in (
        "http://api.example.com:8443/items",
        "https://api.example.com:9443/items",
        "https://other.example.com:8443/items",
    ):
        with pytest.raises(PermissionError, match="configured origin scope"):
            client.prepare_call("fetch_many", {
                "outputs": [str(first)], "request": {"url": changed},
            })


def test_mcp_schema_and_missing_semantic_target_fail_before_remote_call(isolated):
    root = isolated / "root"
    root.mkdir()
    server = mcp.register(
        "validated", "stdio", {"argv": ["fixture"]},
        tool_filter={"include": ["write_file"]},
        effect_policy={"write_file": {"scopes": [{
            "effect": "write", "target": "/path", "target_kind": "path",
            "allowed_paths": [str(root)],
        }]}},
    )
    transport = FakeTransport(tools=[{
        "name": "write_file", "inputSchema": {
            "type": "object", "properties": {"path": {"type": "string"}},
            "required": ["path"], "additionalProperties": False,
        },
    }])
    client = mcp.MCPClient(server, transport=transport)
    with pytest.raises(ValueError, match="missing path"):
        client.call_tool("write_file", {})
    with pytest.raises(ValueError, match="not permitted"):
        client.call_tool("write_file", {"path": str(root / "x"), "other": True})
    assert not any(call["method"] == "tools/call" for call in transport.calls)

    malformed_transport = FakeTransport(tools=[{
        "name": "write_file", "inputSchema": {
            "type": "object", "not": {"$ref": "#/$defs/forbidden"},
        },
    }])
    malformed = mcp.MCPClient(server, transport=malformed_transport)
    with pytest.raises(ValueError, match="unsupported MCP input schema keyword"):
        malformed.call_tool("write_file", {"path": str(root / "x")})
    assert not any(call["method"] == "tools/call" for call in malformed_transport.calls)


def test_mcp_trusted_contract_overrides_remote_read_hint(isolated):
    server = mcp.register(
        "hints", "stdio", {"argv": ["fixture"]},
        tool_filter={"include": ["erase"]},
        effect_policy={"erase": "destructive"},
    )
    transport = FakeTransport(tools=[{
        "name": "erase", "inputSchema": {"type": "object"},
        "annotations": {"readOnlyHint": True},
    }])
    client = mcp.MCPClient(server, transport=transport)
    prepared = client.prepare_call("erase", {"object": "important"})
    request = prepared.requests[0][1]
    decision = policy.ScopeGuard().evaluate(request)
    assert request.destructive is True
    assert decision.effect == "destructive" and decision.action == "ask"


def test_mcp_persistent_approval_is_invalidated_by_server_or_schema_change(isolated):
    schema = {"type": "object", "properties": {"value": {"type": "string"}},
              "required": ["value"], "additionalProperties": False}
    server = mcp.register(
        "revision", "stdio", {"argv": ["fixture-v1"]},
        tool_filter={"include": ["mutate"]}, effect_policy={"mutate": "write"},
    )
    first_transport = FakeTransport(tools=[{"name": "mutate", "inputSchema": schema}])
    first = mcp.MCPClient(server, transport=first_transport)
    prepared = first.prepare_call("mutate", {"value": "alpha"})
    request = prepared.requests[0][1]
    decision = policy.ScopeGuard().evaluate(request)
    approval = ApprovalBroker().request(request, decision, scope="persistent")
    ApprovalBroker().decide(approval.id, approve=True)
    assert first.call_tool("mutate", {"value": "alpha"}).succeeded
    with pytest.raises(PermissionError, match="approved authorization"):
        first.call_tool("mutate", {"value": "bravo"})
    first_transport.tools = [{"name": "mutate", "inputSchema": schema | {
        "description": "changed after approval"}}]
    with pytest.raises(PermissionError, match="approved authorization"):
        first.call_tool("mutate", {"value": "alpha"})
    first_transport.tools = [{"name": "mutate", "inputSchema": schema}]

    changed_server = mcp.register(
        "revision", "stdio", {"argv": ["fixture-v2"]},
        tool_filter={"include": ["mutate"]}, effect_policy={"mutate": "write"},
    )
    changed_transport = FakeTransport(tools=[{"name": "mutate", "inputSchema": schema}])
    changed = mcp.MCPClient(changed_server, transport=changed_transport)
    with pytest.raises(PermissionError, match="approved authorization"):
        changed.call_tool("mutate", {"value": "alpha"})
    assert not any(call["method"] == "tools/call" for call in changed_transport.calls)

    schema_transport = FakeTransport(tools=[{"name": "mutate", "inputSchema": schema | {
        "description": "new contract revision"}}])
    schema_changed = mcp.MCPClient(server, transport=schema_transport)
    with pytest.raises(PermissionError, match="approved authorization"):
        schema_changed.call_tool("mutate", {"value": "alpha"})
    assert not any(call["method"] == "tools/call" for call in schema_transport.calls)


def test_mcp_authorized_argument_snapshot_is_the_one_sent(isolated):
    root = isolated / "snapshot"
    root.mkdir()
    inside = root / "inside"
    outside = isolated / "outside"
    arguments = {"path": str(inside), "content": "approved"}

    class MutatingRuntime(ToolRuntime):
        def authorize(self, requests, approval_ids=None):
            actions = super().authorize(requests, approval_ids)
            arguments["path"] = str(outside)
            arguments["content"] = "changed!"
            return actions

    server = mcp.register(
        "snapshot", "stdio", {"argv": ["fixture"]},
        tool_filter={"include": ["write"]}, effect_policy={"write": {"scopes": [{
            "effect": "write", "target": "/path", "target_kind": "path",
            "allowed_paths": [str(root)],
        }]}},
    )
    policy.add_rule("write", "allow", target=str(root), target_kind="path")
    transport = FakeTransport(tools=[{"name": "write", "inputSchema": {
        "type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"], "additionalProperties": False,
    }}])
    client = mcp.MCPClient(server, transport=transport, runtime=MutatingRuntime())
    assert client.call_tool("write", arguments).succeeded
    assert arguments == {"path": str(outside), "content": "changed!"}
    assert transport.calls[-1]["params"]["arguments"] == {
        "path": str(inside), "content": "approved",
    }


def test_mcp_malformed_authority_contracts_fail_closed_at_registration(isolated):
    root = isolated / "root"
    with pytest.raises(ValueError, match="requires allowed_paths"):
        mcp.register("bad-path", "stdio", {"argv": ["x"]}, effect_policy={
            "write": {"scopes": [{"effect": "write", "target": "/path",
                                    "target_kind": "path"}]},
        })
    with pytest.raises(ValueError, match="requires network effect"):
        mcp.register("bad-network", "stdio", {"argv": ["x"]}, effect_policy={
            "fetch": {"scopes": [{"effect": "write", "target": "/url",
                                    "target_kind": "network",
                                    "allowed_hosts": ["https://example.com"]}]},
        })
    with pytest.raises(ValueError, match="unknown MCP authority scope field"):
        mcp.register("bad-field", "stdio", {"argv": ["x"]}, effect_policy={
            "write": {"scopes": [{"effect": "write", "target_claim": str(root)}]},
        })


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


def test_mcp_connection_authority_and_transport_share_immutable_config(isolated, monkeypatch):
    record = mcp.MCPServerRecord(
        "immutable", "stdio",
        {"argv": ["helper", "--mode", "safe"], "cwd": str(isolated),
         "env": {"TOKEN": "env:MCP_TOKEN"}},
        True, {}, {"echo": "read"}, "created", "updated",
    )
    captured = {}

    class Runtime:
        def authorize(self, requests, approvals):
            captured["request"] = requests[0][1]
            record.config["argv"][2] = "unsafe"
            record.config["cwd"] = "/outside"
            return [SimpleNamespace(id="action", event_uuid="event")]

        def finish(self, *args, **kwargs):
            pass

    class Transport:
        def __init__(self, config, **kwargs):
            captured["config"] = config

        def close(self):
            pass

    monkeypatch.setattr(mcp, "StdioTransport", Transport)
    client = mcp.MCPClient(record, runtime=Runtime())
    client.close()
    request = captured["request"]
    assert request.target == "helper"
    assert request.arguments["config"] == {
        "argv": ["helper", "--mode", "safe"], "cwd": str(isolated),
        "env": {"TOKEN": "env:MCP_TOKEN"},
    }
    assert captured["config"] == request.arguments["config"]
    assert len(request.arguments["server_identity_sha256"]) == 64


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


def test_controlled_mcp_server_uses_same_argument_derived_scope_contract(isolated):
    root = isolated / "controlled"
    root.mkdir()
    calls = []
    policy.add_rule("write", "allow", target=str(root), target_kind="path")
    server = mcp.ControlledMCPServer().register(
        "write", "Write", {
            "type": "object", "properties": {"path": {"type": "string"}},
            "required": ["path"], "additionalProperties": False,
        },
        lambda arguments: calls.append(arguments) or ToolResult("write", "succeeded", {}),
        effect="write", target_arg="path", allowed_paths=(str(root),),
    )
    outside = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "write", "arguments": {"path": str(isolated / "outside")}},
    })
    assert "error" in outside and calls == []
    inside_path = root / "inside"
    inside = server.handle({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "write", "arguments": {"path": str(inside_path)}},
    })
    assert inside["result"]["isError"] is False
    assert calls == [{"path": str(inside_path)}]


def test_controlled_mcp_server_rejects_unbound_scope_metadata(isolated):
    with pytest.raises(ValueError, match="require a target argument"):
        mcp.ControlledMCPServer().register(
            "write", "Write", {"type": "object"},
            lambda arguments: ToolResult("write", "succeeded", {}),
            effect="write", allowed_paths=(str(isolated),),
        )
    with pytest.raises(ValueError, match="host scope requires a network effect"):
        mcp.ControlledMCPServer().register(
            "write", "Write", {"type": "object"},
            lambda arguments: ToolResult("write", "succeeded", {}),
            effect="write", target_arg="url", allowed_hosts=("https://example.com",),
        )
