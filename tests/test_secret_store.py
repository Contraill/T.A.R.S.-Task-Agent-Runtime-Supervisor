import json
import subprocess
from types import SimpleNamespace

import pytest

from tars.core_client import CoreClient
from tars.execution_backends import (ExecutionRequest, HostBackend, SSHBackend,
                                     SSHExecutionTarget, _ExecutionAuthorization)
from tars.mcp import MCPServerRecord, MCPClient
from tars.secret_store import SecretStore, parse_reference
from tars.web_research import TavilyResearch


class Provider:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def get(self, key):
        self.calls.append(key)
        return self.values[key]


def store(values, *, scopes=None):
    return SecretStore({"fixture": Provider(values)}, scopes=scopes)


def test_references_are_scoped_and_values_never_appear_in_reference():
    secrets = store({"token": "raw-secret"}, scopes={
        "fixture:token": ["core:client"]})
    with secrets.resolve("fixture:token", consumer="core:client") as value:
        assert value == "raw-secret"
    with pytest.raises(PermissionError, match="not scoped"):
        with secrets.resolve("fixture:token", consumer="web:tavily"):
            pass
    assert parse_reference("fixture:token").value == "fixture:token"
    assert "raw-secret" not in repr(secrets)
    with pytest.raises(ValueError, match="secret references"):
        ExecutionRequest(("true",), environment_refs={"TOKEN": "raw-secret"})
    with pytest.raises(ValueError, match="secret references"):
        SSHExecutionTarget("bad", "example.com", "user", credential_ref="raw-secret")


def test_configured_scopes_feed_the_store_used_by_consumers(monkeypatch):
    monkeypatch.setenv("SCOPED_TOKEN", "value")
    secrets = SecretStore.from_config({"secrets": {"scopes": {
        "env:SCOPED_TOKEN": ["mcp:allowed"]}}})
    with secrets.resolve("env:SCOPED_TOKEN", consumer="mcp:allowed") as value:
        assert value == "value"
    with pytest.raises(PermissionError):
        with secrets.resolve("env:SCOPED_TOKEN", consumer="mcp:denied"):
            pass


def test_real_host_and_ssh_execution_resolve_at_backend_boundary(tmp_path):
    secrets = store({"env": "resolved", "identity": "/keys/id"})
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, kwargs["env"].get("VALUE", ""), "")

    host = HostBackend(runner=runner, secret_store=secrets)
    result = host.execute(ExecutionRequest(("true",), cwd=str(tmp_path),
                          environment_refs={"VALUE": "fixture:env"}),
                          authorization=_ExecutionAuthorization(("action",)))
    assert result.stdout == "resolved"

    ssh_calls = []
    ssh = SSHBackend((SSHExecutionTarget(
        "lab", "example.com", "operator", credential_ref="fixture:identity",
        allowed_commands=("id",)),), ssh_binary="/usr/bin/ssh",
        runner=lambda argv, **kwargs: ssh_calls.append(argv) or subprocess.CompletedProcess(
            argv, 0, "", ""), secret_store=secrets)
    ssh.execute(ExecutionRequest(("id",), target="lab"),
                authorization=_ExecutionAuthorization(("action",)))
    assert ssh_calls[0][ssh_calls[0].index("-i") + 1] == "/keys/id"


def test_real_mcp_transport_receives_store_and_scoped_consumer():
    record = MCPServerRecord("fixture", "stdio", {"argv": ["fixture"]}, True,
                             {}, {}, "now", "now")
    captured = {}

    class Transport:
        def __init__(self, config, **kwargs):
            captured.update(kwargs)
        def close(self): pass

    runtime = SimpleNamespace(
        authorize=lambda *args, **kwargs: [SimpleNamespace(id="a")],
        finish=lambda *args, **kwargs: None)
    secrets = store({"x": "y"})
    import tars.mcp as mcp
    original = mcp.StdioTransport
    mcp.StdioTransport = Transport
    try:
        MCPClient(record, runtime=runtime, secret_store=secrets).close()
    finally:
        mcp.StdioTransport = original
    assert captured == {"secret_store": secrets, "consumer": "mcp:fixture"}


def test_web_and_core_consumers_resolve_only_at_request_boundary():
    secrets = store({"web": "web-secret", "core": "client.secret"})

    class HTTP:
        def request(self, method, url, **kwargs):
            assert b"web-secret" in kwargs["body"]
            return SimpleNamespace(succeeded=True, data={"content": "{}"},
                                   action_ids=(), evidence_ids=())

    assert TavilyResearch(secret_ref="fixture:web", http=HTTP(),
                          secret_store=secrets).search("truth").succeeded

    captured = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({"ok": True}).encode()
    def transport(request, timeout=None):
        captured["authorization"] = request.headers["Authorization"]
        return Response()
    client = CoreClient("http://127.0.0.1", token_ref="fixture:core",
                        secret_store=secrets, transport=transport)
    assert client.status() == {"ok": True}
    assert captured["authorization"] == "Bearer client.secret"
