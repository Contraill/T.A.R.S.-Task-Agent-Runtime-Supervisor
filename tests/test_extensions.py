from types import SimpleNamespace

import pytest

from tars import action_journal, extensions, runtime, runtime_routing, state_store
from tars.agent_loop import ToolDispatcher
from tars.runtime_backends import (BackendStatus, LifecycleResult, ModelCapabilities,
                                   RuntimeCapabilities, backend_for_model)
from tars.tool_core import ToolResult, ToolRuntime
from tars.generation import GenerationBudget
from tars.policy import ScopeRequest


class EntryPoint:
    def __init__(self, group, name, provider):
        self.group = group
        self.name = name
        self.value = f"fixture:{name}"
        self.dist = SimpleNamespace(name="fixture-dist")
        self.provider = provider
        self.loads = 0

    def load(self):
        self.loads += 1
        return self.provider


class RuntimeBackend:
    identity = "fixture"
    local_only = True
    zero_idle = True

    def __init__(self):
        self.calls = []

    def status(self):
        return BackendStatus(self.identity, True, True, "healthy", "third-party")

    def capabilities(self):
        return RuntimeCapabilities(True, None, None, True, True, True)

    def model_capabilities(self, model):
        return ModelCapabilities(model, 4096)

    def complete(self, request):
        self.calls.append("complete")
        return {"choices": []}

    def stream(self, request):
        return iter(())

    def diagnostics(self):
        return {"backend": self.identity}

    def count_tokens(self, model, messages, *, thinking_mode=None):
        return 10

    def load(self, model):
        self.calls.append("load")
        return LifecycleResult(self.identity, model, "loaded", True, "loaded")

    def unload(self, model):
        self.calls.append("unload")
        return LifecycleResult(self.identity, model, "unloaded", True, "unloaded")


class Provider:
    api_version = 1

    def __init__(self, kind, name, create):
        self.kind = kind
        self.name = name
        self.create = create


def config(identifier):
    return {"extensions": {"enabled": [identifier], "trusted": [identifier]}}


def test_discovery_is_metadata_only_and_marks_third_party_provenance():
    point = EntryPoint("tars.runtime_backends", "fixture",
                       Provider("runtime_backend", "fixture", lambda **kwargs: RuntimeBackend()))
    loader = extensions.ExtensionLoader({}, entry_points=[point])
    descriptor = next(item for item in loader.discover() if item.name == "fixture")
    assert descriptor.provenance == "third-party" and not descriptor.enabled
    assert point.loads == 0


def test_runtime_backend_real_factory_path_requires_enable_and_trust(monkeypatch):
    point = EntryPoint("tars.runtime_backends", "fixture",
                       Provider("runtime_backend", "fixture", lambda **kwargs: RuntimeBackend()))
    monkeypatch.setattr(extensions.metadata, "entry_points", lambda: [point])
    model = SimpleNamespace(backend="fixture")
    with pytest.raises(PermissionError, match="not enabled"):
        backend_for_model(model, {})
    assert point.loads == 0
    value = backend_for_model(model, config("runtime_backend:fixture"))
    assert value.identity == "fixture" and point.loads == 1


def test_runtime_extension_cannot_claim_a_nonlocal_backend(monkeypatch):
    backend = RuntimeBackend()
    backend.local_only = False
    point = EntryPoint("tars.runtime_backends", "fixture",
                       Provider("runtime_backend", "fixture", lambda **kwargs: backend))
    monkeypatch.setattr(extensions.metadata, "entry_points", lambda: [point])
    with pytest.raises(PermissionError, match="local-only"):
        backend_for_model(SimpleNamespace(backend="fixture"),
                          config("runtime_backend:fixture"))


def test_real_inference_path_uses_trusted_runtime_extension(monkeypatch, tmp_path):
    backend = RuntimeBackend()
    point = EntryPoint("tars.runtime_backends", "fixture",
                       Provider("runtime_backend", "fixture", lambda **kwargs: backend))
    monkeypatch.setattr(extensions.metadata, "entry_points", lambda: [point])
    artifact = tmp_path / "fixture.model"
    artifact.write_bytes(b"fixture")
    role = SimpleNamespace(
        id="general", enabled=True, model="fixture-model", runtime_id="fixture-runtime",
        profile="normal", display_name="General", execution="chat",
        capabilities=("conversation",))
    model = SimpleNamespace(
        alias="fixture-model", backend="fixture", path=artifact, integrity_verified=True,
        runtime_compatible=True, thinking_control="unknown")
    monkeypatch.setattr(runtime_routing, "resolve_role_id", lambda value: "general")
    monkeypatch.setattr(runtime_routing, "get_role", lambda value: role)
    monkeypatch.setattr(runtime_routing, "get_model", lambda value: model)
    monkeypatch.setattr(runtime_routing, "get_profile", lambda *args: SimpleNamespace(context=4096))
    monkeypatch.setattr(runtime, "get_role", lambda value: role)
    monkeypatch.setattr(runtime, "get_model", lambda value: model)
    monkeypatch.setattr(runtime, "generation_budget", lambda *args, **kwargs: GenerationBudget(
        "general", 4096, 512, 128, 3456, None))
    response = runtime.chat_completion(
        config("runtime_backend:fixture"), "general",
        [{"role": "user", "content": "hello"}])
    assert response["choices"] == []
    assert backend.calls == ["load", "complete", "unload"]


def test_tool_extension_executes_through_authoritative_policy_runtime():
    executed = []

    class Tool:
        name = "ext.fixture.write"
        retry_safe = False

        def scope_requests(self, arguments):
            return (("target", ScopeRequest(
                self.name, "write", arguments["target"], task_id=arguments.get("task_id"))),)

        def execute(self, target, task_id=None):
            executed.append((target, task_id))
            return ToolResult(self.name, "succeeded", {"target": target})

    provider = Provider("tool", "fixture", lambda: Tool())
    point = EntryPoint("tars.tools", "fixture", provider)
    loader = extensions.ExtensionLoader(
        config("tool:fixture"), entry_points=[point])

    class Runtime:
        def __init__(self, deny=False):
            self.deny = deny
            self.authorized = []
            self.finished = []

        def authorize(self, requests, approval_ids=None):
            self.authorized.append((requests, approval_ids))
            if self.deny:
                raise PermissionError("denied")
            return [SimpleNamespace(id="action-one")]

        def finish(self, actions, *, state, result):
            self.finished.append((state, result))

    runtime = Runtime()
    dispatcher = ToolDispatcher().register_extension("fixture", loader, runtime=runtime)
    result = dispatcher.execute(
        "ext.fixture.write", {"target": "/work", "approval_ids": {"target": "approval-one"}},
        task_id="task-one")
    assert executed == [("/work", "task-one")] and result.action_ids == ("action-one",)
    request = runtime.authorized[0][0][0][1]
    assert request.target == "/work" and request.task_id == "task-one"
    assert runtime.authorized[0][1] == {"target": "approval-one"}
    assert runtime.finished[0][0] == "succeeded"

    denied = Runtime(deny=True)
    blocked = ToolDispatcher().register_extension("fixture", loader, runtime=denied)
    with pytest.raises(PermissionError, match="denied"):
        blocked.execute("ext.fixture.write", {"target": "/blocked"})
    assert executed == [("/work", "task-one")]


def test_tool_extension_real_policy_and_action_journal_path(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index")
    target = tmp_path / "allowed.txt"
    target.write_text("truth")

    class Tool:
        name = "ext.fixture.read"

        def scope_requests(self, arguments):
            return (("target", ScopeRequest(
                self.name, "read", arguments["target"], task_id=arguments.get("task_id"),
                allowed_paths=(str(tmp_path),))),)

        def execute(self, target):
            return ToolResult(self.name, "succeeded", {"content": target})

    loader = extensions.ExtensionLoader(
        config("tool:fixture"), entry_points=[EntryPoint(
            "tars.tools", "fixture", Provider("tool", "fixture", lambda: Tool()))])
    dispatcher = ToolDispatcher().register_extension(
        "fixture", loader, runtime=ToolRuntime())
    result = dispatcher.execute("ext.fixture.read", {"target": str(target)})
    assert result.succeeded and len(result.action_ids) == 1
    action = action_journal.load_action(result.action_ids[0])
    assert action.state == "succeeded" and action.tool == "ext.fixture.read"


def test_extension_api_and_identity_are_validated_before_creation():
    provider = Provider("tool", "fixture", lambda: object())
    provider.api_version = 99
    loader = extensions.ExtensionLoader(
        config("tool:fixture"),
        entry_points=[EntryPoint("tars.tools", "fixture", provider)])
    with pytest.raises(RuntimeError, match="API mismatch"):
        loader.load("tool", "fixture")
