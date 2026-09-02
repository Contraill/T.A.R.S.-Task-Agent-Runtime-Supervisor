from pathlib import Path
from types import SimpleNamespace
import hashlib

import pytest

from tars import ownership, runtime_routing as routing, state_store, tasks
from tars.runtime_backends import (BackendStatus, LifecycleResult, ModelCapabilities,
                                   RuntimeCapabilities)


@pytest.fixture
def route_state(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index")
    monkeypatch.setattr(tasks, "resolve_role_id", lambda value: value)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"local")
    role = SimpleNamespace(
        id="general", display_name="General", enabled=True, model="local",
        runtime_id="daily", profile="normal", execution="chat",
        capabilities=("conversation", "tools"))
    model = SimpleNamespace(
        alias="local", backend="llama.cpp", path=model_path,
        integrity_verified=True, runtime_compatible=True,
        sha256=hashlib.sha256(b"local").hexdigest())
    monkeypatch.setattr(routing, "resolve_role_id", lambda value: value)
    monkeypatch.setattr(routing, "get_role", lambda value: role)
    monkeypatch.setattr(routing, "get_model", lambda value: model)
    monkeypatch.setattr(routing, "backend_binding_ready", lambda value, cfg=None: True)
    monkeypatch.setattr(routing, "get_profile", lambda *args: SimpleNamespace(context=65536))
    return role, model


class Backend:
    identity = "llama.cpp"
    local_only = True
    zero_idle = True

    def __init__(self, *, available=True, healthy=True, tools=True, reasoning=True,
                 context=65536):
        self.available = available
        self.healthy = healthy
        self.tools = tools
        self.reasoning = reasoning
        self.context = context
        self.lifecycle = []

    def status(self):
        return BackendStatus("llama.cpp", self.available, self.healthy,
                             "healthy" if self.healthy else "runtime offline",
                             "reference-tested")

    def capabilities(self):
        return RuntimeCapabilities(True, self.reasoning, self.tools, False, False, True)

    def model_capabilities(self, model):
        return ModelCapabilities(model, self.context, reasoning=self.reasoning,
                                 tool_calls=self.tools)

    def load(self, model):
        self.lifecycle.append(("load", model))
        return LifecycleResult("llama.cpp", model, "on-demand", True, "first request loads")

    def unload(self, model):
        self.lifecycle.append(("unload", model))
        return LifecycleResult("llama.cpp", model, "ttl-managed", True, "finite TTL")


def test_ready_route_preserves_exact_role_and_lifecycle(route_state):
    backend = Backend()
    router = routing.LocalRuntimeRouter({}, backend_factory=lambda model, cfg: backend)
    route = router.resolve(
        "general", required_capabilities=("conversation",), context_tokens=32000,
        require_tools=True, require_reasoning=True)
    assert route.ready and route.requested_role == route.selected_role == "general"
    assert route.backend == "llama.cpp" and route.model_alias == "local"
    assert route.requested["local_only"] and not route.requested["silent_substitution"]
    with ownership.model_execution_scope(operation="route-lifecycle-test"):
        assert router.prepare(route).managed_on_demand
        assert router.release(route).state == "ttl-managed"
    assert backend.lifecycle == [("load", "daily"), ("unload", "daily")]
    assert routing.load_route(route.id).state == "ready"


def test_runtime_lifecycle_cannot_bypass_model_execution_owner(route_state):
    backend = Backend()
    router = routing.LocalRuntimeRouter({}, backend_factory=lambda model, cfg: backend)
    route = router.resolve("general")
    with pytest.raises(routing.RuntimeRouteUnavailable, match="requires model execution"):
        router.prepare(route)
    with pytest.raises(routing.RuntimeRouteUnavailable, match="requires model execution"):
        router.release(route)
    assert backend.lifecycle == []


def test_unhealthy_or_insufficient_binding_is_explicitly_unavailable(route_state):
    backend = Backend(healthy=False, tools=False, reasoning=False)
    route = routing.LocalRuntimeRouter(
        {}, backend_factory=lambda model, cfg: backend).resolve(
            "general", context_tokens=70000, require_tools=True, require_reasoning=True)
    assert not route.ready
    assert any("runtime offline" in reason for reason in route.reasons)
    assert any("exceeds profile context" in reason for reason in route.reasons)
    assert any("tool-call capability is not verified" in reason for reason in route.reasons)
    with pytest.raises(routing.RuntimeRouteUnavailable):
        route.require_ready()


def test_task_continuity_requires_explicit_handoff(route_state, monkeypatch):
    role, _ = route_state
    task = tasks.create_task("owned", "builder", make_active=False)
    route = routing.LocalRuntimeRouter(
        {}, backend_factory=lambda model, cfg: Backend()).resolve(
            "general", task_id=task.id)
    assert not route.ready
    assert "explicit handoff" in " ".join(route.reasons)
    assert routing.load_route(route.id).task_id == task.id


def test_no_cloud_or_silent_model_substitution(route_state):
    role, model = route_state
    model.backend = "openai"
    backend = Backend()
    backend.identity = "openai"
    backend.local_only = False
    route = routing.LocalRuntimeRouter(
        {}, backend_factory=lambda model, cfg: backend).resolve("general")
    assert not route.ready and route.selected_role == "general"
    assert route.model_alias == "local"
    assert any("not an allowed local runtime" in reason for reason in route.reasons)


def test_invalid_llama_cpp_transport_never_becomes_a_local_route(route_state):
    route = routing.LocalRuntimeRouter(
        {"runtime": {"base_url": "http://192.168.1.20:8080"}},
    ).resolve("general")
    assert not route.ready
    assert any("loopback-local" in reason for reason in route.reasons)


def test_missing_role_semantics_do_not_fall_back(route_state):
    route = routing.LocalRuntimeRouter(
        {}, backend_factory=lambda model, cfg: Backend()).resolve(
            "general", required_capabilities=("code",))
    assert not route.ready and route.selected_role == "general"
    assert route.reasons[0] == "Role lacks capabilities: code"


def test_missing_runtime_identity_is_unavailable(route_state):
    route = routing.LocalRuntimeRouter(
        {}, backend_factory=lambda model, cfg: Backend(context=0)).resolve("general")
    assert not route.ready
    assert any("runtime model daily is absent" in reason for reason in route.reasons)


def test_backend_context_is_an_independent_effective_limit(route_state):
    route = routing.LocalRuntimeRouter(
        {}, backend_factory=lambda model, cfg: Backend(context=32768)).resolve(
            "general", context_tokens=48000)
    assert not route.ready
    assert any("exceeds backend model context 32768" in reason for reason in route.reasons)
    assert not any("exceeds profile context" in reason for reason in route.reasons)
