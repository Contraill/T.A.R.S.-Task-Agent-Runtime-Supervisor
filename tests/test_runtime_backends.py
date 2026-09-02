from types import SimpleNamespace
import hashlib
import multiprocessing
from pathlib import Path
import threading

import pytest

from tars import runtime
from tars import runtime_backends as backends
from tars import runtime_routing as routing
from tars import state_store
from tars import ownership
from tars.generation import GenerationBudget


def _hold_inference_slot(database, scratch, ready, release):
    state_store.STATE_DB_PATH = Path(database)
    state_store.TASK_ROOT = Path(scratch) / "legacy"
    state_store.TASK_EVENTS_ROOT = Path(scratch) / "events"
    state_store.TASK_INDEX_PATH = Path(scratch) / "index"
    route = SimpleNamespace(backend="llama.cpp", runtime_id="fixture")
    class Router:
        def prepare(self, value):
            ready.set()
        def release(self, value):
            pass
    with runtime._inference_lifecycle(Router(), route):
        release.wait(10)


class FakeTransport:
    def __init__(self):
        self.calls = []

    def json(self, method, url, *, payload=None, timeout=30):
        self.calls.append((method, url, payload))
        if method == "POST":
            return {"choices": [{"message": {"content": "done"}}]}
        return {"data": [{
            "id": "daily", "context_length": 65536,
            "capabilities": {"function_calling": True},
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        }]}

    def sse(self, url, *, payload, timeout=1200):
        self.calls.append(("SSE", url, payload))
        yield {"choices": [{"delta": {"reasoning_content": "think"}}]}
        yield {"choices": [{"delta": {"content": "answer", "tool_calls": [{"id": "one"}]},
                             "finish_reason": "stop"}], "usage": {"total_tokens": 3}}


@pytest.fixture(autouse=True)
def isolated_runtime_state(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")


def _cfg():
    return {"runtime": {"base_url": "http://127.0.0.1:8080"}}


def test_llama_cpp_backend_contract(monkeypatch, tmp_path):
    transport = FakeTransport()
    binary = tmp_path / "llama-server"
    binary.write_text("binary")
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fixture-model")
    model = SimpleNamespace(
        alias="fixture", path=model_path,
        sha256=hashlib.sha256(b"fixture-model").hexdigest(),
        thinking_control="unknown",
    )
    monkeypatch.setattr(backends, "LLAMA_SERVER_PATH", binary)
    backend = backends.LlamaCppBackend(
        _cfg(), transport=transport, model_record=model)
    assert backend.status().healthy
    assert backend.capabilities().on_demand
    caps = backend.model_capabilities("daily")
    assert caps.context == 65536 and caps.tool_calls
    assert backend.load("daily").managed_on_demand
    assert backend.unload("daily").state == "ttl-managed"
    request = backends.InferenceRequest("daily", ({"role": "user", "content": "hi"},))
    assert backend.complete(request)["choices"]
    events = list(backend.stream(request))
    assert events[0].reasoning == "think"
    assert events[1].content == "answer" and events[1].tool_calls[0]["id"] == "one"


@pytest.mark.parametrize("base_url", [
    "https://example.com",
    "http://192.168.1.10:8080",
    "http://runtime.internal:8080",
    "http://127.0.0.1:8080/v1",
    "http://user:secret@127.0.0.1:8080",
    "http://127.0.0.1:8080?token=secret",
    "http://127.0.0.1:bad",
])
def test_llama_cpp_local_only_rejects_nonlocal_or_nonorigin_transport(base_url):
    with pytest.raises(ValueError, match="runtime base_url"):
        backends.LlamaCppBackend({"runtime": {"base_url": base_url}})


def test_local_runtime_origins_are_literal_and_dns_free():
    ipv4 = backends.LlamaCppBackend(
        {"runtime": {"base_url": "http://127.0.0.1:8080"}},
        transport=FakeTransport(),
    )
    ipv6 = backends.LlamaCppBackend(
        {"runtime": {"base_url": "http://[::1]:8081"}},
        transport=FakeTransport(),
    )
    localhost = backends.LlamaCppBackend(
        {"runtime": {"base_url": "http://localhost:8082"}},
        transport=FakeTransport(),
    )
    assert ipv4.base_url == "http://127.0.0.1:8080"
    assert ipv6.base_url == "http://[::1]:8081"
    assert localhost.base_url == "http://127.0.0.1:8082"


def test_direct_runtime_helpers_share_local_transport_validation(monkeypatch):
    monkeypatch.setattr(
        runtime, "get_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("remote transport was reached")),
    )
    with pytest.raises(ValueError, match="loopback-local"):
        runtime.runtime_models(
            {"runtime": {"base_url": "https://models.example.com"}})


def test_llama_cpp_dispatch_rechecks_current_artifact_bytes(tmp_path):
    payload = b"verified-model"
    path = tmp_path / "model.gguf"
    path.write_bytes(payload)
    model = SimpleNamespace(
        alias="fixture", path=path, sha256=hashlib.sha256(payload).hexdigest(),
        thinking_control="unknown",
    )
    transport = FakeTransport()
    backend = backends.LlamaCppBackend(
        _cfg(), transport=transport, model_record=model)
    request = backends.InferenceRequest(
        "daily", ({"role": "user", "content": "hi"},))
    with ownership.model_execution_scope(operation="artifact-recheck-test"):
        assert backend.complete(request)["choices"]
        path.write_bytes(b"same-path-different-model")
        with pytest.raises(RuntimeError, match="no longer match"):
            backend.complete(request)
    assert [call[0] for call in transport.calls] == ["POST"]


def test_colibri_unconfigured_is_a_supported_truthful_state():
    backend = backends.ColibriBackend({})
    status = backend.status()
    assert not status.available and backend.support == "supported-optional"
    assert "not configured" in status.message
    assert backend.model_capabilities("oracle").context == 0
    with pytest.raises(backends.BackendUnavailable):
        backend.complete(backends.InferenceRequest("oracle", ()))


class ColibriTransport:
    def __init__(self):
        self.calls = []

    def json(self, method, url, *, payload=None, timeout=30):
        self.calls.append((method, url, payload))
        if url.endswith("/v1/health"):
            return {"status": "ok", "version": "1.2.3"}
        if url.endswith("/v1/capabilities"):
            return {"streaming": True, "reasoning": True, "tool_calls": False,
                    "explicit_load": True, "explicit_unload": True, "on_demand": True,
                    "thinking_modes": ["on"]}
        if url.endswith("/v1/models/oracle"):
            return {"context_length": 131072, "reasoning": True,
                    "input_modalities": ["text"], "output_modalities": ["text"]}
        if url.endswith("/load"):
            return {"state": "loaded", "message": "loaded on demand"}
        if url.endswith("/unload"):
            return {"state": "unloaded", "message": "released"}
        if url.endswith("/v1/tokenize"):
            return {"tokens": 42}
        if url.endswith("/v1/chat/completions"):
            return {"choices": [{"message": {"content": "answer"}}]}
        raise AssertionError(url)

    def sse(self, url, *, payload, timeout=1200):
        self.calls.append(("SSE", url, payload))
        yield {"choices": [{"delta": {"reasoning_content": "reason"}}]}
        yield {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]}


def test_colibri_probes_lifecycle_context_and_reasoning_stream_without_real_weights():
    transport = ColibriTransport()
    backend = backends.ColibriBackend(
        {"colibri": {"base_url": "http://127.0.0.1:9988", "ttl_seconds": 45}},
        transport=transport)
    assert backend.status().healthy
    assert backend.capabilities().on_demand
    assert backend.model_capabilities("oracle").context == 131072
    assert not any(call[1].endswith("/load") for call in transport.calls)
    assert backend.count_tokens("oracle", ({"role": "user", "content": "hi"},)) == 42
    assert backend.load("oracle").state == "loaded"
    assert backend.unload("oracle").state == "unloaded"
    request = backends.InferenceRequest("oracle", ({"role": "user", "content": "hi"},),
                                        thinking_mode="on")
    assert backend.complete(request)["choices"]
    events = list(backend.stream(request))
    assert events[0].reasoning == "reason" and events[1].content == "answer"
    assert backend.diagnostics()["ttl_seconds"] == 45


def test_colibri_rejects_nonlocal_endpoint_and_clamps_heavy_ttl():
    backend = backends.ColibriBackend(
        {"colibri": {"base_url": "https://example.com", "ttl_seconds": 3600}})
    assert not backend.status().available
    assert "loopback-local" in backend.status().message
    assert backend.ttl_seconds == 300
    assert backend.diagnostics()["configured"]
    local = backends.ColibriBackend(
        {"colibri": {"base_url": "http://127.0.0.1:9988"}})
    assert isinstance(local.transport, backends.UrllibTransport)
    malformed = backends.ColibriBackend(
        {"colibri": {"base_url": "http://127.0.0.1:9988", "ttl_seconds": "bad"}})
    assert "must be an integer" in malformed.status().message


def test_backend_factory_rejects_cloud_provider_names():
    for name in ("openai", "openai-compatible"):
        with pytest.raises(KeyError, match="unknown runtime backend"):
            backends.backend_for_name(name, {})


def test_runtime_dispatches_role_through_model_backend(monkeypatch):
    role = SimpleNamespace(id="general", enabled=True, model="local", runtime_id="daily",
                           display_name="General", execution="chat")
    model = SimpleNamespace(alias="local", backend="llama.cpp", thinking_control="unknown")

    class Backend:
        def complete(self, request):
            return {"model": request.model, "messages": request.messages,
                    "max_tokens": request.max_tokens}

        def stream(self, request):
            yield backends.StreamEvent(content="chunk")

    backend = Backend()
    calls = []

    class Router:
        def __init__(self, cfg):
            calls.append(("init", cfg))

        def resolve(self, requested_role, **kwargs):
            calls.append(("resolve", requested_role, kwargs))
            return SimpleNamespace(
                selected_role="general", model_alias="local", backend_instance=backend,
                model_capabilities={"context": 100},
                require_ready=lambda: calls.append(("ready",)))

        def require(self, route, **kwargs):
            calls.append(("require", kwargs))
            return route

        def prepare(self, route):
            calls.append(("prepare", route.model_alias))

        def release(self, route):
            calls.append(("release", route.model_alias))

    monkeypatch.setattr(runtime, "get_role", lambda name: role)
    monkeypatch.setattr(runtime, "get_model", lambda alias: model)
    monkeypatch.setattr(runtime, "LocalRuntimeRouter", Router)
    def count_tokens(*args, **kwargs):
        owner = ownership.model_execution_owner()
        assert owner is not None
        assert ownership.held_by("gpu-slot", "local-inference:0", owner)
        calls.append(("count",))
        return 10
    monkeypatch.setattr(runtime, "count_chat_tokens", count_tokens)
    monkeypatch.setattr(runtime, "generation_budget", lambda *args, **kwargs: GenerationBudget(
        "general", 200, 50, 10, 140, None))
    result = runtime.chat_completion({}, "general", [{"role": "user", "content": "hi"}])
    assert result["model"] == "daily"
    assert result["max_tokens"] == 80
    assert result["_tars_generation"]["effective_context_limit"] == 100
    streamed = list(runtime.chat_completion_stream(
        {}, "general", [{"role": "user", "content": "hi"}]))
    assert streamed[0]["content"] == "chunk"
    assert [call[0] for call in calls].count("resolve") == 2
    assert all(call[2]["persist"] is False for call in calls if call[0] == "resolve")
    assert [call[0] for call in calls].count("require") == 2
    assert [call[0] for call in calls].count("prepare") == 2
    assert [call[0] for call in calls].count("release") == 2
    phases = [call[0] for call in calls]
    first_prepare = phases.index("prepare")
    assert first_prepare < phases.index("count", first_prepare) < phases.index(
        "release", first_prepare)


def test_inference_lifecycle_has_one_authoritative_gpu_owner(monkeypatch):
    monkeypatch.setattr(runtime, "INFERENCE_SLOT_WAIT_SECONDS", 0.1)
    entered, release = threading.Event(), threading.Event()
    route = SimpleNamespace(backend="llama.cpp", runtime_id="fixture")
    class Router:
        def prepare(self, value):
            entered.set()
        def release(self, value):
            pass
    router = Router()
    holder = {}
    def hold():
        with runtime._inference_lifecycle(router, route):
            assert release.wait(5)
        holder["done"] = True
    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(5)
    with pytest.raises(RuntimeError, match="slot is busy"):
        with runtime._inference_lifecycle(router, route):
            pass
    release.set(); thread.join(timeout=5)
    assert holder["done"] is True


def test_inference_slot_is_exclusive_across_processes(monkeypatch):
    monkeypatch.setattr(runtime, "INFERENCE_SLOT_WAIT_SECONDS", 0.1)
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_hold_inference_slot,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              ready, release),
    )
    process.start()
    assert ready.wait(5)
    route = SimpleNamespace(backend="llama.cpp", runtime_id="fixture")
    router = SimpleNamespace(prepare=lambda value: None, release=lambda value: None)
    with pytest.raises(RuntimeError, match="slot is busy"):
        with runtime._inference_lifecycle(router, route):
            pass
    release.set()
    process.join(timeout=10)
    assert process.exitcode == 0


def test_dead_model_owner_is_reclaimed_without_waiting_for_false_heartbeat(monkeypatch):
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_hold_inference_slot,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              ready, release),
    )
    process.start()
    assert ready.wait(5)
    process.terminate()
    process.join(timeout=10)
    assert process.exitcode is not None

    with ownership.model_execution_scope(
        operation="dead-owner-recovery", timeout=0.2,
    ) as owner:
        assert ownership.held_by("gpu-slot", "local-inference:0", owner)
    assert not ownership.active("gpu-slot", "local-inference:0")


def test_stream_close_releases_authoritative_runtime_lifecycle(monkeypatch):
    role = SimpleNamespace(id="oracle", model="heavy", runtime_id="oracle",
                           display_name="Oracle", execution="delegate")
    model = SimpleNamespace(alias="heavy", backend="colibri", thinking_control="unknown")
    calls = []

    class Backend:
        def count_tokens(self, *args, **kwargs):
            owner = ownership.model_execution_owner()
            assert owner is not None
            assert ownership.held_by("gpu-slot", "local-inference:0", owner)
            calls.append("count")
            return 10

        def stream(self, request):
            yield backends.StreamEvent(content="first")
            yield backends.StreamEvent(content="second")

    class Router:
        def __init__(self, cfg):
            self.backend = Backend()

        def resolve(self, role_id, **kwargs):
            return SimpleNamespace(
                selected_role="oracle", model_alias="heavy", backend_instance=self.backend,
                model_capabilities={"context": 100}, require_ready=lambda: None)

        def require(self, route, **kwargs):
            return route

        def prepare(self, route):
            calls.append("load")

        def release(self, route):
            calls.append("unload")

    monkeypatch.setattr(runtime, "LocalRuntimeRouter", Router)
    monkeypatch.setattr(runtime, "get_role", lambda value: role)
    monkeypatch.setattr(runtime, "get_model", lambda value: model)
    monkeypatch.setattr(runtime, "generation_budget", lambda *args, **kwargs: GenerationBudget(
        "oracle", 100, 20, 10, 70, None))
    stream = runtime.chat_completion_stream(
        {}, "oracle", [{"role": "user", "content": "review"}])
    assert next(stream)["content"] == "first"
    assert calls == ["load", "count"]
    stream.close()
    assert calls == ["load", "count", "unload"]


def test_real_oracle_completion_path_loads_infers_and_releases(monkeypatch, tmp_path):
    model_path = tmp_path / "heavy.fixture"
    model_path.write_bytes(b"fixture-not-model-weights")
    role = SimpleNamespace(
        id="oracle", enabled=True, model="heavy", runtime_id="oracle", profile="normal",
        display_name="Oracle", execution="delegate", capabilities=("deep-reasoning",))
    model = SimpleNamespace(
        alias="heavy", backend="colibri", path=model_path, integrity_verified=True,
        runtime_compatible=True, thinking_control="unknown")
    transport = ColibriTransport()
    backend = backends.ColibriBackend(
        {"colibri": {"base_url": "http://127.0.0.1:9988", "ttl_seconds": 30}},
        transport=transport)

    monkeypatch.setattr(routing, "resolve_role_id", lambda value: "oracle")
    monkeypatch.setattr(routing, "get_role", lambda value: role)
    monkeypatch.setattr(routing, "get_model", lambda value: model)
    monkeypatch.setattr(routing, "get_profile", lambda *args: SimpleNamespace(context=131072))
    monkeypatch.setattr(routing, "backend_binding_ready", lambda value, cfg=None: True)
    monkeypatch.setattr(runtime, "get_role", lambda value: role)
    monkeypatch.setattr(runtime, "get_model", lambda value: model)
    monkeypatch.setattr(
        runtime, "LocalRuntimeRouter",
        lambda cfg: routing.LocalRuntimeRouter(
            cfg, backend_factory=lambda model_record, config: backend))
    monkeypatch.setattr(runtime, "generation_budget", lambda *args, **kwargs: GenerationBudget(
        "oracle", 131072, 1024, 128, 129920, None))

    response = runtime.chat_completion(
        {"colibri": {"base_url": "http://127.0.0.1:9988"}}, "oracle",
        [{"role": "user", "content": "review"}])
    assert response["choices"][0]["message"]["content"] == "answer"
    urls = [call[1] for call in transport.calls]
    load_index = next(i for i, url in enumerate(urls) if url.endswith("/load"))
    tokenize_index = next(i for i, url in enumerate(urls) if url.endswith("/v1/tokenize"))
    inference_index = next(i for i, url in enumerate(urls)
                           if url.endswith("/v1/chat/completions"))
    unload_index = next(i for i, url in enumerate(urls) if url.endswith("/unload"))
    assert load_index < tokenize_index < inference_index < unload_index
    assert transport.calls[load_index][2] == {"ttl_seconds": 30}


def test_stream_normalization_never_synthesizes_reasoning():
    events = list(backends.normalize_chat_stream([
        {"choices": [{"delta": {"content": "plain"}, "finish_reason": "stop"}]}
    ]))
    assert events[0].content == "plain"
    assert events[0].reasoning == ""


def test_role_token_count_uses_bound_backend_for_context_engine(monkeypatch):
    lifecycle = []

    class Backend:
        def count_tokens(self, model_id, messages, *, thinking_mode=None):
            assert model_id == "oracle" and thinking_mode == "on"
            owner = ownership.model_execution_owner()
            assert owner is not None
            assert ownership.held_by("gpu-slot", "local-inference:0", owner)
            lifecycle.append("count")
            return 77

    class Router:
        def __init__(self, cfg):
            self.backend = Backend()

        def resolve(self, role, **kwargs):
            return SimpleNamespace(
                runtime_id="oracle", backend_instance=self.backend,
                require_ready=lambda: None,
            )

        def prepare(self, route):
            lifecycle.append("load")

        def release(self, route):
            lifecycle.append("unload")

    monkeypatch.setattr(runtime, "LocalRuntimeRouter", Router)
    assert runtime.count_role_chat_tokens(
        {}, "oracle", ({"role": "user", "content": "review"},),
        thinking_mode="on") == 77
    assert lifecycle == ["load", "count", "unload"]
    assert ownership.model_execution_owner() is None


@pytest.mark.parametrize("streaming", [False, True])
def test_end_to_end_chat_does_not_prepare_before_cross_process_model_ownership(
        monkeypatch, streaming):
    monkeypatch.setattr(runtime, "INFERENCE_SLOT_WAIT_SECONDS", 0.1)
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_hold_inference_slot,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              ready, release),
    )
    process.start()
    assert ready.wait(5)

    class Router:
        def __init__(self, cfg):
            raise AssertionError("routing/preparation ran without model ownership")

    monkeypatch.setattr(runtime, "LocalRuntimeRouter", Router)
    try:
        if streaming:
            result = runtime.chat_completion_stream(
                {}, "general", ({"role": "user", "content": "hi"},))
            with pytest.raises(RuntimeError, match="slot is busy"):
                next(result)
        else:
            with pytest.raises(RuntimeError, match="slot is busy"):
                runtime.chat_completion(
                    {}, "general", ({"role": "user", "content": "hi"},))
    finally:
        release.set()
        process.join(timeout=10)
    assert process.exitcode == 0


def test_backend_readiness_requires_matching_local_calibration(monkeypatch):
    payload = b"current-model"
    path = state_store.STATE_DB_PATH.parent / "model.gguf"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    server = state_store.STATE_DB_PATH.parent / "llama-server"
    bench = state_store.STATE_DB_PATH.parent / "llama-bench"
    server.write_bytes(b"server-binary")
    bench.write_bytes(b"bench-binary")
    server_digest = hashlib.sha256(b"server-binary").hexdigest()
    bench_digest = hashlib.sha256(b"bench-binary").hexdigest()
    monkeypatch.setattr(backends, "LLAMA_SERVER_PATH", server)
    monkeypatch.setattr(backends, "LLAMA_BENCH_PATH", bench)
    model = SimpleNamespace(
        alias="local", backend="llama.cpp", sha256=digest, path=path)
    monkeypatch.setattr(backends, "load_calibration", lambda alias: {
        "status": "ready", "model_sha256": digest,
        "fingerprint": {"llama_server_sha256": server_digest,
                        "llama_bench_sha256": bench_digest},
    })
    assert backends.backend_binding_ready(model)
    monkeypatch.setattr(backends, "load_calibration", lambda alias: {
        "status": "ready", "model_sha256": "different"
    })
    assert not backends.backend_binding_ready(model)
    path.write_bytes(b"changed-model")
    monkeypatch.setattr(backends, "load_calibration", lambda alias: {
        "status": "ready", "model_sha256": digest,
        "fingerprint": {"llama_server_sha256": server_digest,
                        "llama_bench_sha256": bench_digest},
    })
    assert not backends.backend_binding_ready(model)


def test_runtime_binary_change_invalidates_calibrated_binding(monkeypatch):
    root = state_store.STATE_DB_PATH.parent
    model_path = root / "model.gguf"
    server = root / "llama-server"
    bench = root / "llama-bench"
    model_path.write_bytes(b"model")
    server.write_bytes(b"server")
    bench.write_bytes(b"bench")
    model = SimpleNamespace(
        alias="local", backend="llama.cpp", path=model_path,
        sha256=hashlib.sha256(b"model").hexdigest(),
    )
    calibration = {
        "status": "ready", "model_sha256": model.sha256,
        "fingerprint": {
            "llama_server_sha256": hashlib.sha256(b"server").hexdigest(),
            "llama_bench_sha256": hashlib.sha256(b"bench").hexdigest(),
        },
    }
    monkeypatch.setattr(backends, "LLAMA_SERVER_PATH", server)
    monkeypatch.setattr(backends, "LLAMA_BENCH_PATH", bench)
    monkeypatch.setattr(backends, "load_calibration", lambda alias: calibration)
    assert backends.backend_binding_ready(model)
    server.write_bytes(b"changed-server")
    assert not backends.backend_binding_ready(model)
    colibri = SimpleNamespace(
        backend="colibri", integrity_verified=True, runtime_compatible=True,
        path=SimpleNamespace(is_file=lambda: True))
    assert backends.backend_binding_ready(colibri)
