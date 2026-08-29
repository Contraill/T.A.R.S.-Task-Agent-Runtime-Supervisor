from types import SimpleNamespace

import pytest

from tars import runtime
from tars import runtime_backends as backends


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


def _cfg():
    return {"runtime": {"base_url": "http://127.0.0.1:8080"}}


def test_llama_cpp_backend_contract(monkeypatch, tmp_path):
    transport = FakeTransport()
    binary = tmp_path / "llama-server"
    binary.write_text("binary")
    monkeypatch.setattr(backends, "LLAMA_SERVER_PATH", binary)
    backend = backends.LlamaCppBackend(_cfg(), transport=transport)
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


def test_colibri_boundary_is_explicitly_unavailable():
    backend = backends.ColibriBackend({})
    status = backend.status()
    assert not status.available and backend.support == "unimplemented"
    assert backend.model_capabilities("oracle").context == 0
    with pytest.raises(backends.BackendUnavailable):
        backend.complete(backends.InferenceRequest("oracle", ()))


def test_backend_factory_rejects_cloud_provider_names():
    for name in ("openai", "openai-compatible"):
        with pytest.raises(KeyError, match="unknown runtime backend"):
            backends.backend_for_name(name, {})


def test_runtime_dispatches_role_through_model_backend(monkeypatch):
    role = SimpleNamespace(enabled=True, model="local", runtime_id="daily", display_name="General")
    model = SimpleNamespace(alias="local", backend="llama.cpp")

    class Backend:
        def complete(self, request):
            return {"model": request.model, "messages": request.messages}

    monkeypatch.setattr(runtime, "get_role", lambda name: role)
    monkeypatch.setattr(runtime, "get_model", lambda alias: model)
    monkeypatch.setattr(runtime, "backend_binding_ready", lambda value: True)
    monkeypatch.setattr(runtime, "backend_for_model", lambda value, cfg: Backend())
    result = runtime.chat_completion({}, "general", [{"role": "user", "content": "hi"}])
    assert result["model"] == "daily"


def test_stream_normalization_never_synthesizes_reasoning():
    events = list(backends.normalize_chat_stream([
        {"choices": [{"delta": {"content": "plain"}, "finish_reason": "stop"}]}
    ]))
    assert events[0].content == "plain"
    assert events[0].reasoning == ""


def test_backend_readiness_requires_matching_local_calibration(monkeypatch):
    model = SimpleNamespace(alias="local", backend="llama.cpp", sha256="abc")
    monkeypatch.setattr(backends, "load_calibration", lambda alias: {
        "status": "ready", "model_sha256": "abc"
    })
    assert backends.backend_binding_ready(model)
    monkeypatch.setattr(backends, "load_calibration", lambda alias: {
        "status": "ready", "model_sha256": "different"
    })
    assert not backends.backend_binding_ready(model)
    assert not backends.backend_binding_ready(SimpleNamespace(backend="colibri"))
