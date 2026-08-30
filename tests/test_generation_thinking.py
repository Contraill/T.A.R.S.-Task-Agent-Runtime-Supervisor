import inspect
from types import SimpleNamespace

import pytest

from tars import activity, chat_tui, delegation, generation, runtime, runtime_backends, temporary, thinking


def binding(*, execution="chat", thinking_control="toggle"):
    role = SimpleNamespace(id="general", enabled=True, model="model", runtime_id="daily",
                           display_name="General", profile="normal", execution=execution)
    model = SimpleNamespace(alias="model", backend="llama.cpp",
                            thinking_control=thinking_control)
    profile = SimpleNamespace(context=65536)
    return role, model, profile


def patch_budget(monkeypatch, *, execution="chat", thinking_control="toggle"):
    role, model, profile = binding(execution=execution, thinking_control=thinking_control)
    monkeypatch.setattr(generation, "resolve_role_id", lambda value: "general")
    monkeypatch.setattr(generation, "get_role", lambda value: role)
    monkeypatch.setattr(generation, "get_profile", lambda alias, name: profile)
    monkeypatch.setattr(thinking, "resolve_role_id", lambda value: "general")
    monkeypatch.setattr(thinking, "get_role", lambda value: role)
    return role, model, profile


def test_dynamic_generation_ceiling_uses_actual_input_not_output_reserve(monkeypatch):
    patch_budget(monkeypatch)
    budget = generation.generation_budget({}, "general")
    assert budget.output_reserve == 8192
    assert budget.usable_input == 65536 - 8192 - 1024
    assert generation.generation_ceiling(budget, 1000) == 65536 - 1000 - 1024
    explicit = generation.generation_budget({}, "general", requested_tokens=512)
    assert explicit.output_reserve == 512
    assert generation.generation_ceiling(explicit, 1000) == 512


def test_profile_bounds_are_enforced(monkeypatch):
    role, model, profile = patch_budget(monkeypatch)
    profile.context = 2048
    with pytest.raises(generation.GenerationBudgetError):
        generation.generation_budget({}, "general")
    budget = generation.generation_budget(
        {"generation": {"output_tokens": 512}}, "general")
    with pytest.raises(generation.GenerationBudgetError):
        generation.generation_ceiling(budget, 1500)


def test_runtime_request_uses_dynamic_ceiling_and_real_thinking_toggle(monkeypatch):
    role, model, _ = patch_budget(monkeypatch)

    class Backend:
        def complete(self, request):
            assert request.max_tokens == 65536 - 100 - 1024
            assert request.thinking_mode == "off"
            return {"choices": [{"message": {"content": "hi"},
                                  "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 3}}

    monkeypatch.setattr(runtime, "_checked_role", lambda value: (role, model))
    monkeypatch.setattr(runtime, "backend_for_model", lambda *args: Backend())
    monkeypatch.setattr(
        runtime, "count_chat_tokens",
        lambda cfg, runtime_id, messages, *, thinking_mode=None: (
            100 if thinking_mode == "off" else 101
        ),
    )
    response = runtime.chat_completion({}, "general", [{"role": "user", "content": "hi"}],
                                       input_tokens=99)
    metadata = response["_tars_generation"]
    assert metadata["configured_generation_limit"] == 64412
    assert metadata["thinking_effective"] == "off"
    assert metadata["prompt_tokens"] == 100 and "reasoning_tokens" not in metadata


def test_template_token_count_uses_effective_thinking_mode(monkeypatch):
    captured = {}

    def apply(cfg, runtime_id, messages, *, thinking_mode=None):
        captured["thinking_mode"] = thinking_mode
        return "rendered"

    monkeypatch.setattr(runtime, "apply_chat_template", apply)
    monkeypatch.setattr(runtime, "tokenize_text", lambda *args: [1, 2, 3])
    assert runtime.count_chat_tokens(
        {}, "daily", [{"role": "user", "content": "hi"}], thinking_mode="off"
    ) == 3
    assert captured == {"thinking_mode": "off"}


def test_ordinary_tui_and_agent_paths_do_not_hardcode_unrelated_1k_limit():
    assert inspect.signature(chat_tui.ChatTUI._stream_completion).parameters["max_tokens"].default is None
    from tars.agent_loop import RuntimeModelAdapter
    assert inspect.signature(RuntimeModelAdapter).parameters["max_tokens"].default is None
    worker = inspect.getsource(chat_tui.ChatTUI._worker)
    assert "self.temporary.send, item.text, thinking=" in worker
    assert "requested_output_tokens=SIDEBAND_GENERATION_TOKENS" in worker


def test_temporary_uses_canonical_default_and_explicit_thinking(monkeypatch):
    captured = {}
    session = temporary.TemporarySession({}, "general")
    monkeypatch.setattr(session, "_messages", lambda *args, **kwargs: [
        {"role": "user", "content": "hello"}])

    def complete(cfg, role, messages, **kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}

    session.send("hello", complete=complete, thinking="off")
    assert captured == {"max_tokens": None, "thinking": "off"}


def test_empty_length_is_exhausted_not_normal_success():
    outcome = generation.generation_outcome("", "length")
    assert outcome == {"state": "exhausted", "normal_success": False,
                       "empty": True, "finish_reason": "length"}
    assert not generation.generation_outcome("partial", "length")["normal_success"]


def test_thinking_capabilities_and_auto_are_truthful_without_inference(monkeypatch):
    role, model, _ = patch_budget(monkeypatch)
    capability = thinking.capability_for_model(model)
    assert capability.modes == ("off", "on") and "low" not in capability.modes
    trivial = thinking.decide({}, "general", capability, requested="auto")
    assert trivial.effective == "off"
    complex_decision = thinking.decide({}, "general", capability, requested="auto",
                                       operation="agent", requires_tools=True)
    assert complex_decision.effective == "on"
    assert thinking.decide({}, "general", capability, requested="off").effective == "off"
    with pytest.raises(ValueError, match="not supported"):
        thinking.decide({}, "general", capability, requested="medium")
    assert not hasattr(complex_decision, "permissions")


def test_reasoning_visibility_does_not_change_thinking_policy():
    assert activity.reasoning_view("hidden", emitted_raw="genuine") == ""
    assert activity.reasoning_view("raw", emitted_raw="genuine") == "genuine"
    assert thinking.ThinkingDecision("auto", "off", "toggle", "chat").effective == "off"


def test_child_explicit_generation_budget_cannot_exceed_profile(monkeypatch):
    patch_budget(monkeypatch)
    with pytest.raises(generation.GenerationBudgetError):
        generation.generation_budget({}, "general", requested_tokens=65536)


def test_llama_payload_translates_only_real_toggle():
    request = runtime_backends.InferenceRequest(
        "daily", ({"role": "user", "content": "hi"},), 1000, thinking_mode="off")
    payload = runtime_backends.LlamaCppBackend._payload(request)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    request = runtime_backends.InferenceRequest(
        "daily", ({"role": "user", "content": "hi"},), 1000, thinking_mode="on")
    assert runtime_backends.LlamaCppBackend._payload(request)["chat_template_kwargs"] == {
        "enable_thinking": True}


def test_backend_advertises_only_verified_model_thinking_modes():
    backend = runtime_backends.LlamaCppBackend(
        {"runtime": {"base_url": "http://127.0.0.1:8080"}},
        transport=SimpleNamespace(json=lambda *args, **kwargs: {"data": []}),
        model_record=SimpleNamespace(thinking_control="toggle"))
    assert backend.model_capabilities("daily").thinking_modes == ("off", "on")
    unknown = runtime_backends.LlamaCppBackend(
        {"runtime": {"base_url": "http://127.0.0.1:8080"}},
        transport=SimpleNamespace(json=lambda *args, **kwargs: {"data": []}))
    assert unknown.model_capabilities("daily").thinking_modes == ()
