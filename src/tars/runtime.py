from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from urllib.parse import quote
import time

from .config import runtime_base_url
from .model_integrity import require_current_model_artifact
from .runtime_backends import InferenceRequest
from .runtime_http import request_json
from .runtime_routing import LocalRuntimeRouter
from .registry import get_model
from .roles import get_role, list_roles
from .generation import generation_budget, generation_ceiling
from .thinking import capability_for_model, decide as decide_thinking
from .ownership import model_execution_owner, model_execution_scope


INFERENCE_SLOT_WAIT_SECONDS = 30.0


def get_json(url, timeout=5):
    return request_json("GET", url, timeout=timeout)


def post_json(url, payload, timeout=600):
    return request_json("POST", url, payload=payload, timeout=timeout)


def runtime_models(cfg):
    with model_execution_scope(
        operation="runtime-models", timeout=INFERENCE_SLOT_WAIT_SECONDS,
    ):
        base = runtime_base_url(cfg)
        return get_json(base + "/v1/models")["data"]


def upstream_post_json(cfg, runtime_id, path, payload, *, timeout=600):
    """Route a llama-server-specific request through llama-swap to one model."""
    with model_execution_scope(
        operation="runtime-upstream",
        timeout=INFERENCE_SLOT_WAIT_SECONDS,
        metadata={"runtime_id": str(runtime_id), "path": str(path)},
    ):
        roles = [
            role for role in list_roles(include_disabled=False)
            if role.runtime_id == str(runtime_id) and role.model
        ]
        if len(roles) != 1:
            raise RuntimeError(
                f"runtime id {runtime_id!r} has no unique verified model binding")
        require_current_model_artifact(get_model(roles[0].model))
        base = runtime_base_url(cfg)
        model_path = quote(str(runtime_id), safe="")
        endpoint = path if str(path).startswith("/") else "/" + str(path)
        return post_json(
            f"{base}/upstream/{model_path}{endpoint}", payload, timeout=timeout
        )


def apply_chat_template(cfg, runtime_id, messages, *, thinking_mode=None):
    payload = {"messages": messages}
    if thinking_mode in {"off", "on"}:
        payload["chat_template_kwargs"] = {
            "enable_thinking": thinking_mode == "on"
        }
    response = upstream_post_json(
        cfg,
        runtime_id,
        "/apply-template",
        payload,
        timeout=1200,
    )
    prompt = response.get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError("llama.cpp /apply-template returned no prompt")
    return prompt


def tokenize_text(cfg, runtime_id, content):
    response = upstream_post_json(
        cfg,
        runtime_id,
        "/tokenize",
        {
            "content": content,
            "add_special": False,
            "parse_special": True,
            "with_pieces": False,
        },
        timeout=1200,
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("llama.cpp /tokenize returned no token list")
    return tokens


def count_chat_tokens(cfg, runtime_id, messages, *, thinking_mode=None):
    with model_execution_scope(
        operation="exact-token-count",
        timeout=INFERENCE_SLOT_WAIT_SECONDS,
        metadata={"runtime_id": str(runtime_id)},
    ):
        prompt = apply_chat_template(
            cfg, runtime_id, messages, thinking_mode=thinking_mode
        )
        return len(tokenize_text(cfg, runtime_id, prompt))


def count_role_chat_tokens(cfg, role, messages, *, thinking_mode=None):
    """Count with the bound backend tokenizer without leaking backend details to Context."""
    with model_execution_scope(
        operation="role-token-count", timeout=INFERENCE_SLOT_WAIT_SECONDS,
        metadata={"role": str(role)},
    ):
        router, route, backend = _inference_route(cfg, role)
        with _inference_lifecycle(router, route):
            if hasattr(backend, "count_tokens"):
                return backend.count_tokens(
                    route.runtime_id, messages, thinking_mode=thinking_mode)
            return count_chat_tokens(
                cfg, route.runtime_id, messages, thinking_mode=thinking_mode)


def _inference_route(cfg, role):
    if model_execution_owner() is None:
        raise RuntimeError("runtime routing requires model execution ownership")
    router = LocalRuntimeRouter(cfg)
    route = router.resolve(role, persist=False)
    route.require_ready()
    backend = route.backend_instance
    if backend is None:
        raise RuntimeError("resolved runtime backend instance is unavailable")
    return router, route, backend


def _inference_request(cfg, role, messages, *, max_tokens=None, input_tokens=None,
                       temperature=0.2, thinking="auto", operation="chat",
                       task_active=False, requires_tools=False, complex_task=False,
                       router, route, backend):
    if model_execution_owner() is None:
        raise RuntimeError("inference preparation requires model execution ownership")
    role_record = get_role(route.selected_role)
    model = get_model(route.model_alias)
    thinking_decision = decide_thinking(
        cfg, role_record.id, capability_for_model(model), requested=thinking,
        operation=operation, task_active=task_active, requires_tools=requires_tools,
        complex_task=complex_task)
    # Thinking controls can alter the rendered chat template. Count the exact
    # template used by inference so the dynamic ceiling is based on the real
    # prompt, even when a context projection supplied an earlier estimate.
    if hasattr(backend, "count_tokens"):
        tokens = backend.count_tokens(
            role_record.runtime_id, messages,
            thinking_mode=thinking_decision.effective)
    else:
        tokens = count_chat_tokens(
            cfg, role_record.runtime_id, messages,
            thinking_mode=thinking_decision.effective,
        )
    router.require(
        route, context_tokens=tokens,
        require_reasoning=thinking_decision.effective == "on",
        require_tools=requires_tools)
    budget = generation_budget(cfg, role_record.id, requested_tokens=max_tokens)
    backend_context = int(route.model_capabilities["context"])
    effective_window = min(budget.context_window, backend_context)
    effective_budget = replace(budget, context_window=effective_window)
    ceiling = generation_ceiling(effective_budget, tokens)
    request = InferenceRequest(
        role_record.runtime_id, tuple(messages), ceiling, temperature,
        thinking_decision.effective, max_tokens, tokens)
    metadata = {"requested_generation_limit": max_tokens,
                "configured_generation_limit": ceiling, "input_tokens": tokens,
                "thinking_policy": thinking_decision.requested,
                "thinking_effective": thinking_decision.effective,
                "thinking_mechanism": thinking_decision.mechanism,
                "profile_context_limit": budget.context_window,
                "backend_context_limit": backend_context,
                "effective_context_limit": effective_window}
    return request, metadata


@contextmanager
def _inference_lifecycle(router, route):
    """Apply the resolved backend lifecycle around every real inference attempt."""
    with model_execution_scope(
        operation="inference-lifecycle",
        timeout=INFERENCE_SLOT_WAIT_SECONDS,
        metadata={"backend": getattr(route, "backend", ""),
                  "runtime_id": getattr(route, "runtime_id", "")},
    ):
        primary_error = None
        try:
            router.prepare(route)
            yield
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                router.release(route)
            except Exception as release_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"runtime release also failed: {release_error}")


def _reported_usage(usage):
    result = {}
    for key in ("prompt_tokens", "completion_tokens"):
        if key in usage:
            result[key] = usage[key]
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict) and "reasoning_tokens" in details:
        result["reasoning_tokens"] = details["reasoning_tokens"]
    return result


def chat_completion(cfg, role, messages, *, max_tokens=None, input_tokens=None,
                    temperature=0.2, thinking="auto", operation="chat",
                    task_active=False, requires_tools=False, complex_task=False):
    overall_started = time.monotonic()
    with model_execution_scope(
        operation="chat-completion", timeout=INFERENCE_SLOT_WAIT_SECONDS,
        metadata={"role": str(role)},
    ):
        router, route, backend = _inference_route(cfg, role)
        with _inference_lifecycle(router, route):
            request, metadata = _inference_request(
                cfg, role, messages, max_tokens=max_tokens,
                input_tokens=input_tokens, temperature=temperature,
                thinking=thinking, operation=operation,
                task_active=task_active, requires_tools=requires_tools,
                complex_task=complex_task, router=router, route=route,
                backend=backend)
            started = time.monotonic()
            response = backend.complete(request)
    choice = (response.get("choices") or [{}])[0]
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    response["_tars_generation"] = metadata | _reported_usage(usage) | {
        "finish_reason": choice.get("finish_reason"),
        "elapsed_seconds": time.monotonic() - started,
        "preparation_seconds": started - overall_started,
    }
    return response


def chat_completion_stream(cfg, role, messages, *, max_tokens=None, input_tokens=None,
                           temperature=0.2, thinking="auto", operation="chat",
                           task_active=False, requires_tools=False,
                           complex_task=False):
    """Yield normalized streaming events.

    Event shape:
      {"content": str, "reasoning": str, "finish_reason": str|None,
       "usage": dict|None, "raw": dict}

    `reasoning` is populated only from backend-emitted reasoning_content (or the
    equivalent reasoning field).  T.A.R.S. never synthesizes Raw reasoning.
    """
    overall_started = time.monotonic()
    with model_execution_scope(
        operation="chat-completion-stream", timeout=INFERENCE_SLOT_WAIT_SECONDS,
        metadata={"role": str(role)},
    ):
        router, route, backend = _inference_route(cfg, role)
        with _inference_lifecycle(router, route):
            request, metadata = _inference_request(
                cfg, role, messages, max_tokens=max_tokens,
                input_tokens=input_tokens, temperature=temperature,
                thinking=thinking, operation=operation,
                task_active=task_active, requires_tools=requires_tools,
                complex_task=complex_task, router=router, route=route,
                backend=backend)
            started = time.monotonic()
            first_token = None
            last_finish = None
            for event in backend.stream(request):
                if first_token is None and (event.content or event.reasoning):
                    first_token = time.monotonic() - started
                usage = event.usage or {}
                if event.finish_reason is not None:
                    last_finish = event.finish_reason
                generation = metadata | _reported_usage(usage) | {
                    "finish_reason": last_finish,
                    "elapsed_seconds": time.monotonic() - started,
                    "preparation_seconds": started - overall_started,
                    "first_token_seconds": first_token,
                }
                yield {
                    "content": event.content,
                    "reasoning": event.reasoning,
                    "tool_calls": list(event.tool_calls),
                    "finish_reason": event.finish_reason,
                    "usage": event.usage,
                    "generation": generation,
                    "raw": event.raw,
                }
