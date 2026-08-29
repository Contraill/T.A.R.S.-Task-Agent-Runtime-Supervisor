from __future__ import annotations

import json
from urllib.parse import quote
import urllib.request

from .config import runtime_base_url
from .runtime_backends import InferenceRequest, backend_binding_ready, backend_for_model
from .registry import get_model
from .roles import get_role


def get_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def post_json(url, payload, timeout=600):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def runtime_models(cfg):
    base = runtime_base_url(cfg)
    return get_json(base + "/v1/models")["data"]


def upstream_post_json(cfg, runtime_id, path, payload, *, timeout=600):
    """Route a llama-server-specific request through llama-swap to one model."""
    base = runtime_base_url(cfg)
    model_path = quote(str(runtime_id), safe="")
    endpoint = path if str(path).startswith("/") else "/" + str(path)
    return post_json(
        f"{base}/upstream/{model_path}{endpoint}", payload, timeout=timeout
    )


def apply_chat_template(cfg, runtime_id, messages):
    response = upstream_post_json(
        cfg,
        runtime_id,
        "/apply-template",
        {"messages": messages},
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


def count_chat_tokens(cfg, runtime_id, messages):
    prompt = apply_chat_template(cfg, runtime_id, messages)
    return len(tokenize_text(cfg, runtime_id, prompt))


def _checked_role(role):
    role_record = get_role(role)
    if not role_record.enabled:
        raise RuntimeError(f"role {role_record.display_name!r} is disabled")
    if not role_record.model:
        raise RuntimeError(f"role {role_record.display_name!r} has no model binding")
    model = get_model(role_record.model)
    if not backend_binding_ready(model):
        raise RuntimeError(f"model binding {model.alias!r} is not runtime-ready")
    return role_record, model


def chat_completion(cfg, role, messages, *, max_tokens=1024, temperature=0.2):
    role_record, model = _checked_role(role)
    backend = backend_for_model(model, cfg)
    request = InferenceRequest(role_record.runtime_id, tuple(messages), max_tokens, temperature)
    return backend.complete(request)


def chat_completion_stream(cfg, role, messages, *, max_tokens=1024, temperature=0.2):
    """Yield normalized streaming events.

    Event shape:
      {"content": str, "reasoning": str, "finish_reason": str|None,
       "usage": dict|None, "raw": dict}

    `reasoning` is populated only from backend-emitted reasoning_content (or the
    equivalent reasoning field).  T.A.R.S. never synthesizes Raw reasoning.
    """
    role_record, model = _checked_role(role)
    backend = backend_for_model(model, cfg)
    request = InferenceRequest(role_record.runtime_id, tuple(messages), max_tokens, temperature)
    for event in backend.stream(request):
        yield {
            "content": event.content,
            "reasoning": event.reasoning,
            "tool_calls": list(event.tool_calls),
            "finish_reason": event.finish_reason,
            "usage": event.usage,
            "raw": event.raw,
        }
