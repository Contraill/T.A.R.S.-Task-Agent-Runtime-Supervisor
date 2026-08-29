from __future__ import annotations

import json
from urllib.parse import quote
import urllib.request

from .config import runtime_base_url
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


def stream_post_json(url, payload, timeout=1200):
    """Yield OpenAI-compatible SSE JSON chunks from an HTTP POST.

    llama.cpp/llama-swap use the conventional `data: {...}` stream shape.  The
    generator deliberately exposes raw decoded JSON objects so higher layers can
    distinguish final-content, backend reasoning_content and usage without
    inventing model-specific text parsing.
    """
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


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


def _completion_payload(role_record, messages, *, max_tokens, temperature, stream):
    payload = {
        "model": role_record.runtime_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def _checked_role(role):
    role_record = get_role(role)
    if not role_record.enabled:
        raise RuntimeError(f"role {role_record.display_name!r} is disabled")
    if not role_record.model:
        raise RuntimeError(f"role {role_record.display_name!r} has no model binding")
    return role_record


def chat_completion(cfg, role, messages, *, max_tokens=1024, temperature=0.2):
    role_record = _checked_role(role)
    base = runtime_base_url(cfg)
    return post_json(
        base + "/v1/chat/completions",
        _completion_payload(
            role_record, messages, max_tokens=max_tokens,
            temperature=temperature, stream=False,
        ),
        timeout=1200,
    )


def chat_completion_stream(cfg, role, messages, *, max_tokens=1024, temperature=0.2):
    """Yield normalized streaming events.

    Event shape:
      {"content": str, "reasoning": str, "finish_reason": str|None,
       "usage": dict|None, "raw": dict}

    `reasoning` is populated only from backend-emitted reasoning_content (or the
    equivalent reasoning field).  T.A.R.S. never synthesizes Raw reasoning.
    """
    role_record = _checked_role(role)
    base = runtime_base_url(cfg)
    payload = _completion_payload(
        role_record, messages, max_tokens=max_tokens,
        temperature=temperature, stream=True,
    )
    for chunk in stream_post_json(base + "/v1/chat/completions", payload, timeout=1200):
        choices = chunk.get("choices") or []
        delta = {}
        finish = None
        if choices:
            choice = choices[0] or {}
            delta = choice.get("delta") or {}
            finish = choice.get("finish_reason")
        reasoning = (
            delta.get("reasoning_content")
            or delta.get("reasoning")
            or ""
        )
        content = delta.get("content") or ""
        usage = chunk.get("usage") if isinstance(chunk.get("usage"), dict) else None
        yield {
            "content": str(content),
            "reasoning": str(reasoning),
            "finish_reason": finish,
            "usage": usage,
            "raw": chunk,
        }
