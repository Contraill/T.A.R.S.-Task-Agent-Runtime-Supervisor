from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable, Protocol
import urllib.request

from .calibration import load_calibration
from .config import LLAMA_SERVER_PATH, runtime_base_url


@dataclass(frozen=True)
class BackendStatus:
    backend: str
    available: bool
    healthy: bool
    message: str
    support: str


@dataclass(frozen=True)
class RuntimeCapabilities:
    streaming: bool
    reasoning: bool | None
    tool_calls: bool | None
    explicit_load: bool
    explicit_unload: bool
    on_demand: bool
    thinking_modes: tuple[str, ...] = ()
    thinking_mechanism: str = "model-dependent"


@dataclass(frozen=True)
class ModelCapabilities:
    model: str
    context: int
    input_modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("text",)
    reasoning: bool | None = None
    tool_calls: bool | None = None
    thinking_modes: tuple[str, ...] = ()
    thinking_mechanism: str = "unavailable"


@dataclass(frozen=True)
class LifecycleResult:
    backend: str
    model: str
    state: str
    managed_on_demand: bool
    message: str


@dataclass(frozen=True)
class InferenceRequest:
    model: str
    messages: tuple[dict, ...]
    max_tokens: int = 1024
    temperature: float = 0.2
    thinking_mode: str | None = None
    requested_generation_limit: int | None = None
    input_tokens: int | None = None


@dataclass(frozen=True)
class StreamEvent:
    content: str = ""
    reasoning: str = ""
    tool_calls: tuple[dict, ...] = ()
    finish_reason: str | None = None
    usage: dict | None = None
    raw: dict | None = None


class BackendUnavailable(RuntimeError):
    pass


class Transport(Protocol):
    def json(self, method: str, url: str, *, payload: dict | None = None,
             timeout: float = 30) -> dict: ...
    def sse(self, url: str, *, payload: dict, timeout: float = 1200) -> Iterable[dict]: ...


class UrllibTransport:
    def json(self, method, url, *, payload=None, timeout=30):
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            url, data=body, method=method, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def sse(self, url, *, payload, timeout=1200):
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
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


class RuntimeBackend(Protocol):
    identity: str
    support: str

    def status(self) -> BackendStatus: ...
    def capabilities(self) -> RuntimeCapabilities: ...
    def model_capabilities(self, model: str) -> ModelCapabilities: ...
    def load(self, model: str) -> LifecycleResult: ...
    def unload(self, model: str) -> LifecycleResult: ...
    def complete(self, request: InferenceRequest) -> dict: ...
    def stream(self, request: InferenceRequest) -> Iterable[StreamEvent]: ...
    def diagnostics(self) -> dict: ...


def normalize_chat_stream(chunks: Iterable[dict]) -> Iterable[StreamEvent]:
    """Normalize backend-emitted content, reasoning, tools, finish and usage fields."""
    for chunk in chunks:
        choices = chunk.get("choices") or []
        choice = choices[0] if choices else {}
        delta = choice.get("delta") or {}
        yield StreamEvent(
            content=str(delta.get("content") or ""),
            reasoning=str(delta.get("reasoning_content") or delta.get("reasoning") or ""),
            tool_calls=tuple(delta.get("tool_calls") or ()),
            finish_reason=choice.get("finish_reason"),
            usage=chunk.get("usage") if isinstance(chunk.get("usage"), dict) else None,
            raw=chunk,
        )


class LlamaCppBackend:
    identity = "llama.cpp"
    support = "reference-tested"

    def __init__(self, cfg, *, transport: Transport | None = None, model_record=None):
        self.cfg = cfg
        self.base_url = runtime_base_url(cfg)
        self.transport = transport or UrllibTransport()
        self.model_record = model_record

    def _models(self):
        return self.transport.json("GET", self.base_url + "/v1/models", timeout=5).get("data", [])

    def status(self):
        if not LLAMA_SERVER_PATH.is_file():
            return BackendStatus(self.identity, False, False, "llama-server binary missing", self.support)
        try:
            self._models()
        except Exception as exc:
            return BackendStatus(self.identity, True, False, str(exc), self.support)
        return BackendStatus(self.identity, True, True, "healthy", self.support)

    def capabilities(self):
        return RuntimeCapabilities(True, True, True, False, False, True,
                                   ("off", "on"),
                                   "chat_template_kwargs.enable_thinking when model metadata is verified")

    def model_capabilities(self, model):
        row = next((item for item in self._models() if item.get("id") == model), {})
        architecture = row.get("architecture") or {}
        capabilities = row.get("capabilities") or {}
        toggle = (self.model_record is not None and
                  getattr(self.model_record, "thinking_control", "unknown") == "toggle")
        return ModelCapabilities(
            model=model,
            context=int(row.get("context_length") or (row.get("meta") or {}).get("n_ctx") or 0),
            input_modalities=tuple(architecture.get("input_modalities") or ("text",)),
            output_modalities=tuple(architecture.get("output_modalities") or ("text",)),
            reasoning=True if row.get("reasoning") is True else None,
            tool_calls=bool(capabilities.get("function_calling")) if capabilities else None,
            thinking_modes=("off", "on") if toggle else (),
            thinking_mechanism=("llama.cpp chat_template_kwargs.enable_thinking"
                                if toggle else "unverified"),
        )

    def load(self, model):
        return LifecycleResult(self.identity, model, "on-demand", True,
                               "llama-swap loads the model on the first inference request")

    def unload(self, model):
        return LifecycleResult(self.identity, model, "ttl-managed", True,
                               "llama-swap unloads the model through the finite TTL policy")

    @staticmethod
    def _payload(request: InferenceRequest, *, stream=False):
        payload = {"model": request.model, "messages": list(request.messages),
                   "max_tokens": request.max_tokens, "temperature": request.temperature}
        if request.thinking_mode is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": request.thinking_mode == "on"}
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    def complete(self, request):
        return self.transport.json("POST", self.base_url + "/v1/chat/completions",
                                   payload=self._payload(request), timeout=1200)

    def stream(self, request):
        chunks = self.transport.sse(self.base_url + "/v1/chat/completions",
                                    payload=self._payload(request, stream=True), timeout=1200)
        yield from normalize_chat_stream(chunks)

    def diagnostics(self):
        status = self.status()
        return {
            "backend": self.identity,
            "support": self.support,
            "available": status.available,
            "healthy": status.healthy,
            "message": status.message,
            "base_url": self.base_url,
            "llama_server": str(LLAMA_SERVER_PATH),
            "lifecycle": "llama-swap on-demand load with finite TTL unload",
        }


class ColibriBackend:
    """Reserved local Heavy-runtime boundary; full Oracle integration lands later."""

    identity = "colibri"
    support = "unimplemented"

    def __init__(self, cfg=None, *, transport=None):
        self.cfg = cfg or {}

    def status(self):
        return BackendStatus(self.identity, False, False,
                             "Colibri runtime integration is not implemented", self.support)

    def capabilities(self):
        return RuntimeCapabilities(False, None, None, False, False, False)

    def model_capabilities(self, model):
        return ModelCapabilities(model, 0, reasoning=None, tool_calls=None)

    def load(self, model):
        raise BackendUnavailable("Colibri load lifecycle is not implemented")

    def unload(self, model):
        raise BackendUnavailable("Colibri unload lifecycle is not implemented")

    def complete(self, request):
        raise BackendUnavailable("Colibri inference is not implemented")

    def stream(self, request):
        raise BackendUnavailable("Colibri streaming is not implemented")
        yield  # pragma: no cover

    def diagnostics(self):
        status = self.status()
        return {"backend": self.identity, "support": self.support,
                "available": status.available, "healthy": status.healthy,
                "message": status.message}


BACKEND_TYPES = {"llama.cpp": LlamaCppBackend, "colibri": ColibriBackend}


def backend_for_name(name: str, cfg, *, transport=None) -> RuntimeBackend:
    try:
        backend_type = BACKEND_TYPES[name]
    except KeyError as exc:
        raise KeyError(f"unknown runtime backend: {name}") from exc
    return backend_type(cfg, transport=transport)


def backend_for_model(model, cfg, *, transport=None) -> RuntimeBackend:
    backend_type = BACKEND_TYPES.get(model.backend)
    if backend_type is None:
        raise KeyError(f"unknown runtime backend: {model.backend}")
    if backend_type is LlamaCppBackend:
        return backend_type(cfg, transport=transport, model_record=model)
    return backend_type(cfg, transport=transport)


def backend_binding_ready(model) -> bool:
    if model.backend == "llama.cpp":
        try:
            calibration = load_calibration(model.alias)
        except (FileNotFoundError, KeyError):
            return False
        return (calibration.get("status") == "ready" and
                calibration.get("model_sha256") == model.sha256)
    return False
