from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote
from typing import Iterable, Protocol

from .calibration import load_calibration
from .config import (
    LLAMA_BENCH_PATH,
    LLAMA_SERVER_PATH,
    local_http_origin,
    runtime_base_url,
)
from .model_integrity import (
    calibration_runtime_artifacts_match,
    model_artifact_matches,
    require_current_model_artifact,
)
from .runtime_http import request_json, request_sse


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
        return request_json(
            method, url, payload=payload, timeout=timeout)

    def sse(self, url, *, payload, timeout=1200):
        yield from request_sse(url, payload=payload, timeout=timeout)


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
    local_only = True
    zero_idle = True

    def __init__(self, cfg, *, transport: Transport | None = None, model_record=None):
        self.cfg = cfg
        self.base_url = runtime_base_url(cfg)
        self.transport = transport or UrllibTransport()
        self.model_record = model_record

    def _models(self):
        return self.transport.json("GET", self.base_url + "/v1/models", timeout=5).get("data", [])

    def _require_current_model(self):
        if self.model_record is None:
            raise BackendUnavailable(
                "llama.cpp inference requires a verified model artifact binding")
        require_current_model_artifact(self.model_record)

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
        self._require_current_model()
        return self.transport.json("POST", self.base_url + "/v1/chat/completions",
                                   payload=self._payload(request), timeout=1200)

    def stream(self, request):
        self._require_current_model()
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
    """Optional local Heavy-runtime adapter; probing never loads a model."""

    identity = "colibri"
    support = "supported-optional"
    local_only = True
    zero_idle = True

    def __init__(self, cfg=None, *, transport=None):
        self.cfg = cfg or {}
        runtime_cfg = self.cfg.get("runtime", {}) if isinstance(self.cfg, dict) else {}
        nested = runtime_cfg.get("colibri", {}) if isinstance(runtime_cfg, dict) else {}
        direct = self.cfg.get("colibri", {}) if isinstance(self.cfg, dict) else {}
        self.settings = dict(nested or direct or {})
        raw_base_url = str(self.settings.get("base_url", "")).strip()
        self.base_url = raw_base_url.rstrip("/")
        self._base_url_error = ""
        if raw_base_url:
            try:
                self.base_url = local_http_origin(
                    raw_base_url, label="Colibri base_url")
            except ValueError as exc:
                self._base_url_error = str(exc)
        self._ttl_error = ""
        try:
            self.ttl_seconds = max(5, min(300, int(self.settings.get("ttl_seconds", 60))))
        except (TypeError, ValueError):
            self.ttl_seconds = 60
            self._ttl_error = "Colibri ttl_seconds must be an integer"
        self.transport = transport or UrllibTransport()

    def _configuration_error(self):
        if not self.base_url:
            return "Colibri is not configured"
        if self._ttl_error:
            return self._ttl_error
        return self._base_url_error

    def _probe(self):
        error = self._configuration_error()
        if error:
            raise BackendUnavailable(error)
        value = self.transport.json("GET", self.base_url + "/v1/health", timeout=5)
        if not isinstance(value, dict):
            raise BackendUnavailable("Colibri health probe returned an invalid response")
        return value

    def status(self):
        error = self._configuration_error()
        if error:
            return BackendStatus(self.identity, False, False, error, self.support)
        try:
            probe = self._probe()
        except Exception as exc:
            return BackendStatus(self.identity, True, False, str(exc), self.support)
        version = str(probe.get("version", "")).strip()
        healthy = probe.get("status") in {"ok", "healthy"} and bool(version)
        message = f"healthy · version {version}" if healthy else "health/version probe failed"
        return BackendStatus(self.identity, True, healthy, message, self.support)

    def capabilities(self):
        if self._configuration_error():
            return RuntimeCapabilities(False, None, None, False, False, False)
        try:
            value = self.transport.json(
                "GET", self.base_url + "/v1/capabilities", timeout=5)
            if not isinstance(value, dict):
                raise TypeError("invalid capability response")
        except Exception:
            return RuntimeCapabilities(False, None, None, False, False, False)
        return RuntimeCapabilities(
            bool(value.get("streaming")),
            True if value.get("reasoning") is True else None,
            True if value.get("tool_calls") is True else None,
            bool(value.get("explicit_load")),
            bool(value.get("explicit_unload")),
            bool(value.get("on_demand")),
            tuple(value.get("thinking_modes") or ()),
            str(value.get("thinking_mechanism") or "backend reasoning mode"),
        )

    def model_capabilities(self, model):
        if self._configuration_error():
            return ModelCapabilities(model, 0, reasoning=None, tool_calls=None)
        try:
            value = self.transport.json(
                "GET", self.base_url + "/v1/models/" + quote(str(model), safe=""), timeout=5)
            if not isinstance(value, dict):
                raise TypeError("invalid model response")
        except Exception:
            return ModelCapabilities(model, 0, reasoning=None, tool_calls=None)
        return ModelCapabilities(
            model, int(value.get("context_length") or 0),
            tuple(value.get("input_modalities") or ()),
            tuple(value.get("output_modalities") or ()),
            True if value.get("reasoning") is True else None,
            True if value.get("tool_calls") is True else None,
            tuple(value.get("thinking_modes") or ()),
            str(value.get("thinking_mechanism") or "backend reasoning mode"),
        )

    def load(self, model):
        self._probe()
        value = self.transport.json(
            "POST", self.base_url + "/v1/models/" + quote(str(model), safe="") + "/load",
            payload={"ttl_seconds": self.ttl_seconds}, timeout=1200)
        state = str(value.get("state") or "failed")
        return LifecycleResult(self.identity, model, state, True,
                               str(value.get("message") or state))

    def unload(self, model):
        self._probe()
        value = self.transport.json(
            "POST", self.base_url + "/v1/models/" + quote(str(model), safe="") + "/unload",
            payload={}, timeout=120)
        state = str(value.get("state") or "failed")
        return LifecycleResult(self.identity, model, state, True,
                               str(value.get("message") or state))

    @staticmethod
    def _payload(request, *, stream=False):
        payload = {"model": request.model, "messages": list(request.messages),
                   "max_tokens": request.max_tokens, "temperature": request.temperature,
                   "stream": bool(stream)}
        if request.thinking_mode is not None:
            payload["reasoning_mode"] = request.thinking_mode
        return payload

    def complete(self, request):
        self._probe()
        return self.transport.json(
            "POST", self.base_url + "/v1/chat/completions",
            payload=self._payload(request), timeout=1200)

    def stream(self, request):
        self._probe()
        chunks = self.transport.sse(
            self.base_url + "/v1/chat/completions",
            payload=self._payload(request, stream=True), timeout=1200)
        yield from normalize_chat_stream(chunks)

    def count_tokens(self, model, messages, *, thinking_mode=None):
        self._probe()
        value = self.transport.json(
            "POST", self.base_url + "/v1/tokenize",
            payload={"model": model, "messages": list(messages),
                     "reasoning_mode": thinking_mode}, timeout=120)
        count = value.get("tokens")
        if not isinstance(count, int) or count < 0:
            raise BackendUnavailable("Colibri tokenizer returned no token count")
        return count

    def diagnostics(self):
        status = self.status()
        capabilities = self.capabilities()
        return {"backend": self.identity, "support": self.support,
                "configured": bool(self.base_url),
                "configuration_error": self._configuration_error() or None,
                "available": status.available, "healthy": status.healthy,
                "message": status.message, "base_url": self.base_url or None,
                "ttl_seconds": self.ttl_seconds,
                "capabilities": capabilities.__dict__,
                "zero_idle": "probes do not load models; inference loads on demand with finite TTL"}


BACKEND_TYPES = {"llama.cpp": LlamaCppBackend, "colibri": ColibriBackend}


def backend_for_name(name: str, cfg, *, transport=None) -> RuntimeBackend:
    backend_type = BACKEND_TYPES.get(name)
    if backend_type is not None:
        return backend_type(cfg, transport=transport)
    from .extensions import ExtensionLoader, validate_runtime_backend
    try:
        provider = ExtensionLoader(cfg).load("runtime_backend", name)
    except KeyError as exc:
        raise KeyError(f"unknown runtime backend: {name}") from exc
    return validate_runtime_backend(provider.create(
        cfg=cfg, model_record=None, transport=transport))


def backend_for_model(model, cfg, *, transport=None) -> RuntimeBackend:
    backend_type = BACKEND_TYPES.get(model.backend)
    if backend_type is None:
        from .extensions import ExtensionLoader, validate_runtime_backend
        try:
            provider = ExtensionLoader(cfg).load("runtime_backend", model.backend)
        except KeyError as exc:
            raise KeyError(f"unknown runtime backend: {model.backend}") from exc
        return validate_runtime_backend(provider.create(
            cfg=cfg, model_record=model, transport=transport))
    if backend_type is LlamaCppBackend:
        return backend_type(cfg, transport=transport, model_record=model)
    return backend_type(cfg, transport=transport)


def backend_binding_ready(model, cfg=None) -> bool:
    if model.backend == "llama.cpp":
        if not model_artifact_matches(model):
            return False
        try:
            calibration = load_calibration(model.alias)
        except (FileNotFoundError, KeyError):
            return False
        return (calibration.get("status") == "ready" and
                calibration.get("model_sha256") == model.sha256 and
                calibration_runtime_artifacts_match(
                    calibration, LLAMA_SERVER_PATH, LLAMA_BENCH_PATH))
    if model.backend == "colibri":
        return (bool(getattr(model, "integrity_verified", False)) and
                bool(getattr(model, "runtime_compatible", False)) and
                model.path.is_file())
    return (bool(getattr(model, "integrity_verified", False)) and
            bool(getattr(model, "runtime_compatible", False)) and
            bool(getattr(model, "path", None)) and model.path.is_file())
