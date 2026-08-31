from __future__ import annotations

from dataclasses import asdict, dataclass, field
import uuid

from .calibration import get_profile
from .registry import get_model
from .roles import get_role, resolve_role_id
from .runtime_backends import (BACKEND_TYPES, BackendStatus, LifecycleResult,
                               ModelCapabilities, RuntimeCapabilities,
                               backend_binding_ready, backend_for_model)
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction
from .tasks import append_event, load_task


LOCAL_BACKENDS = frozenset(BACKEND_TYPES)


class RuntimeRouteUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeRoute:
    id: str
    requested_role: str
    selected_role: str
    model_alias: str
    backend: str
    runtime_id: str
    profile: str
    state: str
    requested: dict
    reasons: tuple[str, ...]
    backend_status: dict
    runtime_capabilities: dict
    model_capabilities: dict
    created_at: str
    task_id: str | None = None
    _backend: object | None = field(default=None, repr=False, compare=False)

    @property
    def ready(self):
        return self.state == "ready"

    def require_ready(self):
        if not self.ready:
            raise RuntimeRouteUnavailable("; ".join(self.reasons) or "runtime route unavailable")
        return self

    @property
    def backend_instance(self):
        return self._backend


def _plain(value):
    return asdict(value) if hasattr(value, "__dataclass_fields__") else {}


class LocalRuntimeRouter:
    """Resolve an exact Role binding to a healthy local runtime without fallback."""

    def __init__(self, cfg, *, backend_factory=backend_for_model):
        self.cfg = cfg
        self.backend_factory = backend_factory

    def resolve(self, role_name, *, task_id=None, required_capabilities=(),
                context_tokens=0, require_reasoning=False, require_tools=False,
                input_modalities=("text",), persist=True) -> RuntimeRoute:
        if persist or task_id:
            ensure_state_store()
        role_id = resolve_role_id(role_name)
        role = get_role(role_id)
        requested = {
            "capabilities": sorted(set(map(str, required_capabilities))),
            "context_tokens": max(0, int(context_tokens)),
            "require_reasoning": bool(require_reasoning),
            "require_tools": bool(require_tools),
            "input_modalities": sorted(set(map(str, input_modalities))),
            "local_only": True,
            "silent_substitution": False,
        }
        reasons = []
        model = None
        backend = None
        status = BackendStatus("", False, False, "not inspected", "unknown")
        runtime_caps = RuntimeCapabilities(False, None, None, False, False, False)
        model_caps = ModelCapabilities("", 0)
        profile_context = 0

        if not role.enabled:
            reasons.append(f"Role {role.id} is disabled")
        if task_id:
            task = load_task(task_id)
            if task.owner_role != role.id:
                reasons.append(
                    f"task owner is {task.owner_role}; explicit handoff is required before routing to {role.id}")
        missing_semantic = sorted(set(requested["capabilities"]) - set(role.capabilities))
        if missing_semantic:
            reasons.append("Role lacks capabilities: " + ", ".join(missing_semantic))
        if not role.model:
            reasons.append(f"Role {role.id} has no model binding")
        else:
            try:
                model = get_model(role.model)
            except Exception as exc:
                reasons.append(f"model binding cannot be loaded: {exc}")

        if model is not None:
            if model.backend not in LOCAL_BACKENDS:
                reasons.append(f"backend is not an allowed local runtime: {model.backend}")
            if not model.integrity_verified:
                reasons.append(f"model {model.alias} integrity is not verified")
            if not model.runtime_compatible:
                reasons.append(f"model {model.alias} is not runtime compatible")
            if not model.path.is_file():
                reasons.append(f"model artifact is missing: {model.path}")
            if not backend_binding_ready(model, self.cfg):
                reasons.append(f"model {model.alias} binding is not runtime-ready")
            try:
                profile = get_profile(model.alias, role.profile)
                profile_context = profile.context
            except Exception as exc:
                reasons.append(f"runtime profile {role.profile} is unavailable: {exc}")
            try:
                backend = self.backend_factory(model, self.cfg)
                status = backend.status()
                runtime_caps = backend.capabilities()
                if status.available and status.healthy:
                    model_caps = backend.model_capabilities(role.runtime_id)
            except Exception as exc:
                reasons.append(f"backend inspection failed: {exc}")
            if not status.available:
                reasons.append(status.message or f"backend {model.backend} is unavailable")
            elif not status.healthy:
                reasons.append(status.message or f"backend {model.backend} is unhealthy")

        if status.available and status.healthy and model_caps.context <= 0:
            reasons.append(
                f"runtime model {role.runtime_id} is absent or its context is unverified")
        reasons.extend(self._requirement_reasons(
            profile_context, model_caps, context_tokens=requested["context_tokens"],
            require_reasoning=require_reasoning, require_tools=require_tools,
            input_modalities=requested["input_modalities"]))
        if not runtime_caps.on_demand:
            reasons.append("backend does not provide an on-demand lifecycle")
        if (model is not None and model.backend == "colibri" and
                not (runtime_caps.explicit_load and runtime_caps.explicit_unload)):
            reasons.append("Colibri does not verify explicit Heavy load/unload lifecycle support")

        route = RuntimeRoute(
            id="rrt-" + uuid.uuid4().hex,
            requested_role=role.id,
            selected_role=role.id,
            model_alias=model.alias if model else "",
            backend=model.backend if model else "",
            runtime_id=role.runtime_id,
            profile=role.profile,
            state="unavailable" if reasons else "ready",
            requested=requested,
            reasons=tuple(dict.fromkeys(reasons)),
            backend_status=_plain(status),
            runtime_capabilities=_plain(runtime_caps),
            model_capabilities=_plain(model_caps) | {"profile_context": profile_context},
            created_at=now_utc(), task_id=task_id, _backend=backend)
        if persist:
            self._persist(route)
        return route

    @staticmethod
    def _requirement_reasons(profile_context, model_caps, *, context_tokens=0,
                             require_reasoning=False, require_tools=False,
                             input_modalities=("text",)):
        reasons = []
        context_tokens = max(0, int(context_tokens))
        if context_tokens and profile_context < context_tokens:
            reasons.append(
                f"requested context {context_tokens} exceeds profile context {profile_context}")
        if context_tokens and model_caps.context < context_tokens:
            reasons.append(
                f"requested context {context_tokens} exceeds backend model context "
                f"{model_caps.context}")
        modalities = set(input_modalities)
        if modalities - set(model_caps.input_modalities):
            reasons.append("model lacks input modalities: " +
                           ", ".join(sorted(modalities - set(model_caps.input_modalities))))
        if require_reasoning and model_caps.reasoning is not True:
            reasons.append("model reasoning capability is not verified")
        if require_tools and model_caps.tool_calls is not True:
            reasons.append("model tool-call capability is not verified")
        return reasons

    def require(self, route, *, context_tokens=0, require_reasoning=False,
                require_tools=False, input_modalities=("text",)):
        route.require_ready()
        model_caps = ModelCapabilities(
            model=str(route.model_capabilities.get("model", route.runtime_id)),
            context=int(route.model_capabilities.get("context", 0)),
            input_modalities=tuple(route.model_capabilities.get("input_modalities", ("text",))),
            output_modalities=tuple(route.model_capabilities.get("output_modalities", ("text",))),
            reasoning=route.model_capabilities.get("reasoning"),
            tool_calls=route.model_capabilities.get("tool_calls"))
        reasons = self._requirement_reasons(
            int(route.model_capabilities.get("profile_context", 0)), model_caps,
            context_tokens=context_tokens, require_reasoning=require_reasoning,
            require_tools=require_tools, input_modalities=input_modalities)
        if reasons:
            raise RuntimeRouteUnavailable("; ".join(reasons))
        return route

    @staticmethod
    def _persist(route):
        with transaction(immediate=True) as conn:
            conn.execute(
                """INSERT INTO runtime_routes(
                   id,task_id,requested_role,selected_role,model_alias,backend,runtime_id,
                   profile,state,requested_json,reasons_json,backend_status_json,
                   runtime_capabilities_json,model_capabilities_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (route.id, route.task_id, route.requested_role, route.selected_role,
                 route.model_alias, route.backend, route.runtime_id, route.profile,
                 route.state, json_dumps(route.requested), json_dumps(route.reasons),
                 json_dumps(route.backend_status), json_dumps(route.runtime_capabilities),
                 json_dumps(route.model_capabilities), route.created_at))
        if route.task_id:
            append_event(
                route.task_id, "routing",
                f"Runtime route {route.state}: {route.selected_role} -> "
                f"{route.backend or 'unavailable'}:{route.runtime_id}",
                role=route.selected_role,
                data={"runtime_route_id": route.id, "state": route.state,
                      "model": route.model_alias, "reasons": list(route.reasons)},
                visibility="normal" if route.ready else "quiet")

    def prepare(self, route: RuntimeRoute) -> LifecycleResult:
        route.require_ready()
        if route._backend is None:
            raise RuntimeRouteUnavailable("runtime backend instance is unavailable")
        result = route._backend.load(route.runtime_id)
        if result.state not in {"loaded", "on-demand", "already-loaded"}:
            raise RuntimeRouteUnavailable(f"backend load failed: {result.message}")
        return result

    def release(self, route: RuntimeRoute) -> LifecycleResult:
        route.require_ready()
        if route._backend is None:
            raise RuntimeRouteUnavailable("runtime backend instance is unavailable")
        return route._backend.unload(route.runtime_id)


def load_route(route_id: str) -> RuntimeRoute:
    ensure_state_store()
    with connect() as conn:
        row = conn.execute("SELECT * FROM runtime_routes WHERE id=?", (route_id,)).fetchone()
    if not row:
        raise KeyError(f"unknown runtime route: {route_id}")
    return RuntimeRoute(
        row["id"], row["requested_role"], row["selected_role"], row["model_alias"],
        row["backend"], row["runtime_id"], row["profile"], row["state"],
        json_loads(row["requested_json"], {}), tuple(json_loads(row["reasons_json"], [])),
        json_loads(row["backend_status_json"], {}),
        json_loads(row["runtime_capabilities_json"], {}),
        json_loads(row["model_capabilities_json"], {}), row["created_at"], row["task_id"])
