from contextlib import contextmanager
from dataclasses import dataclass
import json
import re
import tomllib

from .config import ROLE_REGISTRY_PATH, STATE_ROOT
from .file_transactions import (
    atomic_write_anchored_text,
    exclusive_file_lock,
    installation_transaction,
    read_anchored_text,
    regular_file_exists,
)
from .registry import ensure_registry

ROLE_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")
RUNTIME_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
LEGACY_ROLE_ALIASES = {
    "daily": "general",
    "coder": "builder",
}


@dataclass(frozen=True)
class RoleRecord:
    id: str
    display_name: str
    description: str
    enabled: bool
    runtime_id: str
    model: str
    profile: str
    execution: str
    capabilities: tuple[str, ...]
    aliases: tuple[str, ...]

    @classmethod
    def from_dict(cls, role_id, data):
        runtime_id = validate_runtime_id(data.get("runtime_id", role_id))
        return cls(
            id=role_id,
            display_name=data.get("display_name", role_id.title()),
            description=data.get("description", ""),
            enabled=bool(data.get("enabled", True)),
            runtime_id=runtime_id,
            model=data.get("model", ""),
            profile=data.get("profile", "normal"),
            execution=data.get("execution", "chat"),
            capabilities=tuple(data.get("capabilities", [])),
            aliases=tuple(data.get("aliases", [])),
        )


def _quote(value):
    return json.dumps(str(value), ensure_ascii=False)


def _array(values):
    return "[" + ", ".join(_quote(v) for v in values) + "]"


def validate_runtime_id(value):
    if not isinstance(value, str) or not RUNTIME_ID_RE.fullmatch(value):
        raise ValueError(f"invalid runtime id: {value!r}")
    return value


def default_role_registry():
    return {
        "version": 1,
        "generation": 0,
        "default_role": "general",
        "roles": {
            "general": {
                "display_name": "General",
                "description": "Conversation, planning, everyday reasoning and light assistance.",
                "enabled": True,
                "runtime_id": "daily",
                "model": "qwen3.5-9b",
                "profile": "normal",
                "execution": "chat",
                "capabilities": [
                    "conversation",
                    "planning",
                    "general-reasoning",
                    "light-tools",
                ],
                "aliases": ["daily"],
            },
            "builder": {
                "display_name": "Builder",
                "description": "Build, edit and repair code, projects, documents and automations.",
                "enabled": True,
                "runtime_id": "coder",
                "model": "kat-coder-v2.5-dev",
                "profile": "normal",
                "execution": "loop",
                "capabilities": [
                    "create",
                    "edit",
                    "project-work",
                    "code",
                    "artifact",
                ],
                "aliases": ["coder"],
            },
            "operator": {
                "display_name": "Operator",
                "description": "Tool-first execution on systems, services, files, networks and APIs.",
                "enabled": True,
                "runtime_id": "operator",
                "model": "agents-a1",
                "profile": "normal",
                "execution": "loop",
                "capabilities": [
                    "tools",
                    "system-action",
                    "network-action",
                    "external-action",
                ],
                "aliases": [],
            },
            "oracle": {
                "display_name": "Oracle",
                "description": "Deep analysis, review, architecture and second-opinion reasoning.",
                "enabled": False,
                "runtime_id": "oracle",
                "model": "",
                "profile": "normal",
                "execution": "delegate",
                "capabilities": [
                    "deep-reasoning",
                    "review",
                    "research",
                    "adjudication",
                ],
                "aliases": [],
            },
        },
    }


def serialize_role_registry(data):
    lines = [
        "version = " + str(int(data.get("version", 1))),
        "generation = " + str(int(data.get("generation", 0))),
        f"default_role = {_quote(data.get('default_role', 'general'))}",
    ]

    for role_id in sorted(data.get("roles", {})):
        role = data["roles"][role_id]
        lines.extend([
            "",
            f"[roles.{_quote(role_id)}]",
            f"display_name = {_quote(role.get('display_name', role_id.title()))}",
            f"description = {_quote(role.get('description', ''))}",
            f"enabled = {'true' if role.get('enabled', True) else 'false'}",
            f"runtime_id = {_quote(role.get('runtime_id', role_id))}",
            f"model = {_quote(role.get('model', ''))}",
            f"profile = {_quote(role.get('profile', 'normal'))}",
            f"execution = {_quote(role.get('execution', 'chat'))}",
            f"capabilities = {_array(role.get('capabilities', []))}",
            f"aliases = {_array(role.get('aliases', []))}",
        ])

    return "\n".join(lines) + "\n"


def _role_registry_lock_path():
    return ROLE_REGISTRY_PATH.parent / ".tars-registries.lock"


@contextmanager
def role_registry_transaction():
    with installation_transaction(STATE_ROOT):
        with exclusive_file_lock(_role_registry_lock_path()) as anchor:
            yield anchor


def _save_role_registry_unlocked(data, anchor):
    _validate_registry(data)
    exists = regular_file_exists(anchor, ROLE_REGISTRY_PATH.name)
    if exists:
        current_generation = _load_role_registry_unlocked(anchor).get("generation", 0)
        supplied_generation = data.get("generation")
        if supplied_generation is None or supplied_generation != current_generation:
            raise RuntimeError("stale role registry update refused")
        next_generation = current_generation + 1
    else:
        next_generation = data.get("generation", 0)
    candidate = json.loads(json.dumps(data))
    candidate["generation"] = next_generation
    atomic_write_anchored_text(
        anchor, ROLE_REGISTRY_PATH.name, serialize_role_registry(candidate)
    )
    data["generation"] = next_generation


def save_role_registry(data):
    with role_registry_transaction() as anchor:
        _save_role_registry_unlocked(data, anchor)


def _validate_registry(data):
    generation = data.get("generation", 0)
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("role registry generation must be a non-negative integer")
    roles = data.get("roles", {})
    if not roles:
        raise ValueError("at least one role is required")

    default = data.get("default_role", "")
    if default not in roles:
        raise ValueError("default role must exist")
    if not roles[default].get("enabled", True):
        raise ValueError("default role must be enabled")

    models = ensure_registry().get("models", {})
    seen_names = set()

    for role_id, role in roles.items():
        if not ROLE_ID_RE.match(role_id):
            raise ValueError(f"invalid role id: {role_id}")
        validate_runtime_id(role.get("runtime_id", role_id))

        model = role.get("model", "")
        if model and model not in models:
            raise ValueError(f"role {role_id} references unknown model {model}")

        names = [role_id, *role.get("aliases", [])]
        for name in names:
            key = name.lower()
            if key in seen_names:
                raise ValueError(f"duplicate role/alias name: {name}")
            seen_names.add(key)


def ensure_role_registry():
    with role_registry_transaction() as anchor:
        if not regular_file_exists(anchor, ROLE_REGISTRY_PATH.name):
            _save_role_registry_unlocked(default_role_registry(), anchor)
        return _load_role_registry_unlocked(anchor)


def _load_role_registry_unlocked(anchor):
    data = tomllib.loads(read_anchored_text(anchor, ROLE_REGISTRY_PATH.name))
    data.setdefault("roles", {})
    data.setdefault("default_role", "general")
    data.setdefault("generation", 0)
    _validate_registry(data)
    return data


def load_role_registry():
    with role_registry_transaction() as anchor:
        return _load_role_registry_unlocked(anchor)


def list_roles(include_disabled=True):
    data = ensure_role_registry()
    records = [
        RoleRecord.from_dict(role_id, info)
        for role_id, info in data["roles"].items()
    ]
    records.sort(key=lambda r: r.id)
    if include_disabled:
        return records
    return [r for r in records if r.enabled]


def get_role(role_id):
    with role_registry_transaction():
        data = ensure_role_registry()
        resolved = _resolve_role_id_from(data, role_id)
        return RoleRecord.from_dict(resolved, data["roles"][resolved])


def _resolve_role_id_from(data, name):
    key = name.strip().lower()

    if key in data["roles"]:
        return key

    legacy = LEGACY_ROLE_ALIASES.get(key)
    if legacy and legacy in data["roles"]:
        return legacy

    for role_id, info in data["roles"].items():
        if key in {x.lower() for x in info.get("aliases", [])}:
            return role_id

    raise KeyError(f"unknown role: {name}")


def resolve_role_id(name):
    with role_registry_transaction():
        return _resolve_role_id_from(ensure_role_registry(), name)


def default_role_id():
    return ensure_role_registry().get("default_role", "general")


def create_role(
    role_id,
    *,
    display_name=None,
    model="",
    profile="normal",
    execution="chat",
    runtime_id=None,
    capabilities=None,
    aliases=None,
    description="",
):
    role_id = role_id.lower()
    with role_registry_transaction():
        data = ensure_role_registry()
        if role_id in data["roles"]:
            raise ValueError(f"role already exists: {role_id}")

        data["roles"][role_id] = {
            "display_name": display_name or role_id.title(),
            "description": description,
            "enabled": bool(model),
            "runtime_id": runtime_id or role_id,
            "model": model,
            "profile": profile,
            "execution": execution,
            "capabilities": list(capabilities or []),
            "aliases": list(aliases or []),
        }
        save_role_registry(data)
        return RoleRecord.from_dict(role_id, data["roles"][role_id])


def remove_role(role_id):
    with role_registry_transaction():
        data = ensure_role_registry()
        role_id = _resolve_role_id_from(data, role_id)
        if role_id == data.get("default_role"):
            raise ValueError(
                f"cannot remove default role {role_id!r}; set another default first"
            )
        if len(data["roles"]) <= 1:
            raise ValueError("at least one role is required")
        del data["roles"][role_id]
        save_role_registry(data)


def set_role_enabled(role_id, enabled):
    with role_registry_transaction():
        data = ensure_role_registry()
        role_id = _resolve_role_id_from(data, role_id)
        if not enabled and role_id == data.get("default_role"):
            raise ValueError("cannot disable the default role")
        if enabled and not data["roles"][role_id].get("model"):
            raise ValueError("cannot enable an unbound role")
        data["roles"][role_id]["enabled"] = bool(enabled)
        save_role_registry(data)


def set_default_role(role_id):
    with role_registry_transaction():
        data = ensure_role_registry()
        role_id = _resolve_role_id_from(data, role_id)
        if not data["roles"][role_id].get("enabled", True):
            raise ValueError("default role must be enabled")
        data["default_role"] = role_id
        save_role_registry(data)


def bind_model(role_id, model_alias):
    with role_registry_transaction():
        models = ensure_registry().get("models", {})
        if model_alias and model_alias not in models:
            raise ValueError(f"unknown model alias: {model_alias}")
        data = ensure_role_registry()
        role_id = _resolve_role_id_from(data, role_id)
        data["roles"][role_id]["model"] = model_alias
        save_role_registry(data)


def set_role_profile(role_id, profile):
    if profile not in {"compact", "normal", "extended"}:
        raise ValueError("profile must be compact, normal or extended")
    with role_registry_transaction():
        data = ensure_role_registry()
        role_id = _resolve_role_id_from(data, role_id)
        data["roles"][role_id]["profile"] = profile
        save_role_registry(data)
