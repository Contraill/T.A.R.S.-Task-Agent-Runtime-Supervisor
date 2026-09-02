import json
import re
import tomllib
from contextlib import contextmanager

from .config import DATA_ROOT, REGISTRY_PATH, STATE_ROOT
from .file_transactions import (
    atomic_write_anchored_text,
    exclusive_file_lock,
    installation_transaction,
    read_anchored_text,
    regular_file_exists,
)
from .models import ModelRecord

MODEL_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_model_alias(value):
    if not isinstance(value, str) or not MODEL_ALIAS_RE.fullmatch(value):
        raise ValueError(f"invalid model alias: {value!r}")
    return value


def _validate_registry(data):
    generation = data.get("generation", 0)
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("model registry generation must be a non-negative integer")
    models = data.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("model registry models must be a table")
    for alias in models:
        validate_model_alias(alias)


def _quote(value):
    return json.dumps(str(value), ensure_ascii=False)


def default_registry():
    models_root = DATA_ROOT / "models"
    return {
        "version": 2,
        "generation": 0,
        "models": {
            "qwen3.5-9b": {
                "name": "Qwen3.5-9B",
                "path": str(models_root / "daily/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf"),
                "sha256": "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
                "backend": "llama.cpp",
                "quant": "Q4_K_M",
                "native_context": 262144,
                "thinking_control": "toggle",
            },
            "kat-coder-v2.5-dev": {
                "name": "KAT-Coder V2.5-Dev 35B-A3B",
                "path": str(models_root / "coder/kat-coder-v2.5-dev/KAT-Coder-V2.5-Dev.i1-Q4_K_M.gguf"),
                "sha256": "36b86b60f2eb38e69c1ceaeaa713eb14a5462717b5e67b070cb9b7c8b087c730",
                "backend": "llama.cpp",
                "quant": "Q4_K_M",
                "native_context": 262144,
                "thinking_control": "toggle",
            },
            "agents-a1": {
                "name": "Agents-A1 35B-A3B",
                "path": str(models_root / "operator/agents-a1/Agents-A1-Q4_K_M.gguf"),
                "sha256": "31aefa25b7e1edbde436e643e2b5e3f6e57820a4811d97b131130e48ff0772c2",
                "backend": "llama.cpp",
                "quant": "Q4_K_M",
                "native_context": 262144,
                "thinking_control": "toggle",
            },
        },
    }


def serialize_registry(data):
    lines = [
        "version = " + str(int(data.get("version", 2))),
        "generation = " + str(int(data.get("generation", 0))),
    ]

    for alias in sorted(data.get("models", {})):
        model = data["models"][alias]
        lines.extend([
            "",
            f"[models.{_quote(alias)}]",
            f"name = {_quote(model['name'])}",
            f"path = {_quote(model['path'])}",
            f"sha256 = {_quote(model['sha256'])}",
            f"backend = {_quote(model.get('backend', 'llama.cpp'))}",
            f"quant = {_quote(model.get('quant', 'unknown'))}",
            f"native_context = {int(model.get('native_context', 0))}",
        ])
        for key in ("source", "source_revision", "license", "artifact_sha256"):
            if key in model:
                lines.append(f"{key} = {_quote(model[key])}")
        if "thinking_control" in model:
            lines.append(f"thinking_control = {_quote(model['thinking_control'])}")
        for key in ("size",):
            if key in model:
                lines.append(f"{key} = {int(model[key])}")
        for key in ("integrity_verified", "runtime_compatible"):
            if key in model:
                lines.append(f"{key} = {'true' if model[key] else 'false'}")

    return "\n".join(lines) + "\n"


def _registry_lock_path():
    return REGISTRY_PATH.parent / ".tars-registries.lock"


@contextmanager
def registry_transaction():
    with installation_transaction(STATE_ROOT):
        with exclusive_file_lock(_registry_lock_path()) as anchor:
            yield anchor


def _save_registry_unlocked(data, anchor):
    _validate_registry(data)
    exists = regular_file_exists(anchor, REGISTRY_PATH.name)
    if exists:
        current_generation = _load_registry_unlocked(anchor).get("generation", 0)
        supplied_generation = data.get("generation")
        if supplied_generation is None or supplied_generation != current_generation:
            raise RuntimeError("stale model registry update refused")
        next_generation = current_generation + 1
    else:
        next_generation = data.get("generation", 0)
    candidate = json.loads(json.dumps(data))
    candidate["generation"] = next_generation
    atomic_write_anchored_text(
        anchor, REGISTRY_PATH.name, serialize_registry(candidate)
    )
    data["generation"] = next_generation


def save_registry(data):
    with registry_transaction() as anchor:
        _save_registry_unlocked(data, anchor)


def ensure_registry():
    with registry_transaction() as anchor:
        if not regular_file_exists(anchor, REGISTRY_PATH.name):
            _save_registry_unlocked(default_registry(), anchor)
        return _load_registry_unlocked(anchor)


def _load_registry_unlocked(anchor):
    data = tomllib.loads(read_anchored_text(anchor, REGISTRY_PATH.name))
    data.setdefault("models", {})
    data.setdefault("generation", 0)
    _validate_registry(data)
    for info in data["models"].values():
        info.setdefault("backend", "llama.cpp")
    return data


def load_registry():
    with registry_transaction() as anchor:
        return _load_registry_unlocked(anchor)


def models():
    data = ensure_registry()
    return {
        alias: ModelRecord.from_dict(alias, info)
        for alias, info in data["models"].items()
    }


def get_model(alias):
    validate_model_alias(alias)
    available = models()
    if alias not in available:
        raise KeyError(f"unknown model alias: {alias}")
    return available[alias]


def set_thinking_control(alias, control):
    validate_model_alias(alias)
    if control not in {"unknown", "toggle"}:
        raise ValueError("thinking control must be unknown or toggle")
    with registry_transaction():
        data = ensure_registry()
        if alias not in data["models"]:
            raise KeyError(f"unknown model alias: {alias}")
        data["models"][alias]["thinking_control"] = control
        save_registry(data)
        return ModelRecord.from_dict(alias, data["models"][alias])


def role_for_alias(alias):
    validate_model_alias(alias)
    from .roles import list_roles

    return [
        role.id for role in list_roles()
        if role.model == alias
    ]
