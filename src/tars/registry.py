import json
import os
import tempfile
import tomllib
from pathlib import Path

from .config import DATA_ROOT, REGISTRY_PATH
from .models import ModelRecord


def _quote(value):
    return json.dumps(str(value), ensure_ascii=False)


def default_registry():
    models_root = DATA_ROOT / "models"
    return {
        "version": 2,
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
    lines = ["version = " + str(int(data.get("version", 2)))]

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


def save_registry(data):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_registry(data)

    fd, tmp_name = tempfile.mkstemp(
        prefix=".model-registry.",
        suffix=".toml",
        dir=REGISTRY_PATH.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, REGISTRY_PATH)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def ensure_registry():
    if not REGISTRY_PATH.exists():
        save_registry(default_registry())
    return load_registry()


def load_registry():
    with REGISTRY_PATH.open("rb") as handle:
        data = tomllib.load(handle)

    data.setdefault("models", {})
    for info in data["models"].values():
        info.setdefault("backend", "llama.cpp")
    return data


def models():
    data = ensure_registry()
    return {
        alias: ModelRecord.from_dict(alias, info)
        for alias, info in data["models"].items()
    }


def get_model(alias):
    available = models()
    if alias not in available:
        raise KeyError(f"unknown model alias: {alias}")
    return available[alias]


def set_thinking_control(alias, control):
    if control not in {"unknown", "toggle"}:
        raise ValueError("thinking control must be unknown or toggle")
    data = ensure_registry()
    if alias not in data["models"]:
        raise KeyError(f"unknown model alias: {alias}")
    data["models"][alias]["thinking_control"] = control
    save_registry(data)
    return get_model(alias)


def role_for_alias(alias):
    from .roles import list_roles

    return [
        role.id for role in list_roles()
        if role.model == alias
    ]
