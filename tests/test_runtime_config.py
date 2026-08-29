from pathlib import Path
from types import SimpleNamespace

import pytest

from tars import runtime_config


def _role(role_id, runtime_id, model, *, execution="chat", profile="normal"):
    return SimpleNamespace(
        id=role_id,
        display_name=role_id.title(),
        runtime_id=runtime_id,
        execution=execution,
        model=model,
        profile=profile,
    )


def _model(alias, sha="abc"):
    return SimpleNamespace(
        alias=alias,
        name=alias.upper(),
        path=Path(f"/models/{alias}.gguf"),
        sha256=sha,
        backend="llama.cpp",
        quant="Q4_K_M",
        native_context=262144,
    )


def _calibration(sha="abc", *, context=65536, ngl=-1, tensor_overrides=None):
    profile = {
        "context": context,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "cpus": "0-11",
        "threads": 2,
        "batch_threads": 2,
        "ngl": ngl,
    }
    if tensor_overrides:
        profile["tensor_overrides"] = tensor_overrides
    return {
        "status": "ready",
        "model_sha256": sha,
        "profiles": {"normal": profile},
    }


def test_render_enforces_zero_idle_and_calibration(monkeypatch):
    role = _role("general", "daily", "qwen")
    monkeypatch.setattr(runtime_config, "list_roles", lambda include_disabled=False: [role])
    monkeypatch.setattr(runtime_config, "get_model", lambda alias: _model(alias))
    monkeypatch.setattr(runtime_config, "load_calibration", lambda alias: _calibration())

    rendered = runtime_config.render_runtime_config(runtime_config.build_runtime_plan())

    assert "globalTTL: 30" in rendered
    assert "unloadTimeout: 10" in rendered
    assert "sendLoadingState: false" in rendered
    assert "performance:\n  disabled: true" in rendered
    assert "-c 65536" in rendered
    assert "-ngl all" in rendered
    assert "-ctk q8_0 -ctv q8_0" in rendered
    assert "--port ${PORT}" in rendered


def test_render_preserves_tensor_overrides(monkeypatch):
    role = _role("builder", "coder", "kat", execution="loop")
    monkeypatch.setattr(runtime_config, "list_roles", lambda include_disabled=False: [role])
    monkeypatch.setattr(runtime_config, "get_model", lambda alias: _model(alias))
    monkeypatch.setattr(
        runtime_config,
        "load_calibration",
        lambda alias: _calibration(
            context=196608,
            ngl=41,
            tensor_overrides=r"blk\.5.*=CPU",
        ),
    )

    rendered = runtime_config.render_runtime_config(runtime_config.build_runtime_plan())

    assert "-c 196608 -ngl 41" in rendered
    assert "-ot 'blk\\.5.*=CPU'" in rendered
    assert "tools: true" in rendered
    assert "context: 196608" in rendered


def test_duplicate_runtime_ids_are_rejected(monkeypatch):
    roles = [
        _role("one", "shared", "a"),
        _role("two", "shared", "b"),
    ]
    monkeypatch.setattr(runtime_config, "list_roles", lambda include_disabled=False: roles)
    monkeypatch.setattr(runtime_config, "get_model", lambda alias: _model(alias, sha=alias))
    monkeypatch.setattr(runtime_config, "load_calibration", lambda alias: _calibration(sha=alias))

    with pytest.raises(ValueError, match="duplicate enabled runtime id"):
        runtime_config.build_runtime_plan()


def test_apply_restores_previous_config_when_health_check_fails(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("old config\n", encoding="utf-8")
    backup_root = tmp_path / "backups"

    plan = SimpleNamespace(
        models=(SimpleNamespace(runtime_id="daily"),)
    )

    monkeypatch.setattr(runtime_config, "LLAMA_SWAP_CONFIG_PATH", config_path)
    monkeypatch.setattr(runtime_config, "RUNTIME_CONFIG_BACKUP_ROOT", backup_root)
    monkeypatch.setattr(runtime_config, "_service_active", lambda: True)
    monkeypatch.setattr(
        runtime_config,
        "build_runtime_plan",
        lambda **kwargs: plan,
    )
    monkeypatch.setattr(
        runtime_config,
        "render_runtime_config",
        lambda plan=None: "new config\n",
    )

    restarts = []
    monkeypatch.setattr(
        runtime_config,
        "_restart_service",
        lambda: restarts.append("restart"),
    )

    waits = {"count": 0}

    def fake_wait(*args, **kwargs):
        waits["count"] += 1
        if waits["count"] == 1:
            raise RuntimeError("health probe failed")

    monkeypatch.setattr(runtime_config, "_wait_healthy", fake_wait)

    with pytest.raises(RuntimeError, match="previous config restored"):
        runtime_config.apply_runtime_config(object())

    assert config_path.read_text(encoding="utf-8") == "old config\n"
    assert restarts == ["restart", "restart"]
    assert len(list(backup_root.glob("llama-swap-*.yaml"))) == 1


def test_switch_restores_role_registry_when_runtime_apply_fails(monkeypatch):
    import copy

    before = {
        "roles": {
            "general": {
                "model": "qwen",
                "profile": "normal",
                "enabled": True,
            }
        }
    }
    saved = []

    monkeypatch.setattr(runtime_config, "resolve_role_id", lambda role: "general")
    monkeypatch.setattr(
        runtime_config,
        "get_role",
        lambda role: SimpleNamespace(model="qwen", profile="normal"),
    )
    monkeypatch.setattr(
        runtime_config,
        "get_model",
        lambda alias: SimpleNamespace(alias="qwen"),
    )
    monkeypatch.setattr(
        runtime_config,
        "load_calibration",
        lambda alias: {
            "status": "ready",
            "profiles": {
                "normal": {},
                "compact": {},
            },
        },
    )
    monkeypatch.setattr(
        runtime_config,
        "load_role_registry",
        lambda: copy.deepcopy(before),
    )
    monkeypatch.setattr(
        runtime_config,
        "save_role_registry",
        lambda data: saved.append(copy.deepcopy(data)),
    )

    def fail_apply(cfg):
        raise RuntimeError("runtime apply failed")

    monkeypatch.setattr(runtime_config, "apply_runtime_config", fail_apply)

    with pytest.raises(RuntimeError, match="runtime apply failed"):
        runtime_config.switch_role_runtime(
            object(),
            "general",
            profile_name="compact",
        )

    assert saved[0]["roles"]["general"]["profile"] == "compact"
    assert saved[-1] == before


def test_zero_bound_roles_render_valid_empty_model_map(monkeypatch):
    monkeypatch.setattr(runtime_config, "list_roles", lambda include_disabled=False: [])
    plan = runtime_config.build_runtime_plan()
    assert plan.models == ()
    assert "models: {}" in runtime_config.render_runtime_config(plan)


def test_apply_preserves_stopped_service_state(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    plan = SimpleNamespace(models=())
    monkeypatch.setattr(runtime_config, "LLAMA_SWAP_CONFIG_PATH", config_path)
    monkeypatch.setattr(runtime_config, "RUNTIME_CONFIG_BACKUP_ROOT", tmp_path / "backups")
    monkeypatch.setattr(runtime_config, "build_runtime_plan", lambda **kwargs: plan)
    monkeypatch.setattr(runtime_config, "render_runtime_config", lambda plan=None: "models: {}\n")
    monkeypatch.setattr(runtime_config, "_service_active", lambda: False)
    actions = []
    monkeypatch.setattr(runtime_config, "start_runtime_service", lambda: actions.append("start"))
    monkeypatch.setattr(runtime_config, "stop_runtime_service", lambda: actions.append("stop"))
    monkeypatch.setattr(runtime_config, "_wait_healthy", lambda *args, **kwargs: None)

    runtime_config.apply_runtime_config(object())

    assert actions == ["start", "stop"]


def test_unassign_is_transactional(monkeypatch):
    before = {"roles": {"oracle": {"model": "qwen", "profile": "normal", "enabled": True}}}
    saved = []
    monkeypatch.setattr(runtime_config, "resolve_role_id", lambda role: "oracle")
    monkeypatch.setattr(runtime_config, "get_role", lambda role: SimpleNamespace(model="qwen", profile="normal"))
    monkeypatch.setattr(runtime_config, "load_role_registry", lambda: __import__("copy").deepcopy(before))
    monkeypatch.setattr(runtime_config, "save_role_registry", lambda data: saved.append(__import__("copy").deepcopy(data)))
    monkeypatch.setattr(runtime_config, "apply_runtime_config", lambda cfg: SimpleNamespace(changed=True))

    result = runtime_config.switch_role_runtime(object(), "oracle", unassign=True)

    assert result.model_alias == ""
    assert saved[-1]["roles"]["oracle"]["model"] == ""
    assert saved[-1]["roles"]["oracle"]["enabled"] is True


def test_idempotent_apply_validates_and_restores_stopped_service(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("models: {}\n", encoding="utf-8")
    plan = SimpleNamespace(models=())
    monkeypatch.setattr(runtime_config, "LLAMA_SWAP_CONFIG_PATH", config_path)
    monkeypatch.setattr(runtime_config, "build_runtime_plan", lambda **kwargs: plan)
    monkeypatch.setattr(runtime_config, "render_runtime_config", lambda plan=None: "models: {}\n")
    monkeypatch.setattr(runtime_config, "_service_active", lambda: False)
    actions = []
    monkeypatch.setattr(runtime_config, "start_runtime_service", lambda: actions.append("start"))
    monkeypatch.setattr(runtime_config, "stop_runtime_service", lambda: actions.append("stop"))
    monkeypatch.setattr(runtime_config, "_wait_healthy", lambda *args, **kwargs: actions.append("healthy"))

    result = runtime_config.apply_runtime_config(object())

    assert not result.changed
    assert actions == ["start", "healthy", "stop"]
