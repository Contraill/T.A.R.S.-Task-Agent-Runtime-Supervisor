from __future__ import annotations

import multiprocessing
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from tars import backup, model_lifecycle, registry, roles, runtime_config, themes
from tars.file_transactions import atomic_write_anchored_text, exclusive_file_lock


def _model(alias: str) -> dict:
    return {
        "name": alias.title(),
        "path": f"/models/{alias}.gguf",
        "sha256": alias * 8,
        "backend": "llama.cpp",
        "quant": "Q4_K_M",
        "native_context": 4096,
        "thinking_control": "unknown",
    }


def _configure_registry_paths(root: str) -> None:
    base = Path(root)
    registry.REGISTRY_PATH = base / "model-registry.toml"
    registry.STATE_ROOT = base / "state"
    roles.ROLE_REGISTRY_PATH = base / "role-registry.toml"
    roles.STATE_ROOT = base / "state"


def _hold_model_update(root, ready, release) -> None:
    _configure_registry_paths(root)
    with registry.registry_transaction():
        data = registry.ensure_registry()
        ready.set()
        if not release.wait(10):
            raise TimeoutError("test did not release registry transaction")
        data["models"]["one"]["thinking_control"] = "toggle"
        registry.save_registry(data)


def _set_second_model(root, ready, started, finished) -> None:
    _configure_registry_paths(root)
    if not ready.wait(10):
        raise TimeoutError("registry holder did not start")
    started.set()
    registry.set_thinking_control("two", "toggle")
    finished.set()


def _hold_role_update(root, ready, release) -> None:
    _configure_registry_paths(root)
    with roles.role_registry_transaction():
        data = roles.ensure_role_registry()
        ready.set()
        if not release.wait(10):
            raise TimeoutError("test did not release role registry transaction")
        data["roles"]["general"]["profile"] = "compact"
        roles.save_role_registry(data)


def _set_second_role(root, ready, started, finished) -> None:
    _configure_registry_paths(root)
    if not ready.wait(10):
        raise TimeoutError("role registry holder did not start")
    started.set()
    roles.set_role_profile("builder", "extended")
    finished.set()


def _bind_or_remove_model(root, start, operation, outcomes) -> None:
    _configure_registry_paths(root)
    if not start.wait(10):
        raise TimeoutError("model/role race did not start")
    try:
        if operation == "bind":
            roles.bind_model("general", "shared")
        else:
            model_lifecycle.remove_model("shared")
    except (KeyError, ValueError) as exc:
        outcomes.put((operation, type(exc).__name__))
    else:
        outcomes.put((operation, "ok"))


@pytest.fixture
def isolated_registries(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "model-registry.toml")
    monkeypatch.setattr(registry, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(roles, "ROLE_REGISTRY_PATH", tmp_path / "role-registry.toml")
    monkeypatch.setattr(roles, "STATE_ROOT", tmp_path / "state")
    registry.save_registry({
        "version": 3,
        "models": {"one": _model("one"), "two": _model("two")},
    })
    roles.save_role_registry({
        "version": 1,
        "default_role": "general",
        "roles": {
            "general": {"enabled": True, "runtime_id": "general", "model": ""},
            "builder": {"enabled": True, "runtime_id": "builder", "model": ""},
        },
    })
    return tmp_path


def test_process_registry_transactions_preserve_both_updates(isolated_registries):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    started = context.Event()
    finished = context.Event()
    holder = context.Process(
        target=_hold_model_update,
        args=(str(isolated_registries), ready, release),
    )
    contender = context.Process(
        target=_set_second_model,
        args=(str(isolated_registries), ready, started, finished),
    )
    holder.start()
    contender.start()
    try:
        assert started.wait(10)
        assert not finished.wait(0.2)
        release.set()
        holder.join(timeout=10)
        contender.join(timeout=10)
        assert holder.exitcode == 0
        assert contender.exitcode == 0
    finally:
        release.set()
        for process in (holder, contender):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    data = registry.load_registry()
    assert data["models"]["one"]["thinking_control"] == "toggle"
    assert data["models"]["two"]["thinking_control"] == "toggle"


def test_transaction_refuses_parent_directory_replacement(tmp_path):
    parent = tmp_path / "mutable"
    parent.mkdir()
    displaced = tmp_path / "displaced"

    with pytest.raises(RuntimeError, match="transaction directory changed"):
        with exclusive_file_lock(parent / ".lock") as anchor:
            parent.rename(displaced)
            parent.mkdir()
            atomic_write_anchored_text(anchor, "state.toml", "value = 1\n")

    assert not (parent / "state.toml").exists()
    assert (displaced / "state.toml").read_text(encoding="utf-8") == "value = 1\n"


def test_process_role_transactions_preserve_both_updates(isolated_registries):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    started = context.Event()
    finished = context.Event()
    holder = context.Process(
        target=_hold_role_update,
        args=(str(isolated_registries), ready, release),
    )
    contender = context.Process(
        target=_set_second_role,
        args=(str(isolated_registries), ready, started, finished),
    )
    holder.start()
    contender.start()
    try:
        assert started.wait(10)
        assert not finished.wait(0.2)
        release.set()
        holder.join(timeout=10)
        contender.join(timeout=10)
        assert holder.exitcode == 0
        assert contender.exitcode == 0
    finally:
        release.set()
        for process in (holder, contender):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

    data = roles.load_role_registry()
    assert data["roles"]["general"]["profile"] == "compact"
    assert data["roles"]["builder"]["profile"] == "extended"


def test_backup_lock_excludes_registry_writer(isolated_registries):
    ready = threading.Event()
    release = threading.Event()
    started = threading.Event()
    finished = threading.Event()
    paths = SimpleNamespace(state_root=registry.STATE_ROOT)

    def hold_backup_lock():
        with backup._restore_lock(paths):
            ready.set()
            assert release.wait(10)

    def mutate_registry():
        assert ready.wait(10)
        started.set()
        registry.set_thinking_control("one", "toggle")
        finished.set()

    holder = threading.Thread(target=hold_backup_lock)
    contender = threading.Thread(target=mutate_registry)
    holder.start()
    contender.start()
    try:
        assert started.wait(10)
        assert not finished.wait(0.2)
        release.set()
        holder.join(timeout=10)
        contender.join(timeout=10)
        assert not holder.is_alive() and not contender.is_alive()
    finally:
        release.set()

    assert registry.get_model("one").thinking_control == "toggle"


def test_stale_registry_snapshots_are_refused(isolated_registries):
    model_first = registry.load_registry()
    model_stale = registry.load_registry()
    model_first["models"]["one"]["thinking_control"] = "toggle"
    registry.save_registry(model_first)
    model_stale["models"]["two"]["thinking_control"] = "toggle"
    with pytest.raises(RuntimeError, match="stale model registry"):
        registry.save_registry(model_stale)

    role_first = roles.load_role_registry()
    role_stale = roles.load_role_registry()
    role_first["roles"]["general"]["profile"] = "compact"
    roles.save_role_registry(role_first)
    role_stale["roles"]["builder"]["profile"] = "extended"
    with pytest.raises(RuntimeError, match="stale role registry"):
        roles.save_role_registry(role_stale)


def test_legacy_registries_gain_generation_on_next_commit(isolated_registries):
    for path in (registry.REGISTRY_PATH, roles.ROLE_REGISTRY_PATH):
        legacy = "\n".join(
            line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("generation = ")
        ) + "\n"
        path.write_text(legacy, encoding="utf-8")

    model_data = registry.load_registry()
    assert model_data["generation"] == 0
    registry.save_registry(model_data)
    assert registry.load_registry()["generation"] == 1

    role_data = roles.load_role_registry()
    assert role_data["generation"] == 0
    roles.save_role_registry(role_data)
    assert roles.load_role_registry()["generation"] == 1


def test_model_removal_and_role_binding_cannot_create_dangling_reference(
        isolated_registries):
    data = registry.load_registry()
    data["models"] = {"shared": _model("shared")}
    registry.save_registry(data)

    context = multiprocessing.get_context("spawn")
    start = context.Event()
    outcomes = context.Queue()
    workers = [
        context.Process(
            target=_bind_or_remove_model,
            args=(str(isolated_registries), start, operation, outcomes),
        )
        for operation in ("bind", "remove")
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(timeout=10)
    try:
        assert all(worker.exitcode == 0 for worker in workers)
        results = dict(outcomes.get(timeout=2) for _ in workers)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)
        outcomes.close()

    models = registry.load_registry()["models"]
    bound = roles.get_role("general").model
    assert sorted(results.values()) == ["ValueError", "ok"]
    assert ("shared" in models and bound == "shared") or (
        "shared" not in models and bound == ""
    )


def test_theme_and_logo_updates_do_not_erase_each_other(monkeypatch, tmp_path):
    monkeypatch.setattr(themes, "UI_PREFS_PATH", tmp_path / "ui.toml")
    monkeypatch.setattr(themes, "THEME_ROOT", tmp_path / "themes")
    monkeypatch.setattr(themes, "STATE_ROOT", tmp_path / "state")
    themes.ensure_ui_store()
    ready = threading.Event()
    release = threading.Event()
    started = threading.Event()
    finished = threading.Event()

    def hold_theme_update():
        with exclusive_file_lock(themes._prefs_lock_path()) as anchor:
            prefs = themes._load_prefs_unlocked(anchor)
            ready.set()
            assert release.wait(10)
            prefs["theme"] = "monochrome"
            themes._write_prefs_unlocked(prefs, anchor)

    def set_logo():
        assert ready.wait(10)
        started.set()
        themes.set_logo("minimal")
        finished.set()

    holder = threading.Thread(target=hold_theme_update)
    contender = threading.Thread(target=set_logo)
    holder.start()
    contender.start()
    try:
        assert started.wait(10)
        assert not finished.wait(0.2)
        release.set()
        holder.join(timeout=10)
        contender.join(timeout=10)
        assert not holder.is_alive() and not contender.is_alive()
    finally:
        release.set()

    assert themes.load_ui_prefs() == {
        "theme": "monochrome",
        "logo": "minimal",
    }


def test_failed_runtime_apply_cannot_rollback_over_concurrent_success(
        monkeypatch, tmp_path):
    config_path = tmp_path / "runtime" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(runtime_config, "LLAMA_SWAP_CONFIG_PATH", config_path)
    monkeypatch.setattr(
        runtime_config, "RUNTIME_CONFIG_BACKUP_ROOT", tmp_path / "backups"
    )
    monkeypatch.setattr(roles, "ROLE_REGISTRY_PATH", tmp_path / "role-registry.toml")
    monkeypatch.setattr(
        runtime_config,
        "build_runtime_plan",
        lambda **kwargs: SimpleNamespace(models=()),
    )
    monkeypatch.setattr(
        runtime_config,
        "render_runtime_config",
        lambda plan: f"{threading.current_thread().name}\n",
    )
    monkeypatch.setattr(runtime_config, "_service_active", lambda: True)
    monkeypatch.setattr(runtime_config, "_restart_service", lambda: None)
    first_waiting = threading.Event()
    release_failure = threading.Event()
    second_started = threading.Event()
    second_finished = threading.Event()
    failures = []
    first_probes = 0

    def wait_healthy(*args, **kwargs):
        nonlocal first_probes
        if threading.current_thread().name == "first-apply":
            first_probes += 1
            if first_probes == 1:
                first_waiting.set()
                assert release_failure.wait(10)
                raise RuntimeError("injected health failure")

    monkeypatch.setattr(runtime_config, "_wait_healthy", wait_healthy)

    def first_apply():
        try:
            runtime_config.apply_runtime_config(object())
        except RuntimeError as exc:
            failures.append(str(exc))

    def second_apply():
        assert first_waiting.wait(10)
        second_started.set()
        runtime_config.apply_runtime_config(object())
        second_finished.set()

    first = threading.Thread(target=first_apply, name="first-apply")
    second = threading.Thread(target=second_apply, name="second-apply")
    first.start()
    second.start()
    try:
        assert second_started.wait(10)
        assert not second_finished.wait(0.2)
        release_failure.set()
        first.join(timeout=10)
        second.join(timeout=10)
        assert not first.is_alive() and not second.is_alive()
    finally:
        release_failure.set()

    assert failures and "previous config restored" in failures[0]
    assert config_path.read_text(encoding="utf-8") == "second-apply\n"
