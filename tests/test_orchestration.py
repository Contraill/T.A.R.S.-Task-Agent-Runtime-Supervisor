from types import SimpleNamespace

import pytest

import tars.orchestration as orch
import tars.state_store as state_store
import tars.tasks as tasks


def _role(role_id, capabilities, *, default=False):
    return SimpleNamespace(
        id=role_id,
        display_name=role_id.title(),
        description="",
        enabled=True,
        runtime_id=role_id,
        model=f"{role_id}-model",
        profile="normal",
        execution="loop",
        capabilities=tuple(capabilities),
        aliases=(),
    )


@pytest.fixture()
def isolated_state(tmp_path, monkeypatch):
    db = tmp_path / "state.sqlite3"
    monkeypatch.setattr(state_store, "STATE_DB_PATH", db)
    monkeypatch.setattr(tasks, "resolve_role_id", lambda value: value)
    roles = {
        "chat": _role("chat", ["conversation", "planning"]),
        "maker": _role("maker", ["create", "edit", "code"]),
        "ops": _role("ops", ["tools", "system-action", "network-action"]),
        "wide": _role("wide", ["tools", "system-action", "network-action", "create", "edit"]),
    }
    monkeypatch.setattr(orch, "list_roles", lambda include_disabled=False: list(roles.values()))
    monkeypatch.setattr(orch, "default_role_id", lambda: "chat")
    monkeypatch.setattr(orch, "resolve_role_id", lambda value: value)
    monkeypatch.setattr(orch, "get_role", lambda value: roles[value])
    return roles


def test_router_uses_capabilities_not_role_names(monkeypatch):
    roles = [
        _role("alpha", ["conversation", "planning", "extra"]),
        _role("beta", ["conversation", "planning"]),
    ]
    monkeypatch.setattr(orch, "list_roles", lambda include_disabled=False: roles)
    monkeypatch.setattr(orch, "default_role_id", lambda: "alpha")
    monkeypatch.setattr(orch, "get_model", lambda alias: SimpleNamespace(alias=alias))
    monkeypatch.setattr(orch, "backend_binding_ready", lambda model: True)
    decision = orch.route_for_capabilities(["conversation", "planning"], persist=False)
    assert decision.selected_role == "beta"
    assert decision.requested_capabilities == ("conversation", "planning")


def test_delegation_preserves_parent_owner(isolated_state):
    parent = tasks.create_task("Build something", "maker", make_active=False)
    delegation = orch.create_delegation(
        parent.id,
        "Inspect service health",
        role="ops",
        required_capabilities=["system-action"],
        scope={"service": "demo.service"},
        permissions=["service-status"],
        expected_result="health summary",
    )
    assert tasks.load_task(parent.id).owner_role == "maker"
    child = tasks.load_task(delegation.child_task_id)
    assert child.kind == "delegation"
    assert child.parent_task_id == parent.id
    assert child.owner_role == "ops"
    envelope = orch.delegation_envelope(delegation.id)
    assert envelope["request"]["scope"] == {"service": "demo.service"}

    result = orch.complete_delegation(
        delegation.id,
        status="success",
        summary="Service is healthy",
        result={"active": True},
    )
    assert result.state == "completed"
    assert result.result_status == "success"
    assert tasks.load_task(parent.id).owner_role == "maker"
    assert tasks.load_task(child.id).state == "completed"


def test_delegation_completion_rolls_back_if_child_transition_fails(isolated_state):
    parent = tasks.create_task("parent", "maker", make_active=False)
    item = orch.create_delegation(parent.id, "child", role="ops")
    with state_store.transaction(immediate=True) as conn:
        conn.execute(
            f"""CREATE TRIGGER fail_child_completion BEFORE UPDATE ON tasks
                WHEN OLD.id='{item.child_task_id}'
                BEGIN SELECT RAISE(ABORT, 'injected child failure'); END""")
    with pytest.raises(Exception, match="injected child failure"):
        orch.complete_delegation(item.id, status="success", summary="done")
    assert orch.load_delegation(item.id).state == "requested"
    assert tasks.load_task(item.child_task_id).state == "pending"


def test_handoff_requires_verified_checkpoint_and_changes_owner(isolated_state):
    task = tasks.create_task("Own this task", "maker", make_active=False)
    handoff = orch.handoff_task(task.id, "ops", reason="system execution phase")
    updated = tasks.load_task(task.id)
    assert handoff.from_role == "maker"
    assert handoff.to_role == "ops"
    assert updated.owner_role == "ops"
    assert updated.epoch == 2
    assert orch.verify_checkpoint(handoff.checkpoint_id) is True

    conn = state_store.connect()
    try:
        row = conn.execute("SELECT owner_role FROM checkpoints WHERE id=?", (handoff.checkpoint_id,)).fetchone()
        assert row["owner_role"] == "maker"
    finally:
        conn.close()


def test_handoff_active_run_check_and_checkpoint_are_one_transaction(
        monkeypatch, isolated_state):
    task = tasks.create_task("race", "maker", make_active=False)
    original_transaction = orch.transaction
    inserted = False

    def racing_transaction(*, immediate=False):
        nonlocal inserted
        if not inserted:
            inserted = True
            with state_store.transaction(immediate=True) as conn:
                conn.execute(
                    """INSERT INTO task_runs(id,task_id,role_id,state,epoch,created_at,
                       metadata_json) VALUES('run-race',?,'maker','running',1,?,'{}')""",
                    (task.id, state_store.now_utc()))
        return original_transaction(immediate=immediate)

    monkeypatch.setattr(orch, "transaction", racing_transaction)
    with pytest.raises(RuntimeError, match="active run"):
        orch.handoff_task(task.id, "ops")
    with state_store.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE task_id=?", (task.id,)).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM handoffs WHERE task_id=?", (task.id,)).fetchone()[0] == 0
        assert conn.execute(
            "SELECT owner_role FROM tasks WHERE id=?", (task.id,)).fetchone()[0] == "maker"
