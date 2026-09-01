import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from tars import activity, conversation, identity, prompt_compiler, projects, role_state, sessions
from tars import ownership, runner, state_events, state_store, tasks


def _hold_expired_lease(database, scratch, ready):
    state_store.STATE_DB_PATH = Path(database)
    state_store.TASK_ROOT = Path(scratch) / "legacy-tasks"
    state_store.TASK_EVENTS_ROOT = Path(scratch) / "legacy-events"
    state_store.TASK_INDEX_PATH = Path(scratch) / "legacy-index.json"
    owner = ownership.Owner.create("stalled-owner")
    assert ownership.claim("gpu-slot", "fencing-test", owner, lease_seconds=300)
    assert ownership.claim_workspace(
        str(Path(scratch) / "workspace"), owner, lease_seconds=300)
    ready.send((owner.token, owner.pid, owner.process_start))
    ready.close()
    threading.Event().wait()


@pytest.fixture
def isolated_state(monkeypatch, tmp_path):
    db = tmp_path / "state.sqlite3"
    monkeypatch.setattr(state_store, "STATE_DB_PATH", db)
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy-tasks")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index.json")
    return db


def test_schema_session_event_role_and_project_state(monkeypatch, isolated_state, tmp_path):
    monkeypatch.setattr(sessions, "resolve_role_id", lambda value: value)
    monkeypatch.setattr(role_state, "resolve_role_id", lambda value: value)
    conv = conversation.create_conversation(title="Durable")
    session = sessions.create_session(conversation_id=conv.id, role_id="general")
    event = state_events.append_state_event(
        "user_message", "hello", session_id=session.id, conversation_id=conv.id,
        role="general", payload={"message_id": "one"},
    )
    assert state_events.read_state_events(session_id=session.id)[-1] == event
    message = conversation.add_message(conv.id, "assistant", "welcome", session_id=session.id)
    trace = activity.activity_trace(session_id=session.id)
    assert trace[-1]["type"] == "assistant_response"
    assert trace[-1]["payload"]["message_id"] == message.id
    assert sessions.close_session(session.id).state == "closed"
    assert role_state.save_role_state("general", {"reasoning": "summary"})["reasoning"] == "summary"
    (tmp_path / "TARS.md").write_text("native context")
    (tmp_path / "AGENTS.md").write_text("compatibility context")
    context = projects.register_project(tmp_path)
    assert [path.name for path in context.files] == ["TARS.md", "AGENTS.md"]
    health = state_store.health()
    assert health["ok"] and health["schema_version"] == state_store.SCHEMA_VERSION
    assert health["counts"]["sessions"] == 1 and health["counts"]["state_events"] == 4


def test_state_database_and_directory_permissions_are_repaired(isolated_state):
    isolated_state.parent.chmod(0o755)
    state_store.ensure_state_store()
    isolated_state.chmod(0o644)
    isolated_state.parent.chmod(0o755)
    with state_store.connect():
        pass
    assert os.stat(isolated_state.parent).st_mode & 0o777 == 0o700
    assert os.stat(isolated_state).st_mode & 0o777 == 0o600
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(isolated_state) + suffix)
        if sidecar.exists():
            assert os.stat(sidecar).st_mode & 0o777 == 0o600


def test_expiry_never_reassigns_authority_while_owner_process_is_alive(isolated_state):
    state_store.ensure_state_store()
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    holder = context.Process(
        target=_hold_expired_lease,
        args=(str(isolated_state), str(isolated_state.parent), sender),
    )
    holder.start()
    token, pid, started = receiver.recv()
    receiver.close()
    owner = ownership.Owner(token, pid, started)
    workspace = str(isolated_state.parent / "workspace")
    with state_store.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE resource_leases SET expires_at='1970-01-01T00:00:00+00:00' "
            "WHERE (resource_type='gpu-slot' AND resource_key='fencing-test') "
            "OR resource_type='workspace'"
        )

    contender = ownership.Owner.create("contender")
    copied_token = ownership.Owner(token, contender.pid, contender.process_start)
    try:
        assert ownership.active("gpu-slot", "fencing-test")
        assert not ownership.held_by("gpu-slot", "fencing-test", owner)
        assert not ownership.claim("gpu-slot", "fencing-test", contender)
        assert not ownership.claim("gpu-slot", "fencing-test", copied_token)
        assert not ownership.heartbeat("gpu-slot", "fencing-test", copied_token)
        assert not ownership.release("gpu-slot", "fencing-test", copied_token)
        assert not ownership.claim_workspace(workspace + "/child", contender)
    finally:
        holder.terminate()
        holder.join(timeout=10)
    assert holder.exitcode is not None
    assert not ownership.active("gpu-slot", "fencing-test")
    assert ownership.claim("gpu-slot", "fencing-test", contender)
    assert ownership.claim_workspace(workspace + "/child", contender)


def test_expired_owner_remains_the_only_same_process_renewal_authority(isolated_state):
    state_store.ensure_state_store()
    first = ownership.Owner.create("first-owner")
    second = ownership.Owner.create("second-owner")
    assert ownership.claim("action", "same-process", first)
    with state_store.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE resource_leases SET expires_at='1970-01-01T00:00:00+00:00' "
            "WHERE resource_type='action' AND resource_key='same-process'"
        )
    assert ownership.active("action", "same-process")
    assert not ownership.claim("action", "same-process", second)
    assert ownership.heartbeat("action", "same-process", first)
    assert ownership.held_by("action", "same-process", first)
    assert ownership.release("action", "same-process", first)
    assert ownership.claim("action", "same-process", second)


def test_unverifiable_process_identity_remains_fenced(isolated_state, monkeypatch):
    state_store.ensure_state_store()
    first = ownership.Owner.create("unverifiable-owner")
    second = ownership.Owner.create("blocked-owner")
    assert ownership.claim("gpu-slot", "unverifiable", first)
    with state_store.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE resource_leases SET expires_at='1970-01-01T00:00:00+00:00' "
            "WHERE resource_type='gpu-slot' AND resource_key='unverifiable'"
        )

    def unreadable(_pid):
        raise PermissionError("identity unavailable")

    monkeypatch.setattr(ownership, "_read_process_start", unreadable)
    assert ownership.active("gpu-slot", "unverifiable")
    assert not ownership.claim("gpu-slot", "unverifiable", second)


def test_append_event_transaction_rolls_back_on_invalid_reference(isolated_state):
    with pytest.raises(Exception):
        state_events.append_state_event("activity", task_id="missing")
    assert state_store.health()["counts"]["state_events"] == 0


def test_paused_run_blocks_second_run_and_run_provenance_is_locked(
        monkeypatch, isolated_state):
    monkeypatch.setattr(tasks, "resolve_role_id", lambda value: value)
    conv = conversation.create_conversation()
    task = tasks.create_task("durable", "general", conversation_id=conv.id)
    first = runner.create_run(task.id)
    runner._set_run(first.id, state="paused")
    with pytest.raises(RuntimeError, match="active run"):
        runner.create_run(task.id)


def test_dead_running_task_owner_is_terminalized_before_new_run(
        monkeypatch, isolated_state):
    monkeypatch.setattr(tasks, "resolve_role_id", lambda value: value)
    task = tasks.create_task("recover runner", "general", make_active=False)
    first = runner.create_run(task.id)
    runner._set_run(first.id, state="running")
    dead = ownership.Owner("dead-owner", 999_999_999, "missing")
    with state_store.transaction(immediate=True) as conn:
        assert ownership.claim_in_transaction(
            conn, "task-execution", task.id, dead, lease_seconds=300)
    second = runner.create_run(task.id)
    assert second.state == "queued"
    recovered = runner.load_run(first.id)
    assert recovered.state == "failed" and recovered.finish_reason == "owner-lost"


def test_task_execution_scope_borrows_exact_owner_without_releasing_outer(
        monkeypatch, isolated_state):
    monkeypatch.setattr(tasks, "resolve_role_id", lambda value: value)
    task = tasks.create_task("reentrant owner", "general", make_active=False)
    owner = ownership.Owner.create("outer-engine")
    with ownership.task_execution_scope(
            task.id, engine="outer", owner=owner) as outer:
        assert outer == owner
        assert ownership.held_by("task-execution", task.id, owner)
        with ownership.task_execution_scope(
                task.id, engine="nested", owner=owner) as nested:
            assert nested == owner
        assert ownership.held_by("task-execution", task.id, owner)
    assert not ownership.active("task-execution", task.id)


def test_schema_upgrade_preserves_existing_conversation(isolated_state):
    state_store.ensure_state_store()
    conv = conversation.create_conversation(title="before upgrade")
    with state_store.transaction(immediate=True) as conn:
        conn.execute("UPDATE meta SET value='4' WHERE key='schema_version'")
    state_store.ensure_state_store()
    assert conversation.load_conversation(conv.id).title == "before upgrade"
    assert state_store.health()["schema_version"] == state_store.SCHEMA_VERSION


def test_real_v16_layout_is_migrated_in_order(isolated_state):
    conn = sqlite3.connect(isolated_state)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for version in range(state_store.BASE_SCHEMA_VERSION, 17):
            state_store._apply_schema_level(conn, version)
        conn.execute("INSERT INTO meta(key,value) VALUES('schema_version','16')")
        conn.execute(
            "INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            ("conv-old", "historical", "2026-01-01", "2026-01-01"))
        conn.commit()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='runtime_routes'").fetchone() is None
    finally:
        conn.close()

    state_store.ensure_state_store_no_migration()
    conn = sqlite3.connect(isolated_state)
    try:
        assert conn.execute(
            "SELECT title FROM conversations WHERE id='conv-old'").fetchone()[0] == "historical"
        assert conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == str(
                state_store.SCHEMA_VERSION)
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='runtime_routes'").fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='resource_leases'").fetchone()
    finally:
        conn.close()


def test_real_v17_layout_adds_durable_lease_schema(isolated_state):
    conn = sqlite3.connect(isolated_state)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for version in range(state_store.BASE_SCHEMA_VERSION, 18):
            state_store._apply_schema_level(conn, version)
        conn.execute("INSERT INTO meta(key,value) VALUES('schema_version','17')")
        conn.commit()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='runtime_routes'").fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='resource_leases'").fetchone() is None
    finally:
        conn.close()
    state_store.ensure_state_store_no_migration()
    with sqlite3.connect(isolated_state) as conn:
        assert conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == str(
                state_store.SCHEMA_VERSION)
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='resource_leases'").fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='control_cancellations'").fetchone()


def test_real_v18_layout_adds_control_cancellation_schema(isolated_state):
    conn = sqlite3.connect(isolated_state)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for version in range(state_store.BASE_SCHEMA_VERSION, 19):
            state_store._apply_schema_level(conn, version)
        conn.execute("INSERT INTO meta(key,value) VALUES('schema_version','18')")
        conn.execute(
            """INSERT INTO tasks(id,goal,owner_role,state,created_at,updated_at)
               VALUES('task-legacy','goal','general','running','2025-01-01','2025-01-01')"""
        )
        conn.execute(
            """INSERT INTO task_controls(
               id,task_id,seq,kind,priority,state,message,payload_json,created_at,applied_at)
               VALUES('control-raced','task-legacy',1,'interrupt',1,'applied','stop',
                      '{"applied_at_boundary":true}','2025-01-01','2025-01-01')"""
        )
        conn.execute(
            """INSERT INTO task_controls(
               id,task_id,seq,kind,priority,state,message,payload_json,created_at)
               VALUES('control-returned','task-legacy',2,'cancel',0,'pending','cancel',
                      ?,'2025-01-01')""",
            ('{"cancellable":true,"cancellation_requested":true,'
             '"cancellation_result":{"private":"must-not-survive"}}',),
        )
        conn.execute(
            """INSERT INTO task_controls(
               id,task_id,seq,kind,priority,state,message,payload_json,created_at)
               VALUES('control-malformed','task-legacy',3,'interrupt',1,'pending',
                      'stop','[]','2025-01-01')"""
        )
        conn.commit()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='resource_leases'").fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='control_cancellations'").fetchone() is None
    finally:
        conn.close()
    state_store.ensure_state_store_no_migration()
    with sqlite3.connect(isolated_state) as conn:
        assert conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "19"
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='control_cancellations'").fetchone()
        raced = conn.execute(
            "SELECT state FROM control_cancellations WHERE control_id='control-raced'"
        ).fetchone()
        returned = conn.execute(
            "SELECT state,result_json FROM control_cancellations "
            "WHERE control_id='control-returned'"
        ).fetchone()
        malformed = conn.execute(
            "SELECT state FROM control_cancellations WHERE control_id='control-malformed'"
        ).fetchone()
        payload = conn.execute(
            "SELECT payload_json FROM task_controls WHERE id='control-returned'"
        ).fetchone()[0]
        assert raced[0] == "ambiguous"
        assert returned == ("resolved", '{"requested":true}')
        assert malformed[0] == "ambiguous"
        assert "must-not-survive" not in payload


def test_failed_cancellation_backfill_does_not_advance_v18(monkeypatch, isolated_state):
    conn = sqlite3.connect(isolated_state)
    try:
        for version in range(state_store.BASE_SCHEMA_VERSION, 19):
            state_store._apply_schema_level(conn, version)
        conn.execute("INSERT INTO meta(key,value) VALUES('schema_version','18')")
        conn.commit()
    finally:
        conn.close()

    def fail_backfill(conn):
        raise RuntimeError("injected cancellation backfill failure")

    monkeypatch.setattr(state_store, "_migrate_control_cancellations", fail_backfill)
    with pytest.raises(RuntimeError, match="injected cancellation backfill failure"):
        state_store.ensure_state_store_no_migration()
    with sqlite3.connect(isolated_state) as conn:
        assert conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "18"
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='control_cancellations'"
        ).fetchone() is None

def test_failed_migration_never_advances_version(monkeypatch, isolated_state):
    conn = sqlite3.connect(isolated_state)
    state_store._apply_schema_level(conn, state_store.BASE_SCHEMA_VERSION)
    conn.execute("INSERT INTO meta(key,value) VALUES('schema_version','3')")
    conn.commit()
    conn.close()
    original = state_store._apply_schema_level

    def fail_at_v17(conn, version):
        if version == 17:
            raise RuntimeError("injected migration failure")
        original(conn, version)

    monkeypatch.setattr(state_store, "_apply_schema_level", fail_at_v17)
    with pytest.raises(RuntimeError, match="injected migration failure"):
        state_store.ensure_state_store_no_migration()
    conn = sqlite3.connect(isolated_state)
    try:
        assert conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "3"
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='runtime_routes'").fetchone() is None
    finally:
        conn.close()


def test_legacy_import_failures_are_observable_retryable_and_idempotent(isolated_state):
    state_store.TASK_ROOT.mkdir()
    valid = {"id": "task-valid", "goal": "keep", "owner_role": "general"}
    (state_store.TASK_ROOT / "task-valid.json").write_text(json.dumps(valid))
    malformed = state_store.TASK_ROOT / "task-bad.json"
    malformed.write_text("{")

    state_store.ensure_state_store()
    assert state_store.get_meta("legacy_task_store_migrated") == "0"
    failures = json.loads(state_store.get_meta("legacy_task_store_failures"))
    assert failures and failures[0]["source"].endswith("task-bad.json")

    malformed.write_text(json.dumps({"id": "task-bad", "goal": "retry"}))
    state_store.ensure_state_store()
    assert state_store.get_meta("legacy_task_store_migrated") == "1"
    with state_store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 2


def test_identity_inheritance_and_prompt_explain(monkeypatch, tmp_path):
    persona = tmp_path / "persona"
    monkeypatch.setattr(identity, "IDENTITY_PATH", persona / "IDENTITY.md")
    monkeypatch.setattr(identity, "SOUL_PATH", persona / "SOUL.md")
    monkeypatch.setattr(identity, "ROLE_PERSONA_ROOT", persona / "roles")
    identity.ensure_identity_files()
    (persona / "roles/general.md").write_text("Be concise.")
    monkeypatch.setattr(prompt_compiler, "resolve_role_id", lambda value: "general")
    monkeypatch.setattr(prompt_compiler, "get_role", lambda value: SimpleNamespace(
        display_name="General", description="Daily role", capabilities=("conversation",)
    ))
    monkeypatch.setattr(prompt_compiler, "load_identity", identity.load_identity)
    compiled = prompt_compiler.PromptCompiler().compile(
        role_name="general", personal_memory=["prefers concise answers"],
        pending_controls=["do not modify files"],
    )
    explanation = compiled.explain()
    names = [source["name"] for source in explanation["sources"]]
    assert names == ["base_identity", "role_overlay", "capabilities", "personal_memory", "pending_controls"]
    assert explanation["sources"][-1]["protected"]
    assert len([message for message in compiled.messages if message["role"] == "system"]) == 1
    assert "Be concise" in compiled.messages[0]["content"]


def test_reasoning_visibility_never_synthesizes_raw():
    assert activity.reasoning_view("raw", emitted_raw="") == ""
    assert activity.reasoning_view("raw", emitted_raw="backend thought") == "backend thought"
    assert activity.reasoning_view("summary", summary="short") == "short"
    assert activity.reasoning_view("hidden", emitted_raw="secret") == ""
