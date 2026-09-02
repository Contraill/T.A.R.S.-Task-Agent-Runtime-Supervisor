from types import SimpleNamespace
import sqlite3

import pytest

from tars import (checkpoints, control_queue, conversation, evidence, oracle,
                  sessions, state_events, state_store, tasks)


@pytest.fixture
def lineage_state(monkeypatch, tmp_path):
    database = tmp_path / "state.sqlite3"
    monkeypatch.setattr(state_store, "STATE_DB_PATH", database)
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy-tasks")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index.json")
    monkeypatch.setattr(tasks, "resolve_role_id", lambda value: value)
    monkeypatch.setattr(sessions, "resolve_role_id", lambda value: value)

    first_conversation = conversation.create_conversation(make_active=False)
    second_conversation = conversation.create_conversation(make_active=False)
    first_task = tasks.create_task(
        "first", "general", conversation_id=first_conversation.id, make_active=False)
    second_task = tasks.create_task(
        "second", "general", conversation_id=second_conversation.id,
        make_active=False)
    first_session = sessions.create_session(
        conversation_id=first_conversation.id, role_id="general")
    second_session = sessions.create_session(
        conversation_id=second_conversation.id, role_id="general")
    return SimpleNamespace(
        database=database,
        first_conversation=first_conversation,
        second_conversation=second_conversation,
        first_task=first_task,
        second_task=second_task,
        first_session=first_session,
        second_session=second_session,
    )


def test_state_event_derives_one_canonical_conversation(lineage_state):
    state = lineage_state
    event = state_events.append_state_event(
        "activity", task_id=state.first_task.id,
        session_id=state.first_session.id)
    assert event.conversation_id == state.first_conversation.id

    with pytest.raises(ValueError, match="different conversations"):
        state_events.append_state_event(
            "activity", task_id=state.first_task.id,
            session_id=state.second_session.id)

    with state_store.connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM state_events WHERE task_id=? AND session_id=?",
            (state.first_task.id, state.second_session.id),
        ).fetchone()[0]
    assert count == 0


def test_messages_and_controls_reject_cross_conversation_ids(lineage_state):
    state = lineage_state
    with pytest.raises(ValueError, match="task does not belong"):
        conversation.add_message(
            state.first_conversation.id, "user", "crossed",
            related_task_id=state.second_task.id)
    with pytest.raises(ValueError, match="different conversations"):
        control_queue.enqueue(
            state.first_task.id, "message", "crossed",
            session_id=state.second_session.id)

    with state_store.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE content='crossed'").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM task_controls WHERE message='crossed'").fetchone()[0] == 0


def test_evidence_is_bound_to_its_state_event_and_immutable(lineage_state):
    state = lineage_state
    event = state_events.append_state_event(
        "tool_result", task_id=state.first_task.id)
    record = evidence.record(
        "fixture", "test", "result", task_id=state.first_task.id,
        event_uuid=event.event_uuid)

    with pytest.raises(PermissionError, match="another task"):
        evidence.record(
            "fixture", "test", "crossed", task_id=state.second_task.id,
            event_uuid=event.event_uuid)

    with state_store.transaction(immediate=True) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="evidence event lineage"):
            conn.execute(
                """INSERT INTO evidence_records(
                   id,task_id,event_uuid,evidence_type,source,content_sha256,created_at)
                   VALUES('evidence-injected',?,?, 'fixture','test',?,?)""",
                (state.second_task.id, event.event_uuid, "0" * 64,
                 state_store.now_utc()),
            )
        with pytest.raises(sqlite3.IntegrityError, match="evidence records are immutable"):
            conn.execute(
                "UPDATE evidence_records SET task_id=? WHERE id=?",
                (state.second_task.id, record.id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="evidence records are immutable"):
            conn.execute("DELETE FROM evidence_records WHERE id=?", (record.id,))


def test_task_checkpoint_and_delegation_evidence_follow_parent_lineage(
        lineage_state):
    state = lineage_state
    parent_evidence = evidence.record(
        "input", "parent", "one", task_id=state.first_task.id)
    foreign_evidence = evidence.record(
        "input", "foreign", "two", task_id=state.second_task.id)

    with pytest.raises(sqlite3.IntegrityError, match="outside its lineage"):
        tasks.update_task(
            state.first_task.id, evidence_refs=(foreign_evidence.id,))
    with pytest.raises(sqlite3.IntegrityError, match="outside task lineage"):
        checkpoints.create_checkpoint(
            state.first_task.id, evidence_refs=(foreign_evidence.id,))

    with state_store.transaction(immediate=True) as conn:
        child_id = tasks.create_task_in_transaction(
            conn, "child", "oracle", kind="delegation",
            parent_task_id=state.first_task.id,
            conversation_id=state.first_conversation.id,
            evidence_refs=(parent_evidence.id,), make_active=False)
    child_checkpoint = checkpoints.create_checkpoint(
        child_id, evidence_refs=(parent_evidence.id,))
    assert child_checkpoint.evidence_refs == (parent_evidence.id,)

    with state_store.transaction(immediate=True) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="not owned by its parent"):
            conn.execute(
                """INSERT INTO delegations(
                   id,parent_task_id,child_task_id,requested_role,state,goal,
                   evidence_refs_json,created_at,updated_at)
                   VALUES('dlg-injected',?,?, 'oracle','requested','review',?,?,?)""",
                (state.first_task.id, child_id,
                 state_store.json_dumps([foreign_evidence.id]),
                 state_store.now_utc(), state_store.now_utc()),
            )


def test_oracle_rejects_foreign_input_evidence_before_route_or_child_creation(
        monkeypatch, lineage_state):
    state = lineage_state
    foreign = evidence.record(
        "input", "foreign", "two", task_id=state.second_task.id)

    class UnreachedRouter:
        def __init__(self, _cfg):
            pytest.fail("route resolution was reached")

    monkeypatch.setattr(
        oracle, "create_child",
        lambda *args, **kwargs: pytest.fail("child creation was reached"))
    with pytest.raises(PermissionError, match="another task"):
        oracle.create_oracle_delegation(
            {}, state.first_task.id, "review", evidence_refs=(foreign.id,),
            required_evidence_types=("analysis",), parent_authority={},
            parent_tools=(), router_factory=UnreachedRouter)


def test_database_rejects_cross_task_checkpoint_session_and_event_writers(
        lineage_state):
    state = lineage_state
    checkpoint = checkpoints.create_checkpoint(state.first_task.id)
    stamp = state_store.now_utc()
    with state_store.transaction(immediate=True) as conn:
        attempts = (
            (
                "state event",
                """INSERT INTO state_events(
                   event_uuid,session_id,conversation_id,task_id,timestamp,type)
                   VALUES('sev-injected',?,?,?,?, 'activity')""",
                (state.second_session.id, state.first_conversation.id,
                 state.first_task.id, stamp),
            ),
            (
                "context projection",
                """INSERT INTO context_projections(
                   id,conversation_id,task_id,role_id,model_alias,runtime_id,profile,
                   mode,context_window,output_reserve,safety_margin,usable_input,
                   token_count,exact,included_messages,omitted_messages,created_at)
                   VALUES('ctx-injected',?,?, 'general','model','runtime','normal',
                          'normal',4096,256,128,3712,1,1,1,0,?)""",
                (state.second_conversation.id, state.first_task.id, stamp),
            ),
            (
                "task run",
                """INSERT INTO task_runs(
                   id,task_id,conversation_id,role_id,state,created_at)
                   VALUES('run-injected',?,?, 'general','queued',?)""",
                (state.first_task.id, state.second_conversation.id, stamp),
            ),
            (
                "task control",
                """INSERT INTO task_controls(
                   id,task_id,session_id,seq,kind,priority,state,created_at)
                   VALUES('control-injected',?,?,99,'message',4,'pending',?)""",
                (state.first_task.id, state.second_session.id, stamp),
            ),
            (
                "approval",
                """INSERT INTO approvals(
                   id,state,risk_class,tool,scope,task_id,session_id,created_at)
                   VALUES('approval-injected','pending','high','fixture','call',?,?,?)""",
                (state.first_task.id, state.second_session.id, stamp),
            ),
            (
                "handoff",
                """INSERT INTO handoffs(
                   id,task_id,from_role,to_role,checkpoint_id,created_at)
                   VALUES('handoff-injected',?,'general','builder',?,?)""",
                (state.second_task.id, checkpoint.id, stamp),
            ),
        )
        for label, statement, parameters in attempts:
            with pytest.raises(sqlite3.IntegrityError, match="lineage"):
                conn.execute(statement, parameters)


def test_database_rejects_lineage_rebinding_and_mismatched_action_event(
        lineage_state):
    state = lineage_state
    task_only_event = state_events.append_state_event(
        "tool_result", task_id=state.first_task.id)
    with state_store.transaction(immediate=True) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="session conversation.*immutable"):
            conn.execute(
                "UPDATE sessions SET conversation_id=? WHERE id=?",
                (state.second_conversation.id, state.first_session.id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="task conversation.*bound"):
            conn.execute(
                "UPDATE tasks SET conversation_id=? WHERE id=?",
                (state.second_conversation.id, state.first_task.id),
            )
        with pytest.raises(sqlite3.IntegrityError, match="action lineage"):
            conn.execute(
                """INSERT INTO action_journal(
                   id,task_id,session_id,event_uuid,tool,effect,risk_class,
                   policy_action,state,created_at)
                   VALUES('action-injected',?,?,?,'fixture','execute','low',
                          'allow','proposed',?)""",
                (state.first_task.id, state.first_session.id,
                 task_only_event.event_uuid, state_store.now_utc()),
            )


def test_v19_migration_refuses_preexisting_contradictory_lineage(
        monkeypatch, tmp_path):
    database = tmp_path / "legacy.sqlite3"
    monkeypatch.setattr(state_store, "STATE_DB_PATH", database)
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy-tasks")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index.json")
    conn = sqlite3.connect(database)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for version in range(state_store.BASE_SCHEMA_VERSION, 20):
            state_store._apply_schema_level(conn, version)
        conn.execute("INSERT INTO meta(key,value) VALUES('schema_version','19')")
        stamp = state_store.now_utc()
        conn.execute(
            "INSERT INTO conversations(id,created_at,updated_at) VALUES('a',?,?)",
            (stamp, stamp))
        conn.execute(
            "INSERT INTO conversations(id,created_at,updated_at) VALUES('b',?,?)",
            (stamp, stamp))
        conn.execute(
            """INSERT INTO tasks(
               id,conversation_id,goal,owner_role,state,created_at,updated_at)
               VALUES('task-a','a','goal','general','pending',?,?)""",
            (stamp, stamp))
        conn.execute(
            """INSERT INTO messages(
               id,conversation_id,seq,role,content,related_task_id,created_at)
               VALUES('message-crossed','b',1,'user','crossed','task-a',?)""",
            (stamp,))
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(RuntimeError, match="refused contradictory rows"):
        state_store.ensure_state_store_no_migration()
    with sqlite3.connect(database) as conn:
        assert conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "19"
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='messages_lineage_insert'"
        ).fetchone() is None


def test_v19_migration_backfills_unambiguous_event_conversation(
        monkeypatch, tmp_path):
    database = tmp_path / "legacy.sqlite3"
    monkeypatch.setattr(state_store, "STATE_DB_PATH", database)
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy-tasks")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index.json")
    conn = sqlite3.connect(database)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for version in range(state_store.BASE_SCHEMA_VERSION, 20):
            state_store._apply_schema_level(conn, version)
        conn.execute("INSERT INTO meta(key,value) VALUES('schema_version','19')")
        stamp = state_store.now_utc()
        conn.execute(
            "INSERT INTO conversations(id,created_at,updated_at) VALUES('a',?,?)",
            (stamp, stamp))
        conn.execute(
            """INSERT INTO tasks(
               id,conversation_id,goal,owner_role,state,created_at,updated_at)
               VALUES('task-a','a','goal','general','pending',?,?)""",
            (stamp, stamp))
        conn.execute(
            """INSERT INTO state_events(
               event_uuid,conversation_id,task_id,timestamp,type)
               VALUES('event-a',NULL,'task-a',?,'activity')""",
            (stamp,))
        conn.commit()
    finally:
        conn.close()

    state_store.ensure_state_store_no_migration()
    with sqlite3.connect(database) as conn:
        assert conn.execute(
            "SELECT conversation_id FROM state_events WHERE event_uuid='event-a'"
        ).fetchone()[0] == "a"
        assert conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0] == str(state_store.SCHEMA_VERSION)


def test_schema_validation_rejects_replaced_lineage_trigger(lineage_state):
    state = lineage_state
    with sqlite3.connect(state.database) as conn:
        conn.execute("DROP TRIGGER messages_lineage_insert")
        conn.execute(
            """CREATE TRIGGER messages_lineage_insert
               BEFORE INSERT ON messages BEGIN SELECT 1; END""")
    with pytest.raises(RuntimeError, match="schema definition differs"):
        state_store.ensure_state_store_no_migration()
