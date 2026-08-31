from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import sqlite3
from pathlib import Path

from .config import STATE_DB_PATH, TASK_INDEX_PATH, TASK_ROOT, TASK_EVENTS_ROOT

SCHEMA_VERSION = 19
BASE_SCHEMA_VERSION = 3

# Schema objects were introduced monotonically in the public state-store history.
# Keeping this provenance explicit lets startup apply one real version transition at
# a time instead of confusing an idempotent latest-schema bootstrap with migration.
_SCHEMA_INTRODUCED = {
    "meta": 3, "conversations": 3, "messages": 3, "idx_messages_conversation_seq": 3,
    "tasks": 3, "idx_tasks_state_updated": 3, "idx_tasks_conversation": 3,
    "idx_tasks_schedule": 3, "task_events": 3, "idx_task_events_task_id": 3,
    "checkpoints": 3, "idx_checkpoints_task_seq": 3,
    "checkpoints_immutable_update": 3, "checkpoints_immutable_delete": 3,
    "context_projections": 3, "idx_context_projection_conversation_role": 3,
    "task_runs": 3, "idx_task_runs_task_created": 3, "idx_task_runs_state": 3,
    "delegations": 4, "idx_delegations_parent_created": 4, "handoffs": 4,
    "idx_handoffs_task_created": 4, "routing_decisions": 4,
    "idx_routing_task_created": 4, "sessions": 5,
    "idx_sessions_conversation_updated": 5, "state_events": 5,
    "idx_state_events_session_id": 5, "idx_state_events_task_id": 5,
    "role_state": 5, "project_refs": 5, "memory_index": 6,
    "idx_memory_scope_updated": 6, "memory_fts": 6, "memory_candidates": 6,
    "context_epochs": 7, "idx_context_epochs_task_epoch": 7,
    "memory_maintenance_runs": 8, "idx_memory_maintenance_created": 8,
    "policy_rules": 9, "idx_policy_rules_effect_target": 9, "approvals": 9,
    "idx_approvals_state_created": 9, "action_journal": 9,
    "idx_action_journal_created": 9, "idx_action_journal_task": 9,
    "evidence_records": 10, "idx_evidence_task_created": 10,
    "task_controls": 11, "idx_task_controls_pending": 11,
    "workspace_checkpoints": 12, "idx_workspace_checkpoints_task_created": 12,
    "delegation_contracts": 13, "delegation_memory": 13, "mcp_servers": 14,
    "schedules": 15, "idx_schedules_active_task": 15, "idx_schedules_due": 15,
    "schedule_runs": 15, "idx_schedule_runs_planned": 15,
    "idx_schedule_runs_state": 15, "schedule_deliveries": 15,
    "idx_schedule_deliveries_state": 15, "core_clients": 16,
    "idx_core_clients_state": 16, "core_pairings": 16,
    "idx_core_pairings_expiry": 16, "runtime_routes": 17,
    "idx_runtime_routes_task_created": 17,
    "resource_leases": 18, "idx_resource_leases_expiry": 18,
    "control_cancellations": 19, "idx_control_cancellations_state": 19,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_loads(value, default=None):
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def connect() -> sqlite3.Connection:
    STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(STATE_DB_PATH.parent, 0o700)
    if not STATE_DB_PATH.exists():
        try:
            fd = os.open(STATE_DB_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
    os.chmod(STATE_DB_PATH, 0o600)
    conn = sqlite3.connect(STATE_DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(STATE_DB_PATH) + suffix)
        if path.exists():
            os.chmod(path, 0o600)
    return conn


@contextmanager
def transaction(*, immediate: bool = False):
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _schema_sql() -> str:
    return """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT '',
        state TEXT NOT NULL DEFAULT 'open',
        source TEXT NOT NULL DEFAULT 'chat',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        last_message_at TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        seq INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'message',
        include_in_context INTEGER NOT NULL DEFAULT 1 CHECK(include_in_context IN (0,1)),
        related_task_id TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(conversation_id, seq)
    );
    CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq
        ON messages(conversation_id, seq);

    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        role_id TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'open',
        mode TEXT NOT NULL DEFAULT 'normal',
        started_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        ended_at TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_conversation_updated
        ON sessions(conversation_id, updated_at DESC);

    CREATE TABLE IF NOT EXISTS state_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_uuid TEXT NOT NULL UNIQUE,
        session_id TEXT REFERENCES sessions(id),
        conversation_id TEXT REFERENCES conversations(id),
        task_id TEXT REFERENCES tasks(id),
        timestamp TEXT NOT NULL,
        type TEXT NOT NULL,
        source_role TEXT,
        message TEXT NOT NULL DEFAULT '',
        payload_json TEXT NOT NULL DEFAULT '{}',
        visibility TEXT NOT NULL DEFAULT 'normal'
    );
    CREATE INDEX IF NOT EXISTS idx_state_events_session_id
        ON state_events(session_id, id);
    CREATE INDEX IF NOT EXISTS idx_state_events_task_id
        ON state_events(task_id, id);

    CREATE TABLE IF NOT EXISTS role_state (
        role_id TEXT PRIMARY KEY,
        state_json TEXT NOT NULL DEFAULT '{}',
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS project_refs (
        id TEXT PRIMARY KEY,
        canonical_path TEXT NOT NULL UNIQUE,
        display_name TEXT NOT NULL DEFAULT '',
        context_files_json TEXT NOT NULL DEFAULT '[]',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS memory_index (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        scope TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL,
        source TEXT NOT NULL,
        confidence REAL NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        supersedes TEXT,
        expiry TEXT,
        tags_json TEXT NOT NULL DEFAULT '[]',
        path TEXT NOT NULL UNIQUE
    );
    CREATE INDEX IF NOT EXISTS idx_memory_scope_updated
        ON memory_index(scope, updated_at DESC);
    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        id UNINDEXED, title, content, tags
    );

    CREATE TABLE IF NOT EXISTS memory_candidates (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        scope TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL,
        source TEXT NOT NULL,
        confidence REAL NOT NULL,
        tags_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'staged',
        reason TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        reviewed_at TEXT
    );

    CREATE TABLE IF NOT EXISTS memory_maintenance_runs (
        id TEXT PRIMARY KEY,
        trigger TEXT NOT NULL,
        mode TEXT NOT NULL,
        status TEXT NOT NULL,
        report_json TEXT NOT NULL DEFAULT '{}',
        actions_json TEXT NOT NULL DEFAULT '[]',
        rollback_refs_json TEXT NOT NULL DEFAULT '[]',
        model_provenance_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        completed_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_memory_maintenance_created
        ON memory_maintenance_runs(created_at DESC);

    CREATE TABLE IF NOT EXISTS policy_rules (
        id TEXT PRIMARY KEY,
        effect TEXT NOT NULL,
        action TEXT NOT NULL CHECK(action IN ('allow','deny','ask')),
        target TEXT NOT NULL DEFAULT '',
        scope TEXT NOT NULL DEFAULT 'persistent',
        created_at TEXT NOT NULL,
        expires_at TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_policy_rules_effect_target
        ON policy_rules(effect, target);

    CREATE TABLE IF NOT EXISTS approvals (
        id TEXT PRIMARY KEY,
        state TEXT NOT NULL CHECK(state IN ('pending','approved','denied','expired','consumed')),
        risk_class TEXT NOT NULL,
        tool TEXT NOT NULL,
        target TEXT NOT NULL DEFAULT '',
        request_json TEXT NOT NULL DEFAULT '{}',
        scope TEXT NOT NULL,
        task_id TEXT REFERENCES tasks(id),
        session_id TEXT REFERENCES sessions(id),
        decision_reason TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        decided_at TEXT,
        expires_at TEXT,
        consumed_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_approvals_state_created
        ON approvals(state, created_at DESC);

    CREATE TABLE IF NOT EXISTS action_journal (
        id TEXT PRIMARY KEY,
        task_id TEXT REFERENCES tasks(id),
        session_id TEXT REFERENCES sessions(id),
        event_uuid TEXT NOT NULL,
        tool TEXT NOT NULL,
        normalized_arguments_json TEXT NOT NULL DEFAULT '{}',
        target TEXT NOT NULL DEFAULT '',
        effect TEXT NOT NULL,
        risk_class TEXT NOT NULL,
        policy_action TEXT NOT NULL,
        policy_reason TEXT NOT NULL DEFAULT '',
        approval_id TEXT REFERENCES approvals(id),
        state TEXT NOT NULL CHECK(state IN ('proposed','running','succeeded','failed','denied','cancelled','unknown')),
        result_json TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_action_journal_created
        ON action_journal(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_action_journal_task
        ON action_journal(task_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS evidence_records (
        id TEXT PRIMARY KEY,
        task_id TEXT REFERENCES tasks(id),
        event_uuid TEXT,
        evidence_type TEXT NOT NULL,
        source TEXT NOT NULL,
        content_sha256 TEXT NOT NULL,
        result_ref TEXT NOT NULL DEFAULT '',
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_evidence_task_created
        ON evidence_records(task_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS task_controls (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL REFERENCES tasks(id),
        session_id TEXT REFERENCES sessions(id),
        seq INTEGER NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('cancel','interrupt','approval','redirect','message','pause','resume')),
        priority INTEGER NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('pending','processing','applied','failed')),
        message TEXT NOT NULL DEFAULT '',
        payload_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        applied_at TEXT,
        UNIQUE(task_id, seq)
    );
    CREATE INDEX IF NOT EXISTS idx_task_controls_pending
        ON task_controls(task_id, state, priority, seq);

    CREATE TABLE IF NOT EXISTS control_cancellations (
        control_id TEXT PRIMARY KEY REFERENCES task_controls(id) ON DELETE CASCADE,
        state TEXT NOT NULL CHECK(state IN ('intent','attempting','resolved','ambiguous')),
        operation_id TEXT NOT NULL DEFAULT '',
        active_tool TEXT NOT NULL DEFAULT '',
        cancellable INTEGER NOT NULL DEFAULT 0,
        cancellation_effect TEXT NOT NULL DEFAULT 'execute'
            CHECK(cancellation_effect IN ('execute','destructive')),
        result_json TEXT,
        error TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        attempt_started_at TEXT,
        reconciled_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_control_cancellations_state
        ON control_cancellations(state, created_at);

    CREATE TABLE IF NOT EXISTS workspace_checkpoints (
        id TEXT PRIMARY KEY,
        task_id TEXT REFERENCES tasks(id),
        kind TEXT NOT NULL CHECK(kind IN ('git','filesystem')),
        root TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('ready','restored','failed')),
        storage_ref TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        restored_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_workspace_checkpoints_task_created
        ON workspace_checkpoints(task_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        conversation_id TEXT REFERENCES conversations(id),
        title TEXT NOT NULL DEFAULT '',
        goal TEXT NOT NULL,
        owner_role TEXT NOT NULL,
        state TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'primary',
        phase TEXT NOT NULL DEFAULT '',
        progress REAL CHECK(progress IS NULL OR (progress >= 0.0 AND progress <= 1.0)),
        parent_task_id TEXT REFERENCES tasks(id),
        source TEXT NOT NULL DEFAULT 'cli',
        epoch INTEGER NOT NULL DEFAULT 1 CHECK(epoch >= 1),
        constraints_json TEXT NOT NULL DEFAULT '[]',
        decisions_json TEXT NOT NULL DEFAULT '[]',
        completed_json TEXT NOT NULL DEFAULT '[]',
        open_steps_json TEXT NOT NULL DEFAULT '[]',
        failures_json TEXT NOT NULL DEFAULT '[]',
        evidence_refs_json TEXT NOT NULL DEFAULT '[]',
        schedule_kind TEXT,
        schedule_expr TEXT,
        next_run_at TEXT,
        last_run_at TEXT,
        last_result_status TEXT,
        schedule_enabled INTEGER NOT NULL DEFAULT 1 CHECK(schedule_enabled IN (0,1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_tasks_state_updated
        ON tasks(state, updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_tasks_conversation
        ON tasks(conversation_id, updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_tasks_schedule
        ON tasks(schedule_kind, next_run_at);

    CREATE TABLE IF NOT EXISTS schedules (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL REFERENCES tasks(id),
        kind TEXT NOT NULL CHECK(kind IN ('one-shot','recurring','condition')),
        expression TEXT NOT NULL,
        timezone TEXT NOT NULL DEFAULT 'UTC',
        next_run_at TEXT,
        missed_policy TEXT NOT NULL DEFAULT 'run-once'
            CHECK(missed_policy IN ('skip','run-once','catch-up')),
        max_catch_up INTEGER NOT NULL DEFAULT 1 CHECK(max_catch_up >= 1),
        enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0,1)),
        concurrency_key TEXT NOT NULL DEFAULT 'default',
        max_concurrency INTEGER NOT NULL DEFAULT 1 CHECK(max_concurrency >= 1),
        delivery_target TEXT NOT NULL DEFAULT '',
        condition_state INTEGER NOT NULL DEFAULT 0 CHECK(condition_state IN (0,1)),
        revision INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        removed_at TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_schedules_active_task
        ON schedules(task_id) WHERE removed_at IS NULL;
    CREATE INDEX IF NOT EXISTS idx_schedules_due
        ON schedules(enabled, next_run_at) WHERE removed_at IS NULL;

    CREATE TABLE IF NOT EXISTS schedule_runs (
        id TEXT PRIMARY KEY,
        schedule_id TEXT NOT NULL REFERENCES schedules(id),
        task_id TEXT NOT NULL REFERENCES tasks(id),
        planned_for TEXT NOT NULL,
        idempotency_key TEXT NOT NULL UNIQUE,
        state TEXT NOT NULL CHECK(state IN
            ('claimed','running','succeeded','failed','skipped','cancelled')),
        attempt INTEGER NOT NULL DEFAULT 1,
        checkpoint_id TEXT REFERENCES checkpoints(id),
        result_json TEXT NOT NULL DEFAULT '{}',
        error TEXT NOT NULL DEFAULT '',
        claimed_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_schedule_runs_planned
        ON schedule_runs(schedule_id, planned_for);
    CREATE INDEX IF NOT EXISTS idx_schedule_runs_state
        ON schedule_runs(state, claimed_at);

    CREATE TABLE IF NOT EXISTS schedule_deliveries (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL UNIQUE REFERENCES schedule_runs(id),
        target TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('pending','delivered','failed','suppressed')),
        attempt INTEGER NOT NULL DEFAULT 0,
        result_json TEXT NOT NULL DEFAULT '{}',
        error TEXT NOT NULL DEFAULT '',
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_schedule_deliveries_state
        ON schedule_deliveries(state, updated_at);

    CREATE TABLE IF NOT EXISTS core_clients (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        principal_id TEXT NOT NULL DEFAULT 'local-owner',
        token_salt TEXT NOT NULL,
        token_hash TEXT NOT NULL,
        permissions_json TEXT NOT NULL DEFAULT '[]',
        state TEXT NOT NULL DEFAULT 'active' CHECK(state IN ('active','revoked')),
        created_at TEXT NOT NULL,
        last_seen_at TEXT,
        revoked_at TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_core_clients_state
        ON core_clients(state, created_at);

    CREATE TABLE IF NOT EXISTS core_pairings (
        id TEXT PRIMARY KEY,
        code_hash TEXT NOT NULL UNIQUE,
        permissions_json TEXT NOT NULL DEFAULT '[]',
        principal_id TEXT NOT NULL DEFAULT 'local-owner',
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        consumed_at TEXT,
        client_id TEXT REFERENCES core_clients(id)
    );
    CREATE INDEX IF NOT EXISTS idx_core_pairings_expiry
        ON core_pairings(expires_at, consumed_at);

    CREATE TABLE IF NOT EXISTS runtime_routes (
        id TEXT PRIMARY KEY,
        task_id TEXT REFERENCES tasks(id),
        requested_role TEXT NOT NULL,
        selected_role TEXT NOT NULL,
        model_alias TEXT NOT NULL DEFAULT '',
        backend TEXT NOT NULL DEFAULT '',
        runtime_id TEXT NOT NULL DEFAULT '',
        profile TEXT NOT NULL DEFAULT '',
        state TEXT NOT NULL CHECK(state IN ('ready','unavailable')),
        requested_json TEXT NOT NULL DEFAULT '{}',
        reasons_json TEXT NOT NULL DEFAULT '[]',
        backend_status_json TEXT NOT NULL DEFAULT '{}',
        runtime_capabilities_json TEXT NOT NULL DEFAULT '{}',
        model_capabilities_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_runtime_routes_task_created
        ON runtime_routes(task_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS task_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_uuid TEXT NOT NULL UNIQUE,
        task_id TEXT NOT NULL REFERENCES tasks(id),
        conversation_id TEXT REFERENCES conversations(id),
        timestamp TEXT NOT NULL,
        type TEXT NOT NULL,
        source_role TEXT,
        message TEXT NOT NULL DEFAULT '',
        payload_json TEXT NOT NULL DEFAULT '{}',
        visibility TEXT NOT NULL DEFAULT 'normal'
    );
    CREATE INDEX IF NOT EXISTS idx_task_events_task_id
        ON task_events(task_id, id);

    CREATE TABLE IF NOT EXISTS checkpoints (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL REFERENCES tasks(id),
        epoch INTEGER NOT NULL CHECK(epoch >= 1),
        seq INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        owner_role TEXT NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        state_json TEXT NOT NULL,
        evidence_refs_json TEXT NOT NULL DEFAULT '[]',
        content_sha256 TEXT NOT NULL,
        UNIQUE(task_id, seq)
    );
    CREATE INDEX IF NOT EXISTS idx_checkpoints_task_seq
        ON checkpoints(task_id, seq DESC);

    CREATE TRIGGER IF NOT EXISTS checkpoints_immutable_update
    BEFORE UPDATE ON checkpoints
    BEGIN
        SELECT RAISE(ABORT, 'checkpoints are immutable');
    END;

    CREATE TRIGGER IF NOT EXISTS checkpoints_immutable_delete
    BEFORE DELETE ON checkpoints
    BEGIN
        SELECT RAISE(ABORT, 'checkpoints are immutable');
    END;

    CREATE TABLE IF NOT EXISTS context_projections (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        task_id TEXT REFERENCES tasks(id),
        role_id TEXT NOT NULL,
        model_alias TEXT NOT NULL,
        runtime_id TEXT NOT NULL,
        profile TEXT NOT NULL,
        mode TEXT NOT NULL,
        context_window INTEGER NOT NULL,
        output_reserve INTEGER NOT NULL,
        safety_margin INTEGER NOT NULL,
        usable_input INTEGER NOT NULL,
        token_count INTEGER NOT NULL,
        exact INTEGER NOT NULL CHECK(exact IN (0,1)),
        included_messages INTEGER NOT NULL,
        omitted_messages INTEGER NOT NULL,
        through_message_seq INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_context_projection_conversation_role
        ON context_projections(conversation_id, role_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS context_epochs (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        task_id TEXT NOT NULL REFERENCES tasks(id),
        epoch INTEGER NOT NULL,
        from_message_seq INTEGER NOT NULL,
        through_message_seq INTEGER NOT NULL,
        archived_messages_json TEXT NOT NULL,
        checkpoint_id TEXT NOT NULL REFERENCES checkpoints(id),
        unresolved_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        UNIQUE(task_id, epoch)
    );
    CREATE INDEX IF NOT EXISTS idx_context_epochs_task_epoch
        ON context_epochs(task_id, epoch DESC);

    CREATE TABLE IF NOT EXISTS task_runs (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL REFERENCES tasks(id),
        conversation_id TEXT REFERENCES conversations(id),
        role_id TEXT NOT NULL,
        state TEXT NOT NULL,
        control_request TEXT NOT NULL DEFAULT '',
        epoch INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        started_at TEXT,
        heartbeat_at TEXT,
        finished_at TEXT,
        finish_reason TEXT,
        error TEXT,
        output_message_id TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_task_runs_task_created
        ON task_runs(task_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_task_runs_state
        ON task_runs(state, heartbeat_at DESC);

    CREATE TABLE IF NOT EXISTS resource_leases (
        resource_type TEXT NOT NULL,
        resource_key TEXT NOT NULL,
        owner_token TEXT NOT NULL,
        owner_pid INTEGER NOT NULL,
        owner_start TEXT NOT NULL,
        acquired_at TEXT NOT NULL,
        heartbeat_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY(resource_type, resource_key)
    );
    CREATE INDEX IF NOT EXISTS idx_resource_leases_expiry
        ON resource_leases(expires_at);

    CREATE TABLE IF NOT EXISTS delegations (
        id TEXT PRIMARY KEY,
        parent_task_id TEXT NOT NULL REFERENCES tasks(id),
        child_task_id TEXT NOT NULL UNIQUE REFERENCES tasks(id),
        requested_role TEXT NOT NULL,
        state TEXT NOT NULL,
        goal TEXT NOT NULL,
        scope_json TEXT NOT NULL DEFAULT '{}',
        constraints_json TEXT NOT NULL DEFAULT '[]',
        permissions_json TEXT NOT NULL DEFAULT '[]',
        evidence_refs_json TEXT NOT NULL DEFAULT '[]',
        expected_result TEXT NOT NULL DEFAULT '',
        result_status TEXT,
        result_summary TEXT NOT NULL DEFAULT '',
        result_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_delegations_parent_created
        ON delegations(parent_task_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS delegation_contracts (
        delegation_id TEXT PRIMARY KEY REFERENCES delegations(id),
        parent_delegation_id TEXT REFERENCES delegations(id),
        tool_allowlist_json TEXT NOT NULL DEFAULT '[]',
        authority_json TEXT NOT NULL DEFAULT '{}',
        budget_json TEXT NOT NULL DEFAULT '{}',
        workspace_json TEXT NOT NULL DEFAULT '{}',
        completion_json TEXT NOT NULL DEFAULT '{}',
        state TEXT NOT NULL DEFAULT 'created',
        accepted INTEGER CHECK(accepted IS NULL OR accepted IN (0,1)),
        acceptance_reason TEXT NOT NULL DEFAULT '',
        started_at TEXT,
        deadline_at TEXT,
        finished_at TEXT,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS delegation_memory (
        delegation_id TEXT NOT NULL REFERENCES delegations(id),
        candidate_id TEXT NOT NULL REFERENCES memory_candidates(id),
        state TEXT NOT NULL CHECK(state IN ('staged','accepted','rejected')),
        created_at TEXT NOT NULL,
        reviewed_at TEXT,
        PRIMARY KEY(delegation_id,candidate_id)
    );

    CREATE TABLE IF NOT EXISTS mcp_servers (
        name TEXT PRIMARY KEY,
        transport TEXT NOT NULL CHECK(transport IN ('stdio','streamable-http')),
        config_json TEXT NOT NULL DEFAULT '{}',
        enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0,1)),
        tool_filter_json TEXT NOT NULL DEFAULT '{}',
        effect_policy_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS handoffs (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL REFERENCES tasks(id),
        from_role TEXT NOT NULL,
        to_role TEXT NOT NULL,
        checkpoint_id TEXT NOT NULL REFERENCES checkpoints(id),
        reason TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_handoffs_task_created
        ON handoffs(task_id, created_at DESC);

    CREATE TABLE IF NOT EXISTS routing_decisions (
        id TEXT PRIMARY KEY,
        task_id TEXT REFERENCES tasks(id),
        requested_capabilities_json TEXT NOT NULL DEFAULT '[]',
        selected_role TEXT NOT NULL,
        candidates_json TEXT NOT NULL DEFAULT '[]',
        reason TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_routing_task_created
        ON routing_decisions(task_id, created_at DESC);
    """


def _schema_statements():
    statement = ""
    for line in _schema_sql().splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            value = statement.strip()
            if value:
                yield value
            statement = ""
    if statement.strip():
        raise RuntimeError("incomplete authoritative schema statement")


def _schema_object_name(statement: str) -> str:
    words = statement.replace("(", " ").split()
    try:
        marker = words.index("EXISTS")
    except ValueError as exc:
        raise RuntimeError(f"schema statement lacks IF NOT EXISTS: {statement[:80]}") from exc
    return words[marker + 1]


def _apply_schema_level(conn: sqlite3.Connection, version: int) -> None:
    names = {name for name, introduced in _SCHEMA_INTRODUCED.items()
             if introduced == version}
    applied = set()
    for statement in _schema_statements():
        name = _schema_object_name(statement)
        if name in names:
            conn.execute(statement)
            applied.add(name)
    missing = names - applied
    if missing:
        raise RuntimeError(f"migration v{version - 1}->v{version} has no SQL for: "
                           f"{', '.join(sorted(missing))}")


def _migrate_control_cancellations(conn: sqlite3.Connection) -> None:
    """Give legacy interrupt controls conservative, non-replayable outcome truth."""
    rows = conn.execute(
        """SELECT * FROM task_controls WHERE kind IN ('interrupt','cancel')
           AND NOT EXISTS (
               SELECT 1 FROM control_cancellations x WHERE x.control_id=task_controls.id
           )"""
    ).fetchall()
    stamp = now_utc()
    for row in rows:
        payload = json_loads(row["payload_json"], {})
        if not isinstance(payload, dict):
            payload = {}
        had_truth = "cancellation_requested" in payload
        cancellable = bool(payload.get("cancellable", False))
        requested = payload.get("cancellation_requested")
        returned = "cancellation_result" in payload
        if had_truth and not cancellable and requested is False:
            cancellation_state = "resolved"
            safe_requested = False
            outcome = "legacy-not-cancellable"
        elif had_truth and requested is True and returned:
            cancellation_state = "resolved"
            safe_requested = True
            outcome = "legacy-request-dispatched"
        else:
            cancellation_state = "ambiguous"
            safe_requested = None
            outcome = "legacy-outcome-ambiguous"
        payload.update({
            "cancellation_phase": cancellation_state,
            "cancellation_requested": safe_requested,
            "cancellation_outcome": outcome,
            "cancellation_result": {"requested": safe_requested},
            "cancellable": cancellable,
            "cancellation_effect": "execute",
        })
        payload.pop("cancellation_error", None)
        conn.execute(
            "UPDATE task_controls SET payload_json=? WHERE id=?",
            (json_dumps(payload), row["id"]),
        )
        conn.execute(
            """INSERT INTO control_cancellations(
               control_id,state,operation_id,active_tool,cancellable,cancellation_effect,
               result_json,error,created_at,reconciled_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (row["id"], cancellation_state, "",
             str(payload.get("active_tool", ""))[:256], int(cancellable), "execute",
             json_dumps({"requested": safe_requested}), "", row["created_at"], stamp),
        )


def _expected_schema() -> dict[str, tuple[str, ...]]:
    expected = {}
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for statement in _schema_statements():
            conn.execute(statement)
        for name in _SCHEMA_INTRODUCED:
            row = conn.execute(
                "SELECT type FROM sqlite_master WHERE name=?", (name,)).fetchone()
            if row is None:
                raise RuntimeError(f"authoritative schema object missing: {name}")
            expected[name] = tuple(
                item[1] for item in conn.execute(f'PRAGMA table_info("{name}")'))
    finally:
        conn.close()
    return expected


def schema_errors(conn: sqlite3.Connection) -> list[str]:
    errors = []
    for name, columns in _expected_schema().items():
        row = conn.execute("SELECT type FROM sqlite_master WHERE name=?", (name,)).fetchone()
        if row is None:
            errors.append(f"missing schema object: {name}")
            continue
        actual = tuple(item[1] for item in conn.execute(f'PRAGMA table_info("{name}")'))
        if actual != columns:
            errors.append(f"schema columns differ for {name}: {actual!r} != {columns!r}")
    return errors


def migrate_connection(conn: sqlite3.Connection) -> int:
    """Upgrade an opened state database through ordered, atomic schema transitions."""
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "meta" not in tables:
        version = 0
    else:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            raise RuntimeError("state database has a meta table but no schema_version")
        try:
            version = int(row[0])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid state schema version: {row[0]!r}") from exc
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"state schema {version} is newer than supported {SCHEMA_VERSION}")
    if version and version < BASE_SCHEMA_VERSION:
        raise RuntimeError(
            f"state schema {version} predates the supported SQLite baseline "
            f"{BASE_SCHEMA_VERSION}")

    conn.execute("BEGIN IMMEDIATE")
    try:
        if version == 0:
            _apply_schema_level(conn, BASE_SCHEMA_VERSION)
            version = BASE_SCHEMA_VERSION
            conn.execute(
                "INSERT INTO meta(key,value) VALUES('schema_version',?)", (str(version),))
        while version < SCHEMA_VERSION:
            target = version + 1
            _apply_schema_level(conn, target)
            if target == 19:
                _migrate_control_cancellations(conn)
            conn.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(target),))
            version = target
        errors = schema_errors(conn)
        if errors:
            raise RuntimeError("state schema validation failed: " + "; ".join(errors))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return version


def ensure_state_store() -> Path:
    ensure_state_store_no_migration()
    migrate_legacy_task_store()
    return STATE_DB_PATH


def get_meta(key: str, default=None):
    ensure_state_store_no_migration()
    conn = connect()
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default
    finally:
        conn.close()


def set_meta(key: str, value) -> None:
    ensure_state_store_no_migration()
    with transaction(immediate=True) as conn:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )


def ensure_state_store_no_migration() -> Path:
    STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = connect()
    try:
        migrate_connection(conn)
        with conn:
            conn.execute(
                """UPDATE core_clients SET state='revoked',revoked_at=COALESCE(revoked_at, ?)
                   WHERE state='active' AND token_hash NOT LIKE 'v2$%'""",
                (now_utc(),),
            )
    finally:
        conn.close()
    return STATE_DB_PATH


def health() -> dict:
    ensure_state_store()
    conn = connect()
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        version_row = conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        version = int(version_row[0]) if version_row else 0
        shape_errors = schema_errors(conn)
        cancellation_orphans = conn.execute(
            """SELECT COUNT(*) FROM task_controls c
               WHERE c.kind IN ('interrupt','cancel') AND NOT EXISTS (
                   SELECT 1 FROM control_cancellations x WHERE x.control_id=c.id
               )"""
        ).fetchone()[0]
        state_errors = ([] if not cancellation_orphans else [
            f"{cancellation_orphans} cancellation controls lack durable outcome state"
        ])
        counts = {}
        for table in ("conversations", "messages", "sessions", "state_events", "tasks", "task_events", "checkpoints", "context_projections", "context_epochs", "task_runs", "resource_leases", "delegations", "delegation_contracts", "delegation_memory", "mcp_servers", "handoffs", "routing_decisions", "role_state", "project_refs", "memory_index", "memory_candidates", "memory_maintenance_runs", "policy_rules", "approvals", "action_journal", "evidence_records", "task_controls", "control_cancellations", "workspace_checkpoints", "schedules", "schedule_runs", "schedule_deliveries", "core_clients", "core_pairings", "runtime_routes"):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {
            "ok": (integrity == "ok" and version == SCHEMA_VERSION
                   and not shape_errors and not state_errors),
            "integrity": integrity,
            "schema_errors": shape_errors,
            "state_errors": state_errors,
            "schema_version": version,
            "expected_schema_version": SCHEMA_VERSION,
            "counts": counts,
            "path": str(STATE_DB_PATH),
        }
    finally:
        conn.close()


def _legacy_migrated() -> bool:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='legacy_task_store_migrated'"
        ).fetchone()
        return bool(row and row[0] == "1")
    finally:
        conn.close()


def migrate_legacy_task_store() -> None:
    """Import the v0.3 JSON/JSONL task store once, without deleting source files."""
    ensure_state_store_no_migration()
    if _legacy_migrated():
        return

    # Valid records are committed even when peers are malformed. Failures remain
    # observable and keep the import retryable; stable event IDs make retries safe.
    failures = []
    with transaction(immediate=True) as conn:
        if TASK_ROOT.exists():
            for path in sorted(TASK_ROOT.glob("task-*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    task_id = data["id"]
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO tasks(
                            id, conversation_id, title, goal, owner_role, state, kind,
                            phase, progress, parent_task_id, source, epoch,
                            constraints_json, decisions_json, completed_json,
                            open_steps_json, failures_json, evidence_refs_json,
                            schedule_kind, schedule_expr, next_run_at, last_run_at,
                            last_result_status, schedule_enabled, created_at, updated_at
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            task_id, None, data.get("title", ""), data.get("goal", ""),
                            data.get("owner_role", "general"), data.get("state", "pending"),
                            data.get("kind", "primary"), data.get("phase", ""),
                            data.get("progress"), data.get("parent_task_id"),
                            data.get("source", "legacy-json"), int(data.get("epoch", 1) or 1),
                            json_dumps(data.get("constraints", [])),
                            json_dumps(data.get("decisions", [])),
                            json_dumps(data.get("completed", [])),
                            json_dumps(data.get("open_steps", [])),
                            json_dumps(data.get("failures", [])),
                            json_dumps(data.get("evidence_refs", [])),
                            None, None, None, None, None, 1,
                            data.get("created_at") or now_utc(),
                            data.get("updated_at") or now_utc(),
                        ),
                    )
                    checkpoint = data.get("checkpoint") or {}
                    if checkpoint:
                        # Preserve old mutable checkpoint as first immutable snapshot.
                        import hashlib, uuid
                        payload = json_dumps(checkpoint)
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO checkpoints(
                                id, task_id, epoch, seq, created_at, owner_role, reason,
                                state_json, evidence_refs_json, content_sha256
                            ) VALUES(?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                "cp-" + uuid.uuid4().hex,
                                task_id, 1, 1, data.get("updated_at") or now_utc(),
                                data.get("owner_role", "general"), "legacy migration",
                                payload, "[]", hashlib.sha256(payload.encode()).hexdigest(),
                            ),
                        )
                except Exception as exc:
                    failures.append({"source": str(path), "error": str(exc)})

        if TASK_EVENTS_ROOT.exists():
            for path in sorted(TASK_EVENTS_ROOT.glob("task-*.jsonl")):
                task_id = path.stem
                for line_number, line in enumerate(
                        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    try:
                        event = json.loads(line)
                        exists = conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone()
                        if not exists:
                            raise ValueError(f"unknown legacy task: {task_id}")
                        import hashlib
                        event_uuid = "evt-legacy-" + hashlib.sha256(
                            f"{path.name}:{line_number}:{line}".encode()).hexdigest()
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO task_events(
                                event_uuid, task_id, conversation_id, timestamp, type,
                                source_role, message, payload_json, visibility
                            ) VALUES(?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                event_uuid,
                                task_id, None, event.get("timestamp") or now_utc(),
                                event.get("type", "status"), event.get("role"),
                                event.get("message", ""), json_dumps(event.get("data", {})),
                                "normal",
                            ),
                        )
                    except Exception as exc:
                        failures.append({"source": f"{path}:{line_number}",
                                         "error": str(exc)})

        # Preserve active-task pointer if v0.3 had one.
        if TASK_INDEX_PATH.exists():
            try:
                idx = json.loads(TASK_INDEX_PATH.read_text(encoding="utf-8"))
                active = idx.get("active_task")
                if active and conn.execute("SELECT 1 FROM tasks WHERE id=?", (active,)).fetchone():
                    conn.execute(
                        "INSERT OR REPLACE INTO meta(key,value) VALUES('active_task_id',?)",
                        (active,),
                    )
            except Exception as exc:
                failures.append({"source": str(TASK_INDEX_PATH), "error": str(exc)})

        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (
            "legacy_task_store_failures", json_dumps(failures)))
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (
            "legacy_task_store_migrated", "0" if failures else "1"))
