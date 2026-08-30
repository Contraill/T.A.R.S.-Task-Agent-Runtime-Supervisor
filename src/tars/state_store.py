from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path

from .config import STATE_DB_PATH, TASK_INDEX_PATH, TASK_ROOT, TASK_EVENTS_ROOT

SCHEMA_VERSION = 14


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
    STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(STATE_DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
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


def ensure_state_store() -> Path:
    STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = connect()
    try:
        conn.executescript(_schema_sql())
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
    finally:
        conn.close()
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
        conn.executescript(_schema_sql())
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
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
        counts = {}
        for table in ("conversations", "messages", "sessions", "state_events", "tasks", "task_events", "checkpoints", "context_projections", "context_epochs", "task_runs", "delegations", "delegation_contracts", "delegation_memory", "mcp_servers", "handoffs", "routing_decisions", "role_state", "project_refs", "memory_index", "memory_candidates", "memory_maintenance_runs", "policy_rules", "approvals", "action_journal", "evidence_records", "task_controls", "workspace_checkpoints"):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {
            "ok": integrity == "ok" and version == SCHEMA_VERSION,
            "integrity": integrity,
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

    # Import is deliberately best-effort per record. A malformed historical file
    # never prevents the new state store from opening.
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
                except Exception:
                    continue

        if TASK_EVENTS_ROOT.exists():
            import uuid
            for path in sorted(TASK_EVENTS_ROOT.glob("task-*.jsonl")):
                task_id = path.stem
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        event = json.loads(line)
                        exists = conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone()
                        if not exists:
                            continue
                        conn.execute(
                            """
                            INSERT INTO task_events(
                                event_uuid, task_id, conversation_id, timestamp, type,
                                source_role, message, payload_json, visibility
                            ) VALUES(?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                "evt-" + uuid.uuid4().hex,
                                task_id, None, event.get("timestamp") or now_utc(),
                                event.get("type", "status"), event.get("role"),
                                event.get("message", ""), json_dumps(event.get("data", {})),
                                "normal",
                            ),
                        )
                    except Exception:
                        continue

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
            except Exception:
                pass

        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('legacy_task_store_migrated','1')"
        )
