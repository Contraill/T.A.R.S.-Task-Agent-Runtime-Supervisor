from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid

from .checkpoints import create_checkpoint, latest_checkpoint
from .conversation import active_conversation
from .events import append_event, read_events, EVENT_TYPES
from .ownership import (active_in_transaction, current_owner,
                        held_by_in_transaction)
from .roles import resolve_role_id
from .state_store import connect, ensure_state_store, get_meta, json_dumps, json_loads, now_utc, set_meta, transaction

TASK_STATES = {"pending", "running", "paused", "completed", "failed", "cancelled"}
TASK_KINDS = {"primary", "delegation", "sideband", "scheduled"}
TERMINAL_TASK_STATES = {"completed", "failed", "cancelled"}
EXECUTION_OWNED_COLUMNS = {
    "state", "phase", "progress", "owner_role", "conversation_id", "epoch",
    "constraints_json", "decisions_json", "completed_json", "open_steps_json",
    "failures_json", "evidence_refs_json",
}


@dataclass
class TaskRecord:
    id: str
    goal: str
    owner_role: str
    state: str
    kind: str
    phase: str
    progress: float | None
    parent_task_id: str | None
    created_at: str
    updated_at: str
    checkpoint: dict
    source: str
    conversation_id: str | None = None
    title: str = ""
    epoch: int = 1
    constraints: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    completed: tuple[str, ...] = ()
    open_steps: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    schedule_kind: str | None = None
    schedule_expr: str | None = None
    next_run_at: str | None = None
    last_run_at: str | None = None
    last_result_status: str | None = None
    schedule_enabled: bool = True


def _new_task_id():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"task-{stamp}-{uuid.uuid4().hex[:6]}"


def _from_row(row) -> TaskRecord:
    cp = latest_checkpoint(row["id"])
    return TaskRecord(
        id=row["id"], goal=row["goal"], owner_role=row["owner_role"],
        state=row["state"], kind=row["kind"], phase=row["phase"],
        progress=row["progress"], parent_task_id=row["parent_task_id"],
        created_at=row["created_at"], updated_at=row["updated_at"],
        checkpoint=cp.state if cp else {}, source=row["source"],
        conversation_id=row["conversation_id"], title=row["title"], epoch=row["epoch"],
        constraints=tuple(json_loads(row["constraints_json"], [])),
        decisions=tuple(json_loads(row["decisions_json"], [])),
        completed=tuple(json_loads(row["completed_json"], [])),
        open_steps=tuple(json_loads(row["open_steps_json"], [])),
        failures=tuple(json_loads(row["failures_json"], [])),
        evidence_refs=tuple(json_loads(row["evidence_refs_json"], [])),
        schedule_kind=row["schedule_kind"], schedule_expr=row["schedule_expr"],
        next_run_at=row["next_run_at"], last_run_at=row["last_run_at"],
        last_result_status=row["last_result_status"],
        schedule_enabled=bool(row["schedule_enabled"]),
    )


def ensure_task_store():
    # Compatibility name retained for v0.3 callers. The canonical store is SQLite.
    return ensure_state_store()


def create_task_in_transaction(conn, goal, owner_role, *, task_id=None, kind="primary",
                               parent_task_id=None, source="cli", conversation_id=None,
                               title="", constraints=(), evidence_refs=(), phase="created",
                               schedule_kind=None, schedule_expr=None, next_run_at=None,
                               make_active=False, created_at=None):
    """Insert a task as part of a caller-owned authoritative transaction."""
    if kind not in TASK_KINDS:
        raise ValueError(f"invalid task kind: {kind}")
    if parent_task_id is not None and not conn.execute(
            "SELECT 1 FROM tasks WHERE id=?", (parent_task_id,)).fetchone():
        raise KeyError(f"unknown task: {parent_task_id}")
    task_id = task_id or _new_task_id()
    stamp = created_at or now_utc()
    conn.execute(
        """INSERT INTO tasks(
           id,conversation_id,title,goal,owner_role,state,kind,phase,progress,
           parent_task_id,source,epoch,constraints_json,decisions_json,completed_json,
           open_steps_json,failures_json,evidence_refs_json,schedule_kind,schedule_expr,
           next_run_at,last_run_at,last_result_status,schedule_enabled,created_at,updated_at
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (task_id, conversation_id, title, goal, owner_role, "pending", kind, phase, None,
         parent_task_id, source, 1, json_dumps(list(constraints)), "[]", "[]", "[]", "[]",
         json_dumps(list(evidence_refs)), schedule_kind, schedule_expr, next_run_at, None, None,
         1, stamp, stamp))
    if make_active:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES('active_task_id',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (task_id,))
    return task_id


def create_task(
    goal,
    owner_role,
    *,
    kind="primary",
    parent_task_id=None,
    source="cli",
    make_active=True,
    conversation_id=None,
    title="",
    schedule_kind=None,
    schedule_expr=None,
    next_run_at=None,
):
    ensure_state_store()
    role = resolve_role_id(owner_role)
    if conversation_id is None:
        conv = active_conversation()
        conversation_id = conv.id if conv else None
    with transaction(immediate=True) as conn:
        task_id = create_task_in_transaction(
            conn, goal, role, kind=kind, parent_task_id=parent_task_id, source=source,
            conversation_id=conversation_id, title=title, schedule_kind=schedule_kind,
            schedule_expr=schedule_expr, next_run_at=next_run_at, make_active=make_active)
    append_event(task_id, "status", f"Task created with owner {role}", role=role,
                 data={"state": "pending", "kind": kind})
    return load_task(task_id)


def load_task(task_id):
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown task: {task_id}")
        return _from_row(row)
    finally:
        conn.close()


def list_tasks(limit=50, *, scheduled_only=False, conversation_id=None):
    ensure_state_store()
    sql = "SELECT * FROM tasks WHERE 1=1"
    params = []
    if scheduled_only:
        sql += " AND (kind='scheduled' OR schedule_kind IS NOT NULL)"
    if conversation_id is not None:
        sql += " AND conversation_id=?"
        params.append(conversation_id)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(int(limit))
    conn = connect()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [_from_row(row) for row in rows]
    finally:
        conn.close()


def require_task_write_in_transaction(conn, task_id, *, changes=None, owner=None,
                                      allow_fail_closed=False):
    """Fence execution-owned task truth and terminal-state monotonicity."""
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not row:
        raise KeyError(f"unknown task: {task_id}")
    selected = owner or current_owner()
    if (active_in_transaction(conn, "task-execution", task_id)
            and not allow_fail_closed and not (
                selected and held_by_in_transaction(
                    conn, "task-execution", task_id, selected))):
        raise RuntimeError(
            f"task {task_id} execution-owned state has a live executor")
    changes = dict(changes or {})
    if row["state"] in TERMINAL_TASK_STATES:
        changed = [
            key for key, value in changes.items()
            if key in EXECUTION_OWNED_COLUMNS and key in row.keys() and row[key] != value
        ]
        if changed:
            raise RuntimeError(
                f"terminal task {task_id} cannot mutate execution truth: "
                + ", ".join(sorted(changed)))
    return row



def attach_conversation(task_id, conversation_id):
    """Attach only previously unbound task provenance."""
    from .conversation import load_conversation
    load_conversation(conversation_id)
    with transaction(immediate=True) as conn:
        current = require_task_write_in_transaction(
            conn, task_id, changes={"conversation_id": conversation_id})
        if current["conversation_id"] not in {None, conversation_id}:
            raise RuntimeError("task conversation provenance is already bound")
        conn.execute(
            "UPDATE tasks SET conversation_id=?,updated_at=? WHERE id=?",
            (conversation_id, now_utc(), task_id),
        )
    return load_task(task_id)


def update_task(
    task_id,
    *,
    state=None,
    phase=None,
    progress=None,
    checkpoint=None,
    owner_role=None,
    title=None,
    constraints=None,
    decisions=None,
    completed=None,
    open_steps=None,
    failures=None,
    evidence_refs=None,
    schedule_kind=None,
    schedule_expr=None,
    next_run_at=None,
    last_run_at=None,
    last_result_status=None,
    schedule_enabled=None,
    _execution_owner=None,
):
    load_task(task_id)
    values = {}
    if state is not None:
        if state not in TASK_STATES:
            raise ValueError(f"invalid task state: {state}")
        values["state"] = state
    if phase is not None:
        values["phase"] = phase
    if progress is not None:
        if not 0 <= progress <= 1:
            raise ValueError("progress must be between 0 and 1")
        values["progress"] = progress
    if owner_role is not None:
        values["owner_role"] = resolve_role_id(owner_role)
    if title is not None:
        values["title"] = title
    for key, value in (
        ("constraints_json", constraints), ("decisions_json", decisions),
        ("completed_json", completed), ("open_steps_json", open_steps),
        ("failures_json", failures), ("evidence_refs_json", evidence_refs),
    ):
        if value is not None:
            values[key] = json_dumps(list(value))
    for key, value in (
        ("schedule_kind", schedule_kind), ("schedule_expr", schedule_expr),
        ("next_run_at", next_run_at), ("last_run_at", last_run_at),
        ("last_result_status", last_result_status),
    ):
        if value is not None:
            values[key] = value
    if schedule_enabled is not None:
        values["schedule_enabled"] = 1 if schedule_enabled else 0

    if values:
        values["updated_at"] = now_utc()
        assignments = ", ".join(f"{key}=?" for key in values)
        with transaction(immediate=True) as conn:
            require_task_write_in_transaction(
                conn, task_id, changes=values, owner=_execution_owner)
            conn.execute(
                f"UPDATE tasks SET {assignments} WHERE id=?",
                (*values.values(), task_id),
            )

    # Compatibility bridge: v0.3 callers that supplied checkpoint= now create an
    # immutable snapshot rather than overwriting mutable JSON state.
    if checkpoint is not None:
        create_checkpoint(task_id, checkpoint, reason="task update")

    task = load_task(task_id)
    append_event(
        task.id, "status", f"Task state updated: {task.state}", role=task.owner_role,
        data={"state": task.state, "phase": task.phase, "progress": task.progress,
              "epoch": task.epoch},
    )
    return task


def recover_task(task_id, *, phase="recovered"):
    """Explicitly recover a failed task when no execution owner remains."""
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown task: {task_id}")
        if active_in_transaction(conn, "task-execution", task_id):
            raise RuntimeError(f"task {task_id} still has a live execution owner")
        if row["state"] != "failed":
            raise RuntimeError(f"task {task_id} is not failed")
        if row["phase"] == "cancellation-recovery-required":
            raise RuntimeError("ambiguous cancellation requires dedicated recovery")
        conn.execute(
            "UPDATE tasks SET state='pending',phase=?,updated_at=? "
            "WHERE id=? AND state='failed'",
            (str(phase), stamp, task_id),
        )
    task = load_task(task_id)
    append_event(
        task.id, "status", "Task explicitly recovered", role=task.owner_role,
        data={"state": task.state, "phase": task.phase},
    )
    return task


def canonical_task_state_from_row(row, conn) -> dict:
    redirect = conn.execute(
        "SELECT message FROM task_controls WHERE task_id=? AND kind='redirect' "
        "AND state='applied' ORDER BY seq DESC LIMIT 1", (row["id"],)).fetchone()
    return {
        "task_id": row["id"], "title": row["title"], "goal": row["goal"],
        "current_instruction": redirect["message"] if redirect else row["goal"],
        "owner_role": row["owner_role"], "state": row["state"], "kind": row["kind"],
        "phase": row["phase"], "progress": row["progress"], "epoch": row["epoch"],
        "constraints": json_loads(row["constraints_json"], []),
        "decisions": json_loads(row["decisions_json"], []),
        "completed": json_loads(row["completed_json"], []),
        "open_steps": json_loads(row["open_steps_json"], []),
        "failures": json_loads(row["failures_json"], []),
        "evidence_refs": json_loads(row["evidence_refs_json"], []),
        "parent_task_id": row["parent_task_id"], "conversation_id": row["conversation_id"],
        "schedule": {
            "kind": row["schedule_kind"], "expr": row["schedule_expr"],
            "next_run_at": row["next_run_at"], "last_run_at": row["last_run_at"],
            "last_result_status": row["last_result_status"],
            "enabled": bool(row["schedule_enabled"]),
        },
    }


def canonical_task_state(task_id) -> dict:
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown task: {task_id}")
        return canonical_task_state_from_row(row, conn)
    finally:
        conn.close()


def checkpoint_task(task_id, *, reason="manual checkpoint", advance_epoch=False):
    return create_checkpoint(
        task_id, reason=reason, advance_epoch=advance_epoch,
    )


def set_active_task(task_id):
    if task_id is not None:
        load_task(task_id)
    set_meta("active_task_id", task_id or "")


def active_task():
    task_id = get_meta("active_task_id")
    if not task_id:
        return None
    try:
        return load_task(task_id)
    except KeyError:
        return None


def clear_active_task(task_id=None):
    current = get_meta("active_task_id")
    if task_id is None or current == task_id:
        set_meta("active_task_id", "")


__all__ = [
    "TASK_STATES", "TASK_KINDS", "EVENT_TYPES", "TaskRecord", "ensure_task_store",
    "create_task", "load_task", "list_tasks", "update_task", "append_event",
    "read_events", "set_active_task", "active_task", "clear_active_task",
    "canonical_task_state", "checkpoint_task", "attach_conversation",
    "require_task_write_in_transaction", "recover_task",
]
