from __future__ import annotations

from dataclasses import dataclass
import uuid

from .state_store import (connect, ensure_state_store, json_dumps, json_loads, now_utc,
                          resolve_lineage_in_transaction, transaction)

EVENT_TYPES = {
    "user_message", "assistant_response", "model_invocation", "tool_proposal",
    "policy_decision", "approval", "tool_execution", "tool_result", "checkpoint",
    "task_transition", "delegation", "user_control", "interrupt", "redirect",
    "cancel", "pause", "resume", "session_start", "session_end", "activity",
}


@dataclass(frozen=True)
class StateEvent:
    id: int
    event_uuid: str
    timestamp: str
    type: str
    session_id: str | None
    conversation_id: str | None
    task_id: str | None
    source_role: str | None
    message: str
    payload: dict
    visibility: str


def _from_row(row):
    return StateEvent(
        id=row["id"], event_uuid=row["event_uuid"], timestamp=row["timestamp"],
        type=row["type"], session_id=row["session_id"],
        conversation_id=row["conversation_id"], task_id=row["task_id"],
        source_role=row["source_role"], message=row["message"],
        payload=json_loads(row["payload_json"], {}), visibility=row["visibility"],
    )


def append_state_event(event_type, message="", *, session_id=None, conversation_id=None,
                       task_id=None, role=None, payload=None, visibility="normal"):
    ensure_state_store()
    with transaction(immediate=True) as conn:
        return insert_state_event(
            conn, event_type, message, session_id=session_id,
            conversation_id=conversation_id, task_id=task_id, role=role,
            payload=payload, visibility=visibility,
        )


def insert_state_event(conn, event_type, message="", *, session_id=None,
                       conversation_id=None, task_id=None, role=None, payload=None,
                       visibility="normal"):
    """Insert an event using the caller's transaction for atomic state changes."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"invalid state event type: {event_type}")
    conversation_id = resolve_lineage_in_transaction(
        conn, task_id=task_id, session_id=session_id,
        conversation_id=conversation_id)
    event_uuid = "sev-" + uuid.uuid4().hex
    conn.execute(
        """INSERT INTO state_events(event_uuid,session_id,conversation_id,task_id,
           timestamp,type,source_role,message,payload_json,visibility)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (event_uuid, session_id, conversation_id, task_id, now_utc(), event_type,
         role, str(message), json_dumps(payload or {}), visibility),
    )
    row = conn.execute("SELECT * FROM state_events WHERE event_uuid=?", (event_uuid,)).fetchone()
    return _from_row(row)


def read_state_events(*, session_id=None, task_id=None, after_id=0, limit=200):
    ensure_state_store()
    clauses = ["id>?"]
    params = [int(after_id)]
    if session_id:
        clauses.append("session_id=?")
        params.append(session_id)
    if task_id:
        clauses.append("task_id=?")
        params.append(task_id)
    params.append(int(limit))
    conn = connect()
    try:
        rows = conn.execute(
            f"SELECT * FROM state_events WHERE {' AND '.join(clauses)} ORDER BY id LIMIT ?",
            params,
        ).fetchall()
        return [_from_row(row) for row in rows]
    finally:
        conn.close()
