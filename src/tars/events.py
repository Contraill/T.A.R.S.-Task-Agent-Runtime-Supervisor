from __future__ import annotations

from dataclasses import dataclass
import uuid

from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction

EVENT_TYPES = {
    "progress", "tool", "reasoning", "status", "handoff", "delegation",
    "routing", "sideband", "user", "result", "error", "checkpoint",
}
EVENT_VISIBILITY = {"quiet", "normal", "verbose", "internal"}


@dataclass(frozen=True)
class EventRecord:
    id: int
    event_uuid: str
    task_id: str
    conversation_id: str | None
    timestamp: str
    type: str
    source_role: str | None
    message: str
    payload: dict
    visibility: str


def _from_row(row) -> EventRecord:
    return EventRecord(
        id=row["id"], event_uuid=row["event_uuid"], task_id=row["task_id"],
        conversation_id=row["conversation_id"], timestamp=row["timestamp"],
        type=row["type"], source_role=row["source_role"], message=row["message"],
        payload=json_loads(row["payload_json"], {}), visibility=row["visibility"],
    )


def append_event(task_id, event_type, message, *, role=None, data=None, visibility="normal") -> EventRecord:
    ensure_state_store()
    if event_type not in EVENT_TYPES:
        raise ValueError(f"invalid event type: {event_type}")
    if visibility not in EVENT_VISIBILITY:
        raise ValueError(f"invalid event visibility: {visibility}")
    event_uuid = "evt-" + uuid.uuid4().hex
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        task = conn.execute("SELECT conversation_id FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not task:
            raise KeyError(f"unknown task: {task_id}")
        conn.execute(
            """
            INSERT INTO task_events(
                event_uuid,task_id,conversation_id,timestamp,type,source_role,
                message,payload_json,visibility
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (event_uuid, task_id, task["conversation_id"], stamp, event_type, role,
             str(message), json_dumps(data or {}), visibility),
        )
        row = conn.execute("SELECT * FROM task_events WHERE event_uuid=?", (event_uuid,)).fetchone()
    return _from_row(row)


def read_events(task_id, limit=100) -> list[dict]:
    ensure_state_store()
    conn = connect()
    try:
        if not conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
            raise KeyError(f"unknown task: {task_id}")
        rows = conn.execute(
            "SELECT * FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT ?",
            (task_id, int(limit)),
        ).fetchall()
    finally:
        conn.close()
    result = []
    for row in reversed(rows):
        event = _from_row(row)
        result.append({
            "id": event.id,
            "event_uuid": event.event_uuid,
            "timestamp": event.timestamp,
            "type": event.type,
            "task_id": event.task_id,
            "conversation_id": event.conversation_id,
            "role": event.source_role,
            "message": event.message,
            "data": event.payload,
            "visibility": event.visibility,
        })
    return result



def read_events_since(task_id, after_id=0, limit=200) -> list[dict]:
    """Read task events in ascending order after a durable event id."""
    ensure_state_store()
    conn = connect()
    try:
        if not conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
            raise KeyError(f"unknown task: {task_id}")
        rows = conn.execute(
            "SELECT * FROM task_events WHERE task_id=? AND id>? ORDER BY id ASC LIMIT ?",
            (task_id, int(after_id), int(limit)),
        ).fetchall()
    finally:
        conn.close()
    result = []
    for row in rows:
        event = _from_row(row)
        result.append({
            "id": event.id,
            "event_uuid": event.event_uuid,
            "timestamp": event.timestamp,
            "type": event.type,
            "task_id": event.task_id,
            "conversation_id": event.conversation_id,
            "role": event.source_role,
            "message": event.message,
            "data": event.payload,
            "visibility": event.visibility,
        })
    return result
