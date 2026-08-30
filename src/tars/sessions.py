from __future__ import annotations

from dataclasses import dataclass
import uuid

from .conversation import create_conversation, load_conversation
from .roles import resolve_role_id
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction


@dataclass(frozen=True)
class SessionRecord:
    id: str
    conversation_id: str
    role_id: str
    state: str
    mode: str
    started_at: str
    updated_at: str
    ended_at: str | None
    metadata: dict


def _from_row(row):
    return SessionRecord(
        id=row["id"], conversation_id=row["conversation_id"], role_id=row["role_id"],
        state=row["state"], mode=row["mode"], started_at=row["started_at"],
        updated_at=row["updated_at"], ended_at=row["ended_at"],
        metadata=json_loads(row["metadata_json"], {}),
    )


def create_session(*, conversation_id=None, role_id="general", mode="normal", metadata=None):
    ensure_state_store()
    role_id = resolve_role_id(role_id)
    if mode not in {"normal", "temporary"}:
        raise ValueError("session mode must be normal or temporary")
    if conversation_id is None:
        conversation_id = create_conversation(source="session").id
    else:
        load_conversation(conversation_id)
    session_id = "ses-" + uuid.uuid4().hex
    stamp = now_utc()
    from .state_events import insert_state_event
    with transaction(immediate=True) as conn:
        conn.execute(
            "INSERT INTO sessions(id,conversation_id,role_id,state,mode,started_at,updated_at,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
            (session_id, conversation_id, role_id, "open", mode, stamp, stamp,
             json_dumps(metadata or {})),
        )
        insert_state_event(
            conn, "session_start", f"{mode} session started", session_id=session_id,
            conversation_id=conversation_id, role=role_id, payload={"mode": mode},
        )
    return load_session(session_id)


def load_session(session_id):
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown session: {session_id}")
        return _from_row(row)
    finally:
        conn.close()


def close_session(session_id):
    load_session(session_id)
    stamp = now_utc()
    from .state_events import insert_state_event
    with transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE sessions SET state='closed', updated_at=?, ended_at=? WHERE id=?",
            (stamp, stamp, session_id),
        )
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        insert_state_event(
            conn, "session_end", "session closed", session_id=session_id,
            conversation_id=row["conversation_id"], role=row["role_id"],
            payload={"mode": row["mode"]},
        )
    record = load_session(session_id)
    return record


def list_sessions(*, conversation_id=None, limit=50):
    ensure_state_store()
    sql = "SELECT * FROM sessions"
    params = []
    if conversation_id:
        sql += " WHERE conversation_id=?"
        params.append(conversation_id)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(int(limit))
    conn = connect()
    try:
        return [_from_row(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()
