from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import uuid

from .state_store import (connect, ensure_state_store, get_meta, json_dumps, json_loads,
                          now_utc, resolve_lineage_in_transaction, set_meta, transaction)


@dataclass(frozen=True)
class ConversationRecord:
    id: str
    title: str
    state: str
    source: str
    created_at: str
    updated_at: str
    last_message_at: str | None
    metadata: dict


@dataclass(frozen=True)
class MessageRecord:
    id: str
    conversation_id: str
    seq: int
    role: str
    content: str
    kind: str
    include_in_context: bool
    related_task_id: str | None
    metadata: dict
    created_at: str


def _conversation_from_row(row) -> ConversationRecord:
    return ConversationRecord(
        id=row["id"], title=row["title"], state=row["state"], source=row["source"],
        created_at=row["created_at"], updated_at=row["updated_at"],
        last_message_at=row["last_message_at"], metadata=json_loads(row["metadata_json"], {}),
    )


def _message_from_row(row) -> MessageRecord:
    return MessageRecord(
        id=row["id"], conversation_id=row["conversation_id"], seq=row["seq"],
        role=row["role"], content=row["content"], kind=row["kind"],
        include_in_context=bool(row["include_in_context"]),
        related_task_id=row["related_task_id"], metadata=json_loads(row["metadata_json"], {}),
        created_at=row["created_at"],
    )


def create_conversation(*, title="", source="chat", metadata=None, make_active=True) -> ConversationRecord:
    ensure_state_store()
    now = now_utc()
    conv_id = "conv-" + uuid.uuid4().hex
    with transaction(immediate=True) as conn:
        conn.execute(
            "INSERT INTO conversations(id,title,state,source,created_at,updated_at,metadata_json) VALUES(?,?,?,?,?,?,?)",
            (conv_id, title, "open", source, now, now, json_dumps(metadata or {})),
        )
        if make_active:
            conn.execute(
                "INSERT INTO meta(key,value) VALUES('active_conversation_id',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (conv_id,),
            )
    return load_conversation(conv_id)


def load_conversation(conversation_id: str) -> ConversationRecord:
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown conversation: {conversation_id}")
        return _conversation_from_row(row)
    finally:
        conn.close()


def list_conversations(limit=50) -> list[ConversationRecord]:
    ensure_state_store()
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY COALESCE(last_message_at, updated_at) DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [_conversation_from_row(row) for row in rows]
    finally:
        conn.close()


def active_conversation() -> ConversationRecord | None:
    conv_id = get_meta("active_conversation_id")
    if not conv_id:
        return None
    try:
        return load_conversation(conv_id)
    except KeyError:
        return None


def set_active_conversation(conversation_id: str | None) -> None:
    if conversation_id is not None:
        load_conversation(conversation_id)
    set_meta("active_conversation_id", conversation_id or "")


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    *,
    kind="message",
    include_in_context=True,
    related_task_id=None,
    metadata=None,
    session_id=None,
) -> MessageRecord:
    ensure_state_store()
    now = now_utc()
    message_id = "msg-" + uuid.uuid4().hex
    from .state_events import insert_state_event
    with transaction(immediate=True) as conn:
        if not conn.execute("SELECT 1 FROM conversations WHERE id=?", (conversation_id,)).fetchone():
            raise KeyError(f"unknown conversation: {conversation_id}")
        resolve_lineage_in_transaction(
            conn, task_id=related_task_id, session_id=session_id,
            conversation_id=conversation_id)
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 FROM messages WHERE conversation_id=?",
            (conversation_id,),
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO messages(
                id,conversation_id,seq,role,content,kind,include_in_context,
                related_task_id,metadata_json,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                message_id, conversation_id, seq, role, content, kind,
                1 if include_in_context else 0, related_task_id,
                json_dumps(metadata or {}), now,
            ),
        )
        conn.execute(
            "UPDATE conversations SET updated_at=?, last_message_at=? WHERE id=?",
            (now, now, conversation_id),
        )
        row = conn.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
        if role in {"user", "assistant"}:
            insert_state_event(
                conn,
                "user_message" if role == "user" else "assistant_response",
                session_id=session_id, conversation_id=conversation_id,
                task_id=related_task_id, message=content,
                payload={"message_id": message_id, "kind": kind},
            )
    return _message_from_row(row)


def list_messages(conversation_id: str, *, include_sideband=True, limit=200) -> list[MessageRecord]:
    ensure_state_store()
    conn = connect()
    try:
        sql = "SELECT * FROM messages WHERE conversation_id=?"
        params = [conversation_id]
        if not include_sideband:
            sql += " AND include_in_context=1"
        sql += " ORDER BY seq DESC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return [_message_from_row(row) for row in reversed(rows)]
    finally:
        conn.close()
