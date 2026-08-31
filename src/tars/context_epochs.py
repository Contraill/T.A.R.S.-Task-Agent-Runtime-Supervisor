from __future__ import annotations

from dataclasses import dataclass
import hashlib
import uuid

from .events import append_event
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction
from .tasks import canonical_task_state_from_row


@dataclass(frozen=True)
class ContextEpoch:
    id: str
    conversation_id: str
    task_id: str
    epoch: int
    from_message_seq: int
    through_message_seq: int
    archived_messages: tuple[dict, ...]
    checkpoint_id: str
    unresolved: tuple[dict, ...]
    created_at: str


def _from_row(row):
    return ContextEpoch(
        id=row["id"], conversation_id=row["conversation_id"], task_id=row["task_id"],
        epoch=row["epoch"], from_message_seq=row["from_message_seq"],
        through_message_seq=row["through_message_seq"],
        archived_messages=tuple(json_loads(row["archived_messages_json"], [])),
        checkpoint_id=row["checkpoint_id"],
        unresolved=tuple(json_loads(row["unresolved_json"], [])), created_at=row["created_at"],
    )


def rollover(task_id: str, *, reason="context pressure") -> ContextEpoch:
    """Atomically checkpoint task truth, archive old context, and advance its epoch."""
    ensure_state_store()
    stamp = now_utc()
    checkpoint_id = "cp-" + uuid.uuid4().hex
    epoch_id = "epoch-" + uuid.uuid4().hex
    with transaction(immediate=True) as conn:
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not task:
            raise KeyError(f"unknown task: {task_id}")
        if not task["conversation_id"]:
            raise RuntimeError("context rollover requires a task conversation")
        state = canonical_task_state_from_row(task, conn)
        state_payload = json_dumps(state)
        digest = hashlib.sha256(state_payload.encode("utf-8")).hexdigest()
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id=? AND include_in_context=1 ORDER BY seq",
            (task["conversation_id"],),
        ).fetchall()
        latest_user = max((row["seq"] for row in rows if row["role"] == "user"), default=0)
        archived = []
        unresolved = []
        archive_seqs = []
        for row in rows:
            metadata = json_loads(row["metadata_json"], {})
            protected = (
                row["seq"] >= latest_user or row["kind"] in {"control", "pending_control"}
                or (row["role"] == "tool" and metadata.get("unresolved") is True)
            )
            item = {"id": row["id"], "seq": row["seq"], "role": row["role"],
                    "content": row["content"], "kind": row["kind"], "metadata": metadata}
            if protected:
                if row["role"] == "tool" or row["kind"] in {"control", "pending_control"}:
                    unresolved.append(item)
            else:
                archived.append(item)
                archive_seqs.append(row["seq"])
        if not archived:
            raise RuntimeError("no older context is eligible for epoch rollover")
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 FROM checkpoints WHERE task_id=?", (task_id,)
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO checkpoints(id,task_id,epoch,seq,created_at,owner_role,reason,
               state_json,evidence_refs_json,content_sha256) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (checkpoint_id, task_id, task["epoch"], seq, stamp, task["owner_role"], reason,
             state_payload, task["evidence_refs_json"], digest),
        )
        placeholders = ",".join("?" for _ in archive_seqs)
        conn.execute(
            f"UPDATE messages SET include_in_context=0 WHERE conversation_id=? AND seq IN ({placeholders})",
            (task["conversation_id"], *archive_seqs),
        )
        conn.execute("UPDATE tasks SET epoch=epoch+1,updated_at=? WHERE id=?", (stamp, task_id))
        conn.execute(
            """INSERT INTO context_epochs(id,conversation_id,task_id,epoch,
               from_message_seq,through_message_seq,archived_messages_json,checkpoint_id,
               unresolved_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (epoch_id, task["conversation_id"], task_id, task["epoch"], min(archive_seqs),
             max(archive_seqs), json_dumps(archived), checkpoint_id,
             json_dumps(unresolved), stamp),
        )
        row = conn.execute("SELECT * FROM context_epochs WHERE id=?", (epoch_id,)).fetchone()
    append_event(
        task_id, "checkpoint", f"Context epoch {task['epoch']} archived", role=task["owner_role"],
        data={"context_epoch_id": epoch_id, "checkpoint_id": checkpoint_id,
              "archived_messages": len(archived), "next_epoch": task["epoch"] + 1},
        visibility="verbose",
    )
    return _from_row(row)


def list_epochs(task_id, limit=50):
    ensure_state_store()
    conn = connect()
    try:
        return [_from_row(row) for row in conn.execute(
            "SELECT * FROM context_epochs WHERE task_id=? ORDER BY epoch DESC LIMIT ?",
            (task_id, int(limit)),
        ).fetchall()]
    finally:
        conn.close()


def search_transcript(conversation_id, query, *, limit=50):
    ensure_state_store()
    escaped = str(query).replace("%", r"\%").replace("_", r"\_")
    pattern = f"%{escaped}%"
    conn = connect()
    try:
        rows = conn.execute(
            """SELECT id,seq,role,content,kind,include_in_context,created_at FROM messages
               WHERE conversation_id=? AND content LIKE ? ESCAPE '\\'
               ORDER BY seq DESC LIMIT ?""",
            (conversation_id, pattern, int(limit)),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]
    finally:
        conn.close()
