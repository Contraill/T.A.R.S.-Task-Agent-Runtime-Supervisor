from __future__ import annotations

from dataclasses import dataclass
import hashlib
import uuid

from .events import append_event
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction


@dataclass(frozen=True)
class CheckpointRecord:
    id: str
    task_id: str
    epoch: int
    seq: int
    created_at: str
    owner_role: str
    reason: str
    state: dict
    evidence_refs: tuple[str, ...]
    content_sha256: str


def _from_row(row) -> CheckpointRecord:
    return CheckpointRecord(
        id=row["id"], task_id=row["task_id"], epoch=row["epoch"], seq=row["seq"],
        created_at=row["created_at"], owner_role=row["owner_role"], reason=row["reason"],
        state=json_loads(row["state_json"], {}),
        evidence_refs=tuple(json_loads(row["evidence_refs_json"], [])),
        content_sha256=row["content_sha256"],
    )


def create_checkpoint(task_id, state: dict | None = None, *, reason="", evidence_refs=None,
                      advance_epoch=False) -> CheckpointRecord:
    """Atomically persist an immutable task snapshot.

    The checkpoint row itself can never be UPDATEd or DELETEd due to DB triggers.
    If advance_epoch is true, the task epoch advances only after the snapshot insert
    succeeds in the same SQLite transaction.
    """
    ensure_state_store()
    cp_id = "cp-" + uuid.uuid4().hex
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not task:
            raise KeyError(f"unknown task: {task_id}")
        if state is None:
            from .tasks import canonical_task_state_from_row
            state = canonical_task_state_from_row(task, conn)
        for key, actual in (("task_id", task_id), ("owner_role", task["owner_role"]),
                            ("epoch", int(task["epoch"]))):
            if key in state and state[key] != actual:
                raise RuntimeError(
                    f"checkpoint {key} does not match locked task snapshot")
        payload = json_dumps(state)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 FROM checkpoints WHERE task_id=?",
            (task_id,),
        ).fetchone()[0]
        epoch = int(task["epoch"])
        conn.execute(
            """
            INSERT INTO checkpoints(
                id,task_id,epoch,seq,created_at,owner_role,reason,state_json,
                evidence_refs_json,content_sha256
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (cp_id, task_id, epoch, seq, stamp, task["owner_role"], reason,
             payload, json_dumps(list(evidence_refs) if evidence_refs is not None else
                                 json_loads(task["evidence_refs_json"], [])), digest),
        )
        if advance_epoch:
            conn.execute(
                "UPDATE tasks SET epoch=epoch+1, updated_at=? WHERE id=?",
                (stamp, task_id),
            )
        row = conn.execute("SELECT * FROM checkpoints WHERE id=?", (cp_id,)).fetchone()
    checkpoint = _from_row(row)
    append_event(
        task_id, "checkpoint", f"Checkpoint {checkpoint.seq} persisted",
        role=checkpoint.owner_role,
        data={"checkpoint_id": checkpoint.id, "epoch": checkpoint.epoch,
              "sha256": checkpoint.content_sha256, "advance_epoch": advance_epoch},
        visibility="verbose",
    )
    return checkpoint


def latest_checkpoint(task_id) -> CheckpointRecord | None:
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM checkpoints WHERE task_id=? ORDER BY seq DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        return _from_row(row) if row else None
    finally:
        conn.close()


def list_checkpoints(task_id, limit=50) -> list[CheckpointRecord]:
    ensure_state_store()
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT * FROM checkpoints WHERE task_id=? ORDER BY seq DESC LIMIT ?",
            (task_id, int(limit)),
        ).fetchall()
        return [_from_row(row) for row in rows]
    finally:
        conn.close()


def verify_checkpoint(checkpoint_id: str) -> bool:
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM checkpoints WHERE id=?", (checkpoint_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown checkpoint: {checkpoint_id}")
        digest = hashlib.sha256(row["state_json"].encode("utf-8")).hexdigest()
        return digest == row["content_sha256"]
    finally:
        conn.close()
