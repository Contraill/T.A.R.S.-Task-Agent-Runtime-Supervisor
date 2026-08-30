from __future__ import annotations

from dataclasses import dataclass
import uuid

from .state_events import append_state_event
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction
from .tasks import load_task


PRIORITY = {"cancel": 0, "interrupt": 1, "pause": 1, "approval": 2,
            "redirect": 3, "resume": 3, "message": 4}


@dataclass(frozen=True)
class ControlEvent:
    id: str
    task_id: str
    session_id: str | None
    seq: int
    kind: str
    priority: int
    state: str
    message: str
    payload: dict
    created_at: str
    applied_at: str | None


def _from_row(row):
    return ControlEvent(
        row["id"], row["task_id"], row["session_id"], row["seq"], row["kind"],
        row["priority"], row["state"], row["message"],
        json_loads(row["payload_json"], {}), row["created_at"], row["applied_at"],
    )


def enqueue(task_id, kind, message="", *, session_id=None, payload=None):
    load_task(task_id)
    kind = str(kind).casefold()
    if kind not in PRIORITY:
        raise ValueError(f"unsupported control kind: {kind}")
    control_id = "control-" + uuid.uuid4().hex
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 FROM task_controls WHERE task_id=?", (task_id,),
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO task_controls(id,task_id,session_id,seq,kind,priority,state,
               message,payload_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (control_id, task_id, session_id, seq, kind, PRIORITY[kind], "pending",
             str(message), json_dumps(payload or {}), stamp),
        )
        row = conn.execute("SELECT * FROM task_controls WHERE id=?", (control_id,)).fetchone()
    append_state_event(
        "interrupt" if kind == "interrupt" else kind if kind in {"cancel", "redirect", "pause", "resume"} else "user_control",
        message, session_id=session_id, task_id=task_id,
        payload={"control_id": control_id, "seq": seq, "kind": kind, "state": "pending"},
    )
    return _from_row(row)


def load(control_id):
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM task_controls WHERE id=?", (control_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown control: {control_id}")
        return _from_row(row)
    finally:
        conn.close()


def list_controls(task_id, *, state=None, limit=200):
    ensure_state_store()
    sql = "SELECT * FROM task_controls WHERE task_id=?"
    params = [task_id]
    if state:
        sql += " AND state=?"
        params.append(state)
    sql += " ORDER BY seq LIMIT ?"
    params.append(int(limit))
    conn = connect()
    try:
        return [_from_row(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def claim_next(task_id):
    ensure_state_store()
    with transaction(immediate=True) as conn:
        row = conn.execute(
            "SELECT * FROM task_controls WHERE task_id=? AND state='pending' "
            "ORDER BY priority,seq LIMIT 1", (task_id,),
        ).fetchone()
        if not row:
            return None
        changed = conn.execute(
            "UPDATE task_controls SET state='processing' WHERE id=? AND state='pending'",
            (row["id"],),
        ).rowcount
        if changed != 1:
            return None
        row = conn.execute("SELECT * FROM task_controls WHERE id=?", (row["id"],)).fetchone()
        return _from_row(row)


def finish(control_id, *, success=True, payload=None):
    current = load(control_id)
    if current.state not in {"pending", "processing"}:
        raise RuntimeError(f"control {control_id} is already {current.state}")
    merged = current.payload | (payload or {})
    state = "applied" if success else "failed"
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE task_controls SET state=?,payload_json=?,applied_at=? WHERE id=?",
            (state, json_dumps(merged), stamp, control_id),
        )
    append_state_event(
        "interrupt" if current.kind == "interrupt" else
        current.kind if current.kind in {"cancel", "redirect", "pause", "resume"} else
        "user_control",
        current.message, session_id=current.session_id, task_id=current.task_id,
        payload={"control_id": current.id, "seq": current.seq,
                 "kind": current.kind, "state": state},
    )
    return load(control_id)


def annotate(control_id, payload):
    current = load(control_id)
    if current.state not in {"pending", "processing"}:
        return current
    with transaction(immediate=True) as conn:
        conn.execute("UPDATE task_controls SET payload_json=? WHERE id=?",
                     (json_dumps(current.payload | dict(payload)), control_id))
    return load(control_id)


def recover_processing(task_id):
    """Return controls stranded by a previous process to the pending queue."""
    with transaction(immediate=True) as conn:
        rows = conn.execute(
            "SELECT * FROM task_controls WHERE task_id=? AND state='processing' ORDER BY seq",
            (task_id,),
        ).fetchall()
        for row in rows:
            payload = json_loads(row["payload_json"], {})
            payload["recovered_after_disconnect"] = True
            conn.execute("UPDATE task_controls SET state='pending',payload_json=? WHERE id=?",
                         (json_dumps(payload), row["id"]))
    return len(rows)


def pending_context(task_id):
    return [{"id": item.id, "seq": item.seq, "kind": item.kind,
             "message": item.message, "payload": item.payload}
            for item in list_controls(task_id, state="pending")]


def latest_redirect(task_id):
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM task_controls WHERE task_id=? AND kind='redirect' AND state='applied' "
            "ORDER BY seq DESC LIMIT 1", (task_id,),
        ).fetchone()
        return _from_row(row) if row else None
    finally:
        conn.close()
