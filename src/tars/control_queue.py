from __future__ import annotations

from dataclasses import dataclass
import uuid

from .state_events import insert_state_event
from .ownership import Owner, claim_in_transaction, owner_alive
from .state_store import (connect, ensure_state_store, json_dumps, json_loads, now_utc,
                          resolve_lineage_in_transaction, transaction)
from .tasks import load_task, require_task_write_in_transaction


PRIORITY = {"cancel": 0, "interrupt": 1, "pause": 1, "approval": 2,
            "redirect": 3, "resume": 3, "message": 4}
CANCELLATION_READY = {"resolved", "ambiguous"}
CANCELLATION_RECOVERY_PHASE = "cancellation-recovery-required"


def _fail_task_for_ambiguous_cancellation(conn, task_id, control_id, reason):
    row = conn.execute(
        "SELECT owner_role,failures_json FROM tasks WHERE id=?", (task_id,),
    ).fetchone()
    if not row:
        raise KeyError(f"unknown task: {task_id}")
    failure = "external cancellation outcome is ambiguous; explicit recovery is required"
    failures = json_loads(row["failures_json"], [])
    if failure not in failures:
        failures.append(failure)
    stamp = now_utc()
    require_task_write_in_transaction(
        conn, task_id,
        changes={"state": "failed", "phase": CANCELLATION_RECOVERY_PHASE,
                 "failures_json": json_dumps(failures)},
        allow_fail_closed=True)
    conn.execute(
        "UPDATE tasks SET state='failed',phase=?,failures_json=?,updated_at=? WHERE id=?",
        (CANCELLATION_RECOVERY_PHASE, json_dumps(failures), stamp, task_id),
    )
    insert_state_event(
        conn, "task_transition", failure, task_id=task_id, role=row["owner_role"],
        payload={"control_id": control_id, "state": "recovery-required",
                 "reason": str(reason)[:128]},
    )


def _release_task_fence_if_quiescent(conn, task_id, owner_token=None):
    pending = conn.execute(
        """SELECT 1 FROM task_controls c JOIN control_cancellations x
           ON x.control_id=c.id WHERE c.task_id=? AND c.state='pending'
           AND x.state IN ('intent','attempting') LIMIT 1""",
        (task_id,),
    ).fetchone()
    if not pending:
        if owner_token is None:
            conn.execute(
                "DELETE FROM resource_leases "
                "WHERE resource_type='task-execution-fence' AND resource_key=?",
                (task_id,),
            )
        else:
            conn.execute(
                "DELETE FROM resource_leases WHERE resource_type='task-execution-fence' "
                "AND resource_key=? AND owner_token=?",
                (task_id, owner_token),
            )


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


def enqueue(task_id, kind, message="", *, session_id=None, payload=None,
            _cancellation=None):
    load_task(task_id)
    kind = str(kind).casefold()
    if kind not in PRIORITY:
        raise ValueError(f"unsupported control kind: {kind}")
    if kind in {"interrupt", "cancel"} and _cancellation is None:
        raise ValueError("interrupt/cancel controls require durable cancellation state")
    control_id = "control-" + uuid.uuid4().hex
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        resolve_lineage_in_transaction(
            conn, task_id=task_id, session_id=session_id)
        stored_payload = dict(payload or {})
        cancellation = dict(_cancellation) if _cancellation is not None else None
        if cancellation is not None:
            if kind not in {"interrupt", "cancel"}:
                raise ValueError("only interrupt/cancel controls have cancellation state")
            cancellation_state = cancellation["state"]
            if cancellation_state not in {"intent", "resolved"}:
                raise ValueError("new cancellation state must be intent or resolved")
            operation_id = str(cancellation.get("operation_id", ""))
            if cancellation_state == "intent" and operation_id:
                existing = conn.execute(
                    """SELECT x.control_id FROM control_cancellations x
                       JOIN task_controls c ON c.id=x.control_id
                       WHERE c.task_id=? AND c.state IN ('pending','processing')
                       AND x.operation_id=? ORDER BY c.seq LIMIT 1""",
                    (task_id, operation_id),
                ).fetchone()
                if existing:
                    cancellation["state"] = "resolved"
                    stored_payload.update({
                        "cancellation_phase": "resolved",
                        "cancellation_requested": False,
                        "cancellation_outcome": "already-requested",
                        "cancellation_duplicate_of": existing["control_id"],
                    })
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq),0)+1 FROM task_controls WHERE task_id=?", (task_id,),
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO task_controls(id,task_id,session_id,seq,kind,priority,state,
               message,payload_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (control_id, task_id, session_id, seq, kind, PRIORITY[kind], "pending",
             str(message), json_dumps(stored_payload), stamp),
        )
        if cancellation is not None:
            cancellation_state = cancellation["state"]
            conn.execute(
                """INSERT INTO control_cancellations(
                   control_id,state,operation_id,active_tool,cancellable,cancellation_effect,
                   result_json,error,created_at,reconciled_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (control_id, cancellation_state,
                 str(cancellation.get("operation_id", "")),
                 str(cancellation.get("active_tool", "")),
                 int(bool(cancellation.get("cancellable"))),
                 str(cancellation.get("cancellation_effect", "execute")),
                 json_dumps({"requested": False})
                 if cancellation_state == "resolved" else None,
                 str(cancellation.get("error", "")), stamp,
                 stamp if cancellation_state == "resolved" else None),
            )
            if cancellation_state == "intent":
                owner = cancellation.get("owner")
                if not isinstance(owner, Owner):
                    raise ValueError("unresolved cancellation intent requires an owner")
                if not claim_in_transaction(
                    conn, "control-cancellation", control_id, owner,
                    lease_seconds=float(cancellation.get("lease_seconds", 30.0)),
                    metadata={"kind": kind, "operation_id": operation_id,
                              "active_tool": cancellation.get("active_tool", "")},
                ):
                    raise RuntimeError("could not own new cancellation intent")
                fence = conn.execute(
                    "SELECT owner_token FROM resource_leases "
                    "WHERE resource_type='task-execution-fence' AND resource_key=?",
                    (task_id,),
                ).fetchone()
                if fence and fence["owner_token"] != owner.token:
                    raise RuntimeError("task already has an unresolved cancellation fence")
                if not claim_in_transaction(
                    conn, "task-execution-fence", task_id, owner,
                    lease_seconds=float(cancellation.get("lease_seconds", 30.0)),
                    metadata={"control_id": control_id, "operation_id": operation_id},
                ):
                    raise RuntimeError("could not fence task execution for cancellation")
        event_type = (
            "interrupt" if kind == "interrupt" else kind
            if kind in {"cancel", "redirect", "pause", "resume"} else "user_control")
        insert_state_event(
            conn, event_type, message, session_id=session_id, task_id=task_id,
            payload={"control_id": control_id, "seq": seq, "kind": kind,
                     "state": "pending"},
        )
        row = conn.execute("SELECT * FROM task_controls WHERE id=?", (control_id,)).fetchone()
    return _from_row(row)


def enqueue_cancellation(task_id, kind, message="", *, session_id=None, payload=None,
                         active_tool="", cancellable=False,
                         cancellation_effect="execute", operation_id="", owner=None,
                         ready=False, outcome=""):
    cancellation_effect = str(cancellation_effect)
    if cancellation_effect not in {"execute", "destructive"}:
        raise ValueError("invalid cancellation effect")
    state = "resolved" if ready else "intent"
    cancellation_payload = {
        "cancellation_phase": state,
        "active_tool": str(active_tool),
        "cancellable": bool(cancellable),
        "cancellation_effect": cancellation_effect,
        "cancellation_requested": False,
    }
    if outcome:
        cancellation_payload["cancellation_outcome"] = str(outcome)
    trusted_payload = {
        key: value for key, value in dict(payload or {}).items()
        if not str(key).startswith("cancellation_")
    }
    return enqueue(
        task_id, kind, message, session_id=session_id,
        payload=trusted_payload | cancellation_payload,
        _cancellation={
            "state": state, "active_tool": active_tool,
            "cancellable": cancellable, "cancellation_effect": cancellation_effect,
            "operation_id": operation_id, "owner": owner,
        },
    )


def cancellation_state(control_id):
    ensure_state_store()
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM control_cancellations WHERE control_id=?", (control_id,),
        ).fetchone()
    return dict(row) if row else None


def begin_cancellation(control_id, owner: Owner, *, lease_seconds=30.0):
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        row = conn.execute(
            """SELECT c.state AS control_state,c.kind,c.payload_json,
                      x.state AS cancellation_state,x.active_tool,x.cancellation_effect,
                      x.operation_id
               FROM task_controls c JOIN control_cancellations x ON x.control_id=c.id
               WHERE c.id=?""", (control_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"unknown cancellation control: {control_id}")
        if row["control_state"] != "pending" or row["cancellation_state"] != "intent":
            raise RuntimeError("cancellation attempt is no longer pending")
        if row["cancellation_effect"] == "destructive":
            raise PermissionError(
                "destructive cancellation requires separately authorized process control")
        if not claim_in_transaction(
            conn, "control-cancellation", control_id, owner,
            lease_seconds=lease_seconds,
            metadata={"kind": row["kind"], "operation_id": row["operation_id"],
                      "active_tool": row["active_tool"]},
        ):
            raise RuntimeError("cancellation attempt already has a live owner")
        payload = json_loads(row["payload_json"], {})
        payload["cancellation_phase"] = "attempting"
        changed = conn.execute(
            "UPDATE control_cancellations SET state='attempting',attempt_started_at=? "
            "WHERE control_id=? AND state='intent'",
            (stamp, control_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError("cancellation intent changed concurrently")
        conn.execute(
            "UPDATE task_controls SET payload_json=? WHERE id=? AND state='pending'",
            (json_dumps(payload), control_id),
        )
    return load(control_id)


def resolve_unattempted_cancellation(control_id, owner: Owner, *, outcome, error=""):
    """Resolve an owned intent after proving no cancellation callback was invoked."""
    stamp = now_utc()
    safe_error = str(error)[:256]
    with transaction(immediate=True) as conn:
        row = conn.execute(
            """SELECT c.state AS control_state,c.task_id,c.payload_json,
                      x.state AS cancellation_state
               FROM task_controls c JOIN control_cancellations x ON x.control_id=c.id
               WHERE c.id=?""", (control_id,),
        ).fetchone()
        lease = conn.execute(
            "SELECT owner_token,expires_at FROM resource_leases "
            "WHERE resource_type='control-cancellation' AND resource_key=?",
            (control_id,),
        ).fetchone()
        if (not row or row["control_state"] != "pending"
                or row["cancellation_state"] != "intent"
                or not lease or lease["owner_token"] != owner.token
                or lease["expires_at"] <= stamp):
            raise RuntimeError("unattempted cancellation resolution lost ownership")
        payload = json_loads(row["payload_json"], {})
        payload.update({
            "cancellation_phase": "resolved",
            "cancellation_requested": False,
            "cancellation_outcome": str(outcome)[:128],
        })
        if safe_error:
            payload["cancellation_error"] = safe_error
        else:
            payload.pop("cancellation_error", None)
        changed = conn.execute(
            """UPDATE control_cancellations SET state='resolved',result_json=?,error=?,
               reconciled_at=?
               WHERE control_id=? AND state='intent'""",
            (json_dumps({"requested": False}), safe_error, stamp, control_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError("cancellation intent changed concurrently")
        conn.execute(
            "UPDATE task_controls SET payload_json=? WHERE id=? AND state='pending'",
            (json_dumps(payload), control_id),
        )
        conn.execute(
            "DELETE FROM resource_leases WHERE resource_type='control-cancellation' "
            "AND resource_key=? AND owner_token=?", (control_id, owner.token),
        )
        _release_task_fence_if_quiescent(conn, row["task_id"], owner.token)
    return load(control_id)


def reconcile_cancellation(control_id, owner: Owner, *, result=None, error="",
                           ambiguous=False):
    stamp = now_utc()
    safe_error = str(error)[:256]
    with transaction(immediate=True) as conn:
        row = conn.execute(
            """SELECT c.state AS control_state,c.task_id,c.payload_json,
                      x.state AS cancellation_state
               FROM task_controls c JOIN control_cancellations x ON x.control_id=c.id
               WHERE c.id=?""", (control_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"unknown cancellation control: {control_id}")
        lease = conn.execute(
            "SELECT owner_token,expires_at FROM resource_leases "
            "WHERE resource_type='control-cancellation' AND resource_key=?",
            (control_id,),
        ).fetchone()
        if (row["control_state"] != "pending"
                or row["cancellation_state"] != "attempting"
                or not lease or lease["owner_token"] != owner.token
                or lease["expires_at"] <= stamp):
            raise RuntimeError("cancellation reconciliation lost ownership")
        final_state = "ambiguous" if ambiguous else "resolved"
        payload = json_loads(row["payload_json"], {})
        payload["cancellation_phase"] = final_state
        if ambiguous:
            payload["cancellation_requested"] = None
            payload["cancellation_outcome"] = "ambiguous"
        else:
            requested_value = dict.get(result, "requested") if isinstance(result, dict) else result
            requested = requested_value if type(requested_value) is bool else True
            payload["cancellation_requested"] = requested
            payload["cancellation_outcome"] = (
                "request-dispatched" if requested else "not-requested")
        result_truth = {"requested": payload["cancellation_requested"]}
        payload["cancellation_result"] = result_truth
        if safe_error:
            payload["cancellation_error"] = safe_error
        else:
            payload.pop("cancellation_error", None)
        changed = conn.execute(
            """UPDATE control_cancellations SET state=?,result_json=?,error=?,reconciled_at=?
               WHERE control_id=? AND state='attempting'""",
            (final_state, json_dumps(result_truth), safe_error, stamp, control_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError("cancellation attempt changed concurrently")
        conn.execute(
            "UPDATE task_controls SET payload_json=? WHERE id=? AND state='pending'",
            (json_dumps(payload), control_id),
        )
        conn.execute(
            "DELETE FROM resource_leases WHERE resource_type='control-cancellation' "
            "AND resource_key=? AND owner_token=?", (control_id, owner.token),
        )
        if ambiguous:
            _fail_task_for_ambiguous_cancellation(
                conn, row["task_id"], control_id,
                safe_error or "cancellation callback outcome is ambiguous",
            )
            conn.execute(
                "DELETE FROM resource_leases "
                "WHERE resource_type='task-execution-fence' AND resource_key=?",
                (row["task_id"],),
            )
        else:
            _release_task_fence_if_quiescent(conn, row["task_id"], owner.token)
    return load(control_id)


def abandon_cancellation(control_id, owner: Owner, *, error="shutdown-not-quiescent"):
    """Fence an in-flight local callback without claiming that it stopped."""
    safe_error = str(error)[:256]
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        row = conn.execute(
            """SELECT c.task_id,c.state AS control_state,c.payload_json,
                      x.state AS cancellation_state
               FROM task_controls c JOIN control_cancellations x ON x.control_id=c.id
               WHERE c.id=?""",
            (control_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"unknown cancellation control: {control_id}")
        if row["cancellation_state"] != "ambiguous":
            lease = conn.execute(
                "SELECT owner_token,owner_pid,owner_start FROM resource_leases "
                "WHERE resource_type='control-cancellation' AND resource_key=?",
                (control_id,),
            ).fetchone()
            if (row["control_state"] != "pending"
                    or row["cancellation_state"] != "attempting"
                    or not lease or lease["owner_token"] != owner.token
                    or lease["owner_pid"] != owner.pid
                    or lease["owner_start"] != owner.process_start
                    or not owner_alive(owner.pid, owner.process_start)):
                raise RuntimeError("cancellation cannot be abandoned by this owner")
            payload = json_loads(row["payload_json"], {})
            payload.update({
                "cancellation_phase": "ambiguous",
                "cancellation_requested": None,
                "cancellation_outcome": "ambiguous",
                "cancellation_result": {"requested": None},
                "cancellation_error": safe_error,
            })
            changed = conn.execute(
                """UPDATE control_cancellations SET state='ambiguous',result_json=?,error=?,
                   reconciled_at=? WHERE control_id=? AND state='attempting'""",
                (json_dumps({"requested": None}), safe_error, stamp, control_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("cancellation attempt changed concurrently")
            conn.execute(
                "UPDATE task_controls SET payload_json=? WHERE id=? AND state='pending'",
                (json_dumps(payload), control_id),
            )
            conn.execute(
                "DELETE FROM resource_leases WHERE resource_type='control-cancellation' "
                "AND resource_key=? AND owner_token=?",
                (control_id, owner.token),
            )
            _fail_task_for_ambiguous_cancellation(
                conn, row["task_id"], control_id, safe_error)
            conn.execute(
                "DELETE FROM resource_leases "
                "WHERE resource_type='task-execution-fence' AND resource_key=?",
                (row["task_id"],),
            )
    return load(control_id)


def _resolve_unattempted_in_transaction(conn, row, *, outcome, error=""):
    stamp = now_utc()
    payload = json_loads(row["payload_json"], {})
    payload.update({
        "cancellation_phase": "resolved",
        "cancellation_requested": False,
        "cancellation_outcome": str(outcome),
    })
    if error:
        payload["cancellation_error"] = str(error)
    changed = conn.execute(
        "UPDATE control_cancellations SET state='resolved',result_json=?,error=?,reconciled_at=? "
        "WHERE control_id=? AND state='intent'",
        (json_dumps({"requested": False}), str(error), stamp, row["id"]),
    ).rowcount
    if changed:
        conn.execute(
            "UPDATE task_controls SET payload_json=? WHERE id=? AND state='pending'",
            (json_dumps(payload), row["id"]),
        )
    return changed


def recover_cancellations(task_id, owner: Owner, *, lease_seconds=30.0):
    """Reconcile abandoned intent without ever replaying a cancellation callback."""
    recovered = 0
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        rows = conn.execute(
            """SELECT c.*,x.state AS cancellation_state
               FROM task_controls c JOIN control_cancellations x ON x.control_id=c.id
               WHERE c.task_id=? AND c.state='pending'
               AND x.state IN ('intent','attempting') ORDER BY c.priority,c.seq""",
            (task_id,),
        ).fetchall()
        for row in rows:
            existing = conn.execute(
                "SELECT owner_token,owner_pid,owner_start,expires_at FROM resource_leases "
                "WHERE resource_type='control-cancellation' AND resource_key=?",
                (row["id"],),
            ).fetchone()
            if (existing
                    and owner_alive(existing["owner_pid"], existing["owner_start"])):
                if (existing["expires_at"] > stamp
                        or row["cancellation_state"] == "intent"):
                    continue
                payload = json_loads(row["payload_json"], {})
                payload.update({
                    "cancellation_phase": "ambiguous",
                    "cancellation_requested": None,
                    "cancellation_outcome": "ambiguous",
                    "cancellation_result": {"requested": None},
                    "cancellation_error": (
                        "cancellation heartbeat expired after attempt began; "
                        "the live callback outcome is unknown"),
                })
                changed = conn.execute(
                    """UPDATE control_cancellations SET state='ambiguous',result_json=?,
                       error=?,reconciled_at=? WHERE control_id=? AND state='attempting'""",
                    (json_dumps({"requested": None}), payload["cancellation_error"],
                     stamp, row["id"]),
                ).rowcount
                if changed:
                    conn.execute(
                        "UPDATE task_controls SET payload_json=? "
                        "WHERE id=? AND state='pending'",
                        (json_dumps(payload), row["id"]),
                    )
                    _fail_task_for_ambiguous_cancellation(
                        conn, task_id, row["id"], payload["cancellation_error"])
                    recovered += 1
                    conn.execute(
                        "DELETE FROM resource_leases "
                        "WHERE resource_type='control-cancellation' AND resource_key=? "
                        "AND owner_token=? AND owner_pid=? AND owner_start=?",
                        (row["id"], existing["owner_token"], existing["owner_pid"],
                         existing["owner_start"]),
                    )
                    conn.execute(
                        "DELETE FROM resource_leases "
                        "WHERE resource_type='task-execution-fence' AND resource_key=?",
                        (task_id,),
                    )
                continue
            if not claim_in_transaction(
                conn, "control-cancellation", row["id"], owner,
                lease_seconds=lease_seconds,
                metadata={"recovery": True, "task_id": task_id},
            ):
                continue
            if row["cancellation_state"] == "intent":
                recovered += _resolve_unattempted_in_transaction(
                    conn, row, outcome="recovered-before-attempt")
                conn.execute(
                    "DELETE FROM resource_leases WHERE resource_type='control-cancellation' "
                    "AND resource_key=? AND owner_token=?", (row["id"], owner.token),
                )
                _release_task_fence_if_quiescent(conn, task_id)
                continue
            payload = json_loads(row["payload_json"], {})
            payload.update({
                "cancellation_phase": "ambiguous",
                "cancellation_requested": None,
                "cancellation_outcome": "ambiguous",
                "cancellation_result": {"requested": None},
                "cancellation_error": (
                    "cancellation owner was lost after attempt began; outcome is unknown"),
            })
            changed = conn.execute(
                """UPDATE control_cancellations SET state='ambiguous',result_json=?,error=?,
                   reconciled_at=?
                   WHERE control_id=? AND state='attempting'""",
                (json_dumps({"requested": None}), payload["cancellation_error"],
                 stamp, row["id"]),
            ).rowcount
            if changed:
                conn.execute(
                    "UPDATE task_controls SET payload_json=? WHERE id=? AND state='pending'",
                    (json_dumps(payload), row["id"]),
                )
                _fail_task_for_ambiguous_cancellation(
                    conn, task_id, row["id"], payload["cancellation_error"])
                recovered += 1
            conn.execute(
                "DELETE FROM resource_leases WHERE resource_type='control-cancellation' "
                "AND resource_key=? AND owner_token=?", (row["id"], owner.token),
            )
            conn.execute(
                "DELETE FROM resource_leases "
                "WHERE resource_type='task-execution-fence' AND resource_key=?",
                (task_id,),
            )
    return recovered


def unreconciled_cancellation(task_id):
    ensure_state_store()
    with connect() as conn:
        row = conn.execute(
            """SELECT c.* FROM task_controls c
               JOIN control_cancellations x ON x.control_id=c.id
               WHERE c.task_id=? AND c.state='pending'
               AND x.state IN ('intent','attempting')
               ORDER BY c.priority,c.seq LIMIT 1""", (task_id,),
        ).fetchone()
    return _from_row(row) if row else None


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


def claim_next(task_id, owner: Owner, *, lease_seconds=30.0):
    ensure_state_store()
    with transaction(immediate=True) as conn:
        row = conn.execute(
            "SELECT * FROM task_controls WHERE task_id=? AND state='pending' "
            "ORDER BY priority,seq LIMIT 1", (task_id,),
        ).fetchone()
        if not row:
            return None
        cancellation = conn.execute(
            "SELECT state FROM control_cancellations WHERE control_id=?", (row["id"],),
        ).fetchone()
        if row["kind"] in {"interrupt", "cancel"} and cancellation is None:
            return None
        if cancellation and cancellation["state"] not in CANCELLATION_READY:
            return None
        if not claim_in_transaction(
            conn, "task-control", row["id"], owner, lease_seconds=lease_seconds,
            metadata={"task_id": task_id},
        ):
            return None
        changed = conn.execute(
            "UPDATE task_controls SET state='processing' WHERE id=? AND state='pending'",
            (row["id"],),
        ).rowcount
        if changed != 1:
            conn.execute(
                "DELETE FROM resource_leases WHERE resource_type='task-control' "
                "AND resource_key=? AND owner_token=?", (row["id"], owner.token),
            )
            return None
        row = conn.execute("SELECT * FROM task_controls WHERE id=?", (row["id"],)).fetchone()
        return _from_row(row)


def finish(control_id, owner: Owner, *, success=True, payload=None):
    state = "applied" if success else "failed"
    stamp = now_utc()
    if any(str(key).startswith("cancellation_") for key in (payload or {})):
        raise ValueError("cancellation truth may only be written by reconciliation")
    with transaction(immediate=True) as conn:
        row = conn.execute("SELECT * FROM task_controls WHERE id=?", (control_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown control: {control_id}")
        current = _from_row(row)
        if current.state != "processing":
            raise RuntimeError(f"control {control_id} is already {current.state}")
        cancellation = conn.execute(
            "SELECT state FROM control_cancellations WHERE control_id=?", (control_id,),
        ).fetchone()
        if current.kind in {"interrupt", "cancel"} and cancellation is None:
            raise RuntimeError("cancellation control has no durable outcome state")
        if cancellation and cancellation["state"] not in CANCELLATION_READY:
            raise RuntimeError("cancellation outcome is not reconciled")
        lease = conn.execute(
            "SELECT owner_token,expires_at FROM resource_leases "
            "WHERE resource_type='task-control' AND resource_key=?", (control_id,),
        ).fetchone()
        if not lease or lease["owner_token"] != owner.token or lease["expires_at"] <= stamp:
            raise RuntimeError(f"control {control_id} is not owned by this processor")
        merged = current.payload | (payload or {})
        changed = conn.execute(
            "UPDATE task_controls SET state=?,payload_json=?,applied_at=? WHERE id=? "
            "AND state='processing'",
            (state, json_dumps(merged), stamp, control_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError(f"control {control_id} changed concurrently")
        conn.execute(
            "DELETE FROM resource_leases WHERE resource_type='task-control' "
            "AND resource_key=? AND owner_token=?", (control_id, owner.token),
        )
        event_type = (
            "interrupt" if current.kind == "interrupt" else current.kind
            if current.kind in {"cancel", "redirect", "pause", "resume"}
            else "user_control")
        insert_state_event(
            conn, event_type, current.message, session_id=current.session_id,
            task_id=current.task_id,
            payload={"control_id": current.id, "seq": current.seq,
                     "kind": current.kind, "state": state},
        )
        row = conn.execute(
            "SELECT * FROM task_controls WHERE id=?", (control_id,),
        ).fetchone()
    return _from_row(row)


def annotate(control_id, payload, *, owner: Owner | None = None):
    if any(str(key).startswith("cancellation_") for key in payload):
        raise ValueError("cancellation truth may only be written by reconciliation")
    with transaction(immediate=True) as conn:
        row = conn.execute("SELECT * FROM task_controls WHERE id=?", (control_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown control: {control_id}")
        current = _from_row(row)
        if current.state not in {"pending", "processing"}:
            return current
        if current.state == "processing":
            lease = conn.execute(
                "SELECT owner_token,expires_at FROM resource_leases "
                "WHERE resource_type='task-control' AND resource_key=?", (control_id,),
            ).fetchone()
            if (owner is None or not lease or lease["owner_token"] != owner.token
                    or lease["expires_at"] <= now_utc()):
                raise RuntimeError(f"control {control_id} is owned by another processor")
        conn.execute("UPDATE task_controls SET payload_json=? WHERE id=? "
                     "AND state IN ('pending','processing')",
                     (json_dumps(current.payload | dict(payload)), control_id))
    return load(control_id)


def recover_processing(task_id, owner: Owner, *, lease_seconds=30.0):
    """Return only controls whose previous durable owner is provably gone."""
    recovered = 0
    with transaction(immediate=True) as conn:
        rows = conn.execute(
            "SELECT * FROM task_controls WHERE task_id=? AND state='processing' ORDER BY seq",
            (task_id,),
        ).fetchall()
        for row in rows:
            if not claim_in_transaction(
                conn, "task-control", row["id"], owner, lease_seconds=lease_seconds,
                metadata={"task_id": task_id, "recovery": True},
            ):
                continue
            payload = json_loads(row["payload_json"], {})
            payload["recovered_after_disconnect"] = True
            changed = conn.execute(
                "UPDATE task_controls SET state='pending',payload_json=? "
                "WHERE id=? AND state='processing'",
                (json_dumps(payload), row["id"]),
            ).rowcount
            conn.execute(
                "DELETE FROM resource_leases WHERE resource_type='task-control' "
                "AND resource_key=? AND owner_token=?", (row["id"], owner.token),
            )
            recovered += changed
    return recovered


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
