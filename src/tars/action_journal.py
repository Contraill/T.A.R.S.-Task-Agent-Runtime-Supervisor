from __future__ import annotations

from dataclasses import dataclass
import uuid

from .approvals import ApprovalBroker
from .policy import PolicyDecision, ScopeRequest, redact
from .state_events import insert_state_event
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction

FINAL_STATES = {"succeeded", "failed", "denied", "cancelled", "unknown"}


@dataclass(frozen=True)
class ActionRecord:
    id: str
    task_id: str | None
    session_id: str | None
    event_uuid: str
    tool: str
    normalized_arguments: dict
    target: str
    effect: str
    risk_class: str
    policy_action: str
    policy_reason: str
    approval_id: str | None
    state: str
    result: dict | None
    created_at: str
    started_at: str | None
    completed_at: str | None


def _from_row(row):
    return ActionRecord(
        id=row["id"], task_id=row["task_id"], session_id=row["session_id"],
        event_uuid=row["event_uuid"], tool=row["tool"],
        normalized_arguments=json_loads(row["normalized_arguments_json"], {}),
        target=row["target"], effect=row["effect"], risk_class=row["risk_class"],
        policy_action=row["policy_action"], policy_reason=row["policy_reason"],
        approval_id=row["approval_id"], state=row["state"],
        result=json_loads(row["result_json"], None), created_at=row["created_at"],
        started_at=row["started_at"], completed_at=row["completed_at"],
    )


def record_denied(request: ScopeRequest, decision: PolicyDecision):
    return _create(request, decision, state="denied", result={"error": decision.reason})


def begin_action(request: ScopeRequest, decision: PolicyDecision, *, approval_id=None,
                 broker=None):
    broker = broker or ApprovalBroker()
    if decision.action == "deny":
        record_denied(request, decision)
        raise PermissionError(decision.reason)
    try:
        authorization = broker.authorize(request, decision, approval_id, consume=False)
    except PermissionError as exc:
        _create(request, decision, state="denied", result={"error": str(exc)})
        raise
    return _create(request, decision, approval_id=authorization, state="running")


def _create(request, decision, *, approval_id=None, state, result=None):
    ensure_state_store()
    action_id = "action-" + uuid.uuid4().hex
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        if approval_id:
            approval = conn.execute(
                "SELECT scope,state FROM approvals WHERE id=?", (approval_id,)
            ).fetchone()
            if approval and approval["scope"] == "call":
                changed = conn.execute(
                    "UPDATE approvals SET state='consumed',consumed_at=? "
                    "WHERE id=? AND state='approved'",
                    (stamp, approval_id),
                ).rowcount
                if changed != 1:
                    raise PermissionError("one-call approval was already consumed")
        event = insert_state_event(
            conn, "tool_execution" if state == "running" else "policy_decision",
            f"{request.tool}: {state}", session_id=request.session_id,
            task_id=request.task_id, payload={
                "action_id": action_id, "tool": request.tool, "target": decision.target,
                "effect": decision.effect, "risk_class": decision.risk_class,
                "policy_action": decision.action, "approval_id": approval_id,
            },
        )
        conn.execute(
            """INSERT INTO action_journal(id,task_id,session_id,event_uuid,tool,
               normalized_arguments_json,target,effect,risk_class,policy_action,policy_reason,
               approval_id,state,result_json,created_at,started_at,completed_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (action_id, request.task_id, request.session_id, event.event_uuid, request.tool,
             json_dumps(redact(decision.normalized_arguments)), decision.target,
             decision.effect, decision.risk_class, decision.action, decision.reason,
             approval_id, state, json_dumps(redact(result)) if result is not None else None,
             stamp, stamp if state == "running" else None,
             stamp if state in FINAL_STATES else None),
        )
    return load_action(action_id)


def finish_action(action_id, *, state, result):
    if state not in FINAL_STATES - {"denied"}:
        raise ValueError(f"invalid terminal action state: {state}")
    record = load_action(action_id)
    if record.state != "running":
        raise RuntimeError(f"action is not running: {record.state}")
    safe_result = redact(result)
    with transaction(immediate=True) as conn:
        changed = conn.execute(
            """UPDATE action_journal SET state=?,result_json=?,completed_at=?
               WHERE id=? AND state='running'""",
            (state, json_dumps(safe_result), now_utc(), action_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError("action state changed concurrently")
        insert_state_event(
            conn, "tool_result", f"{record.tool}: {state}",
            session_id=record.session_id, task_id=record.task_id,
            payload={"action_id": action_id, "state": state, "result": safe_result},
        )
    return load_action(action_id)


def load_action(action_id):
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM action_journal WHERE id=?", (action_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown action: {action_id}")
        return _from_row(row)
    finally:
        conn.close()


def list_actions(*, task_id=None, state=None, limit=50):
    ensure_state_store()
    clauses = []
    params = []
    if task_id:
        clauses.append("task_id=?")
        params.append(task_id)
    if state:
        clauses.append("state=?")
        params.append(state)
    sql = "SELECT * FROM action_journal"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(int(limit))
    conn = connect()
    try:
        return [_from_row(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()
