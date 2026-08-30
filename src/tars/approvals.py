from __future__ import annotations

from dataclasses import dataclass
import uuid

from .policy import PolicyDecision, ScopeRequest, add_rule, redact
from .state_events import insert_state_event
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction

APPROVAL_SCOPES = {"call", "task", "session", "target", "persistent"}


@dataclass(frozen=True)
class Approval:
    id: str
    state: str
    risk_class: str
    tool: str
    target: str
    request: dict
    scope: str
    task_id: str | None
    session_id: str | None
    decision_reason: str
    created_at: str
    decided_at: str | None
    expires_at: str | None
    consumed_at: str | None


def _from_row(row):
    return Approval(
        id=row["id"], state=row["state"], risk_class=row["risk_class"],
        tool=row["tool"], target=row["target"], request=json_loads(row["request_json"], {}),
        scope=row["scope"], task_id=row["task_id"], session_id=row["session_id"],
        decision_reason=row["decision_reason"], created_at=row["created_at"],
        decided_at=row["decided_at"], expires_at=row["expires_at"],
        consumed_at=row["consumed_at"],
    )


class ApprovalBroker:
    def request(self, request: ScopeRequest, decision: PolicyDecision, *, scope="call",
                expires_at=None) -> Approval:
        if decision.action != "ask":
            raise ValueError("approvals may only be requested for ask decisions")
        if scope not in APPROVAL_SCOPES:
            raise ValueError(f"invalid approval scope: {scope}")
        if scope == "task" and not request.task_id:
            raise ValueError("task-scoped approval requires a task")
        if scope == "session" and not request.session_id:
            raise ValueError("session-scoped approval requires a session")
        approval_id = "approval-" + uuid.uuid4().hex
        ensure_state_store()
        payload = {
            "effect": decision.effect,
            "arguments": decision.normalized_arguments,
            "policy_reason": decision.reason,
        }
        with transaction(immediate=True) as conn:
            conn.execute(
                """INSERT INTO approvals(id,state,risk_class,tool,target,request_json,scope,
                   task_id,session_id,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (approval_id, "pending", decision.risk_class, request.tool, decision.target,
                 json_dumps(redact(payload)), scope, request.task_id, request.session_id,
                 now_utc(), expires_at),
            )
            insert_state_event(
                conn, "approval", f"{request.tool}: approval requested",
                task_id=request.task_id, session_id=request.session_id,
                payload={"approval_id": approval_id, "state": "pending", "scope": scope,
                         "risk_class": decision.risk_class, "target": decision.target},
            )
        return self.load(approval_id)

    def decide(self, approval_id: str, *, approve: bool, reason="") -> Approval:
        approval = self.load(approval_id)
        if approval.state != "pending":
            raise RuntimeError(f"approval is already {approval.state}")
        state = "approved" if approve else "denied"
        with transaction(immediate=True) as conn:
            conn.execute(
                "UPDATE approvals SET state=?,decision_reason=?,decided_at=? WHERE id=?",
                (state, reason, now_utc(), approval_id),
            )
            insert_state_event(
                conn, "approval", f"{approval.tool}: {state}",
                task_id=approval.task_id, session_id=approval.session_id,
                payload={"approval_id": approval.id, "state": state,
                         "scope": approval.scope, "reason": reason},
            )
        decided = self.load(approval_id)
        if approve and decided.scope == "persistent":
            add_rule(decided.request["effect"], "allow", target=decided.target,
                     target_kind="path" if decided.tool.startswith("fs.") else None,
                     metadata={"approval_id": decided.id, "reason": reason})
        return decided

    def authorize(self, request: ScopeRequest, decision: PolicyDecision,
                  approval_id: str | None = None, *, consume=True) -> str | None:
        if decision.action == "deny":
            raise PermissionError(decision.reason)
        if decision.action == "allow":
            return None
        approval = self.load(approval_id) if approval_id else self._find(request, decision)
        if not approval or approval.state != "approved":
            raise PermissionError("approved authorization is required")
        if not self._matches(approval, request, decision):
            raise PermissionError("approval does not match this action")
        if approval.scope == "call" and consume:
            with transaction(immediate=True) as conn:
                changed = conn.execute(
                    "UPDATE approvals SET state='consumed',consumed_at=? WHERE id=? AND state='approved'",
                    (now_utc(), approval.id),
                ).rowcount
            if changed != 1:
                raise PermissionError("one-call approval was already consumed")
        return approval.id

    def load(self, approval_id):
        ensure_state_store()
        conn = connect()
        try:
            row = conn.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
            if not row:
                raise KeyError(f"unknown approval: {approval_id}")
            return _from_row(row)
        finally:
            conn.close()

    def list(self, *, state=None, limit=50):
        ensure_state_store()
        conn = connect()
        try:
            sql = "SELECT * FROM approvals"
            params = []
            if state:
                sql += " WHERE state=?"
                params.append(state)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(int(limit))
            return [_from_row(row) for row in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    def _find(self, request, decision):
        for approval in self.list(state="approved"):
            if self._matches(approval, request, decision):
                return approval
        return None

    @staticmethod
    def _matches(approval, request, decision):
        if approval.tool != request.tool or approval.target != decision.target:
            return False
        if approval.request.get("effect") != decision.effect:
            return False
        if approval.expires_at and approval.expires_at <= now_utc():
            return False
        if approval.scope == "call" and approval.request.get("arguments") != decision.normalized_arguments:
            return False
        if approval.scope == "task":
            return approval.task_id == request.task_id
        if approval.scope == "session":
            return approval.session_id == request.session_id
        return True
