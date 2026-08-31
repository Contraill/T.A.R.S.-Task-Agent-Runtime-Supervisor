from __future__ import annotations

from dataclasses import dataclass
import uuid

from .checkpoints import (_from_row as _checkpoint_from_row,
                          create_checkpoint_in_transaction, verify_checkpoint)
from .events import append_event
from .roles import default_role_id, get_role, list_roles, resolve_role_id
from .runtime_backends import backend_binding_ready
from .registry import get_model
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction
from .tasks import (canonical_task_state, create_task_in_transaction, load_task,
                    update_task)

DELEGATION_STATES = {"requested", "running", "completed", "failed", "cancelled"}
DELEGATION_RESULT_STATUSES = {"success", "partial", "failed", "cancelled"}


@dataclass(frozen=True)
class RoutingDecision:
    id: str
    requested_capabilities: tuple[str, ...]
    selected_role: str
    candidates: tuple[dict, ...]
    reason: str
    created_at: str
    task_id: str | None = None


@dataclass(frozen=True)
class DelegationRecord:
    id: str
    parent_task_id: str
    child_task_id: str
    requested_role: str
    state: str
    goal: str
    scope: dict
    constraints: tuple[str, ...]
    permissions: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    expected_result: str
    result_status: str | None
    result_summary: str
    result: dict
    created_at: str
    updated_at: str
    completed_at: str | None


@dataclass(frozen=True)
class HandoffRecord:
    id: str
    task_id: str
    from_role: str
    to_role: str
    checkpoint_id: str
    reason: str
    created_at: str


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _require_available_role(role_name: str):
    role = get_role(role_name)
    if not role.enabled:
        raise ValueError(f"role {role.id!r} is disabled")
    if not role.model:
        raise ValueError(f"role {role.id!r} has no model binding")
    return role


def _routing_candidates(required_capabilities) -> list[dict]:
    required = frozenset(str(x).strip() for x in required_capabilities if str(x).strip())
    candidates = []
    for role in list_roles(include_disabled=False):
        if not role.model:
            continue
        if not backend_binding_ready(get_model(role.model)):
            continue
        available = frozenset(role.capabilities)
        missing = sorted(required - available)
        if missing:
            continue
        extra = len(available - required)
        candidates.append({
            "role_id": role.id,
            "required_covered": len(required),
            "extra_capabilities": extra,
            "capabilities": sorted(available),
        })
    return candidates


def route_for_capabilities(required_capabilities=(), *, task_id: str | None = None, persist: bool = True) -> RoutingDecision:
    """Choose an available Role without encoding role names into routing logic.

    Candidates must cover every requested capability, be enabled, and have a model
    binding.  Among valid candidates the narrowest capability surface wins; the
    configured default Role wins a semantic tie.  Role id is only a final stable
    ordering key, never a capability signal.
    """
    if persist or task_id is not None:
        ensure_state_store()
    if task_id is not None:
        load_task(task_id)
    required = tuple(dict.fromkeys(str(x).strip() for x in required_capabilities if str(x).strip()))
    candidates = _routing_candidates(required)
    if not candidates:
        joined = ", ".join(required) or "<none>"
        raise LookupError(f"no enabled bound role covers required capabilities: {joined}")

    default_id = default_role_id()
    candidates.sort(key=lambda x: (
        x["extra_capabilities"],
        0 if x["role_id"] == default_id else 1,
        x["role_id"],
    ))
    selected = candidates[0]["role_id"]
    reason = (
        "all requested capabilities covered; selected narrowest available role"
        + ("; default-role tie-break" if selected == default_id else "")
    )
    decision = RoutingDecision(
        id=_new_id("route"),
        requested_capabilities=required,
        selected_role=selected,
        candidates=tuple(candidates),
        reason=reason,
        created_at=now_utc(),
        task_id=task_id,
    )
    if persist:
        with transaction(immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO routing_decisions(
                    id,task_id,requested_capabilities_json,selected_role,
                    candidates_json,reason,created_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    decision.id, task_id, json_dumps(list(required)), selected,
                    json_dumps(candidates), reason, decision.created_at,
                ),
            )
        if task_id:
            append_event(
                task_id, "routing", f"AUTO selected {selected}", role=selected,
                data={"routing_id": decision.id, "required_capabilities": list(required)},
                visibility="verbose",
            )
    return decision


def _delegation_from_row(row) -> DelegationRecord:
    return DelegationRecord(
        id=row["id"], parent_task_id=row["parent_task_id"], child_task_id=row["child_task_id"],
        requested_role=row["requested_role"], state=row["state"], goal=row["goal"],
        scope=json_loads(row["scope_json"], {}),
        constraints=tuple(json_loads(row["constraints_json"], [])),
        permissions=tuple(json_loads(row["permissions_json"], [])),
        evidence_refs=tuple(json_loads(row["evidence_refs_json"], [])),
        expected_result=row["expected_result"], result_status=row["result_status"],
        result_summary=row["result_summary"], result=json_loads(row["result_json"], {}),
        created_at=row["created_at"], updated_at=row["updated_at"], completed_at=row["completed_at"],
    )


def create_delegation(
    parent_task_id: str,
    goal: str,
    *,
    role: str | None = None,
    required_capabilities=(),
    scope: dict | None = None,
    constraints=(),
    permissions=(),
    evidence_refs=(),
    expected_result: str = "",
    _contract_values: tuple | None = None,
) -> DelegationRecord:
    """Create a bounded child task while preserving parent ownership."""
    ensure_state_store()
    parent = load_task(parent_task_id)
    if parent.state in {"completed", "cancelled"}:
        raise RuntimeError(f"cannot delegate from {parent.state} task {parent.id}")

    if role is None:
        decision = route_for_capabilities(required_capabilities, task_id=parent.id)
        target = _require_available_role(decision.selected_role)
    else:
        target = _require_available_role(resolve_role_id(role))
        required = set(str(x) for x in required_capabilities)
        missing = required - set(target.capabilities)
        if missing:
            raise ValueError(
                f"role {target.id!r} lacks required capabilities: {', '.join(sorted(missing))}"
            )

    now = now_utc()
    delegation_id = _new_id("dlg")
    with transaction(immediate=True) as conn:
        current_parent = conn.execute(
            "SELECT * FROM tasks WHERE id=?", (parent.id,)).fetchone()
        if not current_parent:
            raise KeyError(f"unknown task: {parent.id}")
        if current_parent["state"] in {"completed", "cancelled"}:
            raise RuntimeError(
                f"cannot delegate from {current_parent['state']} task {parent.id}")
        child_id = create_task_in_transaction(
            conn, goal, target.id, kind="delegation", parent_task_id=parent.id,
            source="orchestration", conversation_id=current_parent["conversation_id"],
            title=f"Delegation from {current_parent['title'] or parent.id}",
            constraints=constraints, evidence_refs=evidence_refs, phase="delegated")
        conn.execute(
            """
            INSERT INTO delegations(
                id,parent_task_id,child_task_id,requested_role,state,goal,scope_json,
                constraints_json,permissions_json,evidence_refs_json,expected_result,
                result_status,result_summary,result_json,created_at,updated_at,completed_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                delegation_id, parent.id, child_id, target.id, "requested", goal,
                json_dumps(scope or {}), json_dumps(list(constraints)),
                json_dumps(list(permissions)), json_dumps(list(evidence_refs)),
                expected_result, None, "", "{}", now, now, None,
            ),
        )
        if _contract_values is not None:
            conn.execute(
                "INSERT INTO delegation_contracts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (delegation_id, *_contract_values))
        row = conn.execute("SELECT * FROM delegations WHERE id=?", (delegation_id,)).fetchone()

    append_event(
        child_id, "status", f"Task created with owner {target.id}", role=target.id,
        data={"state": "pending", "kind": "delegation"}, visibility="verbose",
    )
    append_event(
        parent.id, "delegation", f"Delegated to {target.display_name}: {goal}",
        role=parent.owner_role,
        data={
            "delegation_id": delegation_id,
            "child_task_id": child_id,
            "requested_role": target.id,
            "parent_owner_unchanged": parent.owner_role,
        },
        visibility="normal",
    )
    append_event(
        child_id, "delegation", f"Delegation received from {parent.id}", role=target.id,
        data={"delegation_id": delegation_id, "parent_task_id": parent.id},
        visibility="normal",
    )
    return _delegation_from_row(row)


def load_delegation(delegation_id: str) -> DelegationRecord:
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM delegations WHERE id=?", (delegation_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown delegation: {delegation_id}")
        return _delegation_from_row(row)
    finally:
        conn.close()


def list_delegations(parent_task_id: str | None = None, *, limit: int = 50) -> list[DelegationRecord]:
    ensure_state_store()
    conn = connect()
    try:
        if parent_task_id:
            rows = conn.execute(
                "SELECT * FROM delegations WHERE parent_task_id=? ORDER BY created_at DESC LIMIT ?",
                (parent_task_id, int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM delegations ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [_delegation_from_row(row) for row in rows]
    finally:
        conn.close()


def complete_delegation(
    delegation_id: str,
    *,
    status: str,
    summary: str,
    result: dict | None = None,
) -> DelegationRecord:
    status = status.strip().lower()
    if status not in DELEGATION_RESULT_STATUSES:
        raise ValueError("delegation result status must be success, partial, failed or cancelled")
    state = "completed" if status in {"success", "partial"} else status
    child_state = "completed" if status in {"success", "partial"} else status
    now = now_utc()
    with transaction(immediate=True) as conn:
        current = conn.execute(
            "SELECT * FROM delegations WHERE id=?", (delegation_id,)).fetchone()
        if not current:
            raise KeyError(f"unknown delegation: {delegation_id}")
        if current["state"] in {"completed", "failed", "cancelled"}:
            raise RuntimeError(
                f"delegation {delegation_id} is already {current['state']}")
        changed = conn.execute(
            """
            UPDATE delegations
            SET state=?,result_status=?,result_summary=?,result_json=?,updated_at=?,completed_at=?
            WHERE id=? AND state NOT IN ('completed','failed','cancelled')
            """,
            (state, status, summary, json_dumps(result or {}), now, now, delegation_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError(f"delegation {delegation_id} changed concurrently")
        conn.execute(
            """UPDATE tasks SET state=?,phase='delegation-complete',
               progress=CASE WHEN ?='completed' THEN 1.0 ELSE progress END,updated_at=?
               WHERE id=?""",
            (child_state, child_state, now, current["child_task_id"]),)
        delegation = _delegation_from_row(conn.execute(
            "SELECT * FROM delegations WHERE id=?", (delegation_id,)).fetchone())
    append_event(
        delegation.parent_task_id, "delegation", f"Delegation result ({status}): {summary}",
        role=delegation.requested_role,
        data={
            "delegation_id": delegation.id,
            "child_task_id": delegation.child_task_id,
            "status": status,
            "result": result or {},
        },
        visibility="normal" if status in {"success", "partial"} else "quiet",
    )
    return load_delegation(delegation.id)


def delegation_envelope(delegation_id: str) -> dict:
    d = load_delegation(delegation_id)
    return {
        "request": {
            "delegation_id": d.id,
            "parent_task_id": d.parent_task_id,
            "child_task_id": d.child_task_id,
            "requested_role": d.requested_role,
            "goal": d.goal,
            "scope": d.scope,
            "constraints": list(d.constraints),
            "permissions": list(d.permissions),
            "evidence_refs": list(d.evidence_refs),
            "expected_result": d.expected_result,
        },
        "result": {
            "state": d.state,
            "status": d.result_status,
            "summary": d.result_summary,
            "data": d.result,
        },
    }


def _handoff_from_row(row) -> HandoffRecord:
    return HandoffRecord(
        id=row["id"], task_id=row["task_id"], from_role=row["from_role"],
        to_role=row["to_role"], checkpoint_id=row["checkpoint_id"],
        reason=row["reason"], created_at=row["created_at"],
    )


def handoff_task(task_id: str, to_role: str, *, reason: str = "manual handoff") -> HandoffRecord:
    """Transfer task ownership only after a verified immutable checkpoint exists."""
    ensure_state_store()
    task = load_task(task_id)
    target = _require_available_role(resolve_role_id(to_role))
    if task.owner_role == target.id:
        raise ValueError(f"task {task.id} is already owned by {target.id}")
    if task.state in {"completed", "cancelled"}:
        raise RuntimeError(f"cannot hand off {task.state} task {task.id}")

    handoff_id = _new_id("handoff")
    now = now_utc()
    with transaction(immediate=True) as conn:
        current = conn.execute("SELECT * FROM tasks WHERE id=?", (task.id,)).fetchone()
        if not current:
            raise KeyError(f"unknown task: {task.id}")
        if current["owner_role"] != task.owner_role:
            raise RuntimeError("task owner changed while preparing handoff")
        if current["state"] in {"completed", "cancelled"}:
            raise RuntimeError(f"cannot hand off {current['state']} task {task.id}")
        active = conn.execute(
            "SELECT id,state FROM task_runs WHERE task_id=? "
            "AND state IN ('queued','running','paused') LIMIT 1", (task.id,)).fetchone()
        if active:
            raise RuntimeError(
                f"task {task.id} has active run {active['id']} ({active['state']}); "
                "finish or cancel it before handoff")
        checkpoint_row = create_checkpoint_in_transaction(
            conn, task.id,
            reason=f"pre-handoff {task.owner_role} -> {target.id}: {reason}")
        checkpoint = _checkpoint_from_row(checkpoint_row)
        conn.execute(
            """
            INSERT INTO handoffs(id,task_id,from_role,to_role,checkpoint_id,reason,created_at)
            VALUES(?,?,?,?,?,?,?)
            """,
            (handoff_id, task.id, task.owner_role, target.id, checkpoint.id, reason, now),
        )
        conn.execute(
            "UPDATE tasks SET owner_role=?,phase='handoff-complete',epoch=epoch+1,updated_at=? WHERE id=?",
            (target.id, now, task.id),
        )
        row = conn.execute("SELECT * FROM handoffs WHERE id=?", (handoff_id,)).fetchone()

    append_event(
        task.id, "handoff", f"Ownership handed off {task.owner_role} -> {target.id}",
        role=target.id,
        data={
            "handoff_id": handoff_id,
            "from_role": task.owner_role,
            "to_role": target.id,
            "checkpoint_id": checkpoint.id,
            "reason": reason,
        },
        visibility="normal",
    )
    return _handoff_from_row(row)


def list_handoffs(task_id: str | None = None, *, limit: int = 50) -> list[HandoffRecord]:
    ensure_state_store()
    conn = connect()
    try:
        if task_id:
            rows = conn.execute(
                "SELECT * FROM handoffs WHERE task_id=? ORDER BY created_at DESC LIMIT ?",
                (task_id, int(limit)),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM handoffs ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [_handoff_from_row(row) for row in rows]
    finally:
        conn.close()


def task_orchestration_state(task_id: str) -> dict:
    state = canonical_task_state(task_id)
    state["delegations"] = [delegation_envelope(d.id) for d in list_delegations(task_id)]
    state["handoffs"] = [
        {
            "id": h.id,
            "from_role": h.from_role,
            "to_role": h.to_role,
            "checkpoint_id": h.checkpoint_id,
            "reason": h.reason,
            "created_at": h.created_at,
        }
        for h in list_handoffs(task_id)
    ]
    return state


__all__ = [
    "RoutingDecision", "DelegationRecord", "HandoffRecord",
    "route_for_capabilities", "create_delegation", "load_delegation",
    "list_delegations", "complete_delegation", "delegation_envelope",
    "handoff_task", "list_handoffs", "task_orchestration_state",
]
