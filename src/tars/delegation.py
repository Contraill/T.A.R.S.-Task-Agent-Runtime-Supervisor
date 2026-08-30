from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import threading

from . import memory
from .evidence import load as load_evidence
from .events import append_event
from .orchestration import (complete_delegation, create_delegation,
                            load_delegation)
from .policy import canonical_path
from .state_store import (connect, ensure_state_store, json_dumps, json_loads,
                          now_utc, transaction)
from .tasks import load_task, update_task


TERMINAL_STATES = {"completed", "failed", "cancelled", "timed_out", "accepted", "rejected"}
_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="tars-child")
_GPU_SLOT = threading.BoundedSemaphore(1)
_LOCK = threading.RLock()
_RUNS: dict[str, tuple[Future, threading.Event]] = {}
_WORKSPACE_LOCKS: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class DelegationContract:
    delegation_id: str
    parent_delegation_id: str | None
    tool_allowlist: tuple[str, ...]
    authority: dict
    budget: dict
    workspace: dict
    completion: dict
    state: str
    accepted: bool | None
    acceptance_reason: str
    started_at: str | None
    deadline_at: str | None
    finished_at: str | None


def _contract(row):
    return DelegationContract(
        row["delegation_id"], row["parent_delegation_id"],
        tuple(json_loads(row["tool_allowlist_json"], [])),
        json_loads(row["authority_json"], {}), json_loads(row["budget_json"], {}),
        json_loads(row["workspace_json"], {}), json_loads(row["completion_json"], {}),
        row["state"], None if row["accepted"] is None else bool(row["accepted"]),
        row["acceptance_reason"], row["started_at"], row["deadline_at"],
        row["finished_at"],
    )


def load_contract(delegation_id):
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM delegation_contracts WHERE delegation_id=?",
                           (delegation_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown delegation contract: {delegation_id}")
        return _contract(row)
    finally:
        conn.close()


def _paths(values):
    return tuple(dict.fromkeys(canonical_path(value) for value in values))


def _path_subset(child, parent):
    return all(any(value == root or value.startswith(root.rstrip("/") + "/")
                   for root in parent) for value in child)


def _normalize_authority(value):
    value = dict(value or {})
    return {
        "paths": list(_paths(value.get("paths", ()))),
        "effects": sorted(set(map(str, value.get("effects", ())))),
        "network_targets": sorted(set(map(str, value.get("network_targets", ())))),
        "remote_targets": sorted(set(map(str, value.get("remote_targets", ())))),
        "secret_refs": sorted(set(map(str, value.get("secret_refs", ())))),
    }


def _require_subset(child, parent, child_tools, parent_tools):
    if not set(child_tools).issubset(parent_tools):
        raise PermissionError("child tool allowlist exceeds parent authority")
    if not _path_subset(child["paths"], parent["paths"]):
        raise PermissionError("child filesystem scope exceeds parent authority")
    for field in ("effects", "network_targets", "remote_targets", "secret_refs"):
        if not set(child[field]).issubset(parent[field]):
            raise PermissionError(f"child {field} exceeds parent authority")


def authorize_child_action(delegation_id, tool, effect, target=""):
    """Apply the immutable child ceiling before the normal tool policy path."""
    contract = load_contract(delegation_id)
    if tool not in contract.tool_allowlist:
        raise PermissionError(f"tool is outside child allowlist: {tool}")
    if effect not in contract.authority["effects"]:
        raise PermissionError(f"effect is outside child authority: {effect}")
    if effect in {"read", "write", "execute", "destructive"} and target:
        path = canonical_path(target)
        if not _path_subset((path,), contract.authority["paths"]):
            raise PermissionError("target is outside child filesystem authority")
    if effect == "network" and target not in contract.authority["network_targets"]:
        raise PermissionError("destination is outside child network authority")
    if effect == "remote" and target not in contract.authority["remote_targets"]:
        raise PermissionError("remote target is outside child authority")
    return True


def child_inference_options(delegation_id, *, requested_tokens=None, thinking="auto"):
    contract = load_contract(delegation_id)
    ceiling = int(contract.budget["max_tokens"])
    if requested_tokens is not None and int(requested_tokens) > ceiling:
        raise PermissionError("child generation request exceeds delegated token budget")
    return {"max_tokens": int(requested_tokens) if requested_tokens is not None else ceiling,
            "thinking": str(thinking)}


class ChildToolDispatcher:
    """Narrow a canonical dispatcher without copying or expanding its bindings."""
    def __init__(self, delegation_id, dispatcher):
        self.delegation_id = delegation_id
        self.dispatcher = dispatcher

    def execute(self, name, arguments, *, effect, target=""):
        authorize_child_action(self.delegation_id, name, effect, target)
        child_task = load_delegation(self.delegation_id).child_task_id
        return self.dispatcher.execute(name, arguments, task_id=child_task)


def create_child(parent_task_id, goal, *, role=None, required_capabilities=(),
                 tools=(), authority=None, parent_authority=None, parent_tools=None,
                 budget=None, workspace=None, completion=None, constraints=(),
                 evidence_refs=()):
    """Create a bounded child contract; permissions can only narrow its parent."""
    parent = load_task(parent_task_id)
    child_authority = _normalize_authority(authority)
    child_tools = tuple(dict.fromkeys(map(str, tools)))
    conn = connect()
    try:
        parent_row = conn.execute(
            "SELECT d.id,c.* FROM delegations d JOIN delegation_contracts c "
            "ON c.delegation_id=d.id WHERE d.child_task_id=?", (parent.id,),
        ).fetchone()
    finally:
        conn.close()
    parent_delegation_id = parent_row["id"] if parent_row else None
    if parent_row:
        inherited_authority = json_loads(parent_row["authority_json"], {})
        inherited_tools = tuple(json_loads(parent_row["tool_allowlist_json"], []))
    else:
        if parent_authority is None or parent_tools is None:
            raise ValueError("root delegation requires explicit parent authority and tools")
        inherited_authority = _normalize_authority(parent_authority)
        inherited_tools = tuple(map(str, parent_tools))
    _require_subset(child_authority, inherited_authority, child_tools, inherited_tools)

    budget = dict(budget or {})
    normalized_budget = {
        "max_seconds": max(1.0, float(budget.get("max_seconds", 3600))),
        "max_iterations": max(1, int(budget.get("max_iterations", 50))),
        "max_tokens": max(1, int(budget.get("max_tokens", 32768))),
        "inference": bool(budget.get("inference", True)),
    }
    if parent_row:
        parent_budget = json_loads(parent_row["budget_json"], {})
        for key in ("max_seconds", "max_iterations", "max_tokens"):
            if normalized_budget[key] > parent_budget[key]:
                raise PermissionError(f"child {key} exceeds parent budget")
    workspace = dict(workspace or {})
    mode = workspace.get("mode", "isolated")
    access = workspace.get("access", "read-only")
    if mode not in {"isolated", "shared"} or access not in {"read-only", "read-write"}:
        raise ValueError("invalid child workspace policy")
    root = canonical_path(workspace["root"]) if workspace.get("root") else ""
    if root and root not in child_authority["paths"] and not _path_subset((root,), child_authority["paths"]):
        raise PermissionError("workspace root is outside child path authority")
    normalized_workspace = {"mode": mode, "access": access, "root": root,
                            "exclusive": bool(workspace.get("exclusive", access == "read-write"))}
    normalized_completion = {
        "required_evidence_types": sorted(set(map(str, (completion or {}).get(
            "required_evidence_types", ())))),
        "summary_required": bool((completion or {}).get("summary_required", True)),
    }
    delegation = create_delegation(
        parent.id, goal, role=role, required_capabilities=required_capabilities,
        scope=child_authority, constraints=constraints, permissions=child_tools,
        evidence_refs=evidence_refs,
        expected_result=json_dumps(normalized_completion),
    )
    with transaction(immediate=True) as conn:
        conn.execute(
            "INSERT INTO delegation_contracts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (delegation.id, parent_delegation_id, json_dumps(child_tools),
             json_dumps(child_authority), json_dumps(normalized_budget),
             json_dumps(normalized_workspace), json_dumps(normalized_completion), "created",
             None, "", None, None, None, now_utc()),
        )
    return load_contract(delegation.id)


def _set_state(delegation_id, state, **fields):
    allowed = {"started_at", "deadline_at", "finished_at", "accepted", "acceptance_reason"}
    if set(fields) - allowed:
        raise ValueError("unsupported delegation state field")
    assignments = ["state=?", "updated_at=?"] + [f"{key}=?" for key in fields]
    values = [state, now_utc(), *fields.values(), delegation_id]
    with transaction(immediate=True) as conn:
        conn.execute(f"UPDATE delegation_contracts SET {','.join(assignments)} "
                     "WHERE delegation_id=?", values)


def start(delegation_id, executor):
    """Run a child with bounded context and cancellation; executor returns a result dict."""
    contract = load_contract(delegation_id)
    if contract.state != "created":
        raise RuntimeError(f"delegation is already {contract.state}")
    delegation = load_delegation(delegation_id)
    cancel_event = threading.Event()
    start_gate = threading.Event()
    workspace_lock = None
    if contract.workspace.get("exclusive") and contract.workspace.get("root"):
        with _LOCK:
            workspace_lock = _WORKSPACE_LOCKS.setdefault(contract.workspace["root"], threading.Lock())

    def run():
        start_gate.wait()
        acquired_gpu = acquired_workspace = False
        timed_out = threading.Event()
        def expire():
            timed_out.set()
            cancel_event.set()
        timer = threading.Timer(contract.budget["max_seconds"], expire)
        timer.daemon = True
        timer.start()
        try:
            if contract.budget["inference"]:
                acquired_gpu = _GPU_SLOT.acquire(timeout=contract.budget["max_seconds"])
                if not acquired_gpu:
                    raise TimeoutError("local inference slot timeout")
            if workspace_lock:
                acquired_workspace = workspace_lock.acquire(timeout=contract.budget["max_seconds"])
                if not acquired_workspace:
                    raise TimeoutError("workspace lock timeout")
            _set_state(delegation_id, "running", started_at=now_utc())
            update_task(delegation.child_task_id, state="running", phase="delegated-running")
            context = {
                "delegation_id": delegation_id, "task_id": delegation.child_task_id,
                "goal": delegation.goal, "tools": list(contract.tool_allowlist),
                "authority": contract.authority, "budget": contract.budget,
                "workspace": contract.workspace, "completion": contract.completion,
                "generation_limit": contract.budget["max_tokens"],
                "cancel_event": cancel_event,
            }
            result = executor(context)
            if timed_out.is_set():
                complete_delegation(delegation_id, status="failed", summary="child timed out",
                                    result={"timed_out": True})
                _set_state(delegation_id, "timed_out", finished_at=now_utc())
            elif cancel_event.is_set():
                complete_delegation(delegation_id, status="cancelled", summary="cancelled",
                                    result={"cancelled": True})
                _set_state(delegation_id, "cancelled", finished_at=now_utc())
            else:
                if not isinstance(result, dict):
                    raise TypeError("child executor must return a result object")
                summary = str(result.get("summary", "")).strip()
                if contract.completion["summary_required"] and not summary:
                    raise ValueError("child completion summary is required")
                complete_delegation(delegation_id, status="success",
                                    summary=summary, result=result)
                _set_state(delegation_id, "completed", finished_at=now_utc())
            return load_contract(delegation_id)
        except Exception as exc:
            if load_contract(delegation_id).state not in TERMINAL_STATES:
                try:
                    complete_delegation(delegation_id, status="failed", summary=str(exc),
                                        result={"error": str(exc)})
                except RuntimeError:
                    pass
                _set_state(delegation_id, "timed_out" if timed_out.is_set() else "failed",
                           finished_at=now_utc())
            raise
        finally:
            timer.cancel()
            if acquired_workspace:
                workspace_lock.release()
            if acquired_gpu:
                _GPU_SLOT.release()

    deadline = (datetime.now(timezone.utc) + timedelta(
        seconds=contract.budget["max_seconds"])).isoformat()
    _set_state(delegation_id, "scheduled", deadline_at=deadline)
    with _LOCK:
        future = _EXECUTOR.submit(run)
        _RUNS[delegation_id] = (future, cancel_event)
    start_gate.set()
    return future


def cancel(delegation_id):
    contract = load_contract(delegation_id)
    if contract.state in TERMINAL_STATES:
        return {"requested": False, "state": contract.state, "cancelled": contract.state == "cancelled"}
    with _LOCK:
        run = _RUNS.get(delegation_id)
    if not run:
        complete_delegation(delegation_id, status="cancelled", summary="cancelled before start")
        _set_state(delegation_id, "cancelled", finished_at=now_utc())
        return {"requested": True, "state": "cancelled", "cancelled": True}
    future, event = run
    event.set()
    return {"requested": True, "state": "cancellation_requested",
            "cancelled": future.cancel()}


def join(delegation_id, timeout=None):
    with _LOCK:
        run = _RUNS.get(delegation_id)
    if run:
        try:
            run[0].result(timeout=timeout)
        except TimeoutError:
            return {"state": load_contract(delegation_id).state, "joined": False}
    return {"state": load_contract(delegation_id).state, "joined": True}


def accept(delegation_id, *, accept_result=True, reason=""):
    contract = load_contract(delegation_id)
    delegation = load_delegation(delegation_id)
    if contract.state != "completed":
        raise RuntimeError("only completed child work can be accepted")
    evidence_ids = tuple(map(str, delegation.result.get("evidence_ids", ())))
    actual_types = set()
    for evidence_id in evidence_ids:
        record = load_evidence(evidence_id)
        if record.task_id != delegation.child_task_id:
            raise PermissionError("child result references evidence owned by another task")
        actual_types.add(record.evidence_type)
    missing = set(contract.completion["required_evidence_types"]) - actual_types
    if accept_result and missing:
        raise ValueError("child completion evidence is incomplete: " + ", ".join(sorted(missing)))
    if accept_result and contract.completion["summary_required"] and not delegation.result_summary.strip():
        raise ValueError("child completion summary is required")
    _set_state(delegation_id, "accepted" if accept_result else "rejected",
               accepted=1 if accept_result else 0, acceptance_reason=str(reason))
    append_event(delegation.parent_task_id, "delegation",
                 f"Child result {'accepted' if accept_result else 'rejected'}: {delegation_id}",
                 data={"delegation_id": delegation_id, "accepted": accept_result,
                       "evidence_ids": list(evidence_ids), "reason": reason})
    return load_contract(delegation_id)


def stage_child_memory(delegation_id, content, **kwargs):
    delegation = load_delegation(delegation_id)
    candidate_id = memory.stage_candidate(content, source=f"child:{delegation.child_task_id}", **kwargs)
    with transaction(immediate=True) as conn:
        conn.execute("INSERT INTO delegation_memory VALUES(?,?,?,?,NULL)",
                     (delegation_id, candidate_id, "staged", now_utc()))
    return candidate_id


def review_child_memory(delegation_id, candidate_id, *, promote, reason=""):
    contract = load_contract(delegation_id)
    if promote and contract.accepted is not True:
        raise PermissionError("child memory cannot be promoted before parent acceptance")
    conn = connect()
    try:
        owned = conn.execute(
            "SELECT 1 FROM delegation_memory WHERE delegation_id=? AND candidate_id=? "
            "AND state='staged'", (delegation_id, candidate_id),
        ).fetchone()
    finally:
        conn.close()
    if not owned:
        raise KeyError("unknown staged child memory candidate")
    entry = memory.decide_candidate(candidate_id, promote=promote, reason=reason)
    with transaction(immediate=True) as conn:
        changed = conn.execute(
            "UPDATE delegation_memory SET state=?,reviewed_at=? WHERE delegation_id=? "
            "AND candidate_id=? AND state='staged'",
            ("accepted" if promote else "rejected", now_utc(), delegation_id, candidate_id),
        ).rowcount
    if not changed:
        raise KeyError("unknown staged child memory candidate")
    return entry
