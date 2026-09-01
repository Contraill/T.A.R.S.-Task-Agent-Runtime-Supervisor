from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import threading
import time

from . import memory
from .evidence import load as load_evidence
from .events import append_event
from .orchestration import create_delegation, load_delegation
from .ownership import (Heartbeat, Owner, claim, claim_in_transaction,
                        claim_workspace, owner_scope, release as release_lease)
from .policy import canonical_path
from .state_store import (connect, ensure_state_store, json_dumps, json_loads,
                          now_utc, transaction)
from .tasks import load_task, update_task


TERMINAL_STATES = {"completed", "failed", "cancelled", "timed_out", "accepted", "rejected"}
_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="tars-child")
_LOCK = threading.RLock()
_RUNS: dict[str, tuple[Future, threading.Event, Owner]] = {}


class _TaskExecutionConflict(RuntimeError):
    pass


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


@dataclass(frozen=True)
class DelegationStep:
    """One boundary-observable iterative executor step."""
    result: dict
    done: bool = False


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
        _contract_values=(
            parent_delegation_id, json_dumps(child_tools), json_dumps(child_authority),
            json_dumps(normalized_budget), json_dumps(normalized_workspace),
            json_dumps(normalized_completion), "created", None, "", None, None, None,
            now_utc()),
    )
    return load_contract(delegation.id)


def _set_state(delegation_id, state, *, expected=None, owner=None, **fields):
    allowed = {"started_at", "deadline_at", "finished_at", "accepted", "acceptance_reason"}
    if set(fields) - allowed:
        raise ValueError("unsupported delegation state field")
    assignments = ["state=?", "updated_at=?"] + [f"{key}=?" for key in fields]
    values = [state, now_utc(), *fields.values(), delegation_id]
    with transaction(immediate=True) as conn:
        where = "delegation_id=?"
        if expected:
            expected = tuple(expected)
            where += f" AND state IN ({','.join('?' for _ in expected)})"
            values.extend(expected)
        if owner is not None:
            where += (" AND EXISTS (SELECT 1 FROM resource_leases WHERE "
                      "resource_type='delegation' AND resource_key=delegation_id "
                      "AND owner_token=? AND expires_at>?)")
            values.extend((owner.token, now_utc()))
        changed = conn.execute(
            f"UPDATE delegation_contracts SET {','.join(assignments)} WHERE {where}",
            values,
        ).rowcount
        if changed != 1:
            raise RuntimeError(f"delegation {delegation_id} changed concurrently")


def _finish_execution(delegation_id, owner, contract_state, *, status, summary, result,
                      update_child=True):
    stamp = now_utc()
    delegation = load_delegation(delegation_id)
    delegation_state = "completed" if status in {"success", "partial"} else status
    child_state = "completed" if status in {"success", "partial"} else (
        "cancelled" if status == "cancelled" else "failed")
    with transaction(immediate=True) as conn:
        lease = conn.execute(
            "SELECT owner_token,expires_at FROM resource_leases WHERE resource_type='delegation' "
            "AND resource_key=?", (delegation_id,),
        ).fetchone()
        if (not lease or lease["owner_token"] != owner.token
                or lease["expires_at"] <= stamp):
            raise RuntimeError(f"delegation {delegation_id} is not owned by this executor")
        changed = conn.execute(
            """UPDATE delegation_contracts SET state=?,updated_at=?,finished_at=?
               WHERE delegation_id=? AND state IN ('scheduled','running','cancellation_requested')""",
            (contract_state, stamp, stamp, delegation_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError(f"delegation {delegation_id} changed concurrently")
        changed = conn.execute(
            """UPDATE delegations SET state=?,result_status=?,result_summary=?,result_json=?,
               updated_at=?,completed_at=? WHERE id=?
               AND state NOT IN ('completed','failed','cancelled')""",
            (delegation_state, status, str(summary), json_dumps(result or {}), stamp,
             stamp, delegation_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError(f"delegation {delegation_id} result changed concurrently")
        if update_child:
            conn.execute(
                "UPDATE tasks SET state=?,phase=?,updated_at=? WHERE id=?",
                (child_state, f"delegation-{contract_state}", stamp,
                 delegation.child_task_id),
            )
        conn.execute(
            "DELETE FROM resource_leases WHERE resource_type='delegation' "
            "AND resource_key=? AND owner_token=?", (delegation_id, owner.token),
        )
    append_event(
        delegation.child_task_id, "result" if status in {"success", "partial"} else "error",
        str(summary), data={"delegation_id": delegation_id, "status": status},
    )
    return load_contract(delegation_id)


def _result_object(value):
    if not isinstance(value, dict):
        raise TypeError("child executor result must be an object")
    if "iterations" in value:
        raise TypeError("child executor must not self-report iteration counts")
    return dict(value)


def _execute_bounded(executor, context, maximum, cancel_event):
    """Run one atomic call or at most ``maximum`` explicit iterator steps.

    A dictionary return is one opaque boundary invocation. Iterative executors
    must yield ``DelegationStep`` and explicitly mark their terminal result;
    the boundary never resumes an iterator merely to discover an overrun.
    """
    if cancel_event.is_set():
        return {}
    execution = executor(context)
    if isinstance(execution, dict):
        return _result_object(execution)
    if not isinstance(execution, Iterator):
        raise TypeError(
            "child executor must return an atomic result or delegation-step iterator")

    final = None
    primary_error = None
    try:
        for _count in range(1, maximum + 1):
            if cancel_event.is_set():
                return final or {}
            try:
                step = next(execution)
            except StopIteration as exc:
                raise ValueError(
                    "child iterative executor ended without a terminal step") from exc
            if not isinstance(step, DelegationStep):
                raise TypeError("child iteration must yield DelegationStep")
            final = _result_object(step.result)
            if step.done:
                return final
        if cancel_event.is_set():
            return final or {}
        raise RuntimeError("child exceeded delegated iteration budget")
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        close = getattr(execution, "close", None)
        if callable(close):
            try:
                close()
            except Exception as close_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"child iterator close also failed: {close_error}")


def start(delegation_id, executor):
    """Run a bounded atomic child or an explicit ``DelegationStep`` iterator."""
    contract = load_contract(delegation_id)
    delegation = load_delegation(delegation_id)
    owner = Owner.create("delegation")
    cancel_event = threading.Event()
    start_gate = threading.Event()
    deadline = (datetime.now(timezone.utc) + timedelta(
        seconds=contract.budget["max_seconds"])).isoformat()
    stale_running = False
    with transaction(immediate=True) as conn:
        row = conn.execute(
            "SELECT state FROM delegation_contracts WHERE delegation_id=?", (delegation_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"unknown delegation contract: {delegation_id}")
        if row["state"] in TERMINAL_STATES:
            raise RuntimeError(f"delegation is already {row['state']}")
        if row["state"] not in {"created", "scheduled", "running"}:
            raise RuntimeError(f"delegation is already {row['state']}")
        if not claim_in_transaction(
            conn, "delegation", delegation_id, owner, lease_seconds=30,
            metadata={"child_task_id": delegation.child_task_id},
        ):
            raise RuntimeError("delegation already has a live executor")
        stale_running = row["state"] == "running"
        if not stale_running:
            changed = conn.execute(
                "UPDATE delegation_contracts SET state='scheduled',deadline_at=?,updated_at=? "
                "WHERE delegation_id=? AND state IN ('created','scheduled')",
                (deadline, now_utc(), delegation_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError(f"delegation {delegation_id} changed concurrently")
    if stale_running:
        _finish_execution(
            delegation_id, owner, "failed", status="failed",
            summary="previous child execution owner was lost; outcome is ambiguous",
            result={"owner_lost": True, "replayed": False},
        )
        raise RuntimeError("stale running delegation was not replayed")

    def run():
        start_gate.wait()
        acquired = []
        timed_out = threading.Event()
        def expire():
            timed_out.set()
            cancel_event.set()
        timer = threading.Timer(contract.budget["max_seconds"], expire)
        timer.daemon = True
        timer.start()
        monitor_stop = threading.Event()
        def monitor_cancel():
            while not monitor_stop.wait(0.1):
                if load_contract(delegation_id).state == "cancellation_requested":
                    cancel_event.set()
                    return
        monitor = threading.Thread(target=monitor_cancel, name="tars-child-cancel", daemon=True)
        monitor.start()
        try:
            if load_contract(delegation_id).state == "cancellation_requested":
                return _finish_execution(
                    delegation_id, owner, "cancelled", status="cancelled",
                    summary="cancelled before start", result={"cancelled": True})
            resource_lease = max(30.0, float(contract.budget["max_seconds"]) + 5)
            if not claim(
                "task-execution", delegation.child_task_id, owner,
                lease_seconds=resource_lease,
                metadata={"engine": "delegation", "delegation_id": delegation_id},
            ):
                raise _TaskExecutionConflict(
                    f"task {delegation.child_task_id} already has a live execution owner")
            acquired.append(("task-execution", delegation.child_task_id))
            if contract.workspace.get("exclusive") and contract.workspace.get("root"):
                workspace_key = contract.workspace["root"]
                deadline_stamp = time.monotonic() + contract.budget["max_seconds"]
                while not claim_workspace(
                    workspace_key, owner, lease_seconds=resource_lease,
                    metadata={"delegation_id": delegation_id},
                ):
                    if cancel_event.wait(0.1) or time.monotonic() >= deadline_stamp:
                        raise TimeoutError("workspace lock timeout")
                acquired.append(("workspace", workspace_key))
            if cancel_event.is_set():
                return _finish_execution(
                    delegation_id, owner, "cancelled", status="cancelled",
                    summary="cancelled before execution", result={"cancelled": True})
            _set_state(delegation_id, "running", expected=("scheduled",), owner=owner,
                       started_at=now_utc())
            update_task(delegation.child_task_id, state="running", phase="delegated-running")
            context = {
                "delegation_id": delegation_id, "task_id": delegation.child_task_id,
                "goal": delegation.goal, "tools": list(contract.tool_allowlist),
                "authority": contract.authority, "budget": contract.budget,
                "workspace": contract.workspace, "completion": contract.completion,
                "generation_limit": contract.budget["max_tokens"],
                "cancel_event": cancel_event,
            }
            with owner_scope(owner), ExitStack() as heartbeats:
                heartbeats.enter_context(Heartbeat(
                    "delegation", delegation_id, owner, lease_seconds=30))
                for resource_type, resource_key in acquired:
                    heartbeats.enter_context(Heartbeat(
                        resource_type, resource_key, owner, lease_seconds=30))
                result = _execute_bounded(
                    executor, context, int(contract.budget["max_iterations"]),
                    cancel_event)
            if timed_out.is_set():
                _finish_execution(
                    delegation_id, owner, "timed_out", status="failed",
                    summary="child timed out", result={"timed_out": True})
            elif cancel_event.is_set():
                _finish_execution(
                    delegation_id, owner, "cancelled", status="cancelled",
                    summary="cancelled", result={"cancelled": True})
            else:
                summary = str(result.get("summary", "")).strip()
                if contract.completion["summary_required"] and not summary:
                    raise ValueError("child completion summary is required")
                _finish_execution(
                    delegation_id, owner, "completed", status="success",
                    summary=summary, result=result)
            return load_contract(delegation_id)
        except Exception as exc:
            if load_contract(delegation_id).state not in TERMINAL_STATES:
                try:
                    _finish_execution(
                        delegation_id, owner,
                        "timed_out" if timed_out.is_set() else "failed", status="failed",
                        summary=str(exc), result={"error": str(exc)},
                        update_child=not isinstance(exc, _TaskExecutionConflict),
                    )
                except RuntimeError:
                    pass
            raise
        finally:
            timer.cancel()
            monitor_stop.set()
            monitor.join(timeout=1)
            for resource_type, resource_key in reversed(acquired):
                release_lease(resource_type, resource_key, owner)

    try:
        with _LOCK:
            future = _EXECUTOR.submit(run)
            _RUNS[delegation_id] = (future, cancel_event, owner)
    except Exception as exc:
        _finish_execution(
            delegation_id, owner, "failed", status="failed", summary=str(exc),
            result={"submission_failed": True, "error": str(exc)})
        raise
    def cleanup(_future):
        with _LOCK:
            if _RUNS.get(delegation_id, (None,))[0] is _future:
                _RUNS.pop(delegation_id, None)
    future.add_done_callback(cleanup)
    start_gate.set()
    return future


def cancel(delegation_id):
    contract = load_contract(delegation_id)
    if contract.state in TERMINAL_STATES:
        return {"requested": False, "state": contract.state, "cancelled": contract.state == "cancelled"}
    with _LOCK:
        run = _RUNS.get(delegation_id)
    with transaction(immediate=True) as conn:
        changed = conn.execute(
            "UPDATE delegation_contracts SET state='cancellation_requested',updated_at=? "
            "WHERE delegation_id=? AND state IN ('created','scheduled','running')",
            (now_utc(), delegation_id),
        ).rowcount
    if not changed:
        current = load_contract(delegation_id)
        if not run and current.state == "cancellation_requested":
            recovery_owner = Owner.create("delegation-cancel")
            if claim("delegation", delegation_id, recovery_owner, lease_seconds=30,
                     metadata={"cancellation_recovery": True}):
                _finish_execution(
                    delegation_id, recovery_owner, "cancelled", status="cancelled",
                    summary="cancelled after execution owner loss",
                    result={"cancelled": True, "owner_lost": True})
                return {"requested": True, "state": "cancelled", "cancelled": True}
        return {"requested": current.state == "cancellation_requested",
                "state": current.state, "cancelled": current.state == "cancelled"}
    if not run:
        recovery_owner = Owner.create("delegation-cancel")
        if claim("delegation", delegation_id, recovery_owner, lease_seconds=30,
                 metadata={"cancellation_recovery": True}):
            _finish_execution(
                delegation_id, recovery_owner, "cancelled", status="cancelled",
                summary="cancelled before start", result={"cancelled": True})
            return {"requested": True, "state": "cancelled", "cancelled": True}
        return {"requested": True, "state": "cancellation_requested", "cancelled": False}
    future, event, owner = run
    event.set()
    cancelled = future.cancel()
    if cancelled:
        _finish_execution(
            delegation_id, owner, "cancelled", status="cancelled",
            summary="cancelled before start", result={"cancelled": True})
    return {"requested": True,
            "state": "cancelled" if cancelled else "cancellation_requested",
            "cancelled": cancelled}


def join(delegation_id, timeout=None):
    with _LOCK:
        run = _RUNS.get(delegation_id)
    if run:
        try:
            run[0].result(timeout=timeout)
        except TimeoutError:
            return {"state": load_contract(delegation_id).state, "joined": False}
        except CancelledError:
            pass
        except Exception:
            if load_contract(delegation_id).state not in TERMINAL_STATES:
                raise
    state = load_contract(delegation_id).state
    return {"state": state, "joined": state in TERMINAL_STATES}


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
               expected=("completed",),
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
    owner = Owner.create("child-memory-review")
    with transaction(immediate=True) as conn:
        row = conn.execute(
            "SELECT * FROM delegation_memory WHERE delegation_id=? AND candidate_id=? "
            "AND state='staged'", (delegation_id, candidate_id),
        ).fetchone()
        if not row:
            raise KeyError("unknown staged child memory candidate")
        if not claim_in_transaction(
            conn, "delegation-memory-review", f"{delegation_id}:{candidate_id}", owner,
            lease_seconds=30, metadata={"promote": bool(promote)},
        ):
            raise RuntimeError("child memory candidate has a live reviewer")
        changed = conn.execute(
            "UPDATE delegation_memory SET reviewed_at=? WHERE delegation_id=? "
            "AND candidate_id=? AND state='staged'",
            ("processing:" + owner.token, delegation_id, candidate_id),
        ).rowcount
    if not changed:
        raise KeyError("unknown staged child memory candidate")
    try:
        with connect() as conn:
            candidate = conn.execute(
                "SELECT status FROM memory_candidates WHERE id=?", (candidate_id,),
            ).fetchone()
        if not candidate:
            raise KeyError(f"unknown memory candidate: {candidate_id}")
        if candidate["status"] in {"staged", "reviewing"}:
            entry = memory.decide_candidate(candidate_id, promote=promote, reason=reason)
            actual_promote = bool(promote)
        elif candidate["status"] == "promoted":
            actual_promote = True
            entry = memory.inspect("mem-" + candidate_id.removeprefix("cand-"))
        elif candidate["status"] == "rejected":
            actual_promote = False
            entry = None
        else:
            raise RuntimeError(f"unsupported memory candidate state: {candidate['status']}")
        with transaction(immediate=True) as conn:
            stamp = now_utc()
            changed = conn.execute(
                "UPDATE delegation_memory SET state=?,reviewed_at=? WHERE delegation_id=? "
                "AND candidate_id=? AND state='staged' AND reviewed_at=? "
                "AND EXISTS (SELECT 1 FROM resource_leases WHERE "
                "resource_type='delegation-memory-review' AND resource_key=? "
                "AND owner_token=? AND expires_at>?)",
                ("accepted" if actual_promote else "rejected", stamp, delegation_id,
                 candidate_id, "processing:" + owner.token,
                 f"{delegation_id}:{candidate_id}", owner.token, stamp)).rowcount
            if changed != 1:
                raise RuntimeError("child memory review ownership changed")
            conn.execute(
                "DELETE FROM resource_leases WHERE resource_type='delegation-memory-review' "
                "AND resource_key=? AND owner_token=?",
                (f"{delegation_id}:{candidate_id}", owner.token),
            )
        if actual_promote != bool(promote):
            raise RuntimeError("child memory candidate was already finalized differently")
    except Exception:
        with transaction(immediate=True) as conn:
            conn.execute(
                "UPDATE delegation_memory SET reviewed_at=NULL WHERE delegation_id=? "
                "AND candidate_id=? AND state='staged' AND reviewed_at=?",
                (delegation_id, candidate_id, "processing:" + owner.token))
            conn.execute(
                "DELETE FROM resource_leases WHERE resource_type='delegation-memory-review' "
                "AND resource_key=? AND owner_token=?",
                (f"{delegation_id}:{candidate_id}", owner.token),
            )
        raise
    return entry
