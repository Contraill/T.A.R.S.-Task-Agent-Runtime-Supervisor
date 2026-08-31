from __future__ import annotations

from dataclasses import dataclass, field
import threading

from .action_journal import begin_action, finish_action
from .approvals import ApprovalBroker
from .evidence import record as record_evidence
from .ownership import (Heartbeat, Owner, claim_workspace, current_owner, held_by,
                        release as release_lease)
from .policy import ScopeGuard, ScopeRequest, canonical_path


_LEASE_LOCK = threading.Lock()
_ACTION_LEASES = {}


@dataclass(frozen=True)
class ToolResult:
    tool: str
    state: str
    data: dict = field(default_factory=dict)
    error: str = ""
    action_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    @property
    def succeeded(self):
        return self.state == "succeeded"


class ToolRuntime:
    def __init__(self, *, guard=None, broker=None):
        self.guard = guard or ScopeGuard()
        self.broker = broker or ApprovalBroker()
        self._owner_lock = threading.Lock()
        self._thread_owners = {}

    def authorize(self, requests, approval_ids=None):
        approvals = approval_ids or {}
        evaluated = [(key, request, self.guard.evaluate(request))
                     for key, request in requests]
        workspace_keys = sorted({
            canonical_path(decision.target)
            for _, request, decision in evaluated
            if request.allowed_paths and decision.action != "deny" and decision.target
        })
        scoped_owner = current_owner()
        thread_id = threading.get_ident()
        with self._owner_lock:
            local = self._thread_owners.get(thread_id)
        owner = scoped_owner or (local[0] if local else None) or Owner.create("tool")
        claimed = []
        heartbeats = []
        actions = []
        try:
            for workspace_key in workspace_keys:
                borrowed = held_by("workspace", workspace_key, owner)
                if not claim_workspace(
                    workspace_key, owner, lease_seconds=86_400,
                    metadata={"tool": evaluated[0][1].tool if evaluated else ""},
                ):
                    raise RuntimeError(
                        f"workspace is exclusively owned by another executor: {workspace_key}")
                if not borrowed:
                    claimed.append(workspace_key)
                    beat = Heartbeat("workspace", workspace_key, owner, lease_seconds=30)
                    beat.__enter__()
                    heartbeats.append(beat)
            for key, request, decision in evaluated:
                actions.append(begin_action(
                    request, decision, approval_id=approvals.get(key), broker=self.broker,
                    owner=owner,
                ))
            for action in actions:
                beat = Heartbeat("action", action.id, owner, lease_seconds=30)
                beat.__enter__()
                heartbeats.append(beat)
            if actions:
                if scoped_owner is None:
                    with self._owner_lock:
                        existing = self._thread_owners.get(thread_id)
                        self._thread_owners[thread_id] = (
                            owner, (existing[1] if existing else 0) + 1)
                group = {"owner": owner, "keys": tuple(claimed), "remaining": len(actions),
                         "heartbeats": tuple(heartbeats),
                         "runtime": self, "local_owner": scoped_owner is None,
                         "thread_id": thread_id}
                with _LEASE_LOCK:
                    for action in actions:
                        _ACTION_LEASES[action.id] = group
        except Exception:
            for action in actions:
                finish_action(action.id, state="cancelled",
                              result={"error": "a required policy check failed"},
                              owner_token=owner.token)
            for beat in reversed(heartbeats):
                beat.__exit__(None, None, None)
            for workspace_key in reversed(claimed):
                release_lease("workspace", workspace_key, owner)
            raise
        return actions

    def finish(self, actions, *, state, result):
        for action in actions:
            try:
                finish_action(
                    action.id, state=state, result=result,
                    owner_token=action.owner_token,
                )
            finally:
                self._release_action_lease(action.id)

    def _release_action_lease(self, action_id):
        with _LEASE_LOCK:
            group = _ACTION_LEASES.pop(action_id, None)
            if group is None:
                return
            group["remaining"] -= 1
            if group["remaining"]:
                return
        heartbeat_error = None
        for beat in reversed(group["heartbeats"]):
            try:
                beat.__exit__(None, None, None)
            except Exception as exc:
                heartbeat_error = heartbeat_error or exc
        try:
            for workspace_key in reversed(group["keys"]):
                release_lease("workspace", workspace_key, group["owner"])
        finally:
            if group["local_owner"]:
                runtime = group["runtime"]
                with runtime._owner_lock:
                    existing = runtime._thread_owners.get(group["thread_id"])
                    groups = max(0, (existing[1] if existing else 1) - 1)
                    if groups:
                        runtime._thread_owners[group["thread_id"]] = (
                            group["owner"], groups)
                    else:
                        runtime._thread_owners.pop(group["thread_id"], None)
        if heartbeat_error is not None:
            raise heartbeat_error

    @staticmethod
    def evidence(evidence_type, source, content, *, task_id=None, event_uuid=None,
                 result_ref="", metadata=None):
        return record_evidence(
            evidence_type, source, content, task_id=task_id, event_uuid=event_uuid,
            result_ref=result_ref, metadata=metadata,
        )
