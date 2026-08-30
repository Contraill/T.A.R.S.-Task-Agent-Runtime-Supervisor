from __future__ import annotations

from dataclasses import dataclass, field

from .action_journal import begin_action, finish_action
from .approvals import ApprovalBroker
from .evidence import record as record_evidence
from .policy import ScopeGuard, ScopeRequest


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

    def authorize(self, requests, approval_ids=None):
        approvals = approval_ids or {}
        actions = []
        try:
            for key, request in requests:
                decision = self.guard.evaluate(request)
                actions.append(begin_action(
                    request, decision, approval_id=approvals.get(key), broker=self.broker,
                ))
        except Exception:
            for action in actions:
                finish_action(action.id, state="cancelled",
                              result={"error": "a required policy check failed"})
            raise
        return actions

    @staticmethod
    def finish(actions, *, state, result):
        for action in actions:
            finish_action(action.id, state=state, result=result)

    @staticmethod
    def evidence(evidence_type, source, content, *, task_id=None, event_uuid=None,
                 result_ref="", metadata=None):
        return record_evidence(
            evidence_type, source, content, task_id=task_id, event_uuid=event_uuid,
            result_ref=result_ref, metadata=metadata,
        )
