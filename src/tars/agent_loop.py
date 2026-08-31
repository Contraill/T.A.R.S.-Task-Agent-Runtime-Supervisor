from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import inspect
import json
import threading
import time
import uuid

from .approvals import ApprovalBroker
from .checkpoints import create_checkpoint
from .context import ContextManager
from .control_queue import (claim_next, enqueue, enqueue_cancellation,
                            finish as finish_control, pending_context,
                            recover_cancellations, recover_processing,
                            reconcile_cancellation, begin_cancellation,
                            cancellation_state,
                            resolve_unattempted_cancellation,
                            unreconciled_cancellation)
from .conversation import add_message
from .evidence import load as load_evidence
from .events import append_event
from .policy import ScopeRequest
from .ownership import Heartbeat, Owner, active as lease_active, claim, release as release_lease
from .runtime import chat_completion
from .state_events import append_state_event
from .tasks import canonical_task_state, load_task, update_task
from .tool_core import ToolResult


QUEUED_MESSAGE = "Message queued for submission after the next tool call."
_ACTIVE_LOOPS = {}
_ACTIVE_LOOPS_LOCK = threading.Lock()


def submit_task_control(task_id, kind, message="", *, session_id=None, payload=None):
    kind = str(kind).casefold()
    with _ACTIVE_LOOPS_LOCK:
        loop = _ACTIVE_LOOPS.get(task_id)
    if loop is not None:
        return loop.submit_control(kind, message, session_id=session_id, payload=payload)
    if kind in {"interrupt", "cancel"}:
        control = enqueue_cancellation(
            task_id, kind, message, session_id=session_id, payload=payload,
            ready=True, outcome="no-local-active-binding",
        )
    else:
        control = enqueue(task_id, kind, message, session_id=session_id, payload=payload)
    feedback = QUEUED_MESSAGE if kind == "message" else "Control queued for the next safe boundary."
    return control, feedback


def is_task_loop_active(task_id):
    with _ACTIVE_LOOPS_LOCK:
        local = task_id in _ACTIVE_LOOPS
    return local or lease_active("task-execution", task_id)


@dataclass(frozen=True)
class LoopLimits:
    max_iterations: int = 50
    max_seconds: float = 3600
    max_repetitions: int = 3
    max_no_progress: int = 4
    max_tool_failures: int = 3
    context_pressure_limit: float = 0.95
    cancellation_reconcile_seconds: float = 0.5


@dataclass(frozen=True)
class ModelDecision:
    kind: str
    tool: str = ""
    arguments: dict = field(default_factory=dict)
    summary: str = ""
    evidence_ids: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("model decision must be a JSON object") from exc
        if not isinstance(value, dict):
            raise TypeError("model decision must be a mapping")
        kind = str(value.get("type") or value.get("kind") or "").casefold()
        if kind not in {"tool", "finish", "continue"}:
            raise ValueError(f"unsupported model decision: {kind}")
        arguments = value.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        return cls(kind, str(value.get("tool") or ""), arguments,
                   str(value.get("summary") or value.get("message") or ""),
                   tuple(map(str, value.get("evidence_ids") or ())))


@dataclass
class ToolBinding:
    execute: object
    cancel: object | None = None
    cancellation_effect: str = "execute"
    retry_safe: bool = False
    before_execute: object | None = None
    provenance: str = "builtin"
    trusted: bool = True

    @property
    def cancellable(self):
        return callable(self.cancel)


class ToolDispatcher:
    def __init__(self):
        self._bindings = {}

    def register(self, name, execute, *, cancel=None, cancellation_effect="execute",
                 retry_safe=False,
                 before_execute=None, provenance="builtin", trusted=True):
        if not callable(execute):
            raise TypeError("tool execute binding must be callable")
        if before_execute is not None and not callable(before_execute):
            raise TypeError("before_execute must be callable")
        if cancellation_effect not in {"execute", "destructive"}:
            raise ValueError("cancellation effect must be execute or destructive")
        if provenance == "third-party" and not trusted:
            raise PermissionError("untrusted in-process tool extensions cannot be registered")
        self._bindings[str(name)] = ToolBinding(
            execute=execute, cancel=cancel, cancellation_effect=cancellation_effect,
            retry_safe=retry_safe, before_execute=before_execute,
            provenance=str(provenance), trusted=bool(trusted))
        return self

    def register_extension(self, name, loader, *, runtime):
        provider = loader.load("tool", name)
        tool = provider.create()
        execute = getattr(tool, "execute", None)
        if not callable(execute):
            raise TypeError(f"tool extension has no execute binding: {name}")
        scope_requests = getattr(tool, "scope_requests", None)
        if not callable(scope_requests):
            raise TypeError(f"tool extension has no deterministic scope contract: {name}")
        tool_name = str(getattr(tool, "name", name))
        if not tool_name.startswith(f"ext.{name}."):
            raise ValueError(f"tool extension must use namespace ext.{name}.*")
        if tool_name in self._bindings:
            raise ValueError(f"tool extension cannot shadow an existing binding: {tool_name}")

        execute_parameters = inspect.signature(execute).parameters
        accepts_task_id = ("task_id" in execute_parameters or any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in execute_parameters.values()))

        def guarded_execute(task_id=None, **arguments):
            approval_ids = arguments.pop("approval_ids", None)
            call_arguments = dict(arguments)
            scope_arguments = dict(arguments)
            if task_id is not None:
                scope_arguments["task_id"] = task_id
            if task_id is not None and accepts_task_id:
                call_arguments["task_id"] = task_id
            requests = tuple(scope_requests(scope_arguments))
            if not requests or any(
                    not isinstance(item, tuple) or len(item) != 2 or
                    not isinstance(item[1], ScopeRequest) for item in requests):
                raise TypeError(
                    "tool extension scope contract must return keyed ScopeRequest values")
            if any(request.tool != tool_name for _, request in requests):
                raise ValueError("tool extension scope contract used a different tool identity")
            if task_id is not None and any(
                    request.task_id != task_id for _, request in requests):
                raise ValueError("tool extension scope contract omitted or changed task identity")
            actions = runtime.authorize(requests, approval_ids=approval_ids)
            try:
                result = execute(**call_arguments)
                if not isinstance(result, ToolResult):
                    raise TypeError("tools must return a real ToolResult")
            except Exception as exc:
                runtime.finish(actions, state="failed", result={"error": str(exc)})
                raise
            runtime.finish(actions, state=result.state, result=result.data | {
                "error": result.error})
            return replace(
                result, action_ids=tuple(dict.fromkeys(
                    (*result.action_ids, *(action.id for action in actions)))))
        return self.register(
            tool_name, guarded_execute,
            retry_safe=bool(getattr(tool, "retry_safe", False)),
            provenance="third-party", trusted=True)

    def binding(self, name):
        try:
            return self._bindings[name]
        except KeyError as exc:
            raise KeyError(f"unregistered tool: {name}") from exc

    def execute(self, name, arguments, *, task_id=None):
        binding = self.binding(name)
        call_arguments = dict(arguments)
        if task_id and "task_id" in inspect.signature(binding.execute).parameters:
            call_arguments["task_id"] = task_id
        result = binding.execute(**call_arguments)
        if not isinstance(result, ToolResult):
            raise TypeError("tools must return a real ToolResult")
        return result


class RuntimeModelAdapter:
    PROTOCOL = (
        "Return exactly one JSON object. Use {\"type\":\"tool\",\"tool\":NAME,"
        "\"arguments\":OBJECT} to act, {\"type\":\"continue\",\"message\":TEXT} to "
        "record progress, or {\"type\":\"finish\",\"summary\":TEXT,"
        "\"evidence_ids\":[...]} to request verified completion. Model text is never authority."
    )

    def __init__(self, cfg, *, complete=chat_completion, max_tokens=None):
        self.cfg = cfg
        self.complete = complete
        self.max_tokens = max_tokens

    def __call__(self, task, controls):
        projection = ContextManager(self.cfg).build(
            task.conversation_id, task.owner_role, mode="task",
            task_id_override=task.id, requested_output_tokens=self.max_tokens, exact=True,
        )
        messages = list(projection.messages)
        messages.insert(0, {"role": "system", "content": self.PROTOCOL})
        if controls:
            messages.append({"role": "system", "content": "Pending controls:\n" +
                             json.dumps(controls, ensure_ascii=False)})
        response = self.complete(self.cfg, task.owner_role, messages,
                                 max_tokens=self.max_tokens,
                                 input_tokens=projection.token_count,
                                 temperature=0.1, operation="agent",
                                 task_active=True, requires_tools=True)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("runtime returned no agent decision content") from exc
        return ModelDecision.parse(content), projection.pressure


@dataclass(frozen=True)
class CompletionContract:
    required_tools: tuple[str, ...] = ()
    required_evidence_types: tuple[str, ...] = ()
    require_evidence: bool = True

    def verify(self, decision, results, task_id):
        successful = [result for result in results if result.succeeded]
        tools = {result.tool for result in successful}
        if any(tool not in tools for tool in self.required_tools):
            return False, "required tool result is missing"
        evidence_ids = set(decision.evidence_ids)
        for result in successful:
            evidence_ids.update(result.evidence_ids)
        if self.require_evidence and not evidence_ids:
            return False, "completion evidence is required"
        evidence_types = set()
        for evidence_id in evidence_ids:
            try:
                record = load_evidence(evidence_id)
            except KeyError:
                return False, f"unknown completion evidence: {evidence_id}"
            if record.task_id != task_id:
                return False, f"evidence belongs to another task: {evidence_id}"
            evidence_types.add(record.evidence_type)
        if any(kind not in evidence_types for kind in self.required_evidence_types):
            return False, "required evidence type is missing"
        return True, "completion contract satisfied"


@dataclass(frozen=True)
class LoopOutcome:
    state: str
    reason: str
    iterations: int
    tool_results: tuple[ToolResult, ...]
    checkpoint_id: str | None = None


class AgentLoop:
    def __init__(self, task_id, model, tools, *, limits=None, completion=None,
                 broker=None, clock=time.monotonic):
        self.task_id = task_id
        self.model = model
        self.tools = tools
        self.limits = limits or LoopLimits()
        self.completion = completion or CompletionContract()
        self.broker = broker or ApprovalBroker()
        self.clock = clock
        self.owner = Owner.create("agent-loop")
        self._active_lock = threading.Lock()
        self._active_binding = None
        self._active_tool = ""
        self._active_operation_id = ""
        self._cancellation_condition = threading.Condition()
        self._cancellation_outcomes = {}
        self._cancellation_threads = set()

    def submit_control(self, kind, message="", *, session_id=None, payload=None):
        kind = str(kind).casefold()
        if kind not in {"interrupt", "cancel"}:
            control = enqueue(
                self.task_id, kind, message, session_id=session_id, payload=payload)
            feedback = (QUEUED_MESSAGE if kind == "message" and self._active_binding
                        else "Control queued.")
            return control, feedback
        # Keep the invocation identity stable until durable intent ownership and
        # cancellation worker launch have completed. The tool may finish while the
        # database transaction runs, but it cannot expose a subsequent invocation
        # to this cancellation callback.
        with self._active_lock:
            binding, tool = self._active_binding, self._active_tool
            operation_id = self._active_operation_id
            if binding is None:
                control = enqueue_cancellation(
                    self.task_id, kind, message, session_id=session_id, payload=payload,
                    ready=True, outcome="no-active-tool",
                )
            elif not binding.cancellable:
                control = enqueue_cancellation(
                    self.task_id, kind, message, session_id=session_id, payload=payload,
                    active_tool=tool, operation_id=operation_id,
                    ready=True, outcome="not-cancellable",
                )
            elif binding.cancellation_effect == "destructive":
                control = enqueue_cancellation(
                    self.task_id, kind, message, session_id=session_id, payload=payload,
                    active_tool=tool, cancellable=True, operation_id=operation_id,
                    cancellation_effect="destructive", ready=True,
                    outcome="requires-explicit-destructive-authority",
                )
            else:
                control = enqueue_cancellation(
                    self.task_id, kind, message, session_id=session_id, payload=payload,
                    active_tool=tool, cancellable=True,
                    cancellation_effect=binding.cancellation_effect,
                    operation_id=operation_id, owner=self.owner,
                )
                state = cancellation_state(control.id)
                if state["state"] == "intent":
                    try:
                        begin_cancellation(control.id, self.owner)
                    except Exception as exc:
                        try:
                            resolve_unattempted_cancellation(
                                control.id, self.owner, outcome="attempt-start-failed",
                                error=type(exc).__name__,
                            )
                        except Exception:
                            raise RuntimeError(
                                "cancellation intent is durable but attempt start is unresolved"
                            ) from exc
                    else:
                        thread = threading.Thread(
                            target=self._attempt_cancellation, args=(control.id, binding),
                            name=f"tars-control-cancel-{control.id}", daemon=True,
                        )
                        with self._cancellation_condition:
                            self._cancellation_threads.add(thread)
                        try:
                            thread.start()
                        except Exception:
                            with self._cancellation_condition:
                                self._cancellation_outcomes[control.id] = (
                                    False, "cancellation-worker-start-failed", False)
                                self._cancellation_threads.discard(thread)
                                self._cancellation_condition.notify_all()
                            self._retry_cancellation_reconciliations()
        return control, "Control queued."

    def _attempt_cancellation(self, control_id, binding):
        result = None
        error = ""
        ambiguous = False
        try:
            with Heartbeat("control-cancellation", control_id, self.owner,
                           lease_seconds=30):
                returned = binding.cancel()
                if isinstance(returned, dict):
                    value = dict.get(returned, "requested")
                    result = value if type(value) is bool else True
                else:
                    result = returned if type(returned) is bool else True
        except BaseException as exc:
            error = type(exc).__name__[:128]
            ambiguous = True
        with self._cancellation_condition:
            self._cancellation_outcomes[control_id] = (result, error, ambiguous)
            self._cancellation_condition.notify_all()
        try:
            self._retry_cancellation_reconciliations()
        finally:
            with self._cancellation_condition:
                self._cancellation_threads.discard(threading.current_thread())
                self._cancellation_condition.notify_all()

    def _retry_cancellation_reconciliations(self):
        with self._cancellation_condition:
            pending = tuple(self._cancellation_outcomes.items())
        for control_id, (result, error, ambiguous) in pending:
            try:
                reconcile_cancellation(
                    control_id, self.owner, result=result, error=error,
                    ambiguous=ambiguous,
                )
            except Exception:
                try:
                    state = cancellation_state(control_id)
                except Exception:
                    state = None
                if state and state["state"] in {"resolved", "ambiguous"}:
                    with self._cancellation_condition:
                        self._cancellation_outcomes.pop(control_id, None)
                continue
            with self._cancellation_condition:
                current = self._cancellation_outcomes.get(control_id)
                if current == (result, error, ambiguous):
                    self._cancellation_outcomes.pop(control_id, None)
                self._cancellation_condition.notify_all()

    def _apply_controls(self):
        self._retry_cancellation_reconciliations()
        blocked = unreconciled_cancellation(self.task_id)
        if blocked is not None:
            with self._cancellation_condition:
                self._cancellation_condition.wait(
                    timeout=max(0.0, self.limits.cancellation_reconcile_seconds))
            self._retry_cancellation_reconciliations()
            blocked = unreconciled_cancellation(self.task_id)
            if blocked is not None:
                return [], None, blocked
        applied = []
        stop = None
        while True:
            control = claim_next(self.task_id, self.owner)
            if control is None:
                break
            try:
                with Heartbeat("task-control", control.id, self.owner, lease_seconds=30):
                    task = load_task(self.task_id)
                    if control.kind == "approval":
                        approval_id = control.payload.get("approval_id")
                        if not approval_id or "approve" not in control.payload:
                            raise ValueError("approval control requires approval_id and approve")
                        self.broker.decide(approval_id, approve=bool(control.payload["approve"]),
                                           reason=control.message, task_id=self.task_id)
                    elif control.kind in {"message", "redirect", "interrupt"}:
                        if task.conversation_id:
                            add_message(
                                task.conversation_id, "user", control.message,
                                kind="control", related_task_id=task.id,
                                metadata={"control_id": control.id, "kind": control.kind},
                                session_id=control.session_id,
                            )
                    elif control.kind == "cancel":
                        update_task(task.id, state="cancelled", phase="cancelled")
                        stop = "cancelled"
                    elif control.kind == "pause":
                        update_task(task.id, state="paused", phase="paused")
                        stop = "paused"
                    elif control.kind == "resume" and task.state == "paused":
                        update_task(task.id, state="running", phase="agent-loop")
                    finish_control(control.id, self.owner,
                                   payload={"applied_at_boundary": True})
                applied.append(control)
            except Exception as exc:
                finish_control(control.id, self.owner, success=False,
                               payload={"error": str(exc)})
                raise
            if stop:
                break
        return applied, stop, None

    @staticmethod
    def _signature(decision):
        return hashlib.sha256(json.dumps({"tool": decision.tool,
                                          "arguments": decision.arguments},
                                         sort_keys=True, default=str).encode()).hexdigest()

    def run(self):
        task = load_task(self.task_id)
        if task.state in {"completed", "cancelled"}:
            raise RuntimeError(f"task {task.id} is {task.state}")
        if not claim("task-execution", task.id, self.owner, lease_seconds=30,
                     metadata={"engine": "agent-loop"}):
            raise RuntimeError(f"task {task.id} already has a live execution owner")
        try:
            # Reconcile dead control owners even when task execution itself is
            # ambiguous and must fail closed rather than replay.
            recover_processing(task.id, self.owner)
            recover_cancellations(task.id, self.owner)
            if task.state == "running":
                update_task(task.id, state="failed", phase="execution-owner-lost",
                            failures=[*task.failures,
                                      "previous execution outcome is ambiguous"])
                raise RuntimeError(
                    f"task {task.id} lost its previous execution owner; "
                    "automatic replay is unsafe")
            with _ACTIVE_LOOPS_LOCK:
                if task.id in _ACTIVE_LOOPS:
                    raise RuntimeError(f"task {task.id} already has an active agent loop")
                _ACTIVE_LOOPS[task.id] = self
            with Heartbeat("task-execution", task.id, self.owner, lease_seconds=30):
                return self._run_registered(task)
        finally:
            with _ACTIVE_LOOPS_LOCK:
                if _ACTIVE_LOOPS.get(task.id) is self:
                    del _ACTIVE_LOOPS[task.id]
            release_lease("task-execution", task.id, self.owner)

    def _run_registered(self, task):
        update_task(task.id, state="running", phase="agent-loop")
        started = self.clock()
        results = []
        repetitions = {}
        no_progress = tool_failures = 0
        checkpoint_id = None
        for iteration in range(1, self.limits.max_iterations + 1):
            if self.clock() - started > self.limits.max_seconds:
                return self._stop("paused", "time budget exhausted", iteration - 1, results)
            try:
                controls, stop, blocked = self._apply_controls()
            except Exception as exc:
                return self._stop("failed", f"control application failed: {exc}",
                                  iteration - 1, results, checkpoint_id)
            if blocked is not None:
                return self._stop(
                    "paused", "cancellation reconciliation pending",
                    iteration - 1, results, checkpoint_id)
            if stop:
                return LoopOutcome(stop, f"{stop} by user control", iteration - 1, tuple(results))
            task = load_task(task.id)
            try:
                produced = self.model(task, pending_context(task.id))
            except Exception as exc:
                return self._stop("failed", f"model invocation failed: {exc}",
                                  iteration - 1, results, checkpoint_id)
            if isinstance(produced, tuple) and len(produced) == 2:
                raw_decision, pressure = produced
            else:
                raw_decision, pressure = produced, 0.0
            if pressure >= self.limits.context_pressure_limit:
                return self._stop("paused", "context pressure guard", iteration - 1, results)
            try:
                decision = ModelDecision.parse(raw_decision)
            except Exception as exc:
                return self._stop("failed", f"invalid model decision: {exc}",
                                  iteration, results, checkpoint_id)
            try:
                boundary_controls, stop, blocked = self._apply_controls()
            except Exception as exc:
                return self._stop("failed", f"control application failed: {exc}",
                                  iteration, results, checkpoint_id)
            if blocked is not None:
                return self._stop(
                    "paused", "cancellation reconciliation pending",
                    iteration, results, checkpoint_id)
            if stop:
                return LoopOutcome(stop, f"{stop} by user control", iteration - 1,
                                   tuple(results), checkpoint_id)
            if boundary_controls:
                append_event(task.id, "status", "Model decision superseded by user control",
                             role=task.owner_role,
                             data={"controls": [item.id for item in boundary_controls]},
                             visibility="verbose")
                continue
            append_state_event("model_invocation", decision.kind, task_id=task.id,
                               role=task.owner_role,
                               payload={"iteration": iteration, "decision": decision.kind,
                                        "tool": decision.tool})
            if decision.kind == "tool":
                if not decision.tool:
                    return self._stop("failed", "tool decision omitted tool name", iteration, results)
                signature = self._signature(decision)
                repetitions[signature] = repetitions.get(signature, 0) + 1
                if repetitions[signature] > self.limits.max_repetitions:
                    return self._stop("paused", "repetition guard", iteration, results)
                append_event(task.id, "tool", f"Proposed {decision.tool}", role=task.owner_role,
                             data={"tool": decision.tool, "arguments": decision.arguments})
                try:
                    binding = self.tools.binding(decision.tool)
                except KeyError as exc:
                    append_event(task.id, "error", str(exc), role=task.owner_role,
                                 data={"tool": decision.tool})
                    tool_failures += 1
                    if tool_failures >= self.limits.max_tool_failures:
                        return self._stop("paused", "tool failure guard", iteration, results,
                                          checkpoint_id)
                    continue
                if binding.before_execute is not None:
                    try:
                        hook_arguments = dict(decision.arguments)
                        signature = inspect.signature(binding.before_execute).parameters
                        if "task_id" in signature:
                            hook_arguments["task_id"] = task.id
                        checkpoint_result = binding.before_execute(**hook_arguments)
                        if not isinstance(checkpoint_result, ToolResult):
                            raise TypeError("pre-execution checkpoint must return ToolResult")
                        if not checkpoint_result.succeeded:
                            return self._stop("paused", "workspace checkpoint failed",
                                              iteration, results, checkpoint_id)
                        results.append(checkpoint_result)
                        refs = tuple(dict.fromkeys((*load_task(task.id).evidence_refs,
                                                    *checkpoint_result.evidence_ids)))
                        update_task(task.id, evidence_refs=refs,
                                    phase="workspace-checkpointed")
                        checkpoint_id = checkpoint_result.data.get("checkpoint_id")
                    except Exception as exc:
                        append_event(task.id, "error", f"Workspace checkpoint failed: {exc}",
                                     role=task.owner_role,
                                     data={"tool": decision.tool})
                        return self._stop("paused", "workspace checkpoint failed",
                                          iteration, results, checkpoint_id)
                with self._active_lock:
                    self._active_binding, self._active_tool = binding, decision.tool
                    self._active_operation_id = "operation-" + uuid.uuid4().hex
                try:
                    result = self.tools.execute(decision.tool, decision.arguments,
                                                task_id=task.id)
                except Exception as exc:
                    append_event(task.id, "error", f"{decision.tool}: {exc}",
                                 role=task.owner_role,
                                 data={"tool": decision.tool, "exception": type(exc).__name__})
                    tool_failures += 1
                    no_progress += 1
                    if tool_failures >= self.limits.max_tool_failures:
                        return self._stop("paused", "tool failure guard", iteration, results,
                                          checkpoint_id)
                    continue
                finally:
                    with self._active_lock:
                        self._active_binding, self._active_tool = None, ""
                        self._active_operation_id = ""
                results.append(result)
                append_event(task.id, "tool", f"{decision.tool}: {result.state}",
                             role=task.owner_role,
                             data={"tool": result.tool, "state": result.state,
                                   "error": result.error, "evidence_ids": result.evidence_ids})
                if task.conversation_id:
                    add_message(task.conversation_id, "tool", json.dumps({
                        "tool": result.tool, "state": result.state, "data": result.data,
                        "error": result.error, "evidence_ids": result.evidence_ids,
                    }, ensure_ascii=False, default=str)[:64_000], kind="tool-result",
                        related_task_id=task.id, metadata={"unresolved": False})
                refs = tuple(dict.fromkeys((*load_task(task.id).evidence_refs,
                                            *result.evidence_ids)))
                update_task(task.id, evidence_refs=refs, phase="verifying-progress")
                checkpoint = create_checkpoint(
                    task.id, canonical_task_state(task.id),
                    reason=f"agent boundary after {decision.tool}", evidence_refs=refs,
                )
                checkpoint_id = checkpoint.id
                if result.succeeded:
                    no_progress = 0
                else:
                    tool_failures += 1
                    no_progress += 1
                    if not binding.retry_safe:
                        return self._stop("paused", "unsafe retry guard", iteration, results,
                                          checkpoint_id)
                    if tool_failures >= self.limits.max_tool_failures:
                        return self._stop("paused", "tool failure guard", iteration, results,
                                          checkpoint_id)
                _, boundary_stop, blocked = self._apply_controls()
                if blocked is not None:
                    return self._stop(
                        "paused", "cancellation reconciliation pending",
                        iteration, results, checkpoint_id)
                if boundary_stop:
                    return LoopOutcome(
                        boundary_stop, f"{boundary_stop} at safe tool boundary",
                        iteration, tuple(results), checkpoint_id)
                current = load_task(task.id)
                if current.state in {"paused", "cancelled"}:
                    return LoopOutcome(current.state, f"{current.state} at safe tool boundary",
                                       iteration, tuple(results), checkpoint_id)
                continue
            if decision.kind == "continue":
                no_progress += 1
                if task.conversation_id and decision.summary:
                    add_message(task.conversation_id, "assistant", decision.summary,
                                kind="agent-progress", related_task_id=task.id)
            else:
                verified, reason = self.completion.verify(decision, results, task.id)
                if verified:
                    refs = tuple(dict.fromkeys((*load_task(task.id).evidence_refs,
                                                *decision.evidence_ids)))
                    update_task(task.id, state="completed", phase="completed", progress=1.0,
                                evidence_refs=refs)
                    checkpoint = create_checkpoint(
                        task.id, canonical_task_state(task.id), reason="verified completion",
                        evidence_refs=refs,
                    )
                    append_event(task.id, "result", decision.summary, role=task.owner_role,
                                 data={"verified": True, "reason": reason,
                                       "evidence_ids": list(refs)})
                    return LoopOutcome("completed", reason, iteration, tuple(results), checkpoint.id)
                no_progress += 1
                append_event(task.id, "status", "Completion rejected", role=task.owner_role,
                             data={"verified": False, "reason": reason})
                if task.conversation_id:
                    add_message(task.conversation_id, "tool", f"Completion rejected: {reason}",
                                kind="tool-result", related_task_id=task.id,
                                metadata={"unresolved": True})
            if no_progress >= self.limits.max_no_progress:
                return self._stop("paused", "no-progress guard", iteration, results,
                                  checkpoint_id)
        return self._stop("paused", "iteration budget exhausted", self.limits.max_iterations,
                          results, checkpoint_id)

    def _stop(self, state, reason, iterations, results, checkpoint_id=None):
        task = load_task(self.task_id)
        failures = task.failures
        if state == "failed":
            failures = (*failures, reason)
        update_task(task.id, state=state, phase=reason, failures=failures)
        append_event(task.id, "status" if state == "paused" else "error", reason,
                     role=task.owner_role, data={"guard": reason})
        return LoopOutcome(state, reason, iterations, tuple(results), checkpoint_id)
