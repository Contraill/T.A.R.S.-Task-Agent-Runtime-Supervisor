from __future__ import annotations

from dataclasses import dataclass
import uuid

from .checkpoints import create_checkpoint
from .context import ContextManager
from .conversation import active_conversation, add_message, create_conversation
from .events import append_event
from .roles import get_role
from .runtime import chat_completion_stream
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction
from .tasks import attach_conversation, canonical_task_state, load_task, update_task

RUN_STATES = {"queued", "running", "paused", "completed", "failed", "cancelled"}
CONTROL_ACTIONS = {"pause", "resume", "cancel", ""}


@dataclass(frozen=True)
class TaskRun:
    id: str
    task_id: str
    conversation_id: str | None
    role_id: str
    state: str
    control_request: str
    epoch: int
    created_at: str
    started_at: str | None
    heartbeat_at: str | None
    finished_at: str | None
    finish_reason: str | None
    error: str | None
    output_message_id: str | None
    metadata: dict


def _from_row(row) -> TaskRun:
    return TaskRun(
        id=row["id"], task_id=row["task_id"], conversation_id=row["conversation_id"],
        role_id=row["role_id"], state=row["state"], control_request=row["control_request"],
        epoch=row["epoch"], created_at=row["created_at"], started_at=row["started_at"],
        heartbeat_at=row["heartbeat_at"], finished_at=row["finished_at"],
        finish_reason=row["finish_reason"], error=row["error"],
        output_message_id=row["output_message_id"],
        metadata=json_loads(row["metadata_json"], {}),
    )


def create_run(task_id: str, conversation_id: str | None = None) -> TaskRun:
    ensure_state_store()
    task = load_task(task_id)
    if task.state in {"completed", "cancelled"}:
        raise RuntimeError(f"task {task.id} is {task.state}")
    if conversation_id is None:
        conversation_id = task.conversation_id
    if conversation_id is None:
        conv = active_conversation() or create_conversation(
            title=f"Task {task.id}", source="task-runner", make_active=False
        )
        conversation_id = conv.id
    if task.conversation_id is None:
        task = attach_conversation(task.id, conversation_id)
    run_id = "run-" + uuid.uuid4().hex
    now = now_utc()
    with transaction(immediate=True) as conn:
        existing = conn.execute(
            "SELECT id FROM task_runs WHERE task_id=? AND state IN ('queued','running') LIMIT 1",
            (task.id,),
        ).fetchone()
        if existing:
            raise RuntimeError(f"task {task.id} already has active run {existing['id']}")
        conn.execute(
            """
            INSERT INTO task_runs(
                id,task_id,conversation_id,role_id,state,control_request,epoch,
                created_at,metadata_json
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (run_id, task.id, conversation_id, task.owner_role, "queued", "", task.epoch, now, "{}"),
        )
        row = conn.execute("SELECT * FROM task_runs WHERE id=?", (run_id,)).fetchone()
    append_event(task.id, "status", f"Runner queued {run_id}", role=task.owner_role,
                 data={"run_id": run_id, "state": "queued"}, visibility="verbose")
    return _from_row(row)


def load_run(run_id: str) -> TaskRun:
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM task_runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown run: {run_id}")
        return _from_row(row)
    finally:
        conn.close()


def list_runs(task_id: str | None = None, limit=50) -> list[TaskRun]:
    ensure_state_store()
    conn = connect()
    try:
        if task_id:
            rows = conn.execute(
                "SELECT * FROM task_runs WHERE task_id=? ORDER BY created_at DESC LIMIT ?",
                (task_id, int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM task_runs ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [_from_row(row) for row in rows]
    finally:
        conn.close()


def active_run(task_id: str) -> TaskRun | None:
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM task_runs WHERE task_id=? AND state IN ('queued','running','paused') "
            "ORDER BY created_at DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        return _from_row(row) if row else None
    finally:
        conn.close()


def _set_run(run_id: str, **values) -> TaskRun:
    if not values:
        return load_run(run_id)
    with transaction(immediate=True) as conn:
        values["heartbeat_at"] = values.get("heartbeat_at") or now_utc()
        assignments = ", ".join(f"{key}=?" for key in values)
        conn.execute(
            f"UPDATE task_runs SET {assignments} WHERE id=?",
            (*values.values(), run_id),
        )
    return load_run(run_id)


def request_control(task_id: str, action: str) -> TaskRun | None:
    action = str(action).strip().lower()
    if action not in {"pause", "resume", "cancel"}:
        raise ValueError("control action must be pause, resume or cancel")
    run = active_run(task_id)
    if run is None:
        # No live runner: preserve intuitive task-state controls.
        if action == "pause":
            update_task(task_id, state="paused", phase="paused")
        elif action == "resume":
            update_task(task_id, state="pending", phase="resume-requested")
        else:
            update_task(task_id, state="cancelled", phase="cancelled")
        return None
    if action == "resume":
        if run.state == "paused":
            _set_run(
                run.id, state="completed", control_request="",
                finished_at=run.finished_at or now_utc(), finish_reason=run.finish_reason or "paused",
            )
            update_task(task_id, state="pending", phase="resume-requested")
            append_event(
                task_id, "status", "Resume requested; task is ready for a new runner epoch",
                role=run.role_id, data={"previous_run_id": run.id}, visibility="normal",
            )
            return load_run(run.id)
        raise RuntimeError(f"task {task_id} already has {run.state} run {run.id}")

    _set_run(run.id, control_request=action)
    append_event(
        task_id, "status", f"{action} requested; will apply at the next safe boundary",
        role=run.role_id, data={"run_id": run.id, "control_request": action},
        visibility="normal",
    )
    return load_run(run.id)


def pending_control(run_id: str) -> str:
    return load_run(run_id).control_request


def _apply_boundary_control(run: TaskRun) -> TaskRun:
    request = pending_control(run.id)
    if request == "cancel":
        _set_run(run.id, state="cancelled", control_request="", finished_at=now_utc(), finish_reason="cancelled")
        update_task(run.task_id, state="cancelled", phase="cancelled")
        append_event(run.task_id, "status", "Task cancelled at safe boundary", role=run.role_id)
        return load_run(run.id)
    if request == "pause":
        _set_run(run.id, state="paused", control_request="", finished_at=now_utc(), finish_reason="paused")
        update_task(run.task_id, state="paused", phase="paused")
        append_event(run.task_id, "status", "Task paused at safe boundary", role=run.role_id)
        return load_run(run.id)
    return run


def run_task_epoch(cfg, run_id: str, *, on_stream=None, max_tokens=None) -> dict:
    """Execute one reasoning-only task epoch using real backend streaming.

    This is intentionally *not* the future ToolRegistry agent loop.  It produces one
    model epoch from canonical task state, persists the genuine final model output as
    an event/message and checkpoints the resulting durable task state.  No file/tool
    action is simulated.
    """
    run = load_run(run_id)
    task = load_task(run.task_id)
    role = get_role(task.owner_role)
    if run.state not in {"queued", "paused"}:
        raise RuntimeError(f"run {run.id} is {run.state}")

    # Resume clears a previous boundary pause; queued runs simply begin.
    now = now_utc()
    run = _set_run(
        run.id, state="running", control_request="", started_at=run.started_at or now,
        finished_at=None, finish_reason=None, error=None,
    )
    update_task(task.id, state="running", phase="reasoning-epoch")
    append_event(
        task.id, "progress", f"{role.display_name} reasoning epoch {task.epoch} started",
        role=task.owner_role, data={"run_id": run.id, "epoch": task.epoch},
        visibility="normal",
    )

    # Safe-boundary control before expensive inference.
    controlled = _apply_boundary_control(load_run(run.id))
    if controlled.state in {"paused", "cancelled"}:
        return {"run": controlled, "content": "", "reasoning": "", "finish_reason": controlled.finish_reason}

    manager = ContextManager(cfg)
    projection = manager.build(
        run.conversation_id,
        task.owner_role,
        mode="task",
        task_id_override=task.id,
        requested_output_tokens=max_tokens,
        exact=True,
    )

    content_parts = []
    reasoning_parts = []
    finish_reason = None
    usage = None
    try:
        for event in chat_completion_stream(
            cfg, task.owner_role, list(projection.messages), max_tokens=max_tokens,
            input_tokens=projection.token_count, operation="task", task_active=True
        ):
            content = event.get("content") or ""
            reasoning = event.get("reasoning") or ""
            if content:
                content_parts.append(content)
            if reasoning:
                reasoning_parts.append(reasoning)
            if event.get("finish_reason") is not None:
                finish_reason = event.get("finish_reason")
            if event.get("usage"):
                usage = event.get("usage")
            _set_run(run.id, heartbeat_at=now_utc())
            if on_stream:
                on_stream(event)

        final = "".join(content_parts)
        raw_reasoning = "".join(reasoning_parts)
        if not final:
            final = f"[no final content · {finish_reason or 'unknown'}]"
        if finish_reason == "length" and not "".join(content_parts):
            raise RuntimeError("generation exhausted its context-bounded ceiling before final content")

        # Inference completion is a safe boundary.  A cancellation requested while
        # tokens were streaming is honored here before the generated output is
        # promoted into canonical conversation/task state.
        control = pending_control(run.id)
        if control == "cancel":
            _set_run(
                run.id, state="cancelled", control_request="", finished_at=now_utc(),
                finish_reason="cancelled-at-boundary",
                metadata_json=json_dumps({"discarded_output_chars": len(final), "reasoning_chars": len(raw_reasoning)}),
            )
            update_task(task.id, state="cancelled", phase="cancelled")
            append_event(
                task.id, "status", "Cancellation applied at inference boundary; generated output was not promoted",
                role=task.owner_role, data={"run_id": run.id}, visibility="normal",
            )
            return {
                "run": load_run(run.id), "content": "", "reasoning": raw_reasoning,
                "finish_reason": "cancelled-at-boundary", "projection": projection,
            }

        message = add_message(
            run.conversation_id, "assistant", final,
            kind="task-run", include_in_context=True, related_task_id=task.id,
            metadata={
                "role_id": task.owner_role,
                "run_id": run.id,
                "context_projection_id": projection.id,
                "finish_reason": finish_reason,
                "prompt_tokens": projection.token_count,
                "prompt_tokens_exact": projection.exact,
            },
        )
        append_event(
            task.id, "result", final, role=task.owner_role,
            data={
                "run_id": run.id,
                "epoch": task.epoch,
                "finish_reason": finish_reason,
                "reasoning_chars": len(raw_reasoning),
                "usage": usage or {},
            },
            visibility="normal",
        )

        # A reasoning-only epoch cannot honestly declare verified task completion.
        # Persist the result, advance epoch, then pause for user/tool-loop continuation.
        cp_state = canonical_task_state(task.id)
        cp_state["last_runner_result"] = {
            "run_id": run.id,
            "message_id": message.id,
            "finish_reason": finish_reason,
        }
        cp = create_checkpoint(
            task.id, cp_state, reason=f"runner epoch {task.epoch} complete",
            evidence_refs=task.evidence_refs, advance_epoch=True,
        )
        update_task(task.id, state="paused", phase="epoch-complete")
        run = _set_run(
            run.id, state="completed", control_request="", finished_at=now_utc(),
            finish_reason=finish_reason or "stop", output_message_id=message.id,
            metadata_json=json_dumps({"reasoning_chars": len(raw_reasoning), "usage": usage or {}}),
        )
        return {
            "run": run,
            "content": final,
            "reasoning": raw_reasoning,
            "finish_reason": finish_reason,
            "projection": projection,
            "checkpoint": cp,
        }
    except Exception as exc:
        _set_run(run.id, state="failed", finished_at=now_utc(), error=str(exc), finish_reason="error")
        update_task(task.id, state="failed", phase="runner-error", failures=[*task.failures, str(exc)])
        append_event(task.id, "error", str(exc), role=task.owner_role,
                     data={"run_id": run.id}, visibility="quiet")
        raise
