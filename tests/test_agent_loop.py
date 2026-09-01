import multiprocessing
from pathlib import Path
import threading
import time

import pytest

from tars import (agent_loop, control_queue, conversation, evidence, ownership,
                  runner, state_store, tasks)
from tars.cli import build_parser
from tars.tool_core import ToolResult


@pytest.fixture
def loop_state(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    monkeypatch.setattr(tasks, "resolve_role_id", lambda value: value)
    conv = conversation.create_conversation(make_active=False)
    task = tasks.create_task("verify work", "general", conversation_id=conv.id)
    return task, conv


def result_for(task_id, *, tool="fs.write", state="succeeded", error=""):
    record = evidence.record("filesystem", "/workspace/file", "diff", task_id=task_id)
    return ToolResult(tool, state, {"verified": state == "succeeded"}, error=error,
                      action_ids=("action-real",), evidence_ids=(record.id,))


def _configure_process_state(database, scratch):
    state_store.STATE_DB_PATH = Path(database)
    state_store.TASK_ROOT = Path(scratch) / "legacy"
    state_store.TASK_EVENTS_ROOT = Path(scratch) / "events"
    state_store.TASK_INDEX_PATH = Path(scratch) / "index"


def _hold_agent_loop(database, scratch, task_id, ready, release, crash=False):
    _configure_process_state(database, scratch)
    def model(task, controls):
        ready.set()
        if crash:
            import os
            os._exit(17)
        release.wait(10)
        return {"type": "finish", "summary": "done"}
    agent_loop.AgentLoop(
        task_id, model, agent_loop.ToolDispatcher(),
        completion=agent_loop.CompletionContract(require_evidence=False),
    ).run()


def _claim_control_and_exit(database, scratch, task_id):
    _configure_process_state(database, scratch)
    owner = ownership.Owner.create("control-worker")
    assert control_queue.claim_next(task_id, owner) is not None


def _strand_cancellation(database, scratch, task_id, attempted, marker, connection):
    _configure_process_state(database, scratch)
    owner = ownership.Owner.create("cancel-worker")
    control = control_queue.enqueue_cancellation(
        task_id, "interrupt", "stop", active_tool="terminal.run",
        cancellable=True, operation_id="operation-crash", owner=owner,
    )
    if attempted:
        control_queue.begin_cancellation(control.id, owner)
        Path(marker).write_text("external cancellation may have happened", encoding="utf-8")
    connection.send(control.id)
    connection.close()
    import os
    os._exit(23)


def _recover_cancellations_in_process(database, scratch, task_id, connection):
    _configure_process_state(database, scratch)
    recovered = control_queue.recover_cancellations(
        task_id, ownership.Owner.create("cancel-recovery"))
    connection.send(recovered)
    connection.close()


def _attempt_task_execution_in_process(database, scratch, task_id, connection):
    _configure_process_state(database, scratch)
    try:
        with ownership.task_execution_scope(task_id, engine="contender"):
            connection.send(("entered", ""))
    except Exception as exc:
        connection.send(("blocked", str(exc)))
    finally:
        connection.close()


def _wait_for_cancellation_state(control_id, expected, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = control_queue.cancellation_state(control_id)
        if state and state["state"] == expected:
            return state
        threading.Event().wait(0.01)
    raise AssertionError(f"cancellation {control_id} did not reach {expected}")


def test_agent_loop_executes_real_tool_result_and_requires_completion_evidence(loop_state):
    task, _ = loop_state
    calls = iter((
        {"type": "tool", "tool": "fs.write", "arguments": {"content": "value"}},
        {"type": "finish", "summary": "verified"},
    ))
    dispatcher = agent_loop.ToolDispatcher().register(
        "fs.write", lambda content, task_id=None: result_for(task_id),
    )
    outcome = agent_loop.AgentLoop(
        task.id, lambda current, controls: next(calls), dispatcher,
        completion=agent_loop.CompletionContract(required_tools=("fs.write",),
                                                  required_evidence_types=("filesystem",)),
    ).run()
    assert outcome.state == "completed" and outcome.tool_results[0].succeeded
    assert tasks.load_task(task.id).state == "completed" and outcome.checkpoint_id


def test_agent_loop_has_one_live_owner_across_processes(loop_state):
    task, _ = loop_state
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_hold_agent_loop,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              task.id, ready, release),
    )
    process.start()
    assert ready.wait(5)
    contender = agent_loop.AgentLoop(
        task.id, lambda task, controls: {"type": "finish", "summary": "duplicate"},
        agent_loop.ToolDispatcher(),
        completion=agent_loop.CompletionContract(require_evidence=False),
    )
    with pytest.raises(RuntimeError, match="live execution owner"):
        contender.run()
    release.set(); process.join(timeout=10)
    assert process.exitcode == 0


def test_same_agent_loop_instance_cannot_borrow_its_owner_concurrently(loop_state):
    task, _ = loop_state
    entered, release = threading.Event(), threading.Event()

    def model(current, controls):
        entered.set()
        assert release.wait(5)
        return {"type": "finish", "summary": "done"}

    loop = agent_loop.AgentLoop(
        task.id, model, agent_loop.ToolDispatcher(),
        completion=agent_loop.CompletionContract(require_evidence=False),
    )
    holder = {}
    thread = threading.Thread(target=lambda: holder.setdefault("outcome", loop.run()))
    thread.start()
    assert entered.wait(5)
    with pytest.raises(RuntimeError, match="active agent loop"):
        loop.run()
    release.set()
    thread.join(5)
    assert not thread.is_alive() and holder["outcome"].state == "completed"


def test_dead_agent_loop_owner_is_not_automatically_replayed(loop_state):
    task, _ = loop_state
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_hold_agent_loop,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              task.id, ready, release, True),
    )
    process.start()
    assert ready.wait(5)
    process.join(timeout=10)
    assert process.exitcode == 17
    called = []
    contender = agent_loop.AgentLoop(
        task.id,
        lambda task, controls: called.append(True) or {
            "type": "finish", "summary": "duplicate"},
        agent_loop.ToolDispatcher(),
        completion=agent_loop.CompletionContract(require_evidence=False),
    )
    with pytest.raises(RuntimeError, match="automatic replay is unsafe"):
        contender.run()
    assert called == []
    assert tasks.load_task(task.id).state == "failed"


def test_model_completion_without_evidence_is_rejected_by_no_progress_guard(loop_state):
    task, _ = loop_state
    loop = agent_loop.AgentLoop(
        task.id, lambda current, controls: {"type": "finish", "summary": "trust me"},
        agent_loop.ToolDispatcher(),
        limits=agent_loop.LoopLimits(max_iterations=5, max_no_progress=2),
    )
    outcome = loop.run()
    assert outcome.state == "paused" and outcome.reason == "no-progress guard"
    assert tasks.load_task(task.id).state == "paused"


def test_queued_messages_preserve_order_and_reach_next_inference(loop_state):
    task, conv = loop_state
    started = threading.Event()
    release = threading.Event()
    observations = []

    def slow(task_id=None):
        started.set()
        assert release.wait(5)
        return result_for(task_id, tool="terminal.run")

    decisions = iter((
        {"type": "tool", "tool": "terminal.run", "arguments": {}},
        {"type": "finish", "summary": "done"},
    ))
    def model(current, controls):
        observations.append([item.content for item in conversation.list_messages(conv.id)
                             if item.kind == "control"])
        return next(decisions)

    loop = agent_loop.AgentLoop(task.id, model,
                                agent_loop.ToolDispatcher().register("terminal.run", slow))
    holder = {}
    thread = threading.Thread(target=lambda: holder.setdefault("outcome", loop.run()))
    thread.start()
    assert started.wait(5)
    first, first_feedback = loop.submit_control("message", "first")
    second, second_feedback = loop.submit_control("message", "second")
    assert first_feedback == second_feedback == agent_loop.QUEUED_MESSAGE
    release.set()
    thread.join(5)
    assert not thread.is_alive() and holder["outcome"].state == "completed"
    assert observations[-1] == ["first", "second"]
    assert [item.state for item in control_queue.list_controls(task.id)] == ["applied", "applied"]


@pytest.mark.parametrize("cancellable", [True, False])
def test_immediate_interrupt_records_actual_cancellation_truth(loop_state, cancellable):
    task, conv = loop_state
    started = threading.Event()
    release = threading.Event()
    cancel_called = []

    def execute(task_id=None):
        started.set()
        assert release.wait(5)
        return result_for(task_id, tool="terminal.run",
                          state="failed" if cancellable else "succeeded",
                          error="cancelled" if cancellable else "")

    def cancel():
        cancel_called.append(True)
        release.set()
        return {"requested": True, "signal": "TERM"}

    decisions = iter((
        {"type": "tool", "tool": "terminal.run", "arguments": {}},
        {"type": "finish", "summary": "redirected after interrupt"},
    ))
    dispatcher = agent_loop.ToolDispatcher().register(
        "terminal.run", execute, cancel=cancel if cancellable else None,
        retry_safe=True,
    )
    loop = agent_loop.AgentLoop(
        task.id, lambda current, controls: next(decisions), dispatcher,
        completion=agent_loop.CompletionContract(require_evidence=False),
    )
    holder = {}
    thread = threading.Thread(target=lambda: holder.setdefault("outcome", loop.run()))
    thread.start()
    assert started.wait(5)
    control, _ = loop.submit_control("interrupt", "stop and inspect")
    if not cancellable:
        release.set()
    thread.join(5)
    assert not thread.is_alive()
    recorded = control_queue.load(control.id)
    assert recorded.state == "applied"
    assert recorded.payload.get("cancellable") is cancellable
    assert recorded.payload["cancellation_phase"] == "resolved"
    assert recorded.payload["cancellation_requested"] is cancellable
    assert bool(cancel_called) is cancellable
    messages = [item.content for item in conversation.list_messages(conv.id)
                if item.kind == "control"]
    assert messages == ["stop and inspect"]


def test_worker_cannot_terminalize_or_duplicate_cancel_before_reconciliation(loop_state):
    task, _ = loop_state
    started = threading.Event()
    release_tool = threading.Event()
    tool_returned = threading.Event()
    cancel_entered = threading.Event()
    allow_cancel_return = threading.Event()
    cancel_calls = []
    model_calls = []

    def execute(task_id=None):
        started.set()
        assert release_tool.wait(5)
        tool_returned.set()
        return result_for(task_id, tool="terminal.run", state="failed", error="cancelled")

    def cancel():
        cancel_calls.append(True)
        cancel_entered.set()
        release_tool.set()
        assert allow_cancel_return.wait(5)
        return {"requested": True, "sensitive_backend_detail": "must-not-persist"}

    decisions = iter((
        {"type": "tool", "tool": "terminal.run", "arguments": {}},
        {"type": "finish", "summary": "done"},
    ))

    def model(current, controls):
        model_calls.append(True)
        return next(decisions)

    loop = agent_loop.AgentLoop(
        task.id, model,
        agent_loop.ToolDispatcher().register(
            "terminal.run", execute, cancel=cancel, retry_safe=True),
        limits=agent_loop.LoopLimits(cancellation_reconcile_seconds=5),
        completion=agent_loop.CompletionContract(require_evidence=False),
    )
    holder = {}
    thread = threading.Thread(target=lambda: holder.setdefault("outcome", loop.run()))
    thread.start()
    assert started.wait(5)
    control, _ = loop.submit_control("interrupt", "first interrupt")
    assert cancel_entered.wait(5)
    assert tool_returned.wait(5)

    duplicate, _ = loop.submit_control("interrupt", "duplicate interrupt")
    assert control_queue.cancellation_state(control.id)["state"] == "attempting"
    assert control_queue.load(control.id).state == "pending"
    assert control_queue.cancellation_state(duplicate.id)["state"] == "resolved"
    assert duplicate.payload["cancellation_outcome"] == "already-requested"
    assert cancel_calls == [True]
    assert len(model_calls) == 1

    allow_cancel_return.set()
    thread.join(5)
    assert not thread.is_alive() and holder["outcome"].state == "completed"
    recorded = control_queue.load(control.id)
    assert recorded.state == "applied"
    assert recorded.payload["cancellation_outcome"] == "request-dispatched"
    assert recorded.payload["cancellation_result"] == {"requested": True}
    assert "must-not-persist" not in str(recorded.payload)
    assert control_queue.load(duplicate.id).state == "applied"
    assert cancel_calls == [True]


def test_blocking_cancel_pauses_loop_without_advancing_and_can_reconcile_later(loop_state):
    task, _ = loop_state
    started = threading.Event()
    release_tool = threading.Event()
    cancel_entered = threading.Event()
    allow_cancel_return = threading.Event()
    model_calls = []

    def execute(task_id=None):
        started.set()
        assert release_tool.wait(5)
        return result_for(task_id, tool="terminal.run", state="failed", error="cancelled")

    def cancel():
        cancel_entered.set()
        release_tool.set()
        assert allow_cancel_return.wait(5)
        return True

    def model(current, controls):
        model_calls.append(True)
        return {"type": "tool", "tool": "terminal.run", "arguments": {}}

    loop = agent_loop.AgentLoop(
        task.id, model,
        agent_loop.ToolDispatcher().register(
            "terminal.run", execute, cancel=cancel, retry_safe=True),
        limits=agent_loop.LoopLimits(cancellation_reconcile_seconds=0.01),
    )
    holder = {}
    thread = threading.Thread(target=lambda: holder.setdefault("outcome", loop.run()))
    thread.start()
    assert started.wait(5)
    control, _ = loop.submit_control("interrupt", "stop")
    assert cancel_entered.wait(5)
    thread.join(5)
    assert not thread.is_alive()
    assert holder["outcome"].reason == "cancellation reconciliation pending"
    assert len(model_calls) == 1
    assert control_queue.load(control.id).state == "pending"
    assert control_queue.cancellation_state(control.id)["state"] == "attempting"

    allow_cancel_return.set()
    _wait_for_cancellation_state(control.id, "resolved")
    resumed = agent_loop.AgentLoop(
        task.id, lambda current, controls: {"type": "finish", "summary": "done"},
        agent_loop.ToolDispatcher(),
        completion=agent_loop.CompletionContract(require_evidence=False),
    ).run()
    assert resumed.state == "completed"
    assert control_queue.load(control.id).state == "applied"


def test_hung_cancellation_fences_every_executor_and_close_is_truthful(loop_state):
    task, _ = loop_state
    tool_started = threading.Event()
    release_tool = threading.Event()
    cancel_entered = threading.Event()
    release_cancel = threading.Event()
    cancel_calls = []

    def execute(task_id=None):
        tool_started.set()
        assert release_tool.wait(5)
        return result_for(task_id, tool="terminal.run", state="failed", error="cancelled")

    def cancel():
        cancel_calls.append(True)
        cancel_entered.set()
        release_tool.set()
        release_cancel.wait()
        return True

    loop = agent_loop.AgentLoop(
        task.id,
        lambda *_: {"type": "tool", "tool": "terminal.run", "arguments": {}},
        agent_loop.ToolDispatcher().register(
            "terminal.run", execute, cancel=cancel, retry_safe=True),
        limits=agent_loop.LoopLimits(cancellation_reconcile_seconds=0.01),
    )
    holder = {}
    worker = threading.Thread(target=lambda: holder.setdefault("outcome", loop.run()))
    worker.start()
    assert tool_started.wait(5)
    control, _ = loop.submit_control("interrupt", "stop")
    assert cancel_entered.wait(5)
    worker.join(5)
    assert not worker.is_alive()
    assert holder["outcome"].reason == "cancellation reconciliation pending"
    assert not ownership.active("task-execution", task.id)
    assert agent_loop.is_task_loop_active(task.id)
    with state_store.connect() as conn:
        assert conn.execute(
            "SELECT 1 FROM resource_leases "
            "WHERE resource_type='task-execution-fence' AND resource_key=?",
            (task.id,),
        ).fetchone()
    assert not ownership.claim(
        "task-execution", task.id, ownership.Owner.create("delegation-contender"))

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    contender = context.Process(
        target=_attempt_task_execution_in_process,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              task.id, sender),
    )
    contender.start()
    state, _ = receiver.recv()
    contender.join(timeout=10)
    assert contender.exitcode == 0 and state == "blocked"
    with pytest.raises(RuntimeError, match="live execution owner"):
        runner.create_run(task.id)

    assert loop.close(timeout=0.01) is False
    assert control_queue.cancellation_state(control.id)["state"] == "ambiguous"
    failed = tasks.load_task(task.id)
    assert failed.state == "failed"
    assert failed.phase == control_queue.CANCELLATION_RECOVERY_PHASE
    with pytest.raises(RuntimeError, match="explicit recovery"):
        agent_loop.AgentLoop(
            task.id, lambda *_: pytest.fail("model must remain fenced"),
            agent_loop.ToolDispatcher(),
        ).run()
    assert cancel_calls == [True]

    release_cancel.set()
    assert loop.close(timeout=5) is True
    assert cancel_calls == [True]


def test_close_cannot_overtake_cancellation_worker_admission(loop_state, monkeypatch):
    task, _ = loop_state
    enqueue_entered = threading.Event()
    release_enqueue = threading.Event()
    callback_called = threading.Event()
    close_waiting = threading.Event()
    close_finished = threading.Event()
    real_enqueue = agent_loop.enqueue_cancellation

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if threading.current_thread().name == "closing-agent-loop":
                close_waiting.set()
            self.lock.acquire()
            return self

        def __exit__(self, *exc):
            self.lock.release()

    def delayed_enqueue(*args, **kwargs):
        enqueue_entered.set()
        assert release_enqueue.wait(5)
        return real_enqueue(*args, **kwargs)

    loop = agent_loop.AgentLoop(
        task.id, lambda *_: {"type": "done", "summary": "unused"},
        agent_loop.ToolDispatcher(),
    )
    loop._active_binding = agent_loop.ToolBinding(
        execute=lambda **_: result_for(task.id),
        cancel=lambda: callback_called.set() or True,
        retry_safe=True,
    )
    loop._active_tool = "terminal.run"
    loop._active_operation_id = "operation-admission"
    loop._lifecycle_lock = ObservedLock()
    monkeypatch.setattr(agent_loop, "enqueue_cancellation", delayed_enqueue)

    submitted = {}
    submitter = threading.Thread(
        target=lambda: submitted.setdefault("control", loop.submit_control("interrupt")))
    submitter.start()
    assert enqueue_entered.wait(5)
    closed = {}
    def close_loop():
        closed["quiescent"] = loop.close(timeout=5)
        close_finished.set()

    closer = threading.Thread(target=close_loop, name="closing-agent-loop")
    closer.start()
    assert close_waiting.wait(5)
    assert not close_finished.is_set()

    release_enqueue.set()
    submitter.join(5)
    closer.join(5)
    assert not submitter.is_alive() and not closer.is_alive()
    assert callback_called.is_set()
    assert closed["quiescent"] is True
    assert submitted["control"][0].state in {"pending", "applied"}


def test_expired_live_cancellation_owner_becomes_fail_closed_not_replayable(loop_state):
    task, _ = loop_state
    tool_started = threading.Event()
    release_tool = threading.Event()
    cancel_entered = threading.Event()
    release_cancel = threading.Event()
    cancel_calls = []

    def execute(task_id=None):
        tool_started.set()
        assert release_tool.wait(5)
        return result_for(task_id, tool="terminal.run", state="failed", error="cancelled")

    def cancel():
        cancel_calls.append(True)
        cancel_entered.set()
        release_tool.set()
        release_cancel.wait()
        return True

    loop = agent_loop.AgentLoop(
        task.id,
        lambda *_: {"type": "tool", "tool": "terminal.run", "arguments": {}},
        agent_loop.ToolDispatcher().register(
            "terminal.run", execute, cancel=cancel, retry_safe=True),
        limits=agent_loop.LoopLimits(cancellation_reconcile_seconds=0.01),
    )
    worker = threading.Thread(target=loop.run)
    worker.start()
    assert tool_started.wait(5)
    control, _ = loop.submit_control("interrupt", "stop")
    assert cancel_entered.wait(5)
    worker.join(5)
    assert not worker.is_alive()

    heartbeat = loop._cancellation_heartbeats[control.id]
    heartbeat.stop_event.set()
    heartbeat.thread.join(timeout=5)
    with state_store.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE resource_leases SET expires_at='1970-01-01T00:00:00+00:00' "
            "WHERE resource_type='control-cancellation' AND resource_key=?",
            (control.id,),
        )

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    contender = context.Process(
        target=_attempt_task_execution_in_process,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              task.id, sender),
    )
    contender.start()
    state, reason = receiver.recv()
    contender.join(timeout=10)
    assert contender.exitcode == 0 and state == "blocked"
    assert "explicit recovery" in reason
    assert control_queue.cancellation_state(control.id)["state"] == "ambiguous"
    assert tasks.load_task(task.id).phase == control_queue.CANCELLATION_RECOVERY_PHASE
    assert cancel_calls == [True]

    release_cancel.set()
    assert loop.close(timeout=5) is True
    assert cancel_calls == [True]


def test_cancellation_exception_is_ambiguous_and_does_not_persist_error_text(loop_state):
    task, _ = loop_state
    started = threading.Event()
    release_tool = threading.Event()

    def execute(task_id=None):
        started.set()
        assert release_tool.wait(5)
        return result_for(task_id, tool="terminal.run", state="failed", error="cancelled")

    def cancel():
        release_tool.set()
        raise RuntimeError("resolved-secret-must-not-persist")

    decisions = iter((
        {"type": "tool", "tool": "terminal.run", "arguments": {}},
        {"type": "finish", "summary": "done"},
    ))
    loop = agent_loop.AgentLoop(
        task.id, lambda current, controls: next(decisions),
        agent_loop.ToolDispatcher().register(
            "terminal.run", execute, cancel=cancel, retry_safe=True),
        completion=agent_loop.CompletionContract(require_evidence=False),
    )
    holder = {}
    thread = threading.Thread(target=lambda: holder.setdefault("outcome", loop.run()))
    thread.start()
    assert started.wait(5)
    control, _ = loop.submit_control("interrupt", "stop")
    thread.join(5)
    assert not thread.is_alive() and holder["outcome"].state == "failed"
    recorded = control_queue.load(control.id)
    assert control_queue.cancellation_state(control.id)["state"] == "ambiguous"
    assert recorded.payload["cancellation_requested"] is None
    assert recorded.payload["cancellation_error"] == "RuntimeError"
    assert "resolved-secret" not in str(recorded.payload)
    failed = tasks.load_task(task.id)
    assert failed.state == "failed"
    assert failed.phase == control_queue.CANCELLATION_RECOVERY_PHASE


def test_reconciliation_retry_never_replays_external_cancellation(loop_state, monkeypatch):
    task, _ = loop_state
    started = threading.Event()
    release_tool = threading.Event()
    cancel_calls = []
    real_reconcile = control_queue.reconcile_cancellation
    attempts = []

    def execute(task_id=None):
        started.set()
        assert release_tool.wait(5)
        return result_for(task_id, tool="terminal.run", state="failed", error="cancelled")

    def cancel():
        cancel_calls.append(True)
        release_tool.set()
        return True

    def flaky_reconcile(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("database temporarily unavailable")
        return real_reconcile(*args, **kwargs)

    monkeypatch.setattr(agent_loop, "reconcile_cancellation", flaky_reconcile)
    decisions = iter((
        {"type": "tool", "tool": "terminal.run", "arguments": {}},
        {"type": "finish", "summary": "done"},
    ))
    loop = agent_loop.AgentLoop(
        task.id, lambda current, controls: next(decisions),
        agent_loop.ToolDispatcher().register(
            "terminal.run", execute, cancel=cancel, retry_safe=True),
        completion=agent_loop.CompletionContract(require_evidence=False),
    )
    holder = {}
    thread = threading.Thread(target=lambda: holder.setdefault("outcome", loop.run()))
    thread.start()
    assert started.wait(5)
    control, _ = loop.submit_control("interrupt", "stop")
    thread.join(5)
    assert not thread.is_alive() and holder["outcome"].state == "completed"
    assert len(attempts) >= 2
    assert cancel_calls == [True]
    assert control_queue.cancellation_state(control.id)["state"] == "resolved"


def test_attempt_start_failure_resolves_without_external_cancellation(loop_state, monkeypatch):
    task, _ = loop_state
    started = threading.Event()
    release_tool = threading.Event()
    cancel_calls = []

    def execute(task_id=None):
        started.set()
        assert release_tool.wait(5)
        return result_for(task_id, tool="terminal.run")

    def fail_begin(*args, **kwargs):
        raise OSError("database unavailable")

    monkeypatch.setattr(agent_loop, "begin_cancellation", fail_begin)
    dispatcher = agent_loop.ToolDispatcher().register(
        "terminal.run", execute, cancel=lambda: cancel_calls.append(True))
    decisions = iter((
        {"type": "tool", "tool": "terminal.run", "arguments": {}},
        {"type": "finish", "summary": "done"},
    ))
    loop = agent_loop.AgentLoop(
        task.id, lambda current, controls: next(decisions), dispatcher,
        completion=agent_loop.CompletionContract(require_evidence=False),
    )
    holder = {}
    thread = threading.Thread(target=lambda: holder.setdefault("outcome", loop.run()))
    thread.start()
    assert started.wait(5)
    control, _ = loop.submit_control("interrupt", "stop")
    assert cancel_calls == []
    assert control_queue.cancellation_state(control.id)["state"] == "resolved"
    assert control_queue.load(control.id).payload["cancellation_outcome"] == (
        "attempt-start-failed")
    release_tool.set()
    thread.join(5)
    assert not thread.is_alive() and holder["outcome"].state == "completed"


def test_cancellation_enqueue_rolls_back_intent_lease_when_event_insert_fails(
        loop_state, monkeypatch):
    task, _ = loop_state
    owner = ownership.Owner.create("cancel-owner")

    def fail_event(*args, **kwargs):
        raise OSError("event insert failed")

    monkeypatch.setattr(control_queue, "insert_state_event", fail_event)
    with pytest.raises(OSError, match="event insert failed"):
        control_queue.enqueue_cancellation(
            task.id, "interrupt", "stop", active_tool="terminal.run",
            cancellable=True, operation_id="operation-rollback", owner=owner,
        )
    with state_store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_controls").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM control_cancellations").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM resource_leases "
            "WHERE resource_type IN ('control-cancellation','task-execution-fence')"
        ).fetchone()[0] == 0


def test_ambiguous_cancellation_and_fail_closed_task_transition_are_atomic(
        loop_state, monkeypatch):
    task, _ = loop_state
    owner = ownership.Owner.create("cancel-atomic")
    control = control_queue.enqueue_cancellation(
        task.id, "interrupt", "stop", active_tool="terminal.run",
        cancellable=True, operation_id="operation-atomic", owner=owner,
    )
    control_queue.begin_cancellation(control.id, owner)

    def fail_event(*args, **kwargs):
        raise OSError("task transition event failed")

    monkeypatch.setattr(control_queue, "insert_state_event", fail_event)
    with pytest.raises(OSError, match="task transition event failed"):
        control_queue.reconcile_cancellation(
            control.id, owner, error="callback-failed", ambiguous=True)
    assert control_queue.cancellation_state(control.id)["state"] == "attempting"
    assert tasks.load_task(task.id).state == "pending"
    with state_store.connect() as conn:
        assert conn.execute(
            "SELECT 1 FROM resource_leases "
            "WHERE resource_type='task-execution-fence' AND resource_key=?",
            (task.id,),
        ).fetchone()


def test_destructive_cancellation_callback_is_not_exercised_by_interrupt(loop_state):
    task, _ = loop_state
    started = threading.Event()
    release_tool = threading.Event()
    cancel_calls = []

    def execute(task_id=None):
        started.set()
        assert release_tool.wait(5)
        return result_for(task_id, tool="terminal.run")

    dispatcher = agent_loop.ToolDispatcher().register(
        "terminal.run", execute,
        cancel=lambda: cancel_calls.append(True),
        cancellation_effect="destructive",
    )
    decisions = iter((
        {"type": "tool", "tool": "terminal.run", "arguments": {}},
        {"type": "finish", "summary": "done"},
    ))
    loop = agent_loop.AgentLoop(
        task.id, lambda current, controls: next(decisions), dispatcher,
        completion=agent_loop.CompletionContract(require_evidence=False),
    )
    holder = {}
    thread = threading.Thread(target=lambda: holder.setdefault("outcome", loop.run()))
    thread.start()
    assert started.wait(5)
    control, _ = loop.submit_control("interrupt", "stop")
    release_tool.set()
    thread.join(5)
    assert not thread.is_alive() and holder["outcome"].state == "completed"
    assert cancel_calls == []
    assert control_queue.load(control.id).payload["cancellation_outcome"] == (
        "requires-explicit-destructive-authority")


def test_live_cancellation_intent_cannot_be_recovered_by_second_process(loop_state):
    task, _ = loop_state
    owner = ownership.Owner.create("cancel-owner")
    control = control_queue.enqueue_cancellation(
        task.id, "interrupt", "stop", active_tool="terminal.run",
        cancellable=True, operation_id="operation-live", owner=owner,
    )
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_recover_cancellations_in_process,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              task.id, sender),
    )
    process.start()
    assert receiver.recv() == 0
    process.join(timeout=10)
    assert process.exitcode == 0
    assert control_queue.cancellation_state(control.id)["state"] == "intent"
    control_queue.resolve_unattempted_cancellation(
        control.id, owner, outcome="test-cleanup")


def test_expired_cancellation_attempt_is_recovered_as_ambiguous(loop_state):
    task, _ = loop_state
    owner = ownership.Owner.create("stale-cancel-owner")
    control = control_queue.enqueue_cancellation(
        task.id, "interrupt", "stop", active_tool="terminal.run",
        cancellable=True, operation_id="operation-stale", owner=owner,
    )
    control_queue.begin_cancellation(control.id, owner)
    with state_store.transaction(immediate=True) as conn:
        conn.execute(
            """UPDATE resource_leases SET expires_at='2000-01-01T00:00:00+00:00'
               WHERE resource_type='control-cancellation' AND resource_key=?""",
            (control.id,),
        )
    assert control_queue.recover_cancellations(
        task.id, ownership.Owner.create("recovery")) == 1
    assert control_queue.cancellation_state(control.id)["state"] == "ambiguous"
    assert control_queue.load(control.id).payload["cancellation_requested"] is None


@pytest.mark.parametrize("attempted", [False, True])
def test_dead_cancellation_owner_recovers_without_replay(loop_state, tmp_path, attempted):
    task, _ = loop_state
    marker = tmp_path / "external-effect"
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_strand_cancellation,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              task.id, attempted, str(marker), sender),
    )
    process.start()
    control_id = receiver.recv()
    process.join(timeout=10)
    assert process.exitcode == 23

    tasks.update_task(task.id, state="running", phase="stranded-execution")
    model_calls = []
    recovering_loop = agent_loop.AgentLoop(
        task.id,
        lambda current, controls: model_calls.append(True) or {
            "type": "finish", "summary": "must not run"},
        agent_loop.ToolDispatcher(),
        completion=agent_loop.CompletionContract(require_evidence=False),
    )
    with pytest.raises(
            RuntimeError,
            match="automatic replay is unsafe|explicit recovery after ambiguous cancellation"):
        recovering_loop.run()
    assert model_calls == []
    state = control_queue.cancellation_state(control_id)
    recorded = control_queue.load(control_id)
    if attempted:
        assert marker.read_text(encoding="utf-8") == "external cancellation may have happened"
        assert state["state"] == "ambiguous"
        assert recorded.payload["cancellation_requested"] is None
    else:
        assert not marker.exists()
        assert state["state"] == "resolved"
        assert recorded.payload["cancellation_requested"] is False


def test_unreconciled_cancellation_cannot_be_claimed_or_finished(loop_state):
    task, _ = loop_state
    attempt_owner = ownership.Owner.create("cancel-attempt")
    processor = ownership.Owner.create("control-processor")
    control = control_queue.enqueue_cancellation(
        task.id, "interrupt", "stop", active_tool="terminal.run",
        cancellable=True, operation_id="operation-terminalization", owner=attempt_owner,
    )
    control_queue.begin_cancellation(control.id, attempt_owner)
    assert control_queue.claim_next(task.id, processor) is None
    assert control_queue.load(control.id).state == "pending"
    control_queue.reconcile_cancellation(control.id, attempt_owner, result=True)
    claimed = control_queue.claim_next(task.id, processor)
    assert claimed.id == control.id
    assert control_queue.finish(control.id, processor).state == "applied"


def test_generic_control_api_cannot_bypass_cancellation_state_machine(loop_state):
    task, _ = loop_state
    with pytest.raises(ValueError, match="durable cancellation state"):
        control_queue.enqueue(task.id, "interrupt", "stop")
    control = control_queue.enqueue_cancellation(
        task.id, "interrupt", "stop", ready=True, outcome="no-active-tool",
        payload={"cancellation_requested": True,
                 "cancellation_outcome": "forged-success",
                 "cancellation_result": {"secret": "must-not-persist"}},
    )
    assert control.payload["cancellation_requested"] is False
    assert control.payload["cancellation_outcome"] == "no-active-tool"
    assert "must-not-persist" not in str(control.payload)


def test_orphaned_cancellation_control_fails_closed_and_is_unhealthy(loop_state):
    task, _ = loop_state
    with state_store.transaction(immediate=True) as conn:
        conn.execute(
            """INSERT INTO task_controls(
               id,task_id,seq,kind,priority,state,message,payload_json,created_at)
               VALUES('control-orphan',?,1,'interrupt',1,'pending','stop','{}',?)""",
            (task.id, state_store.now_utc()),
        )
    owner = ownership.Owner.create("control-processor")
    assert control_queue.claim_next(task.id, owner) is None
    report = state_store.health()
    assert not report["ok"]
    assert report["state_errors"] == [
        "1 cancellation controls lack durable outcome state"]


def test_redirect_updates_canonical_instruction_and_priority_preempts_message(loop_state):
    task, _ = loop_state
    message = control_queue.enqueue(task.id, "message", "ordinary")
    redirect = control_queue.enqueue(task.id, "redirect", "Do not touch the database layer.")
    owner = ownership.Owner.create("test-control")
    assert control_queue.claim_next(task.id, owner).id == redirect.id
    control_queue.finish(redirect.id, owner)
    assert tasks.canonical_task_state(task.id)["current_instruction"] == redirect.message
    assert control_queue.claim_next(task.id, owner).id == message.id


def test_live_processing_control_is_not_reclaimed_only_because_lease_expired(loop_state):
    task, _ = loop_state
    queued = control_queue.enqueue(task.id, "message", "survive reconnect")
    first_owner = ownership.Owner.create("test-control")
    assert control_queue.claim_next(task.id, first_owner).state == "processing"
    recovery_owner = ownership.Owner.create("recovery")
    assert control_queue.recover_processing(task.id, recovery_owner) == 0
    with state_store.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE resource_leases SET expires_at='2000-01-01T00:00:00+00:00' "
            "WHERE resource_type='task-control' AND resource_key=?", (queued.id,))
    assert control_queue.recover_processing(task.id, recovery_owner) == 0
    assert control_queue.load(queued.id).state == "processing"
    assert ownership.heartbeat("task-control", queued.id, first_owner)
    assert control_queue.finish(queued.id, first_owner).state == "applied"


def test_processing_control_owned_by_dead_process_is_recovered(loop_state):
    task, _ = loop_state
    queued = control_queue.enqueue(task.id, "message", "recover after crash")
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_claim_control_and_exit,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent), task.id),
    )
    process.start(); process.join(timeout=10)
    assert process.exitcode == 0
    assert control_queue.recover_processing(
        task.id, ownership.Owner.create("recovery")) == 1
    assert control_queue.load(queued.id).state == "pending"


def test_non_tool_return_is_not_promoted_to_fabricated_tool_result(loop_state):
    task, _ = loop_state
    dispatcher = agent_loop.ToolDispatcher().register("bad.tool", lambda: {"success": True})
    loop = agent_loop.AgentLoop(
        task.id, lambda current, controls: {"type": "tool", "tool": "bad.tool"}, dispatcher,
        limits=agent_loop.LoopLimits(max_iterations=2, max_tool_failures=1),
    )
    outcome = loop.run()
    assert outcome.state == "paused" and not outcome.tool_results
    assert outcome.reason == "tool failure guard"


def test_model_arguments_cannot_declare_trusted_execution_identity():
    observed = []

    def execute(*, task_id=None, session_id=None, principal_id=None):
        observed.append((task_id, session_id, principal_id))
        return ToolResult("fixture", "succeeded")

    dispatcher = agent_loop.ToolDispatcher().register("fixture", execute)
    dispatcher.execute(
        "fixture",
        {"task_id": "attacker-task", "session_id": "attacker-session",
         "principal_id": "attacker-principal"},
        task_id="trusted-task",
    )
    dispatcher.execute(
        "fixture",
        {"task_id": "attacker-task", "session_id": "attacker-session",
         "principal_id": "attacker-principal"},
    )
    assert observed == [("trusted-task", None, None), (None, None, None)]


def test_model_identity_arguments_do_not_reach_preexecution_hook(loop_state):
    task, _ = loop_state
    observed = []

    def before_execute(*, task_id=None, session_id=None, principal_id=None):
        observed.append((task_id, session_id, principal_id))
        return ToolResult("workspace.checkpoint", "succeeded", {"checkpoint_id": "cp"})

    dispatcher = agent_loop.ToolDispatcher().register(
        "fixture", lambda task_id=None: result_for(task_id),
        before_execute=before_execute)
    agent_loop.AgentLoop(
        task.id,
        lambda *_: {
            "type": "tool", "tool": "fixture",
            "arguments": {"task_id": "attacker-task",
                          "session_id": "attacker-session",
                          "principal_id": "attacker-principal"},
        },
        dispatcher,
        limits=agent_loop.LoopLimits(max_iterations=1),
    ).run()
    assert observed == [(task.id, None, None)]


def test_failed_control_terminalizes_loop_instead_of_leaving_task_running(loop_state):
    task, _ = loop_state
    control_queue.enqueue(task.id, "approval", "malformed", payload={})
    outcome = agent_loop.AgentLoop(
        task.id, lambda task, controls: {"type": "finish", "summary": "unused"},
        agent_loop.ToolDispatcher(),
        completion=agent_loop.CompletionContract(require_evidence=False),
    ).run()
    assert outcome.state == "failed"
    assert "control application failed" in outcome.reason
    assert tasks.load_task(task.id).state == "failed"
    assert control_queue.list_controls(task.id)[0].state == "failed"


def test_task_control_cli_surface_parses_redirect_and_inspection():
    parser = build_parser()
    redirect = parser.parse_args(["task", "control", "task-one", "redirect", "new scope"])
    assert redirect.kind == "redirect" and redirect.message == "new scope"
    controls = parser.parse_args(["task", "controls", "task-one"])
    assert controls.task_command == "controls"


def test_repetition_context_and_unsafe_retry_guards(loop_state):
    task, _ = loop_state
    repeated = agent_loop.AgentLoop(
        task.id, lambda current, controls: {"type": "tool", "tool": "read", "arguments": {}},
        agent_loop.ToolDispatcher().register(
            "read", lambda task_id=None: result_for(task_id, tool="read"), retry_safe=True,
        ),
        limits=agent_loop.LoopLimits(max_iterations=5, max_repetitions=1),
    ).run()
    assert repeated.reason == "repetition guard"

    tasks.update_task(task.id, state="pending", phase="retry")
    context = agent_loop.AgentLoop(
        task.id, lambda current, controls: ({"type": "continue", "message": "x"}, .99),
        agent_loop.ToolDispatcher(),
    ).run()
    assert context.reason == "context pressure guard"

    tasks.update_task(task.id, state="pending", phase="retry")
    failed = agent_loop.AgentLoop(
        task.id, lambda current, controls: {"type": "tool", "tool": "write", "arguments": {}},
        agent_loop.ToolDispatcher().register(
            "write", lambda task_id=None: result_for(task_id, tool="write", state="failed",
                                                      error="partial effect unknown"),
        ),
    ).run()
    assert failed.reason == "unsafe retry guard"


def test_risky_tool_checkpoint_hook_runs_before_mutation(loop_state):
    task, _ = loop_state
    order = []
    checkpoint_evidence = evidence.record(
        "workspace_checkpoint", "/workspace", "checkpoint", task_id=task.id,
    )
    def checkpoint(**kwargs):
        order.append("checkpoint")
        return ToolResult("workspace.checkpoint", "succeeded",
                          {"checkpoint_id": "wcp-real"},
                          evidence_ids=(checkpoint_evidence.id,))
    def mutate(task_id=None):
        order.append("mutate")
        return result_for(task_id)
    decisions = iter((
        {"type": "tool", "tool": "fs.write", "arguments": {}},
        {"type": "finish", "summary": "done"},
    ))
    dispatcher = agent_loop.ToolDispatcher().register(
        "fs.write", mutate, before_execute=checkpoint,
    )
    outcome = agent_loop.AgentLoop(task.id, lambda current, controls: next(decisions),
                                   dispatcher).run()
    assert outcome.state == "completed" and order == ["checkpoint", "mutate"]
