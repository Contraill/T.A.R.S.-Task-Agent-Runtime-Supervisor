import threading
import time

import pytest

from tars import agent_loop, control_queue, conversation, evidence, state_store, tasks
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
    assert bool(cancel_called) is cancellable
    messages = [item.content for item in conversation.list_messages(conv.id)
                if item.kind == "control"]
    assert messages == ["stop and inspect"]


def test_redirect_updates_canonical_instruction_and_priority_preempts_message(loop_state):
    task, _ = loop_state
    message = control_queue.enqueue(task.id, "message", "ordinary")
    redirect = control_queue.enqueue(task.id, "redirect", "Do not touch the database layer.")
    assert control_queue.claim_next(task.id).id == redirect.id
    control_queue.finish(redirect.id)
    assert tasks.canonical_task_state(task.id)["current_instruction"] == redirect.message
    assert control_queue.claim_next(task.id).id == message.id


def test_processing_control_recovers_after_client_or_process_disconnect(loop_state):
    task, _ = loop_state
    queued = control_queue.enqueue(task.id, "message", "survive reconnect")
    assert control_queue.claim_next(task.id).state == "processing"
    assert control_queue.recover_processing(task.id) == 1
    recovered = control_queue.load(queued.id)
    assert recovered.state == "pending" and recovered.payload["recovered_after_disconnect"]


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
