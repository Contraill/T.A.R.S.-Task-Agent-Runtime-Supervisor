import multiprocessing
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from tars import (agent_loop, delegation, evidence, fs_tools, memory, orchestration,
                  ownership, runner, state_store, tasks)
from tars.agent_loop import ToolDispatcher
from tars.tool_core import ToolResult
from tars.cli import build_parser


def _configure_child_state(database, scratch, memory_root, history_root):
    state_store.STATE_DB_PATH = Path(database)
    state_store.TASK_ROOT = Path(scratch) / "legacy"
    state_store.TASK_EVENTS_ROOT = Path(scratch) / "events"
    state_store.TASK_INDEX_PATH = Path(scratch) / "index"
    memory.MEMORY_ROOT = Path(memory_root)
    memory.MEMORY_HISTORY_ROOT = Path(history_root)


def _run_delegation_process(database, scratch, memory_root, history_root,
                            delegation_id, ready, release, crash):
    _configure_child_state(database, scratch, memory_root, history_root)
    def execute(context):
        ready.set()
        if crash:
            import os
            os._exit(17)
        release.wait(10)
        return {"summary": "process complete"}
    future = delegation.start(delegation_id, execute)
    future.result(timeout=15)


def _run_agent_task_process(database, scratch, task_id, entered, release, output):
    _configure_child_state(database, scratch, Path(scratch) / "memory",
                           Path(scratch) / "history")

    def model(current, controls):
        entered.set()
        release.wait(10)
        return {"type": "finish", "summary": "agent complete"}

    try:
        outcome = agent_loop.AgentLoop(
            task_id, model, agent_loop.ToolDispatcher(),
            completion=agent_loop.CompletionContract(require_evidence=False),
        ).run()
    except Exception as exc:
        output.put(("error", str(exc)))
    else:
        output.put(("ok", outcome.state))


def _strand_memory_review(database, scratch, memory_root, history_root, candidate_id):
    _configure_child_state(database, scratch, memory_root, history_root)
    from tars import ownership
    owner = ownership.Owner.create("stranded-review")
    with state_store.transaction(immediate=True) as conn:
        assert ownership.claim_in_transaction(
            conn, "memory-candidate-review", candidate_id, owner, lease_seconds=300)
        conn.execute("UPDATE memory_candidates SET status='reviewing' WHERE id=?",
                     (candidate_id,))


def _strand_scheduled_delegation(database, scratch, memory_root, history_root,
                                 delegation_id):
    _configure_child_state(database, scratch, memory_root, history_root)
    from tars import ownership
    owner = ownership.Owner.create("stranded-delegation")
    with state_store.transaction(immediate=True) as conn:
        assert ownership.claim_in_transaction(
            conn, "delegation", delegation_id, owner, lease_seconds=300)
        conn.execute(
            "UPDATE delegation_contracts SET state='scheduled' WHERE delegation_id=?",
            (delegation_id,))


def _role(role_id):
    return SimpleNamespace(id=role_id, display_name=role_id.title(), enabled=True,
                           model=f"{role_id}-model", capabilities=("code", "tools"))


@pytest.fixture
def delegated(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    monkeypatch.setattr(memory, "MEMORY_ROOT", tmp_path / "memory")
    monkeypatch.setattr(memory, "MEMORY_HISTORY_ROOT", tmp_path / "history")
    monkeypatch.setattr(tasks, "resolve_role_id", lambda value: value)
    monkeypatch.setattr(orchestration, "resolve_role_id", lambda value: value)
    monkeypatch.setattr(orchestration, "get_role", lambda value: _role(value))
    root = tmp_path / "workspace"
    root.mkdir()
    parent = tasks.create_task("parent", "maker", make_active=False)

    def create(**overrides):
        values = dict(
            role="maker", tools=("fs.read", "fs.write"),
            authority={"paths": (root,), "effects": ("read", "write")},
            parent_authority={"paths": (tmp_path,), "effects": ("read", "write", "execute")},
            parent_tools=("fs.read", "fs.write", "terminal.run"),
            budget={"max_seconds": 5, "max_iterations": 10, "max_tokens": 1000,
                    "inference": False},
            workspace={"root": root, "mode": "shared", "access": "read-write"},
            completion={"required_evidence_types": (), "summary_required": True},
        )
        values.update(overrides)
        return delegation.create_child(parent.id, "child", **values)
    return parent, root, create


def test_child_authority_can_only_narrow_parent(delegated, tmp_path):
    parent, root, create = delegated
    contract = create()
    assert contract.authority["paths"] == [str(root.resolve())]
    child = tasks.load_task(orchestration.load_delegation(contract.delegation_id).child_task_id)
    with pytest.raises(PermissionError, match="filesystem scope"):
        delegation.create_child(
            child.id, "nested", role="maker", tools=("fs.read",),
            authority={"paths": (tmp_path,), "effects": ("read",)},
            budget={"max_seconds": 2, "max_iterations": 2, "max_tokens": 100},
        )
    with pytest.raises(PermissionError, match="tool allowlist"):
        delegation.create_child(
            child.id, "nested", role="maker", tools=("terminal.run",),
            authority={"paths": (root,), "effects": ("read",)},
            budget={"max_seconds": 2, "max_iterations": 2, "max_tokens": 100},
        )


def test_child_task_and_contract_creation_roll_back_together(delegated):
    parent, _, create = delegated
    with state_store.transaction(immediate=True) as conn:
        conn.execute(
            """CREATE TRIGGER fail_contract BEFORE INSERT ON delegation_contracts
               BEGIN SELECT RAISE(ABORT, 'injected contract failure'); END""")
    with pytest.raises(Exception, match="injected contract failure"):
        create()
    with state_store.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE parent_task_id=?", (parent.id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM delegations").fetchone()[0] == 0


def test_child_tool_dispatcher_enforces_tool_effect_and_target_ceiling(delegated, tmp_path):
    _, root, create = delegated
    contract = create()
    parent = ToolDispatcher().register(
        "fs.read", lambda path, task_id=None: ToolResult(
            "fs.read", "succeeded", {"path": path, "task_id": task_id}))
    tools = delegation.ChildToolDispatcher(contract.delegation_id, parent)
    assert tools.execute("fs.read", {"path": str(root / "a")}, effect="read",
                         target=root / "a").succeeded
    with pytest.raises(PermissionError, match="outside child allowlist"):
        tools.execute("terminal.run", {}, effect="execute", target=root)
    with pytest.raises(PermissionError, match="outside child filesystem"):
        tools.execute("fs.read", {"path": str(tmp_path / "outside")}, effect="read",
                      target=tmp_path / "outside")


def test_child_run_join_and_parent_evidence_acceptance(delegated):
    _, _, create = delegated
    contract = create(completion={"required_evidence_types": ("terminal_result",)})
    child_id = orchestration.load_delegation(contract.delegation_id).child_task_id

    def execute(context):
        assert context["tools"] == ["fs.read", "fs.write"]
        record = evidence.record("terminal_result", "fixture", "passed", task_id=child_id)
        return {"summary": "verified", "evidence_ids": [record.id]}

    delegation.start(contract.delegation_id, execute)
    assert delegation.join(contract.delegation_id, timeout=2) == {
        "state": "completed", "joined": True}
    accepted = delegation.accept(contract.delegation_id, reason="evidence verified")
    assert accepted.state == "accepted" and accepted.accepted is True
    assert delegation.cancel(contract.delegation_id) == {
        "requested": False, "state": "accepted", "cancelled": False}


def test_delegation_can_compose_agent_loop_under_same_task_owner(delegated):
    _, _, create = delegated
    contract = create()

    def execute(context):
        outcome = agent_loop.AgentLoop(
            context["task_id"],
            lambda current, controls: {"type": "finish", "summary": "child done"},
            agent_loop.ToolDispatcher(),
            completion=agent_loop.CompletionContract(require_evidence=False),
        ).run()
        return {"summary": outcome.reason}

    future = delegation.start(contract.delegation_id, execute)
    assert future.result(timeout=10).state == "completed"


def test_delegation_has_one_live_executor_across_processes(delegated):
    _, root, create = delegated
    (root / "value.txt").write_text("value")
    contract = create()
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_run_delegation_process,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              str(memory.MEMORY_ROOT), str(memory.MEMORY_HISTORY_ROOT),
              contract.delegation_id, ready, release, False),
    )
    process.start()
    assert ready.wait(5)
    with pytest.raises(RuntimeError, match="live executor"):
        delegation.start(contract.delegation_id, lambda context: {"summary": "duplicate"})
    with pytest.raises(RuntimeError, match="exclusively owned"):
        fs_tools.FilesystemTools((root,)).read(root / "value.txt")
    release.set()
    process.join(timeout=10)
    assert process.exitcode == 0
    assert delegation.load_contract(contract.delegation_id).state == "completed"


def test_delegation_child_lease_blocks_runner_and_cross_process_agent(delegated):
    _, _, create = delegated
    contract = create()
    child_id = orchestration.load_delegation(contract.delegation_id).child_task_id
    entered, release = threading.Event(), threading.Event()

    def execute(context):
        entered.set()
        assert release.wait(10)
        return {"summary": "delegation complete"}

    future = delegation.start(contract.delegation_id, execute)
    assert entered.wait(5)
    with pytest.raises(RuntimeError, match="live execution owner"):
        runner.create_run(child_id)

    context = multiprocessing.get_context("spawn")
    agent_entered, agent_release = context.Event(), context.Event()
    output = context.Queue()
    process = context.Process(
        target=_run_agent_task_process,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              child_id, agent_entered, agent_release, output),
    )
    process.start()
    assert output.get(timeout=10) == (
        "error", f"task {child_id} already has a live execution owner")
    process.join(timeout=10)
    assert process.exitcode == 0 and not agent_entered.is_set()

    release.set()
    assert future.result(timeout=10).state == "completed"
    assert not ownership.active("task-execution", child_id)


def test_live_cross_process_agent_blocks_delegation_without_clobbering_task(delegated):
    _, _, create = delegated
    contract = create()
    child_id = orchestration.load_delegation(contract.delegation_id).child_task_id
    context = multiprocessing.get_context("spawn")
    entered, release = context.Event(), context.Event()
    output = context.Queue()
    process = context.Process(
        target=_run_agent_task_process,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              child_id, entered, release, output),
    )
    process.start()
    assert entered.wait(5)

    calls = []
    future = delegation.start(
        contract.delegation_id,
        lambda child_context: calls.append(True) or {"summary": "duplicate"},
    )
    with pytest.raises(RuntimeError, match="live execution owner"):
        future.result(timeout=10)
    assert calls == []
    assert delegation.load_contract(contract.delegation_id).state == "failed"
    assert tasks.load_task(child_id).state == "running"

    release.set()
    assert output.get(timeout=10) == ("ok", "completed")
    process.join(timeout=10)
    assert process.exitcode == 0
    assert not ownership.active("task-execution", child_id)


def test_dead_running_delegation_is_not_replayed(delegated):
    _, _, create = delegated
    contract = create()
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_run_delegation_process,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              str(memory.MEMORY_ROOT), str(memory.MEMORY_HISTORY_ROOT),
              contract.delegation_id, ready, release, True),
    )
    process.start()
    assert ready.wait(5)
    process.join(timeout=10)
    assert process.exitcode == 17
    called = []
    with pytest.raises(RuntimeError, match="not replayed"):
        delegation.start(contract.delegation_id,
                         lambda context: called.append(True) or {"summary": "duplicate"})
    assert called == []
    current = delegation.load_contract(contract.delegation_id)
    assert current.state == "failed"


def test_child_thinking_cannot_bypass_generation_budget(delegated):
    _, _, create = delegated
    contract = create(budget={"max_seconds": 5, "max_iterations": 2,
                              "max_tokens": 100, "inference": True})
    assert delegation.child_inference_options(
        contract.delegation_id, requested_tokens=80, thinking="on") == {
            "max_tokens": 80, "thinking": "on"}
    with pytest.raises(PermissionError, match="token budget"):
        delegation.child_inference_options(
            contract.delegation_id, requested_tokens=101, thinking="off")


def test_child_memory_is_staged_until_parent_acceptance(delegated):
    _, _, create = delegated
    contract = create()
    candidate = delegation.stage_child_memory(
        contract.delegation_id, "candidate fact", kind="reference", scope="project")
    unrelated = memory.stage_candidate("unrelated", kind="reference", scope="project")
    with pytest.raises(PermissionError):
        delegation.review_child_memory(contract.delegation_id, candidate, promote=True)
    delegation.start(contract.delegation_id, lambda context: {"summary": "done"})
    delegation.join(contract.delegation_id, timeout=2)
    delegation.accept(contract.delegation_id)
    with pytest.raises(KeyError):
        delegation.review_child_memory(contract.delegation_id, unrelated, promote=True)
    assert memory.review_candidates()[0]["id"] in {candidate, unrelated}
    entry = delegation.review_child_memory(contract.delegation_id, candidate, promote=True)
    assert entry.content == "candidate fact"


def test_cancel_is_cooperative_and_truthful(delegated):
    _, _, create = delegated
    contract = create()
    entered = threading.Event()

    def execute(context):
        entered.set()
        while not context["cancel_event"].wait(0.01):
            pass
        return {"summary": "stopped at safe boundary"}

    delegation.start(contract.delegation_id, execute)
    assert entered.wait(1)
    result = delegation.cancel(contract.delegation_id)
    assert result["requested"] is True
    joined = delegation.join(contract.delegation_id, timeout=2)
    assert joined == {"state": "cancelled", "joined": True}


def test_unstarted_delegation_can_be_cancelled_terminally(delegated):
    _, _, create = delegated
    contract = create()
    assert delegation.cancel(contract.delegation_id) == {
        "requested": True, "state": "cancelled", "cancelled": True}
    assert delegation.join(contract.delegation_id) == {
        "state": "cancelled", "joined": True}


def test_queued_future_cancellation_is_terminal(monkeypatch, delegated):
    _, _, create = delegated
    executor = delegation.ThreadPoolExecutor(max_workers=1)
    gate = threading.Event()
    blocker = executor.submit(gate.wait, 5)
    monkeypatch.setattr(delegation, "_EXECUTOR", executor)
    contract = create()
    future = delegation.start(contract.delegation_id,
                              lambda context: {"summary": "must not run"})
    result = delegation.cancel(contract.delegation_id)
    assert result == {"requested": True, "state": "cancelled", "cancelled": True}
    assert future.cancelled()
    assert delegation.join(contract.delegation_id) == {
        "state": "cancelled", "joined": True}
    gate.set()
    blocker.result(timeout=2)
    executor.shutdown()


def test_cancel_reclaims_dead_scheduled_owner(delegated):
    _, _, create = delegated
    contract = create()
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_strand_scheduled_delegation,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              str(memory.MEMORY_ROOT), str(memory.MEMORY_HISTORY_ROOT),
              contract.delegation_id),
    )
    process.start(); process.join(timeout=10)
    assert process.exitcode == 0
    assert delegation.cancel(contract.delegation_id) == {
        "requested": True, "state": "cancelled", "cancelled": True}
    assert delegation.join(contract.delegation_id) == {
        "state": "cancelled", "joined": True}


def test_children_with_same_exclusive_workspace_serialize(delegated):
    _, _, create = delegated
    first = create(budget={"max_seconds": 5, "max_iterations": 2,
                           "max_tokens": 100, "inference": False})
    second = create(budget={"max_seconds": 5, "max_iterations": 2,
                            "max_tokens": 100, "inference": False})
    active = 0
    maximum = 0
    lock = threading.Lock()

    def execute(context):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"summary": "done"}

    delegation.start(first.delegation_id, execute)
    delegation.start(second.delegation_id, execute)
    assert delegation.join(first.delegation_id, timeout=2)["joined"]
    assert delegation.join(second.delegation_id, timeout=2)["joined"]
    assert maximum == 1


def test_exclusive_workspace_blocks_sibling_tool_surfaces(delegated):
    _, root, create = delegated
    (root / "value.txt").write_text("value")
    contract = create()
    entered, release = threading.Event(), threading.Event()
    def execute(context):
        assert fs_tools.FilesystemTools((root,)).read(root / "value.txt").succeeded
        entered.set()
        assert release.wait(5)
        return {"summary": "done"}
    delegation.start(contract.delegation_id, execute)
    assert entered.wait(5)
    with pytest.raises(RuntimeError, match="exclusively owned"):
        fs_tools.FilesystemTools((root,)).read(root / "value.txt")
    release.set()
    assert delegation.join(contract.delegation_id, timeout=2)["joined"]
    assert fs_tools.FilesystemTools((root,)).read(root / "value.txt").succeeded


def test_child_timeout_requests_cooperative_stop(delegated):
    _, _, create = delegated
    contract = create(budget={"max_seconds": 1, "max_iterations": 2,
                              "max_tokens": 100, "inference": False})

    def execute(context):
        assert context["cancel_event"].wait(2)
        return {"summary": "safe boundary reached"}

    delegation.start(contract.delegation_id, execute)
    assert delegation.join(contract.delegation_id, timeout=2) == {
        "state": "timed_out", "joined": True}


def test_iteration_budget_is_enforced_by_child_executor_protocol(delegated):
    _, _, create = delegated
    contract = create(budget={"max_seconds": 5, "max_iterations": 2,
                              "max_tokens": 100, "inference": False})
    resumed = []
    closed = []
    def execute(context):
        try:
            resumed.append("one")
            yield delegation.DelegationStep({"summary": "one"})
            resumed.append("two")
            yield delegation.DelegationStep({"summary": "two"})
            resumed.append("three-side-effect")
            yield delegation.DelegationStep({"summary": "three"}, done=True)
        finally:
            closed.append(True)
    future = delegation.start(contract.delegation_id, execute)
    with pytest.raises(RuntimeError, match="iteration budget"):
        future.result(timeout=2)
    assert resumed == ["one", "two"]
    assert closed == [True]
    assert delegation.load_contract(contract.delegation_id).state == "failed"
    assert delegation.join(contract.delegation_id) == {
        "state": "failed", "joined": True}


def test_iterative_executor_must_finish_within_authoritative_budget(delegated):
    _, _, create = delegated
    contract = create(budget={"max_seconds": 5, "max_iterations": 2,
                              "max_tokens": 100, "inference": False})

    def execute(context):
        yield delegation.DelegationStep({"summary": "working"})
        yield delegation.DelegationStep({"summary": "done"}, done=True)

    delegation.start(contract.delegation_id, execute)
    assert delegation.join(contract.delegation_id, timeout=2) == {
        "state": "completed", "joined": True}


@pytest.mark.parametrize("claimed", [0, 1, 999])
def test_opaque_executor_cannot_self_report_iteration_compliance(
        delegated, claimed):
    _, _, create = delegated
    contract = create(budget={"max_seconds": 5, "max_iterations": 2,
                              "max_tokens": 100, "inference": False})
    future = delegation.start(
        contract.delegation_id,
        lambda context: {"summary": "claimed compliance", "iterations": claimed},
    )
    with pytest.raises(TypeError, match="must not self-report"):
        future.result(timeout=2)
    assert delegation.load_contract(contract.delegation_id).state == "failed"


def test_iterative_executor_cannot_use_untyped_results_as_hidden_protocol(delegated):
    _, _, create = delegated
    contract = create(budget={"max_seconds": 5, "max_iterations": 2,
                              "max_tokens": 100, "inference": False})

    def execute(context):
        yield {"summary": "untyped terminal", "done": True}

    future = delegation.start(contract.delegation_id, execute)
    with pytest.raises(TypeError, match="must yield DelegationStep"):
        future.result(timeout=2)


def test_iterator_is_not_resumed_after_cooperative_cancellation(delegated):
    _, _, create = delegated
    contract = create(budget={"max_seconds": 5, "max_iterations": 2,
                              "max_tokens": 100, "inference": False})
    calls = []

    class Executor:
        def __call__(self, context):
            class Steps:
                def __iter__(self):
                    return self

                def __next__(self):
                    calls.append("resume")
                    if len(calls) == 1:
                        context["cancel_event"].set()
                        return delegation.DelegationStep({"summary": "stopping"})
                    raise AssertionError("iterator resumed after cancellation")

            return Steps()

    delegation.start(contract.delegation_id, Executor())
    assert delegation.join(contract.delegation_id, timeout=2) == {
        "state": "cancelled", "joined": True}
    assert calls == ["resume"]


def test_accept_reject_review_is_compare_and_swap(delegated):
    _, _, create = delegated
    contract = create()
    delegation.start(contract.delegation_id, lambda context: {"summary": "done"})
    assert delegation.join(contract.delegation_id, timeout=2)["joined"]
    barrier = threading.Barrier(2)
    outcomes = []
    def decide(value):
        barrier.wait()
        try:
            outcomes.append(delegation.accept(
                contract.delegation_id, accept_result=value).state)
        except RuntimeError:
            outcomes.append("lost")
    first = threading.Thread(target=decide, args=(True,))
    second = threading.Thread(target=decide, args=(False,))
    first.start(); second.start(); first.join(); second.join()
    assert outcomes.count("lost") == 1
    assert set(outcomes) & {"accepted", "rejected"}


def test_crashed_memory_review_is_reclaimable(delegated):
    _, _, create = delegated
    contract = create()
    candidate = delegation.stage_child_memory(
        contract.delegation_id, "recover review", kind="reference", scope="project")
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_strand_memory_review,
        args=(str(state_store.STATE_DB_PATH), str(state_store.STATE_DB_PATH.parent),
              str(memory.MEMORY_ROOT), str(memory.MEMORY_HISTORY_ROOT), candidate),
    )
    process.start(); process.join(timeout=10)
    assert process.exitcode == 0
    assert memory.decide_candidate(candidate, promote=True).content == "recover review"


def test_child_cli_surfaces_parse():
    parser = build_parser()
    create = parser.parse_args(["task", "child-create", "task-one", "inspect",
                                "--contract-json", "{}"])
    assert create.task_command == "child-create"
    assert parser.parse_args(["task", "child-cancel", "dlg-one"]).delegation_id == "dlg-one"
    assert parser.parse_args(["task", "child-accept", "dlg-one", "--reject"]).reject
