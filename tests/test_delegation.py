import threading
import time
from types import SimpleNamespace

import pytest

from tars import delegation, evidence, memory, orchestration, state_store, tasks
from tars.agent_loop import ToolDispatcher
from tars.tool_core import ToolResult
from tars.cli import build_parser


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


def test_local_inference_children_serialize(delegated):
    _, _, create = delegated
    first = create(budget={"max_seconds": 5, "max_iterations": 2,
                           "max_tokens": 100, "inference": True})
    second = create(budget={"max_seconds": 5, "max_iterations": 2,
                            "max_tokens": 100, "inference": True})
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


def test_child_cli_surfaces_parse():
    parser = build_parser()
    create = parser.parse_args(["task", "child-create", "task-one", "inspect",
                                "--contract-json", "{}"])
    assert create.task_command == "child-create"
    assert parser.parse_args(["task", "child-cancel", "dlg-one"]).delegation_id == "dlg-one"
    assert parser.parse_args(["task", "child-accept", "dlg-one", "--reject"]).reject
