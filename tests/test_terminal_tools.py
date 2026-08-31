import sys
import time
import multiprocessing
import os
from pathlib import Path

import pytest

from tars import approvals, execution_backends as execution, policy, state_store, terminal_tools


def _start_background_then_exit(database, state_root, log_root, cwd, output):
    state_store.STATE_DB_PATH = Path(database)
    state_store.TASK_ROOT = Path(state_root) / "legacy"
    state_store.TASK_EVENTS_ROOT = Path(state_root) / "events"
    state_store.TASK_INDEX_PATH = Path(state_root) / "index"
    policy.add_rule("sandbox_escape", "allow", target="host")
    manager = terminal_tools.ProcessManager(log_root=log_root)
    request = execution.ExecutionRequest(
        (sys.executable, "-c", "import time; time.sleep(60)"),
        cwd=cwd, allowed_paths=(cwd,),
    )
    result = manager.start(request)
    output.put(result.data["pid"])


@pytest.fixture
def terminal(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    policy.add_rule("sandbox_escape", "allow", target="host")
    manager = terminal_tools.ProcessManager(log_root=tmp_path / "logs")
    return terminal_tools.TerminalTools(
        executor=execution.GuardedExecutor({"host": execution.HostBackend()}),
        processes=manager, output_limit=32, log_root=tmp_path / "foreground-logs",
    ), tmp_path


def _approve(tool, effect, target, arguments=None, *, destructive=False):
    request = policy.ScopeRequest(tool, effect, target, arguments or {}, destructive=destructive)
    decision = policy.ScopeGuard().evaluate(request)
    broker = approvals.ApprovalBroker()
    pending = broker.request(request, decision)
    broker.decide(pending.id, approve=True)
    return pending.id


def test_terminal_direct_argv_returns_structured_truth_and_truncates(terminal):
    tools, root = terminal
    result = tools.run(
        (sys.executable, "-c", "print('x' * 100)"), cwd=str(root),
        allowed_paths=(str(root),),
    )
    assert result.succeeded and result.data["exit_code"] == 0
    assert result.data["truncated"] and len(result.data["stdout"]) == 32
    assert Path(result.data["stdout_ref"]).read_text().strip() == "x" * 100
    assert result.action_ids and result.evidence_ids


def test_terminal_pty_is_explicit_and_reports_terminal_truth(terminal):
    tools, root = terminal
    result = tools.run(
        (sys.executable, "-c", "import os; print(os.isatty(1))"), cwd=str(root),
        allowed_paths=(str(root),), tty=True,
    )
    assert result.succeeded and result.data["pty"] and "True" in result.data["stdout"]


def test_background_process_lifecycle_and_durable_logs(terminal):
    tools, root = terminal
    started = tools.run(
        (sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(30)"),
        cwd=str(root), allowed_paths=(str(root),), background=True,
    )
    process_id = started.data["process_id"]
    for _ in range(50):
        logs = tools.processes.logs(process_id)
        if "ready" in logs.data["content"]:
            break
        time.sleep(.02)
    assert "ready" in logs.data["content"] and logs.data["full_log_ref"]
    approval = _approve("process.signal", "execute", process_id, {"signal": "TERM"})
    truth = tools.processes.signal(process_id, approval_id=approval)
    assert truth.data["requested"] and truth.data["cancellable"]
    waited = tools.processes.wait(process_id, timeout=5)
    assert waited.data["state"] == "exited" and waited.data["exit_code"] is not None


def test_background_environment_secret_is_redacted_from_log(monkeypatch, terminal):
    tools, root = terminal
    monkeypatch.setenv("TARS_BG_SECRET", "do-not-log")
    started = tools.run(
        (sys.executable, "-u", "-c", "import os; print(os.environ['VALUE'])"),
        cwd=str(root), allowed_paths=(str(root),), background=True,
        environment_refs={"VALUE": "env:TARS_BG_SECRET"},
    )
    tools.processes.wait(started.data["process_id"], timeout=5)
    logs = tools.processes.logs(started.data["process_id"])
    assert logs.data["content"].strip() == "[REDACTED]"


def test_process_kill_is_destructive_and_denied_without_approval(terminal):
    tools, root = terminal
    started = tools.run(
        (sys.executable, "-c", "import time; time.sleep(30)"), cwd=str(root),
        allowed_paths=(str(root),), background=True,
    )
    process_id = started.data["process_id"]
    with pytest.raises(PermissionError):
        tools.processes.kill(process_id)
    approval = _approve(
        "process.kill", "destructive", process_id, {"signal": "KILL"}, destructive=True,
    )
    killed = tools.processes.kill(process_id, approval_id=approval)
    assert killed.tool == "process.kill" and killed.succeeded
    assert tools.processes.wait(process_id, timeout=5).data["state"] == "exited"


def test_background_process_dies_with_owning_manager_process(tmp_path):
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=_start_background_then_exit,
        args=(tmp_path / "state.sqlite3", tmp_path, tmp_path / "logs",
              str(tmp_path), output),
    )
    process.start()
    child_pid = output.get(timeout=10)
    process.join(timeout=10)
    assert process.exitcode == 0
    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        os.kill(child_pid, 9)
        pytest.fail("managed background child survived its owning process")
