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
    business_pid_path = Path(state_root) / "business.pid"
    descendant_pid_path = Path(state_root) / "descendant.pid"
    descendant_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(descendant_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    business_code = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{descendant_code!r}]); "
        f"Path({str(business_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    request = execution.ExecutionRequest(
        (sys.executable, "-c", business_code),
        cwd=cwd, allowed_paths=(cwd,),
    )
    result = manager.start(request)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if business_pid_path.exists() and descendant_pid_path.exists():
            break
        time.sleep(0.01)
    output.put((result.data["pid"], int(business_pid_path.read_text()),
                int(descendant_pid_path.read_text())))


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


def test_graceful_term_reaches_business_process_without_cleanup_escalation(terminal):
    tools, root = terminal
    marker = root / "term-handled"
    code = (
        "import signal,sys,time; from pathlib import Path; "
        f"marker=Path({str(marker)!r}); "
        "signal.signal(signal.SIGTERM, "
        "lambda *_: (marker.write_text('TERM'), sys.exit(0))); "
        "print('ready', flush=True); time.sleep(30)"
    )
    started = tools.run(
        (sys.executable, "-u", "-c", code), cwd=str(root),
        allowed_paths=(str(root),), background=True,
    )
    process_id = started.data["process_id"]
    for _ in range(100):
        if "ready" in tools.processes.logs(process_id).data["content"]:
            break
        time.sleep(0.01)
    else:
        pytest.fail("business process did not become ready")
    approval = _approve("process.signal", "execute", process_id, {"signal": "TERM"})
    truth = tools.processes.signal(process_id, "TERM", approval_id=approval)
    waited = tools.processes.wait(process_id, timeout=5)
    assert truth.data["signal"] == "TERM"
    assert waited.data["state"] == "exited" and waited.data["exit_code"] == 0
    assert marker.read_text() == "TERM"


def test_term_cannot_turn_into_hidden_group_kill(terminal):
    tools, root = terminal
    started = tools.run(
        (sys.executable, "-u", "-c",
         "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
         "print('ready', flush=True); time.sleep(30)"),
        cwd=str(root), allowed_paths=(str(root),), background=True,
    )
    process_id = started.data["process_id"]
    for _ in range(100):
        if "ready" in tools.processes.logs(process_id).data["content"]:
            break
        time.sleep(0.01)
    else:
        pytest.fail("business process did not become ready")
    approval = _approve("process.signal", "execute", process_id, {"signal": "TERM"})
    tools.processes.signal(process_id, "TERM", approval_id=approval)
    waited = tools.processes.wait(process_id, timeout=0.2)
    assert waited.data["state"] == "running" and waited.data["timed_out"]

    kill_approval = _approve(
        "process.kill", "destructive", process_id, {"signal": "KILL"}, destructive=True)
    tools.processes.kill(process_id, approval_id=kill_approval)
    assert tools.processes.wait(process_id, timeout=5).data["state"] == "exited"


@pytest.mark.parametrize("signal_name", ["KILL", "SEGV", "RTMIN"])
def test_process_signal_rejects_authority_widening_signals(terminal, signal_name):
    tools, root = terminal
    started = tools.run(
        (sys.executable, "-c", "import time; time.sleep(30)"), cwd=str(root),
        allowed_paths=(str(root),), background=True,
    )
    process_id = started.data["process_id"]
    with pytest.raises(ValueError, match="process.signal supports"):
        tools.processes.signal(process_id, signal_name)
    with state_store.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM action_journal WHERE tool='process.signal'"
        ).fetchone()[0] == 0
    kill_approval = _approve(
        "process.kill", "destructive", process_id, {"signal": "KILL"}, destructive=True)
    tools.processes.kill(process_id, approval_id=kill_approval)
    tools.processes.wait(process_id, timeout=5)


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


def test_signal_after_exit_does_not_fabricate_dispatch(terminal):
    tools, root = terminal
    started = tools.run(
        (sys.executable, "-c", "pass"), cwd=str(root),
        allowed_paths=(str(root),), background=True,
    )
    process_id = started.data["process_id"]
    tools.processes.wait(process_id, timeout=5)
    approval = _approve("process.signal", "execute", process_id, {"signal": "TERM"})
    truth = tools.processes.signal(process_id, "TERM", approval_id=approval)
    assert truth.succeeded
    assert truth.data["requested"] is False
    assert truth.data["outcome"] == "already-exited"


def test_background_process_dies_with_owning_manager_process(tmp_path):
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=_start_background_then_exit,
        args=(tmp_path / "state.sqlite3", tmp_path, tmp_path / "logs",
              str(tmp_path), output),
    )
    process.start()
    managed_pids = output.get(timeout=10)
    process.join(timeout=10)
    assert process.exitcode == 0
    remaining = set(managed_pids)
    for _ in range(200):
        for pid in tuple(remaining):
            try:
                os.kill(pid, 0)
                stat = Path(f"/proc/{pid}/stat").read_text()
                if stat[stat.rfind(")") + 2:].split()[0] == "Z":
                    remaining.discard(pid)
            except (ProcessLookupError, FileNotFoundError):
                remaining.discard(pid)
        if not remaining:
            break
        time.sleep(0.01)
    if remaining:
        for pid in remaining:
            os.kill(pid, 9)
        pytest.fail(f"managed process group survived its owning process: {remaining}")
