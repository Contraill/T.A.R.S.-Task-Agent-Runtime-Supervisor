import sys
import time
import multiprocessing
import os
from pathlib import Path
import subprocess
import sqlite3
import threading
from dataclasses import replace

import pytest

from tars import (approvals, execution_backends as execution, policy, state_store,
                  sessions, tasks, terminal_tools)


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
        f"subprocess.Popen([sys.executable,'-c',{descendant_code!r}],start_new_session=True); "
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
    output.put((result.data["process_id"], (
        result.data["supervisor_pid"], result.data["pid"],
        int(business_pid_path.read_text()), int(descendant_pid_path.read_text()))))


def _start_background_signal_then_wait(database, state_root, log_root, cwd,
                                       output, commands, acknowledged):
    state_store.STATE_DB_PATH = Path(database)
    state_store.TASK_ROOT = Path(state_root) / "legacy"
    state_store.TASK_EVENTS_ROOT = Path(state_root) / "events"
    state_store.TASK_INDEX_PATH = Path(state_root) / "index"
    policy.add_rule("sandbox_escape", "allow", target="host")
    manager = terminal_tools.ProcessManager(log_root=log_root)
    business_pid_path = Path(state_root) / "signal-business.pid"
    descendant_pid_path = Path(state_root) / "signal-descendant.pid"
    ignored = "(signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGUSR1,signal.SIGUSR2)"
    descendant_code = (
        "import os,signal,time; from pathlib import Path; "
        f"[signal.signal(value,signal.SIG_IGN) for value in {ignored}]; "
        f"Path({str(descendant_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    business_code = (
        "import os,signal,subprocess,sys,time; from pathlib import Path; "
        f"[signal.signal(value,signal.SIG_IGN) for value in {ignored}]; "
        f"subprocess.Popen([sys.executable,'-c',{descendant_code!r}],start_new_session=True); "
        f"Path({str(business_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    result = manager.start(execution.ExecutionRequest(
        (sys.executable, "-c", business_code), cwd=cwd, allowed_paths=(cwd,)))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if business_pid_path.exists() and descendant_pid_path.exists():
            break
        time.sleep(0.01)
    output.put((result.data["process_id"], result.data["supervisor_pid"],
                result.data["pid"], int(descendant_pid_path.read_text())))
    signal_name = commands.get(timeout=10)
    approval = _approve(
        "process.signal", "execute", result.data["process_id"],
        {"signal": signal_name})
    truth = manager.signal(
        result.data["process_id"], signal_name, approval_id=approval)
    acknowledged.put(truth.data)
    threading.Event().wait(60)


def _process_state(pid):
    try:
        value = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return ""
    return value[value.rfind(")") + 2:].split()[0]


def _assert_processes_gone(pids):
    remaining = set(pids)
    for _ in range(200):
        for pid in tuple(remaining):
            state = _process_state(pid)
            if not state or state == "Z":
                remaining.discard(pid)
        if not remaining:
            return
        time.sleep(0.01)
    for pid in remaining:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    pytest.fail(f"managed process topology survived its owner: {remaining}")


@pytest.fixture
def terminal(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    policy.add_rule("sandbox_escape", "allow", target="host")
    manager = terminal_tools.ProcessManager(log_root=tmp_path / "logs")
    tools = terminal_tools.TerminalTools(
        executor=execution.GuardedExecutor({"host": execution.HostBackend()}),
        processes=manager, output_limit=32, log_root=tmp_path / "foreground-logs",
    )
    try:
        yield tools, tmp_path
    finally:
        manager.close(timeout=10)


def _approve(tool, effect, target, arguments=None, *, destructive=False,
             task_id=None, session_id=None):
    request = policy.ScopeRequest(
        tool, effect, target, arguments or {}, destructive=destructive,
        task_id=task_id, session_id=session_id)
    decision = policy.ScopeGuard().evaluate(request)
    broker = approvals.ApprovalBroker()
    pending = broker.request(request, decision)
    broker.decide(
        pending.id, approve=True, task_id=task_id, session_id=session_id)
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


def test_process_operations_require_exact_trusted_task_origin(terminal):
    tools, root = terminal
    owner = tasks.create_task("own managed process", "general")
    stranger = tasks.create_task("must not control another task process", "general")
    owner_session = sessions.create_session()
    stranger_session = sessions.create_session()
    started = tools.run(
        (sys.executable, "-c", "import time; time.sleep(30)"), cwd=str(root),
        allowed_paths=(str(root),), background=True, task_id=owner.id,
        session_id=owner_session.id,
    )
    process_id = started.data["process_id"]

    assert tools.processes.list(task_id=stranger.id).data["processes"] == []
    for operation in (
        lambda: tools.processes.poll(process_id, task_id=stranger.id),
        lambda: tools.processes.wait(process_id, timeout=0, task_id=stranger.id),
        lambda: tools.processes.logs(process_id, task_id=stranger.id),
        lambda: tools.processes.write(process_id, "x", task_id=stranger.id),
        lambda: tools.processes.signal(process_id, "TERM", task_id=stranger.id),
        lambda: tools.processes.kill(process_id, task_id=stranger.id),
        lambda: tools.processes.poll(
            process_id, task_id=owner.id, session_id=stranger_session.id),
        lambda: tools.processes.poll(process_id),
    ):
        with pytest.raises(PermissionError, match="another task or session"):
            operation()
    assert tools.processes.poll(
        process_id, task_id=owner.id,
        session_id=owner_session.id).data["state"] == "running"

    approval = _approve(
        "process.kill", "destructive", process_id, {"signal": "KILL"},
        destructive=True, task_id=owner.id, session_id=owner_session.id)
    tools.processes.kill(
        process_id, approval_id=approval, task_id=owner.id,
        session_id=owner_session.id)
    assert tools.processes.wait(
        process_id, timeout=5, task_id=owner.id,
        session_id=owner_session.id).data["state"] == "exited"


def test_fake_process_records_cannot_control_unrelated_process(terminal):
    tools, root = terminal
    started = tools.run(
        (sys.executable, "-c", "import time; time.sleep(30)"), cwd=str(root),
        allowed_paths=(str(root),), background=True,
    )
    process_id = started.data["process_id"]
    real_record = tools.processes._processes[process_id]
    unrelated = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(30)"), start_new_session=True)
    fake_id = "process-forged"
    try:
        tools.processes._processes["process-raw-popen"] = unrelated
        with pytest.raises(PermissionError, match="trusted creation provenance"):
            tools.processes.poll("process-raw-popen")

        tools.processes._processes[fake_id] = replace(
            real_record, id=fake_id, process=unrelated,
            business_pid=unrelated.pid,
            business_start=terminal_tools.process_start(unrelated.pid),
            process_group_id=os.getpgid(unrelated.pid),
            supervisor_start=terminal_tools.process_start(unrelated.pid),
        )
        with pytest.raises(PermissionError, match="trusted creation provenance"):
            tools.processes.signal(fake_id, "TERM")

        tools.processes._processes[process_id] = replace(
            real_record, process=unrelated, business_pid=unrelated.pid,
            business_start=terminal_tools.process_start(unrelated.pid),
            process_group_id=os.getpgid(unrelated.pid),
            supervisor_start=terminal_tools.process_start(unrelated.pid),
        )
        with pytest.raises(PermissionError, match="trusted creation provenance"):
            tools.processes.kill(process_id)
        assert unrelated.poll() is None
    finally:
        tools.processes._processes[process_id] = real_record
        tools.processes._processes.pop(fake_id, None)
        tools.processes._processes.pop("process-raw-popen", None)
        unrelated.terminate()
        unrelated.wait(timeout=5)
        approval = _approve(
            "process.kill", "destructive", process_id, {"signal": "KILL"},
            destructive=True)
        tools.processes.kill(process_id, approval_id=approval)
        tools.processes.wait(process_id, timeout=5)


def test_second_manager_cannot_reconstruct_live_process_authority(terminal):
    tools, root = terminal
    started = tools.run(
        (sys.executable, "-c", "import time; time.sleep(30)"), cwd=str(root),
        allowed_paths=(str(root),), background=True,
    )
    process_id = started.data["process_id"]
    second = terminal_tools.ProcessManager(log_root=root / "other-logs")
    with pytest.raises(KeyError, match="unknown process"):
        second.signal(process_id, "TERM")
    assert tools.processes.poll(process_id).data["state"] == "running"
    approval = _approve(
        "process.kill", "destructive", process_id, {"signal": "KILL"}, destructive=True)
    tools.processes.kill(process_id, approval_id=approval)
    tools.processes.wait(process_id, timeout=5)


def test_stale_supervisor_identity_fails_closed_before_signal(monkeypatch, terminal):
    tools, root = terminal
    started = tools.run(
        (sys.executable, "-c", "import time; time.sleep(30)"), cwd=str(root),
        allowed_paths=(str(root),), background=True,
    )
    process_id = started.data["process_id"]
    supervisor_pid = started.data["supervisor_pid"]
    real_process_start = terminal_tools.process_start

    def stale(pid):
        return "reused-process-identity" if pid == supervisor_pid else real_process_start(pid)

    monkeypatch.setattr(terminal_tools, "process_start", stale)
    with pytest.raises(PermissionError, match="supervisor identity changed"):
        tools.processes.signal(process_id, "TERM")
    monkeypatch.setattr(terminal_tools, "process_start", real_process_start)
    approval = _approve(
        "process.kill", "destructive", process_id, {"signal": "KILL"}, destructive=True)
    tools.processes.kill(process_id, approval_id=approval)
    tools.processes.wait(process_id, timeout=5)


def test_lost_process_authority_lease_never_rebinds_control(terminal):
    tools, root = terminal
    started = tools.run(
        (sys.executable, "-c", "import time; time.sleep(30)"), cwd=str(root),
        allowed_paths=(str(root),), background=True,
    )
    process_id = started.data["process_id"]
    record = tools.processes._processes[process_id]
    heartbeat = tools.processes._handles[process_id][4]
    heartbeat.stop_event.set()
    heartbeat.thread.join(timeout=5)
    with state_store.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE resource_leases SET expires_at='1970-01-01T00:00:00+00:00' "
            "WHERE resource_type='managed-process' AND resource_key=?",
            (process_id,),
        )

    with pytest.raises(PermissionError, match="durable provenance"):
        tools.processes.signal(process_id, "TERM")
    assert _process_state(record.business_pid)

    os.kill(record.process.pid, terminal_tools.signals.SIGRTMIN)
    record.process.wait(timeout=5)
    tools.processes._refresh(record)
    assert tools.processes.poll(process_id).data["state"] == "exited"


def test_lost_signal_acknowledgement_is_ambiguous_and_fenced(terminal):
    tools, root = terminal
    marker = root / "usr1-count"
    code = (
        "import signal,time; from pathlib import Path; "
        f"marker=Path({str(marker)!r}); "
        "signal.signal(signal.SIGUSR1, lambda *_: marker.write_text("
        "marker.read_text() + 'x' if marker.exists() else 'x')); "
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
    record = tools.processes._processes[process_id]
    os.close(record.response_read)

    first_approval = _approve(
        "process.signal", "execute", process_id, {"signal": "USR1"})
    outcome = tools.processes.signal(
        process_id, "USR1", approval_id=first_approval)
    assert outcome.state == "unknown"
    assert outcome.data["requested"] is None
    assert outcome.data["outcome"] == "dispatch-ambiguous"
    for _ in range(100):
        if marker.exists():
            break
        time.sleep(0.01)
    assert marker.read_text() == "x"

    second_approval = _approve(
        "process.signal", "execute", process_id, {"signal": "USR1"})
    refused = tools.processes.signal(
        process_id, "USR1", approval_id=second_approval)
    assert refused.state == "failed" and refused.data["cancellable"] is False
    assert marker.read_text() == "x"

    os.kill(record.process.pid, terminal_tools.signals.SIGRTMIN)
    record.process.wait(timeout=5)
    tools.processes._refresh(record)


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


def test_process_manager_async_lifetime_stays_bound_to_creation_database(
        monkeypatch, tmp_path):
    first_database = tmp_path / "first" / "state.sqlite3"
    second_database = tmp_path / "second" / "state.sqlite3"
    monkeypatch.setattr(state_store, "STATE_DB_PATH", first_database)
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    policy.add_rule("sandbox_escape", "allow", target="host")
    manager = terminal_tools.ProcessManager(log_root=tmp_path / "logs")
    started = manager.start(execution.ExecutionRequest(
        (sys.executable, "-c", "import time; time.sleep(30)"),
        cwd=str(tmp_path), allowed_paths=(str(tmp_path,),),
    ))
    record = manager._processes[started.data["process_id"]]

    monkeypatch.setattr(state_store, "STATE_DB_PATH", second_database)
    state_store.ensure_state_store()
    assert manager.close(timeout=10) is True
    assert record.finalized.is_set() and not manager._watch_errors

    with sqlite3.connect(first_database) as first:
        assert first.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert first.execute(
            "SELECT state FROM action_journal WHERE id=?", (record.action_id,)
        ).fetchone()[0] in {"succeeded", "failed"}
        assert first.execute(
            "SELECT COUNT(*) FROM resource_leases WHERE resource_key=?",
            (record.id,),
        ).fetchone()[0] == 0
    with sqlite3.connect(second_database) as second:
        assert second.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert second.execute(
            "SELECT COUNT(*) FROM action_journal WHERE id=?", (record.action_id,)
        ).fetchone()[0] == 0


def test_lease_heartbeat_keeps_its_creation_database_context(monkeypatch, tmp_path):
    first_database = tmp_path / "heartbeat-first" / "state.sqlite3"
    second_database = tmp_path / "heartbeat-second" / "state.sqlite3"
    monkeypatch.setattr(state_store, "STATE_DB_PATH", first_database)
    owner = terminal_tools.Owner.create("database-bound-heartbeat")
    assert terminal_tools.claim("fixture", "resource", owner, lease_seconds=1)
    with sqlite3.connect(first_database) as first:
        original = first.execute(
            "SELECT heartbeat_at FROM resource_leases WHERE resource_type='fixture' "
            "AND resource_key='resource'"
        ).fetchone()[0]
    heartbeat = terminal_tools.Heartbeat(
        "fixture", "resource", owner, lease_seconds=1)
    heartbeat.__enter__()
    monkeypatch.setattr(state_store, "STATE_DB_PATH", second_database)
    state_store.ensure_state_store()
    deadline = time.monotonic() + 3
    changed = False
    while time.monotonic() < deadline:
        with sqlite3.connect(first_database) as first:
            changed = first.execute(
                "SELECT heartbeat_at FROM resource_leases WHERE resource_type='fixture' "
                "AND resource_key='resource'"
            ).fetchone()[0] != original
        if changed:
            break
        time.sleep(0.01)
    heartbeat.__exit__(None, None, None)
    assert changed and not heartbeat.lost and heartbeat.error is None
    with sqlite3.connect(second_database) as second:
        assert second.execute(
            "SELECT COUNT(*) FROM resource_leases WHERE resource_type='fixture' "
            "AND resource_key='resource'"
        ).fetchone()[0] == 0
    with state_store.state_db_path_scope(first_database):
        terminal_tools.release("fixture", "resource", owner)


def test_process_manager_close_fences_concurrent_start(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    policy.add_rule("sandbox_escape", "allow", target="host")
    manager = terminal_tools.ProcessManager(log_root=tmp_path / "logs")
    entered = threading.Event()
    release_start = threading.Event()
    actual_start = manager._start

    def delayed_start(*args, **kwargs):
        entered.set()
        assert release_start.wait(timeout=5)
        return actual_start(*args, **kwargs)

    monkeypatch.setattr(manager, "_start", delayed_start)
    outcome = {}
    starter = threading.Thread(target=lambda: outcome.setdefault(
        "started", manager.start(execution.ExecutionRequest(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            cwd=str(tmp_path), allowed_paths=(str(tmp_path),),
        ))))
    starter.start()
    assert entered.wait(timeout=5)
    closer = threading.Thread(target=lambda: outcome.setdefault(
        "closed", manager.close(timeout=10)))
    closer.start()
    with manager._lifecycle:
        assert manager._lifecycle.wait_for(lambda: manager._closed, timeout=5)
    assert closer.is_alive()
    release_start.set()
    starter.join(timeout=10)
    closer.join(timeout=10)
    assert not starter.is_alive() and not closer.is_alive()
    assert outcome["closed"] is True
    record = manager._processes[outcome["started"].data["process_id"]]
    assert record.finalized.is_set() and record.process.poll() is not None
    with pytest.raises(RuntimeError, match="closed"):
        manager.start(execution.ExecutionRequest(
            (sys.executable, "-c", "pass"), cwd=str(tmp_path),
            allowed_paths=(str(tmp_path),),
        ))


def test_background_process_dies_with_owning_manager_process(monkeypatch, tmp_path):
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=_start_background_then_exit,
        args=(tmp_path / "state.sqlite3", tmp_path, tmp_path / "logs",
              str(tmp_path), output),
    )
    process.start()
    process_id, managed_pids = output.get(timeout=10)
    process.join(timeout=10)
    assert process.exitcode == 0
    _assert_processes_gone(managed_pids)
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    restarted = terminal_tools.ProcessManager(log_root=tmp_path / "restarted-logs")
    with state_store.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM resource_leases "
            "WHERE resource_type='managed-process' AND resource_key=?",
            (process_id,),
        ).fetchone()[0] == 0
    with pytest.raises(KeyError, match="unknown process"):
        restarted.poll(process_id)


@pytest.mark.parametrize(
    "signal_name", ["TERM", "INT", "HUP", "USR1", "USR2", "STOP", "CONT"])
def test_user_signal_cannot_disable_owner_death_cleanup(tmp_path, signal_name):
    context = multiprocessing.get_context("spawn")
    output, commands, acknowledged = context.Queue(), context.Queue(), context.Queue()
    manager = context.Process(
        target=_start_background_signal_then_wait,
        args=(tmp_path / "state.sqlite3", tmp_path, tmp_path / "logs",
              str(tmp_path), output, commands, acknowledged),
    )
    manager.start()
    process_id, supervisor_pid, business_pid, descendant_pid = output.get(timeout=10)
    assert process_id.startswith("process-")
    commands.put(signal_name)
    truth = acknowledged.get(timeout=10)
    assert truth["requested"] is True and truth["signal"] == signal_name
    if signal_name == "STOP":
        assert _process_state(business_pid) in {"T", "t"}
        assert _process_state(supervisor_pid) not in {"T", "t"}
    manager.terminate()
    manager.join(timeout=10)
    assert manager.exitcode is not None
    _assert_processes_gone((supervisor_pid, business_pid, descendant_pid))
