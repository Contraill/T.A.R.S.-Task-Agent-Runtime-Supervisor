import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from tars.mcp import StdioTransport


def _process_state(pid):
    try:
        value = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return ""
    return value[value.rfind(")") + 2:].split()[0]


def _assert_processes_gone(pids):
    remaining = set(pids)
    for _ in range(300):
        for pid in tuple(remaining):
            if _process_state(pid) in {"", "Z"}:
                remaining.discard(pid)
        if not remaining:
            return
        time.sleep(0.01)
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    pytest.fail(f"supervised MCP process topology survived its owner: {remaining}")


def _helper_program(helper_pid_path, descendant_pid_path, request_path):
    descendant = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(descendant_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    return (
        "import os,signal,subprocess,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(helper_pid_path)!r}).write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}],start_new_session=True); "
        "line=sys.stdin.readline(); "
        f"Path({str(request_path)!r}).write_text('received' if line else 'eof'); "
        "time.sleep(60)"
    )


def _owner_starts_request_then_dies(root, connection):
    root = Path(root)
    helper_pid_path = root / "helper.pid"
    descendant_pid_path = root / "descendant.pid"
    request_path = root / "request.received"
    os.environ["TARS_MCP_LIFETIME_SECRET"] = "ephemeral-owner-secret"
    transport = StdioTransport({
        "argv": [sys.executable, "-u", "-c", _helper_program(
            helper_pid_path, descendant_pid_path, request_path)],
        "env": {"TOKEN": "env:TARS_MCP_LIFETIME_SECRET"},
        "timeout": 60,
    })
    errors = []
    worker = threading.Thread(
        target=lambda: _record_request_error(transport, errors), daemon=True)
    worker.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (helper_pid_path.exists() and descendant_pid_path.exists()
                and request_path.exists()):
            break
        time.sleep(0.01)
    connection.send((
        transport.process.pid, int(helper_pid_path.read_text()),
        int(descendant_pid_path.read_text()), transport.supervised.child_pid,
    ))
    connection.close()
    os._exit(23)


def _record_request_error(transport, errors):
    try:
        transport.request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    except Exception as exc:
        errors.append(exc)


def test_stdio_mcp_helper_and_detached_descendant_die_with_owner(tmp_path):
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    owner = context.Process(
        target=_owner_starts_request_then_dies,
        args=(tmp_path, child_connection),
    )
    owner.start()
    assert parent_connection.poll(15), "MCP owner did not report supervised identities"
    wrapper_pid, helper_pid, descendant_pid, supervised_child = parent_connection.recv()
    assert helper_pid == supervised_child
    owner.join(timeout=10)
    assert owner.exitcode == 23
    _assert_processes_gone((wrapper_pid, helper_pid, descendant_pid))


def test_stdio_mcp_close_interrupts_hung_request_and_quiesces(tmp_path, monkeypatch):
    helper_pid_path = tmp_path / "helper.pid"
    descendant_pid_path = tmp_path / "descendant.pid"
    request_path = tmp_path / "request.received"
    monkeypatch.setenv("TARS_MCP_LIFETIME_SECRET", "ephemeral-close-secret")
    transport = StdioTransport({
        "argv": [sys.executable, "-u", "-c", _helper_program(
            helper_pid_path, descendant_pid_path, request_path)],
        "env": {"TOKEN": "env:TARS_MCP_LIFETIME_SECRET"},
        "timeout": 60,
    })
    errors = []
    worker = threading.Thread(
        target=_record_request_error, args=(transport, errors), daemon=True)
    worker.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not (
            request_path.exists() and helper_pid_path.exists()
            and descendant_pid_path.exists()):
        time.sleep(0.01)
    try:
        assert request_path.exists() and descendant_pid_path.exists()
        wrapper_pid = transport.process.pid
        helper_pid = int(helper_pid_path.read_text())
        descendant_pid = int(descendant_pid_path.read_text())

        started = time.monotonic()
        assert transport.close(timeout=5) is True
        elapsed = time.monotonic() - started
        worker.join(timeout=2)

        assert elapsed < 5
        assert not worker.is_alive() and errors
        assert transport._quiesced.is_set()
        assert not transport._stderr_thread.is_alive()
        assert transport.supervised.control_write == -1
        assert transport.supervised.response_read == -1
        assert transport.secret_values == ()
        assert transport.process.stdin.closed
        assert transport.process.stdout.closed
        assert transport.process.stderr.closed
        _assert_processes_gone((wrapper_pid, helper_pid, descendant_pid))
    finally:
        if transport.process.poll() is None:
            transport.close(timeout=5)


def test_stdio_mcp_initialization_failure_leaves_no_live_wrapper(tmp_path):
    marker = tmp_path / "started"
    code = (
        "from pathlib import Path; "
        f"Path({str(marker)!r}).write_text('started')"
    )
    wrappers = []

    def recording_popen(*args, **kwargs):
        process = subprocess.Popen(*args, **kwargs)
        wrappers.append(process.pid)
        return process

    try:
        transport = StdioTransport({
            "argv": [sys.executable, "-c", code], "timeout": 1,
        }, popen=recording_popen)
    except RuntimeError:
        transport = None
    if transport is not None:
        wrapper_pid = transport.process.pid
        transport.close(timeout=5)
        _assert_processes_gone((wrapper_pid,))
    assert wrappers
    _assert_processes_gone(wrappers)


def test_stdio_mcp_continuously_drains_bounded_stderr():
    code = (
        "import json,sys; "
        "sys.stderr.write('x'*200000); sys.stderr.flush(); "
        "request=json.loads(sys.stdin.readline()); "
        "print(json.dumps({'jsonrpc':'2.0','id':request['id'],'result':{'ok':True}}),flush=True)"
    )
    transport = StdioTransport({
        "argv": [sys.executable, "-u", "-c", code], "timeout": 5,
    })
    wrapper_pid = transport.process.pid
    helper_pid = transport.supervised.child_pid
    try:
        response = transport.request({"jsonrpc": "2.0", "id": 9, "method": "ping"})
        assert response == {"jsonrpc": "2.0", "id": 9, "result": {"ok": True}}
    finally:
        transport.close(timeout=5)
    assert 0 < len(transport._stderr_tail_value) <= 65_536
    _assert_processes_gone((wrapper_pid, helper_pid))
