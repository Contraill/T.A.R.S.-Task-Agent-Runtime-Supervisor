from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
import os
from pathlib import Path
import select
import signal as signals
import shlex
import subprocess
import threading
import time
import uuid

from .config import STATE_ROOT
from .execution_backends import ExecutionRequest, GuardedExecutor, HostBackend, ResourceLimits
from .action_journal import load_action
from .ownership import Heartbeat, Owner, claim, owner_gone, process_start, release
from .policy import ScopeRequest, canonical_path
from .process_supervision import SupervisedProcess, spawn_supervised
from .state_store import (connect, current_state_db_path, ensure_state_store, json_loads,
                          now_utc, state_db_path_scope, transaction)
from .tool_core import ToolResult, ToolRuntime
from .secret_store import SecretStore


_USER_PROCESS_SIGNALS = {
    "TERM": signals.SIGTERM,
    "INT": signals.SIGINT,
    "HUP": signals.SIGHUP,
    "USR1": signals.SIGUSR1,
    "USR2": signals.SIGUSR2,
    "STOP": signals.SIGSTOP,
    "CONT": signals.SIGCONT,
}


@dataclass(frozen=True)
class ManagedProcess:
    id: str
    process: subprocess.Popen
    business_pid: int
    business_start: str
    process_group_id: int
    supervisor_start: str
    argv: tuple[str, ...]
    cwd: str
    target: str
    started_at: str
    stdout_path: Path
    stderr_path: Path
    action_id: str
    task_id: str | None
    session_id: str | None
    provenance_owner: Owner = field(repr=False)
    supervision: SupervisedProcess = field(repr=False)
    control_write: int = field(repr=False)
    response_read: int = field(repr=False)
    control_lock: threading.Lock = field(default_factory=threading.Lock, repr=False,
                                         compare=False)
    finalized: threading.Event = field(default_factory=threading.Event, repr=False,
                                       compare=False)
    cancellable: bool = True
    completed_at: str | None = field(default=None, compare=False)


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _manager_state(method):
    @wraps(method)
    def bound(self, *args, **kwargs):
        with state_db_path_scope(self.state_db_path):
            return method(self, *args, **kwargs)
    return bound


class ProcessManager:
    def __init__(self, *, log_root=None, runtime=None, secret_store=None):
        self.state_db_path = current_state_db_path()
        self.log_root = Path(log_root or (STATE_ROOT / "process-logs"))
        self.log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.log_root.chmod(0o700)
        self.runtime = runtime or ToolRuntime()
        self.secret_store = secret_store or SecretStore()
        self._processes = {}
        self._handles = {}
        self._watch_errors = {}
        self._lock = threading.Lock()
        self._lifecycle = threading.Condition()
        self._starting = 0
        self._closed = False
        with state_db_path_scope(self.state_db_path):
            self._recover_dead_authority()

    @staticmethod
    def _recover_dead_authority():
        ensure_state_store()
        with transaction(immediate=True) as conn:
            rows = conn.execute(
                "SELECT resource_key,owner_token,owner_pid,owner_start "
                "FROM resource_leases WHERE resource_type='managed-process'"
            ).fetchall()
            for row in rows:
                if not owner_gone(row["owner_pid"], row["owner_start"]):
                    continue
                conn.execute(
                    "DELETE FROM resource_leases WHERE resource_type='managed-process' "
                    "AND resource_key=? AND owner_token=?",
                    (row["resource_key"], row["owner_token"]),
                )

    @_manager_state
    def start(self, request, *, approval_id=None, task_id=None, session_id=None):
        with self._lifecycle:
            if self._closed:
                raise RuntimeError("process manager is closed")
            self._starting += 1
        try:
            return self._start(
                request, approval_id=approval_id, task_id=task_id,
                session_id=session_id,
            )
        finally:
            with self._lifecycle:
                self._starting -= 1
                self._lifecycle.notify_all()

    def _start(self, request, *, approval_id=None, task_id=None, session_id=None):
        cwd = canonical_path(request.cwd or os.getcwd())
        path_request = ScopeRequest(
            "fs.read", "read", cwd, task_id=task_id, session_id=session_id,
            allowed_paths=request.allowed_paths,
        )
        execute_request = ScopeRequest(
            "terminal.run", "execute", "host",
            {"argv": list(request.argv), "cwd": cwd,
             "environment_refs": request.environment_refs, "background": True,
             "authority_contract": {"filesystem": "unrestricted-host",
                                    "network": "unrestricted-host",
                                    "resource_limits": "none"}},
            task_id=task_id, session_id=session_id,
            sandbox_escape=True,
        )
        actions = self.runtime.authorize(
            (("cwd", path_request), ("execute", execute_request)),
            {"execute": approval_id},
        )
        process_id = "process-" + uuid.uuid4().hex
        root = self.log_root / process_id
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
        stdout_path, stderr_path = root / "stdout.log", root / "stderr.log"
        stdout_handle = stdout_path.open("wb")
        stderr_handle = stderr_path.open("wb")
        stdout_path.chmod(0o600)
        stderr_path.chmod(0o600)
        argv = ["/bin/bash", "-lc", request.argv[0]] if request.shell else list(request.argv)
        environment = os.environ.copy()
        process = None
        supervised = None
        provenance_owner = Owner.create("managed-process")
        provenance_heartbeat = None
        secrets = ()
        try:
            resolved = self.secret_store.resolve_many(
                request.environment_refs, consumer="execution:background")
            environment.update(resolved)
            secrets = tuple(environment[name].encode() for name in request.environment_refs)
            supervised = spawn_supervised(
                argv, cwd=cwd, env=environment, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            process = supervised.process
            business_pid = supervised.child_pid
            business_start = supervised.child_start
            supervisor_start = supervised.supervisor_start
            control_write = supervised.control_write
            response_read = supervised.response_read
            metadata = {
                "action_id": actions[-1].id, "task_id": task_id,
                "session_id": session_id, "supervisor_pid": process.pid,
                "supervisor_start": supervisor_start, "business_pid": business_pid,
                "business_start": business_start, "process_group_id": business_pid,
                "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
            }
            if not claim(
                "managed-process", process_id, provenance_owner,
                lease_seconds=30, metadata=metadata,
            ):
                raise RuntimeError("managed process authority registration failed")
            provenance_heartbeat = Heartbeat(
                "managed-process", process_id, provenance_owner, lease_seconds=30)
            provenance_heartbeat.__enter__()
        except Exception as exc:
            if provenance_heartbeat is not None:
                try:
                    provenance_heartbeat.__exit__(type(exc), exc, exc.__traceback__)
                except Exception:
                    pass
            try:
                release("managed-process", process_id, provenance_owner)
            except Exception:
                pass
            if supervised is not None:
                try:
                    supervised.stop(timeout=5)
                except Exception:
                    pass
                finally:
                    supervised.close_control()
            stdout_handle.close()
            stderr_handle.close()
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        record = ManagedProcess(
            process_id, process, business_pid, business_start, business_pid,
            supervisor_start, request.argv, cwd, "host", _stamp(), stdout_path,
            stderr_path, actions[-1].id, task_id, session_id, provenance_owner,
            supervised, control_write, response_read,
        )
        def pump(source, destination):
            try:
                for chunk in iter(source.readline, b""):
                    for secret in secrets:
                        if secret:
                            chunk = chunk.replace(secret, b"[REDACTED]")
                    destination.write(chunk)
                    destination.flush()
            finally:
                source.close()
        threads = (
            threading.Thread(target=pump, args=(process.stdout, stdout_handle), daemon=True),
            threading.Thread(target=pump, args=(process.stderr, stderr_handle), daemon=True),
        )
        for thread in threads:
            thread.start()
        with self._lock:
            self._processes[process_id] = record
            self._handles[process_id] = (
                stdout_handle, stderr_handle, actions, threads, provenance_heartbeat)
        watcher = threading.Thread(
            target=self._watch, args=(record,),
            name=f"tars-process-watch-{process_id}", daemon=True,
        )
        watcher.start()
        data = {
            "process_id": process_id, "pid": business_pid,
            "process_group_id": business_pid, "supervisor_pid": process.pid,
            "target": "host", "cwd": cwd,
            "state": "running", "cancellable": True,
            "stdout_ref": str(stdout_path), "stderr_ref": str(stderr_path),
        }
        evidence = self.runtime.evidence(
            "process", process_id, repr(data), task_id=task_id,
            event_uuid=actions[-1].event_uuid, result_ref=str(stdout_path),
        )
        return ToolResult("process.start", "succeeded", data,
                          action_ids=tuple(action.id for action in actions),
                          evidence_ids=(evidence.id,))

    def _get(self, process_id):
        with self._lock:
            try:
                record = self._processes[process_id]
            except KeyError as exc:
                raise KeyError(f"unknown process: {process_id}") from exc
        self._verify_provenance(record)
        return record

    @staticmethod
    def _provenance_metadata(record):
        return {
            "action_id": record.action_id, "task_id": record.task_id,
            "session_id": record.session_id, "supervisor_pid": record.process.pid,
            "supervisor_start": record.supervisor_start,
            "business_pid": record.business_pid,
            "business_start": record.business_start,
            "process_group_id": record.process_group_id,
            "stdout_path": str(record.stdout_path), "stderr_path": str(record.stderr_path),
        }

    def _verify_provenance(self, record):
        if (type(record) is not ManagedProcess
                or type(record.supervision) is not SupervisedProcess
                or not isinstance(record.process, subprocess.Popen)
                or record.supervision.process is not record.process
                or record.supervision.child_pid != record.business_pid
                or record.supervision.child_start != record.business_start
                or record.supervision.supervisor_start != record.supervisor_start):
            raise PermissionError("process record has no trusted creation provenance")
        with connect() as conn:
            row = conn.execute(
                "SELECT * FROM resource_leases "
                "WHERE resource_type='managed-process' AND resource_key=?",
                (record.id,),
            ).fetchone()
        owner = record.provenance_owner
        lease_valid = bool(
            row and row["owner_token"] == owner.token
            and row["owner_pid"] == owner.pid
            and row["owner_start"] == owner.process_start
            and row["expires_at"] > now_utc()
            and process_start(row["owner_pid"]) == row["owner_start"]
            and json_loads(row["metadata_json"], {}) == self._provenance_metadata(record)
        )
        if lease_valid:
            if record.process.poll() is None:
                if process_start(record.process.pid) != record.supervisor_start:
                    raise PermissionError("managed process supervisor identity changed")
                current_business = process_start(record.business_pid)
                if current_business and current_business != record.business_start:
                    raise PermissionError("managed business process identity changed")
            return

        action = load_action(record.action_id)
        result = action.result or {}
        terminal_matches = (
            action.tool == "terminal.run"
            and action.task_id == record.task_id
            and action.session_id == record.session_id
            and action.state in {"succeeded", "failed"}
            and result.get("process_id") == record.id
            and result.get("pid") == record.business_pid
            and result.get("supervisor_pid") == record.process.pid
            and result.get("stdout_ref") == str(record.stdout_path)
            and result.get("stderr_ref") == str(record.stderr_path)
            and record.process.poll() is not None
        )
        if not terminal_matches:
            raise PermissionError("process record is not backed by trusted durable provenance")

    @staticmethod
    def _verify_context(record, *, task_id=None, session_id=None):
        if record.task_id != task_id or record.session_id != session_id:
            raise PermissionError("managed process belongs to another task or session")

    def _owned(self, process_id, *, task_id=None, session_id=None):
        record = self._get(process_id)
        self._verify_context(record, task_id=task_id, session_id=session_id)
        return record

    @_manager_state
    def _watch(self, record):
        record.process.wait()
        try:
            self._refresh(record)
        except Exception as exc:
            with self._lock:
                self._watch_errors[record.id] = str(exc)

    @_manager_state
    def _refresh(self, record):
        code = record.process.poll()
        handles = None
        finalizing = False
        if code is not None:
            with self._lock:
                if record.completed_at is None:
                    object.__setattr__(record, "completed_at", _stamp())
                    handles = self._handles.pop(record.id, None)
                else:
                    finalizing = not record.finalized.is_set()
            if handles:
                try:
                    for thread in handles[3]:
                        thread.join(timeout=5)
                    handles[0].close()
                    handles[1].close()
                    state = "succeeded" if code == 0 else "failed"
                    self.runtime.finish(handles[2], state=state, result={
                        "process_id": record.id, "pid": record.business_pid,
                        "supervisor_pid": record.process.pid, "exit_code": code,
                        "stdout_ref": str(record.stdout_path),
                        "stderr_ref": str(record.stderr_path),
                    })
                finally:
                    record.supervision.close_control()
                    try:
                        handles[4].__exit__(None, None, None)
                    finally:
                        try:
                            release("managed-process", record.id, record.provenance_owner)
                        finally:
                            record.finalized.set()
            elif finalizing and not record.finalized.wait(timeout=5):
                raise RuntimeError("managed process finalization did not quiesce")
        return code

    def _read_result(self, tool, target, producer, *, task_id=None, session_id=None):
        request = ScopeRequest(tool, "read", target, task_id=task_id, session_id=session_id)
        actions = self.runtime.authorize((("read", request),))
        try:
            data = producer()
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=data)
        evidence = self.runtime.evidence(
            "process", target, repr(data), task_id=task_id,
            event_uuid=actions[0].event_uuid,
            result_ref=data.get("full_log_ref", "") if isinstance(data, dict) else "",
        )
        return ToolResult(tool, "succeeded", {"processes": data} if isinstance(data, list) else data,
                          action_ids=tuple(action.id for action in actions),
                          evidence_ids=(evidence.id,))

    @_manager_state
    def list(self, *, task_id=None, session_id=None):
        with self._lock:
            records = tuple(self._processes.values())
        visible = []
        for record in records:
            try:
                self._verify_provenance(record)
                self._verify_context(record, task_id=task_id, session_id=session_id)
            except (KeyError, PermissionError, RuntimeError):
                continue
            visible.append(record)
        return self._read_result(
            "process.list", "process-manager",
            lambda: [self._status(record) for record in visible],
            task_id=task_id, session_id=session_id,
        )

    def _status(self, record):
        code = self._refresh(record)
        return {"process_id": record.id, "pid": record.business_pid,
                "process_group_id": record.process_group_id,
                "supervisor_pid": record.process.pid,
                "state": "running" if code is None else "exited", "exit_code": code,
                "cancellable": record.cancellable, "started_at": record.started_at,
                "completed_at": record.completed_at,
                "management_error": self._watch_errors.get(record.id, "")}

    @_manager_state
    def poll(self, process_id, *, task_id=None, session_id=None):
        record = self._owned(process_id, task_id=task_id, session_id=session_id)
        return self._read_result(
            "process.poll", process_id, lambda: self._status(record),
            task_id=task_id, session_id=session_id,
        )

    @_manager_state
    def wait(self, process_id, timeout=None, *, task_id=None, session_id=None):
        record = self._owned(process_id, task_id=task_id, session_id=session_id)
        def producer():
            try:
                record.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return self._status(record) | {"timed_out": True}
            return self._status(record) | {"timed_out": False}
        return self._read_result("process.wait", process_id, producer,
                                 task_id=task_id, session_id=session_id)

    @_manager_state
    def logs(self, process_id, *, stream="stdout", offset=0, limit=64_000,
             task_id=None, session_id=None):
        record = self._owned(process_id, task_id=task_id, session_id=session_id)
        def producer():
            path = record.stdout_path if stream == "stdout" else record.stderr_path
            with path.open("rb") as handle:
                handle.seek(offset)
                payload = handle.read(limit + 1)
            return {"process_id": process_id, "stream": stream, "offset": offset,
                    "content": payload[:limit].decode(errors="replace"),
                    "next_offset": offset + min(len(payload), limit),
                    "truncated": len(payload) > limit, "full_log_ref": str(path)}
        return self._read_result("process.logs", process_id, producer,
                                 task_id=task_id, session_id=session_id)

    @_manager_state
    def write(self, process_id, data, *, approval_id=None, task_id=None, session_id=None):
        record = self._owned(process_id, task_id=task_id, session_id=session_id)
        request = ScopeRequest(
            "process.write", "execute", process_id, {"data": data.encode()},
            task_id=task_id, session_id=session_id,
        )
        actions = self.runtime.authorize((("execute", request),), {"execute": approval_id})
        if record.process.poll() is not None or record.process.stdin is None:
            self.runtime.finish(actions, state="failed", result={"error": "input unavailable"})
            raise RuntimeError("process input is unavailable")
        record.process.stdin.write(data.encode())
        record.process.stdin.flush()
        result = {"process_id": process_id, "bytes": len(data.encode())}
        self.runtime.finish(actions, state="succeeded", result=result)
        evidence = self.runtime.evidence("process", process_id, repr(result), task_id=task_id,
                                         event_uuid=actions[0].event_uuid)
        return ToolResult("process.write", "succeeded", result,
                          action_ids=tuple(action.id for action in actions),
                          evidence_ids=(evidence.id,))

    @_manager_state
    def signal(self, process_id, signal_name="TERM", *, approval_id=None,
               task_id=None, session_id=None):
        return self._send_signal(
            process_id, signal_name, tool="process.signal", approval_id=approval_id,
            task_id=task_id, session_id=session_id,
        )

    @staticmethod
    def _dispatch_signal(record, signal_name):
        if record.process.poll() is not None:
            return "already-exited"
        with record.control_lock:
            if record.process.poll() is not None:
                return "already-exited"
            try:
                command = (signal_name + "\n").encode("ascii")
                if os.write(record.control_write, command) != len(command):
                    object.__setattr__(record, "cancellable", False)
                    return "ambiguous"
                readable, _, _ = select.select((record.response_read,), (), (), 5.0)
                if not readable:
                    object.__setattr__(record, "cancellable", False)
                    return "ambiguous"
                response = os.read(record.response_read, 64).strip().decode(
                    "ascii", errors="replace")
            except OSError:
                object.__setattr__(record, "cancellable", False)
                return "ambiguous"
        if response in {"dispatched", "already-exited"}:
            return response
        object.__setattr__(record, "cancellable", False)
        return "ambiguous"

    def _send_signal(self, process_id, signal_name, *, tool, approval_id=None,
                     task_id=None, session_id=None):
        record = self._owned(process_id, task_id=task_id, session_id=session_id)
        signal_name = str(signal_name).upper()
        if tool == "process.kill":
            if signal_name != "KILL":
                raise ValueError("process.kill only supports KILL")
            destructive = True
        else:
            if signal_name not in _USER_PROCESS_SIGNALS:
                raise ValueError(
                    "process.signal supports TERM, INT, HUP, USR1, USR2, STOP, and CONT"
                )
            destructive = False
        request = ScopeRequest(
            tool, "destructive" if destructive else "execute", process_id,
            {"signal": signal_name}, task_id=task_id, session_id=session_id,
            destructive=destructive,
        )
        actions = self.runtime.authorize((("control", request),), {"control": approval_id})
        if not record.cancellable:
            result = {"process_id": process_id, "requested": False, "cancellable": False}
            self.runtime.finish(actions, state="failed", result=result)
            return ToolResult(tool, "failed", result,
                              error="process is not cancellable",
                              action_ids=tuple(action.id for action in actions))
        outcome = self._dispatch_signal(record, signal_name)
        if outcome == "ambiguous":
            result = {"process_id": process_id, "requested": None,
                      "cancellable": False, "signal": signal_name,
                      "outcome": "dispatch-ambiguous"}
            self.runtime.finish(actions, state="unknown", result=result)
            return ToolResult(tool, "unknown", result,
                              error="process signal outcome is ambiguous",
                              action_ids=tuple(action.id for action in actions))
        requested = outcome == "dispatched"
        result = {"process_id": process_id, "requested": requested,
                  "cancellable": True, "signal": signal_name,
                  "outcome": outcome}
        self.runtime.finish(actions, state="succeeded", result=result)
        evidence = self.runtime.evidence("process", process_id, repr(result), task_id=task_id,
                                         event_uuid=actions[0].event_uuid)
        return ToolResult(tool, "succeeded", result,
                          action_ids=tuple(action.id for action in actions),
                          evidence_ids=(evidence.id,))

    @_manager_state
    def kill(self, process_id, **kwargs):
        return self._send_signal(process_id, "KILL", tool="process.kill", **kwargs)

    @_manager_state
    def close(self, timeout=10.0):
        deadline = time.monotonic() + max(0.0, float(timeout))
        errors = []
        with self._lifecycle:
            self._closed = True
            self._lifecycle.notify_all()
            while self._starting:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    errors.append("process startup did not quiesce")
                    break
                self._lifecycle.wait(remaining)
        with self._lock:
            records = tuple(self._processes.values())
        for record in records:
            try:
                self._verify_provenance(record)
                remaining = deadline - time.monotonic()
                if record.process.poll() is None:
                    if remaining <= 0:
                        raise TimeoutError("managed process shutdown deadline expired")
                    record.supervision.stop(timeout=max(0.1, remaining))
                self._refresh(record)
                remaining = deadline - time.monotonic()
                if not record.finalized.wait(timeout=max(0.0, remaining)):
                    raise TimeoutError("managed process finalization did not quiesce")
            except Exception as exc:
                errors.append(f"{record.id}: {exc}")
        if errors:
            raise RuntimeError("process manager shutdown was incomplete: " + "; ".join(errors))
        return True


class TerminalTools:
    def __init__(self, *, executor=None, processes=None, output_limit=64_000, log_root=None):
        self.executor = executor or GuardedExecutor({"host": HostBackend()})
        self.processes = processes or ProcessManager()
        self.output_limit = output_limit
        self.log_root = Path(log_root or (STATE_ROOT / "process-logs"))
        self.log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.log_root.chmod(0o700)
        self.runtime = self.processes.runtime

    def run(self, argv, *, cwd=None, target="host", timeout=300, shell=False,
            background=False, tty=False, environment_refs=None, allowed_paths=(), approval_id=None,
            task_id=None, session_id=None, cpus=None, memory=None, processes=256,
            workspace=None, workspace_mode="ephemeral", mounts=(), image=None,
            network=False, network_hosts=(), persistent=False, container_name=None,
            ssh_target=None):
        if tty:
            if target != "host" or background:
                raise ValueError("PTY execution requires a foreground host target")
            command = argv[0] if shell else shlex.join(tuple(argv))
            argv = ("script", "-qefc", command, "/dev/null")
            shell = False
        request = ExecutionRequest(
            tuple(argv), cwd=cwd, shell=shell, environment_refs=environment_refs or {},
            limits=ResourceLimits(cpus=cpus, memory=memory, processes=processes, timeout=timeout),
            allowed_paths=tuple(allowed_paths), workspace=workspace,
            workspace_mode=workspace_mode, mounts=tuple(mounts), image=image,
            network=network, network_hosts=tuple(network_hosts), persistent=persistent,
            container_name=container_name, target=ssh_target,
        )
        if background:
            if target != "host":
                raise ValueError("background lifecycle is currently supported on host targets")
            return self.processes.start(
                request, approval_id=approval_id, task_id=task_id, session_id=session_id,
            )
        result = self.executor.execute(
            target, request, approval_id=approval_id, task_id=task_id, session_id=session_id,
        )
        stdout, stderr = result.stdout, result.stderr
        truncated = len(stdout) > self.output_limit or len(stderr) > self.output_limit
        stdout_ref = stderr_ref = ""
        if truncated:
            root = self.log_root / result.execution_id
            root.mkdir(parents=True, exist_ok=False)
            root.chmod(0o700)
            stdout_path, stderr_path = root / "stdout.log", root / "stderr.log"
            stdout_path.write_text(stdout)
            stderr_path.write_text(stderr)
            stdout_path.chmod(0o600)
            stderr_path.chmod(0o600)
            stdout_ref, stderr_ref = str(stdout_path), str(stderr_path)
        data = {
            "execution_id": result.execution_id, "exit_code": result.exit_code,
            "stdout": stdout[:self.output_limit], "stderr": stderr[:self.output_limit],
            "target": result.target, "cwd": result.cwd, "started_at": result.started_at,
            "completed_at": result.completed_at, "timed_out": result.timed_out,
            "truncated": truncated, "pty": tty,
            "stdout_ref": stdout_ref, "stderr_ref": stderr_ref,
        }
        evidence = self.runtime.evidence(
            "terminal", result.target, repr(data), task_id=task_id,
            result_ref=stdout_ref or stderr_ref,
            metadata={"exit_code": result.exit_code, "state": result.state,
                      "truncated": truncated},
        )
        return ToolResult("terminal.run", result.state, data,
                          error=result.stderr if not result.succeeded else "",
                          action_ids=result.action_ids, evidence_ids=(evidence.id,))
