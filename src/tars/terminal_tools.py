from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import signal as signals
import shlex
import subprocess
import threading
import uuid

from .config import STATE_ROOT
from .execution_backends import ExecutionRequest, GuardedExecutor, HostBackend, ResourceLimits
from .policy import ScopeRequest, canonical_path
from .tool_core import ToolResult, ToolRuntime
from .secret_store import SecretStore


@dataclass
class ManagedProcess:
    id: str
    process: subprocess.Popen
    argv: tuple[str, ...]
    cwd: str
    target: str
    started_at: str
    stdout_path: Path
    stderr_path: Path
    action_id: str
    cancellable: bool = True
    completed_at: str | None = None


def _stamp():
    return datetime.now(timezone.utc).isoformat()


class ProcessManager:
    def __init__(self, *, log_root=None, runtime=None, secret_store=None):
        self.log_root = Path(log_root or (STATE_ROOT / "process-logs"))
        self.log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.log_root.chmod(0o700)
        self.runtime = runtime or ToolRuntime()
        self.secret_store = secret_store or SecretStore()
        self._processes = {}
        self._handles = {}
        self._lock = threading.Lock()

    def start(self, request, *, approval_id=None, task_id=None, session_id=None):
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
        try:
            resolved = self.secret_store.resolve_many(
                request.environment_refs, consumer="execution:background")
            environment.update(resolved)
            secrets = tuple(environment[name].encode() for name in request.environment_refs)
            process = subprocess.Popen(
                argv, cwd=cwd, env=environment, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
            )
        except Exception as exc:
            stdout_handle.close()
            stderr_handle.close()
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        record = ManagedProcess(
            process_id, process, request.argv, cwd, "host", _stamp(), stdout_path,
            stderr_path, actions[-1].id,
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
            self._handles[process_id] = (stdout_handle, stderr_handle, actions, threads)
        data = {
            "process_id": process_id, "pid": process.pid, "target": "host", "cwd": cwd,
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
        try:
            return self._processes[process_id]
        except KeyError as exc:
            raise KeyError(f"unknown process: {process_id}") from exc

    def _refresh(self, record):
        code = record.process.poll()
        if code is not None and record.completed_at is None:
            record.completed_at = _stamp()
            handles = self._handles.pop(record.id, None)
            if handles:
                for thread in handles[3]:
                    thread.join(timeout=5)
                handles[0].close()
                handles[1].close()
                state = "succeeded" if code == 0 else "failed"
                self.runtime.finish(handles[2], state=state, result={
                    "process_id": record.id, "exit_code": code,
                    "stdout_ref": str(record.stdout_path), "stderr_ref": str(record.stderr_path),
                })
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

    def list(self, *, task_id=None, session_id=None):
        return self._read_result(
            "process.list", "process-manager",
            lambda: [self._status(record) for record in self._processes.values()],
            task_id=task_id, session_id=session_id,
        )

    def _status(self, record):
        code = self._refresh(record)
        return {"process_id": record.id, "pid": record.process.pid,
                "state": "running" if code is None else "exited", "exit_code": code,
                "cancellable": record.cancellable, "started_at": record.started_at,
                "completed_at": record.completed_at}

    def poll(self, process_id, *, task_id=None, session_id=None):
        return self._read_result(
            "process.poll", process_id, lambda: self._status(self._get(process_id)),
            task_id=task_id, session_id=session_id,
        )

    def wait(self, process_id, timeout=None, *, task_id=None, session_id=None):
        def producer():
            record = self._get(process_id)
            try:
                record.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return self._status(record) | {"timed_out": True}
            return self._status(record) | {"timed_out": False}
        return self._read_result("process.wait", process_id, producer,
                                 task_id=task_id, session_id=session_id)

    def logs(self, process_id, *, stream="stdout", offset=0, limit=64_000,
             task_id=None, session_id=None):
        def producer():
            record = self._get(process_id)
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

    def write(self, process_id, data, *, approval_id=None, task_id=None, session_id=None):
        record = self._get(process_id)
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

    def signal(self, process_id, signal_name="TERM", *, approval_id=None,
               task_id=None, session_id=None, _tool="process.signal"):
        record = self._get(process_id)
        destructive = signal_name.upper() == "KILL"
        request = ScopeRequest(
            _tool, "destructive" if destructive else "execute", process_id,
            {"signal": signal_name.upper()}, task_id=task_id, session_id=session_id,
            destructive=destructive,
        )
        actions = self.runtime.authorize((("control", request),), {"control": approval_id})
        if not record.cancellable:
            result = {"process_id": process_id, "requested": False, "cancellable": False}
            self.runtime.finish(actions, state="failed", result=result)
            return ToolResult(_tool, "failed", result,
                              error="process is not cancellable",
                              action_ids=tuple(action.id for action in actions))
        try:
            number = getattr(signals, "SIG" + signal_name.upper())
        except AttributeError as exc:
            raise ValueError(f"unknown signal: {signal_name}") from exc
        if record.process.poll() is None:
            os.killpg(record.process.pid, number)
        result = {"process_id": process_id, "requested": True,
                  "cancellable": True, "signal": signal_name.upper()}
        self.runtime.finish(actions, state="succeeded", result=result)
        evidence = self.runtime.evidence("process", process_id, repr(result), task_id=task_id,
                                         event_uuid=actions[0].event_uuid)
        return ToolResult(_tool, "succeeded", result,
                          action_ids=tuple(action.id for action in actions),
                          evidence_ids=(evidence.id,))

    def kill(self, process_id, **kwargs):
        return self.signal(process_id, "KILL", _tool="process.kill", **kwargs)


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
