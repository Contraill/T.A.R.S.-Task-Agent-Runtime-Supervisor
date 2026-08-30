from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
from typing import Protocol

from .policy import ScopeRequest
from .tool_core import ToolResult, ToolRuntime


class ServiceTools:
    def __init__(self, *, runtime=None, runner=subprocess.run):
        self.runtime = runtime or ToolRuntime()
        self.runner = runner

    def _systemctl(self, arguments):
        return self.runner(["systemctl", "--user", *arguments], capture_output=True,
                           text=True, check=False)

    def status(self, unit, *, task_id=None, session_id=None):
        request = ScopeRequest("service.status", "read", unit, task_id=task_id,
                               session_id=session_id)
        actions = self.runtime.authorize((("read", request),))
        proc = self._systemctl(["show", unit, "--property=ActiveState,SubState,LoadState"])
        values = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
        data = {"unit": unit, "exit_code": proc.returncode, **values, "stderr": proc.stderr}
        state = "succeeded" if proc.returncode == 0 else "failed"
        self.runtime.finish(actions, state=state, result=data)
        evidence = self.runtime.evidence("service", unit, repr(data), task_id=task_id,
                                         event_uuid=actions[0].event_uuid)
        return ToolResult("service.status", state, data, error=proc.stderr if proc.returncode else "",
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def mutate(self, operation, unit, *, approval_id=None, task_id=None, session_id=None):
        if operation not in {"start", "stop", "restart"}:
            raise ValueError(f"unsupported service operation: {operation}")
        request = ScopeRequest(
            f"service.{operation}", "service", unit, {"operation": operation},
            task_id=task_id, session_id=session_id,
        )
        actions = self.runtime.authorize((("service", request),), {"service": approval_id})
        before = self._systemctl(["show", unit, "--property=ActiveState,SubState"])
        changed = self._systemctl([operation, unit])
        after = self._systemctl(["show", unit, "--property=ActiveState,SubState"])
        values = dict(line.split("=", 1) for line in after.stdout.splitlines() if "=" in line)
        expected = "inactive" if operation == "stop" else "active"
        verified = changed.returncode == 0 and values.get("ActiveState") == expected
        data = {"unit": unit, "operation": operation, "exit_code": changed.returncode,
                "before": before.stdout, "after": values, "verified": verified,
                "stderr": changed.stderr}
        state = "succeeded" if verified else "failed"
        self.runtime.finish(actions, state=state, result=data)
        evidence = self.runtime.evidence(
            "service", unit, repr(data), task_id=task_id,
            event_uuid=actions[0].event_uuid,
        )
        return ToolResult(f"service.{operation}", state, data,
                          error="" if verified else changed.stderr or "post-state verification failed",
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def logs(self, unit, *, lines=100, task_id=None, session_id=None):
        request = ScopeRequest("service.logs", "read", unit, task_id=task_id,
                               session_id=session_id)
        actions = self.runtime.authorize((("read", request),))
        proc = self.runner(
            ["journalctl", "--user", "-u", unit, "-n", str(int(lines)), "--no-pager"],
            capture_output=True, text=True, check=False,
        )
        data = {"unit": unit, "exit_code": proc.returncode, "logs": proc.stdout,
                "stderr": proc.stderr}
        state = "succeeded" if proc.returncode == 0 else "failed"
        self.runtime.finish(actions, state=state, result=data)
        evidence = self.runtime.evidence("service", unit, repr(data), task_id=task_id,
                                         event_uuid=actions[0].event_uuid)
        return ToolResult("service.logs", state, data, error=proc.stderr if proc.returncode else "",
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))


class PackageBackend(Protocol):
    identity: str
    support: str

    def status(self) -> dict: ...
    def search(self, query, **kwargs) -> ToolResult: ...
    def info(self, package, **kwargs) -> ToolResult: ...
    def installed(self, package=None, **kwargs) -> ToolResult: ...
    def install(self, packages, **kwargs) -> ToolResult: ...
    def remove(self, packages, **kwargs) -> ToolResult: ...
    def upgrade(self, **kwargs) -> ToolResult: ...
    def orphans(self, **kwargs) -> ToolResult: ...


class PacmanBackend:
    identity = "pacman"
    support = "reference-tested"

    def __init__(self, *, runtime=None, runner=subprocess.run, binary=None):
        self.runtime = runtime or ToolRuntime()
        self.runner = runner
        self.binary = binary or shutil.which("pacman")

    def status(self):
        return {"backend": self.identity, "available": bool(self.binary),
                "support": self.support,
                "message": "available" if self.binary else "pacman is unavailable"}

    def _run(self, tool, arguments, *, mutation=False, approval_id=None,
             task_id=None, session_id=None):
        if not self.binary:
            raise RuntimeError("Pacman backend is unavailable")
        effect = "elevated" if mutation else "read"
        request = ScopeRequest(
            tool, effect, "pacman", {"argv": [self.binary, *arguments]},
            task_id=task_id, session_id=session_id, elevated=mutation,
        )
        actions = self.runtime.authorize((("package", request),), {"package": approval_id})
        proc = self.runner([self.binary, *arguments], capture_output=True, text=True, check=False)
        data = {"backend": self.identity, "exit_code": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr, "arguments": arguments}
        state = "succeeded" if proc.returncode == 0 else "failed"
        self.runtime.finish(actions, state=state, result=data)
        return ToolResult(tool, state, data, error=proc.stderr if proc.returncode else "",
                          action_ids=tuple(a.id for a in actions))

    def search(self, query, **kwargs): return self._run("package.search", ["-Ss", query], **kwargs)
    def info(self, package, **kwargs): return self._run("package.info", ["-Si", package], **kwargs)
    def installed(self, package=None, **kwargs): return self._run("package.installed", ["-Q", *( [package] if package else [])], **kwargs)
    def orphans(self, **kwargs): return self._run("package.orphans", ["-Qdt"], **kwargs)
    def install(self, packages, *, approval_id=None, **kwargs):
        return self._run("package.install", ["-Syu", "--needed", "--noconfirm", *packages],
                         mutation=True, approval_id=approval_id, **kwargs)
    def remove(self, packages, *, approval_id=None, **kwargs):
        return self._run("package.remove", ["-Rns", "--noconfirm", *packages],
                         mutation=True, approval_id=approval_id, **kwargs)
    def upgrade(self, *, approval_id=None, **kwargs):
        return self._run("package.upgrade", ["-Syu", "--noconfirm"], mutation=True,
                         approval_id=approval_id, **kwargs)


class SystemInspection:
    def __init__(self, *, runtime=None):
        self.runtime = runtime or ToolRuntime()

    def _collect(self, tool, producer, *, task_id=None, session_id=None):
        request = ScopeRequest(tool, "read", "host", task_id=task_id, session_id=session_id)
        actions = self.runtime.authorize((("read", request),))
        try:
            data = producer()
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=data)
        evidence = self.runtime.evidence(
            "system", "host", repr(data), task_id=task_id, event_uuid=actions[0].event_uuid,
            metadata={"tool": tool},
        )
        return ToolResult(tool, "succeeded", {"result": data},
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def info(self, **kwargs):
        return self._collect("system.info", lambda: {
            "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "python": platform.python_version(),
            "hostname": socket.gethostname(),
        }, **kwargs)

    def processes(self, limit=200, **kwargs):
        def collect():
            rows = []
            for path in sorted(Path("/proc").glob("[0-9]*"), key=lambda value: int(value.name)):
                if len(rows) >= limit:
                    break
                try:
                    rows.append({"pid": int(path.name),
                                 "command": (path / "comm").read_text().strip(),
                                 "state": (path / "stat").read_text().split()[2]})
                except (OSError, IndexError):
                    continue
            return rows
        return self._collect("system.processes", collect, **kwargs)

    def storage(self, paths=("/",), **kwargs):
        return self._collect("system.storage", lambda: [
            {"path": path, "total": usage.total, "used": usage.used, "free": usage.free}
            for path in paths for usage in (shutil.disk_usage(path),)
        ], **kwargs)

    def network(self, **kwargs):
        return self._collect("system.network", lambda: {
            "hostname": socket.gethostname(),
            "interfaces": sorted(name for _, name in socket.if_nameindex()),
        }, **kwargs)

    def hardware(self, **kwargs):
        def collect():
            cpu = platform.processor() or platform.machine()
            memory_kib = 0
            try:
                for line in Path("/proc/meminfo").read_text().splitlines():
                    if line.startswith("MemTotal:"):
                        memory_kib = int(line.split()[1])
                        break
            except OSError:
                pass
            return {"cpu": cpu, "logical_cpus": os.cpu_count(), "memory_kib": memory_kib}
        return self._collect("system.hardware", collect, **kwargs)

    def logs(self, *, lines=100, runner=subprocess.run, **kwargs):
        def collect():
            proc = runner(["journalctl", "--user", "-n", str(int(lines)), "--no-pager"],
                          capture_output=True, text=True, check=False)
            return {"exit_code": proc.returncode, "logs": proc.stdout, "stderr": proc.stderr}
        return self._collect("system.logs", collect, **kwargs)
