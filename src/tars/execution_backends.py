from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import os
import math
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
from typing import Protocol
import uuid

from .action_journal import begin_action, finish_action, record_denied
from .approvals import ApprovalBroker
from .policy import ScopeGuard, ScopeRequest, canonical_path, normalize_network_target, redact
from .secret_store import SecretStore, parse_reference


@dataclass(frozen=True)
class ExecutionStatus:
    backend: str
    available: bool
    support: str
    message: str


@dataclass(frozen=True)
class ResourceLimits:
    cpus: float | None = None
    memory: str | None = None
    processes: int = 256
    timeout: float = 300.0


@dataclass(frozen=True)
class ContainerMount:
    source: str
    destination: str
    read_only: bool = True


@dataclass(frozen=True)
class SSHExecutionTarget:
    name: str
    host: str
    user: str
    port: int = 22
    credential_ref: str | None = None
    allowed_commands: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()

    def __post_init__(self):
        if self.credential_ref:
            parse_reference(self.credential_ref)


@dataclass(frozen=True)
class ExecutionRequest:
    argv: tuple[str, ...]
    cwd: str | None = None
    shell: bool = False
    environment_refs: dict[str, str] = field(default_factory=dict)
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    workspace: str | None = None
    mounts: tuple[ContainerMount, ...] = ()
    image: str | None = None
    network: bool = False
    network_hosts: tuple[str, ...] = ()
    persistent: bool = False
    container_name: str | None = None
    target: str | None = None
    allowed_paths: tuple[str, ...] = ()
    workspace_mode: str = "ephemeral"

    def __post_init__(self):
        if not self.argv or not all(isinstance(item, str) and item for item in self.argv):
            raise ValueError("execution argv must contain non-empty strings")
        if self.shell and len(self.argv) != 1:
            raise ValueError("an explicit shell request accepts one command string")
        if self.limits.timeout <= 0 or self.limits.processes <= 0:
            raise ValueError("execution limits must be positive")
        if self.workspace_mode not in {"ephemeral", "read_only", "read_write"}:
            raise ValueError("invalid container workspace mode")
        for reference in self.environment_refs.values():
            parse_reference(reference)


@dataclass(frozen=True)
class ExecutionResult:
    execution_id: str
    backend: str
    target: str
    argv: tuple[str, ...]
    cwd: str
    exit_code: int | None
    stdout: str
    stderr: str
    state: str
    started_at: str
    completed_at: str
    timed_out: bool = False
    action_ids: tuple[str, ...] = ()

    @property
    def succeeded(self):
        return self.state == "succeeded" and self.exit_code == 0

    def audit_result(self):
        limit = 16_384
        return redact({
            "execution_id": self.execution_id, "backend": self.backend,
            "target": self.target, "exit_code": self.exit_code,
            "stdout": self.stdout[:limit], "stderr": self.stderr[:limit],
            "state": self.state, "timed_out": self.timed_out,
            "truncated": len(self.stdout) > limit or len(self.stderr) > limit,
        })


class Runner(Protocol):
    def __call__(self, argv, **kwargs): ...


class ExecutionBackend(Protocol):
    identity: str
    support: str

    def status(self) -> ExecutionStatus: ...
    def execute(self, request: ExecutionRequest, *, authorization) -> ExecutionResult: ...


class _ExecutionAuthorization:
    def __init__(self, action_ids):
        self.action_ids = tuple(action_ids)


def _require_authorization(value):
    if not isinstance(value, _ExecutionAuthorization) or not value.action_ids:
        raise PermissionError("execution backends require guarded authorization")


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _environment(refs, *, secret_store=None, consumer="execution:host"):
    store = secret_store or SecretStore()
    result = os.environ.copy()
    for name, reference in refs.items():
        if not name or not name.replace("_", "a").isalnum() or name[0].isdigit():
            raise ValueError(f"invalid environment name: {name}")
        parse_reference(reference)
        with store.resolve(reference, consumer=consumer) as value:
            result[name] = value
    return result


def _secret_values(refs, *, secret_store=None, consumer="execution:host"):
    return tuple((secret_store or SecretStore()).resolve_many(
        refs, consumer=consumer).values())


def _mask_text(value, secrets):
    result = value
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result


def _completed(backend, target, request, started, proc=None, *, error="", timed_out=False):
    exit_code = None if proc is None else proc.returncode
    stdout = "" if proc is None else str(proc.stdout or "")
    stderr = error or ("" if proc is None else str(proc.stderr or ""))
    state = "succeeded" if exit_code == 0 and not timed_out else "failed"
    return ExecutionResult(
        "exec-" + uuid.uuid4().hex, backend, target, request.argv,
        request.cwd or "", exit_code, stdout, stderr, state, started, _stamp(), timed_out,
    )


class HostBackend:
    identity = "host"
    support = "explicit-escape-hatch"

    def __init__(self, *, runner: Runner = subprocess.run, secret_store=None):
        self.runner = runner
        self.secret_store = secret_store or SecretStore()

    def status(self):
        return ExecutionStatus(
            self.identity, True, self.support,
            "unconfined host execution; filesystem and network declarations are not enforced")

    def execute(self, request, *, authorization=None):
        _require_authorization(authorization)
        cwd = canonical_path(request.cwd or os.getcwd())
        if not Path(cwd).is_dir():
            raise ValueError(f"execution cwd is not a directory: {cwd}")
        argv = ("/bin/bash", "-lc", request.argv[0]) if request.shell else request.argv
        started = _stamp()
        try:
            proc = self.runner(
                list(argv), cwd=cwd, env=_environment(
                    request.environment_refs, secret_store=self.secret_store,
                    consumer="execution:host"),
                capture_output=True, text=True, timeout=request.limits.timeout,
                check=False,
            )
            return _completed(self.identity, "host", request, started, proc)
        except subprocess.TimeoutExpired as exc:
            proc = subprocess.CompletedProcess(argv, None, exc.stdout or "", exc.stderr or "")
            return _completed(self.identity, "host", request, started, proc,
                              error="execution timed out", timed_out=True)


class ContainerBackend:
    identity = "container"
    support = "tested"

    def __init__(self, *, runtime=None, runner: Runner = subprocess.run, rootless_verified=None,
                 secret_store=None):
        self.runtime = runtime or shutil.which("podman") or shutil.which("docker")
        self.runner = runner
        self.secret_store = secret_store or SecretStore()
        self.rootless_verified = (
            rootless_verified if rootless_verified is not None
            else bool(self.runtime and Path(self.runtime).name == "podman" and os.geteuid() != 0)
        )

    def status(self):
        if not self.runtime:
            return ExecutionStatus(self.identity, False, self.support,
                                   "podman or docker is required")
        if not self.rootless_verified:
            return ExecutionStatus(self.identity, False, self.support,
                                   f"{Path(self.runtime).name} rootless mode is not verified")
        return ExecutionStatus(self.identity, True, self.support,
                               f"rootless execution through {Path(self.runtime).name}")

    @staticmethod
    def sandbox_escape(request):
        if request.workspace_mode == "ephemeral" and not request.mounts:
            return False
        if not request.workspace:
            return bool(request.mounts)
        workspace = Path(canonical_path(request.workspace))
        for mount in request.mounts:
            try:
                Path(canonical_path(mount.source)).relative_to(workspace)
            except ValueError:
                return True
        return False

    def _command(self, request, *, allow_host_mounts=False):
        if not self.runtime:
            raise RuntimeError("container backend unavailable: podman or docker is required")
        if not self.rootless_verified:
            raise RuntimeError("container backend requires verified rootless execution")
        if not request.image:
            raise ValueError("container image is required")
        workspace = Path(canonical_path(request.workspace)) if request.workspace else None
        if request.workspace_mode != "ephemeral" and workspace is None:
            raise ValueError("host-backed container workspace is required")
        if workspace is not None and not workspace.is_dir():
            raise ValueError(f"container workspace is not a directory: {workspace}")
        if request.persistent and not request.container_name:
            raise ValueError("persistent containers require an explicit name")
        runtime_name = request.container_name or "tars-" + uuid.uuid4().hex[:12]
        command = [self.runtime, "run", "--name", runtime_name]
        if not request.persistent:
            command.append("--rm")
        if Path(self.runtime).name == "podman":
            command += ["--userns=keep-id", "--timeout", str(math.ceil(request.limits.timeout))]
        command += ["--pull=never", "--cpus", str(request.limits.cpus or 1.0),
                    "--memory", request.limits.memory or "1g",
                    "--pids-limit", str(request.limits.processes)]
        network_mode = "slirp4netns" if Path(self.runtime).name == "podman" else "bridge"
        command += ["--network", network_mode if request.network else "none"]
        if request.workspace_mode == "ephemeral":
            command += ["--tmpfs", "/workspace:rw", "--workdir", "/workspace"]
        else:
            mode = "ro" if request.workspace_mode == "read_only" else "rw"
            command += ["--volume", f"{workspace}:/workspace:{mode}",
                        "--workdir", "/workspace"]
        for mount in request.mounts:
            source = Path(canonical_path(mount.source))
            destination = PurePosixPath(mount.destination)
            if not source.exists():
                raise ValueError(f"container mount source does not exist: {source}")
            if not destination.is_absolute() or ".." in destination.parts:
                raise ValueError(f"invalid container mount destination: {destination}")
            try:
                if workspace is None:
                    raise ValueError
                source.relative_to(workspace)
            except ValueError:
                if not allow_host_mounts:
                    raise PermissionError("mount outside workspace requires sandbox-escape approval")
            mode = "ro" if mount.read_only else "rw"
            command += ["--volume", f"{source}:{destination}:{mode}"]
        runtime_env = _environment(request.environment_refs, secret_store=self.secret_store,
                                   consumer="execution:container")
        for name in request.environment_refs:
            command += ["--env", name]
        command.append(request.image)
        command += ["/bin/bash", "-lc", request.argv[0]] if request.shell else list(request.argv)
        return command, runtime_env

    def execute(self, request, *, authorization=None, allow_host_mounts=False):
        _require_authorization(authorization)
        command, environment = self._command(request, allow_host_mounts=allow_host_mounts)
        started = _stamp()
        try:
            proc = self.runner(
                command, env=environment, capture_output=True, text=True,
                timeout=request.limits.timeout, check=False,
            )
            return _completed(self.identity, request.container_name or "ephemeral", request,
                              started, proc)
        except subprocess.TimeoutExpired as exc:
            name_index = command.index("--name") + 1
            runtime_name = command[name_index]
            cleanup = ([self.runtime, "stop", runtime_name] if request.persistent
                       else [self.runtime, "rm", "-f", runtime_name])
            try:
                self.runner(cleanup, capture_output=True, text=True, timeout=30, check=False)
            except Exception:
                pass
            proc = subprocess.CompletedProcess(command, None, exc.stdout or "", exc.stderr or "")
            return _completed(self.identity, request.container_name or "ephemeral", request,
                              started, proc, error="execution timed out", timed_out=True)


class SSHBackend:
    identity = "ssh"
    support = "experimental"

    def __init__(self, targets=(), *, ssh_binary=None, runner: Runner = subprocess.run,
                 secret_store=None):
        self.targets = {target.name: target for target in targets}
        self.ssh_binary = ssh_binary or shutil.which("ssh")
        self.runner = runner
        self.secret_store = secret_store or SecretStore()

    def status(self):
        if not self.ssh_binary:
            return ExecutionStatus(self.identity, False, self.support, "OpenSSH client missing")
        return ExecutionStatus(self.identity, True, self.support,
                               "client available; remote support is experimental")

    def execute(self, request, *, authorization=None):
        _require_authorization(authorization)
        if not self.ssh_binary:
            raise RuntimeError("SSH backend unavailable: OpenSSH client missing")
        try:
            target = self.targets[request.target]
        except KeyError as exc:
            raise PermissionError("SSH target is not explicitly registered") from exc
        command_name = request.argv[0]
        if not target.allowed_commands or command_name not in target.allowed_commands:
            raise PermissionError(f"SSH command is outside target scope: {command_name}")
        if request.cwd:
            cwd = PurePosixPath(request.cwd)
            if not cwd.is_absolute() or ".." in cwd.parts:
                raise PermissionError("SSH cwd must be an absolute normalized path")
            if not any(cwd == PurePosixPath(root) or PurePosixPath(root) in cwd.parents
                       for root in target.allowed_paths):
                raise PermissionError("SSH cwd is outside target scope")
        destination = f"{target.user}@{target.host}"
        command = [self.ssh_binary, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                   "-o", f"ConnectTimeout={max(1, min(int(request.limits.timeout), 30))}",
                   "-p", str(target.port)]
        if target.credential_ref:
            with self.secret_store.resolve(
                    target.credential_ref, consumer=f"ssh:{target.name}") as identity:
                command += ["-i", identity]
        remote = shlex.join(request.argv)
        if request.cwd:
            remote = f"cd -- {shlex.quote(request.cwd)} && exec {remote}"
        command += [destination, remote]
        started = _stamp()
        try:
            proc = self.runner(
                command, capture_output=True, text=True, timeout=request.limits.timeout,
                check=False,
            )
            return _completed(self.identity, target.name, request, started, proc)
        except subprocess.TimeoutExpired as exc:
            proc = subprocess.CompletedProcess(command, None, exc.stdout or "", exc.stderr or "")
            return _completed(self.identity, target.name, request, started, proc,
                              error="execution timed out", timed_out=True)


class GuardedExecutor:
    def __init__(self, backends=None, *, guard=None, broker=None):
        self.backends = backends or {
            "host": HostBackend(), "container": ContainerBackend(), "ssh": SSHBackend(),
        }
        self.guard = guard or ScopeGuard()
        self.broker = broker or ApprovalBroker()

    def execute(self, backend_name, request, *, approval_id=None, task_id=None, session_id=None):
        try:
            backend = self.backends[backend_name]
        except KeyError as exc:
            raise KeyError(f"unknown execution backend: {backend_name}") from exc
        escape = (backend_name == "host" or
                  (backend_name == "container" and backend.sandbox_escape(request)))
        effect = "remote" if backend_name == "ssh" else "execute"
        target = request.target or backend_name
        if backend_name == "ssh":
            try:
                target = "https://" + backend.targets[request.target].host
            except KeyError as exc:
                raise PermissionError("SSH target is not explicitly registered") from exc
        path_target = (request.workspace if backend_name == "container"
                       and request.workspace_mode != "ephemeral" else None)
        path_checks = []
        if path_target:
            path_effect = (
                "write" if backend_name == "container" and request.workspace_mode == "read_write"
                else "read"
            )
            path_request = ScopeRequest(
                "fs.write" if path_effect == "write" else "fs.read", path_effect, path_target,
                task_id=task_id, session_id=session_id,
                allowed_paths=request.allowed_paths,
            )
            path_decision = self.guard.evaluate(path_request)
            if path_decision.action == "deny":
                record_denied(path_request, path_decision)
                raise PermissionError(path_decision.reason)
            path_checks.append(("workspace", path_request, path_decision))
        if backend_name == "container":
            for mount in request.mounts:
                mount_effect = "read" if mount.read_only else "write"
                mount_request = ScopeRequest(
                    "fs.read" if mount.read_only else "fs.write", mount_effect,
                    mount.source, task_id=task_id, session_id=session_id,
                    allowed_paths=request.allowed_paths,
                )
                mount_decision = self.guard.evaluate(mount_request)
                if mount_decision.action == "deny":
                    record_denied(mount_request, mount_decision)
                    raise PermissionError(mount_decision.reason)
                path_checks.append((f"mount:{mount.source}", mount_request, mount_decision))
        scope_request = ScopeRequest(
            "terminal.run", effect, target,
            arguments={
                "argv": list(request.argv), "cwd": request.cwd,
                "environment_refs": request.environment_refs,
                "workspace": request.workspace,
                "mounts": [mount.__dict__ for mount in request.mounts],
                "image": request.image, "network": request.network,
                "network_hosts": list(request.network_hosts),
                "persistent": request.persistent,
                "authority_contract": ({
                    "filesystem": "unrestricted-host", "network": "unrestricted-host",
                    "resource_limits": "timeout-only",
                } if backend_name == "host" else {
                    "filesystem": "container-mounts", "network": (
                        "unrestricted-public-egress" if request.network else "disabled"),
                    "resource_limits": "container-runtime",
                }),
            },
            task_id=task_id, session_id=session_id, sandbox_escape=escape,
        )
        decision = self.guard.evaluate(scope_request)
        if backend_name == "ssh" and decision.action != "deny":
            try:
                normalize_network_target(target, resolve_dns=True)
            except ValueError as exc:
                decision = replace(decision, action="deny", reason=str(exc))
        approval_map = approval_id if isinstance(approval_id, dict) else {"primary": approval_id}
        actions = [begin_action(
            scope_request, decision, approval_id=approval_map.get("primary"), broker=self.broker,
        )]
        for key, guarded_path, guarded_decision in path_checks:
            if guarded_decision.action == "ask":
                try:
                    actions.append(begin_action(
                        guarded_path, guarded_decision, approval_id=approval_map.get(key),
                        broker=self.broker,
                    ))
                except Exception:
                    for active in actions:
                        finish_action(active.id, state="cancelled",
                                      result={"error": "workspace authorization failed"})
                    raise
        if request.network:
            if not request.network_hosts:
                finish_action(actions[0].id, state="cancelled",
                              result={"error": "network destinations are required"})
                raise PermissionError("network-enabled execution requires explicit destinations")
            try:
                for host in request.network_hosts:
                    normalize_network_target(host, resolve_dns=True)
            except ValueError as exc:
                for active in actions:
                    finish_action(active.id, state="cancelled", result={"error": str(exc)})
                raise PermissionError(str(exc)) from exc
            network_request = ScopeRequest(
                "terminal.network.unrestricted", "network",
                "https://public-internet.invalid/",
                arguments={"backend": backend_name, "argv": list(request.argv),
                           "declared_destinations": list(request.network_hosts),
                           "enforcement": "unrestricted-public-egress"},
                task_id=task_id, session_id=session_id,
            )
            network_decision = self.guard.evaluate(network_request)
            try:
                actions.append(begin_action(
                    network_request, network_decision,
                    approval_id=approval_map.get("network"), broker=self.broker,
                ))
            except Exception:
                for active in actions:
                    finish_action(active.id, state="cancelled",
                                  result={"error": "network authorization failed"})
                raise
        try:
            authorization = _ExecutionAuthorization(action.id for action in actions)
            result = backend.execute(
                request, authorization=authorization, allow_host_mounts=escape,
            ) if backend_name == "container" else backend.execute(
                request, authorization=authorization,
            )
            secret_store = getattr(backend, "secret_store", None)
            secrets = _secret_values(
                request.environment_refs, secret_store=secret_store,
                consumer=f"execution:{backend_name}")
            if secrets:
                result = replace(
                    result, stdout=_mask_text(result.stdout, secrets),
                    stderr=_mask_text(result.stderr, secrets),
                )
        except Exception as exc:
            for action in actions:
                finish_action(action.id, state="failed", result={"error": str(exc)})
            raise
        for action in actions:
            finish_action(action.id, state=result.state, result=result.audit_result())
        return replace(result, action_ids=tuple(action.id for action in actions))
