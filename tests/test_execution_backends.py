from pathlib import Path
import os
import socket
import subprocess

import pytest

from tars import approvals, execution_backends as execution, policy, state_store


class FakeRunner:
    def __init__(self, *, stdout="ok", returncode=0):
        self.calls = []
        self.stdout = stdout
        self.returncode = returncode

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, "")


class TimeoutRunner(FakeRunner):
    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if len(self.calls) == 1:
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 1))
        return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture
def isolated_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ])
    return tmp_path


def _allow(effect, target):
    policy.add_rule(effect, "allow", target=target)


def test_host_backend_runs_direct_argv_through_guard_and_records_truth(isolated_execution):
    runner = FakeRunner(stdout="hello\n")
    backend = execution.HostBackend(runner=runner)
    _allow("sandbox_escape", "host")
    request = execution.ExecutionRequest(
        ("printf", "hello"), cwd=str(isolated_execution),
        allowed_paths=(str(isolated_execution),),
    )
    result = execution.GuardedExecutor({"host": backend}).execute("host", request)
    assert result.succeeded and result.stdout == "hello\n"
    assert runner.calls[0][0] == ["printf", "hello"]
    assert "shell" not in runner.calls[0][1]
    assert backend.status().support == "explicit-escape-hatch"
    assert "unconfined" in backend.status().message


def test_host_shell_is_explicit_and_environment_uses_references(monkeypatch, isolated_execution):
    runner = FakeRunner()
    backend = execution.HostBackend(runner=runner)
    _allow("sandbox_escape", "host")
    monkeypatch.setenv("TARS_TEST_SOURCE", "resolved-value")
    request = execution.ExecutionRequest(
        ("printf '%s' \"$VALUE\"",), cwd=str(isolated_execution), shell=True,
        environment_refs={"VALUE": "env:TARS_TEST_SOURCE"},
        allowed_paths=(str(isolated_execution),),
    )
    execution.GuardedExecutor({"host": backend}).execute("host", request)
    argv, kwargs = runner.calls[0]
    assert argv[:2] == ["/bin/bash", "-lc"] and kwargs["env"]["VALUE"] == "resolved-value"


def test_resolved_environment_values_are_redacted_from_results(monkeypatch, isolated_execution):
    runner = FakeRunner(stdout="value=resolved-secret")
    _allow("sandbox_escape", "host")
    monkeypatch.setenv("TARS_RESULT_SECRET", "resolved-secret")
    request = execution.ExecutionRequest(
        ("printenv", "VALUE"), cwd=str(isolated_execution),
        environment_refs={"VALUE": "env:TARS_RESULT_SECRET"},
        allowed_paths=(str(isolated_execution),),
    )
    result = execution.GuardedExecutor({"host": execution.HostBackend(runner=runner)}).execute(
        "host", request,
    )
    assert result.stdout == "value=[REDACTED]"


def test_backend_cannot_be_called_without_guarded_authorization(isolated_execution):
    backend = execution.HostBackend(runner=FakeRunner())
    with pytest.raises(PermissionError, match="guarded authorization"):
        backend.execute(execution.ExecutionRequest(("true",), cwd=str(isolated_execution)))


def test_denied_host_command_never_reaches_runner(isolated_execution):
    runner = FakeRunner()
    request = execution.ExecutionRequest(
        ("id",), cwd=str(isolated_execution), allowed_paths=(str(isolated_execution),),
    )
    with pytest.raises(PermissionError, match="approved authorization"):
        execution.GuardedExecutor({"host": execution.HostBackend(runner=runner)}).execute(
            "host", request,
        )
    assert runner.calls == []


def test_container_command_owns_isolation_limits_and_mount_semantics(isolated_execution):
    workspace = isolated_execution / "workspace"
    workspace.mkdir()
    dependency = workspace / "dependency"
    dependency.mkdir()
    runner = FakeRunner()
    backend = execution.ContainerBackend(
        runtime="/usr/bin/podman", runner=runner, rootless_verified=True,
    )
    _allow("execute", "container")
    policy.add_rule("write", "allow", target=str(workspace), target_kind="path")
    request = execution.ExecutionRequest(
        ("python", "-m", "pytest"), workspace=str(workspace), image="python:3.13",
        mounts=(execution.ContainerMount(str(dependency), "/deps", True),),
        limits=execution.ResourceLimits(cpus=2, memory="2g", processes=64, timeout=30),
        allowed_paths=(str(workspace),), workspace_mode="read_write",
    )
    result = execution.GuardedExecutor({"container": backend}).execute("container", request)
    command = runner.calls[0][0]
    assert result.succeeded and command[:2] == ["/usr/bin/podman", "run"]
    assert "--userns=keep-id" in command and "--rm" in command and "--pull=never" in command
    assert command[command.index("--network") + 1] == "none"
    assert "2g" in command and "64" in command
    workspace_volume = next(value for value in command if value.endswith(":/workspace:rw"))
    dependency_volume = next(value for value in command if value.endswith(":/deps:ro"))
    assert workspace_volume.startswith("/proc/")
    assert dependency_volume.startswith("/proc/")


def test_container_mount_identity_is_bound_before_authorization(
        isolated_execution):
    workspace = isolated_execution / "workspace"
    workspace.mkdir()
    (workspace / "identity.txt").write_text("inside")
    outside = isolated_execution / "outside"
    outside.mkdir()
    (outside / "identity.txt").write_text("outside")
    bound_targets = []

    class IdentityRunner(FakeRunner):
        def __call__(self, argv, **kwargs):
            volume = next(value for value in argv if value.endswith(":/workspace:ro"))
            bound_targets.append(os.readlink(volume.removesuffix(":/workspace:ro")))
            return super().__call__(argv, **kwargs)

    runner = IdentityRunner()
    backend = execution.ContainerBackend(
        runtime="/usr/bin/podman", runner=runner, rootless_verified=True,
    )
    _allow("execute", "container")
    policy.add_rule("read", "allow", target=str(workspace), target_kind="path")

    class SwappingGuard(policy.ScopeGuard):
        swapped = False

        def evaluate(self, request):
            decision = super().evaluate(request)
            if request.tool == "terminal.run" and not self.swapped:
                self.swapped = True
                workspace.rename(isolated_execution / "displaced")
                workspace.symlink_to(outside, target_is_directory=True)
            return decision

    request = execution.ExecutionRequest(
        ("true",), workspace=str(workspace), image="python:3.13",
        allowed_paths=(str(workspace),), workspace_mode="read_only",
    )
    result = execution.GuardedExecutor(
        {"container": backend}, guard=SwappingGuard(),
    ).execute("container", request)
    assert result.succeeded
    assert "displaced" in bound_targets[0]
    assert "outside" not in bound_targets[0]


def test_container_mount_outside_workspace_requires_separate_escape_approval(isolated_execution):
    workspace = isolated_execution / "workspace"
    outside = isolated_execution / "outside"
    workspace.mkdir()
    outside.mkdir()
    runner = FakeRunner()
    backend = execution.ContainerBackend(
        runtime="/usr/bin/podman", runner=runner, rootless_verified=True,
    )
    request = execution.ExecutionRequest(
        ("ls", "/host"), workspace=str(workspace), image="base",
        mounts=(execution.ContainerMount(str(outside), "/host"),),
        allowed_paths=(str(workspace), str(outside)), workspace_mode="read_only",
    )
    guard = policy.ScopeGuard()
    scope_request = policy.ScopeRequest(
        "terminal.run", "execute", "container", sandbox_escape=True,
    )
    decision = guard.evaluate(scope_request)
    broker = approvals.ApprovalBroker()
    pending = broker.request(scope_request, decision, scope="target")
    broker.decide(pending.id, approve=True)
    result = execution.GuardedExecutor(
        {"container": backend}, guard=guard, broker=broker,
    ).execute("container", request, approval_id=pending.id)
    assert result.succeeded and runner.calls


def test_writable_host_workspace_requires_write_approval(isolated_execution):
    workspace = isolated_execution / "workspace"
    workspace.mkdir()
    runner = FakeRunner()
    backend = execution.ContainerBackend(
        runtime="/usr/bin/podman", runner=runner, rootless_verified=True,
    )
    _allow("execute", "container")
    request = execution.ExecutionRequest(
        ("touch", "result"), workspace=str(workspace), workspace_mode="read_write",
        image="base", allowed_paths=(str(workspace),),
    )
    executor = execution.GuardedExecutor({"container": backend})
    with pytest.raises(PermissionError, match="approved authorization"):
        executor.execute("container", request)
    assert not runner.calls
    scope_request = policy.ScopeRequest(
        "fs.write", "write", str(workspace), allowed_paths=(str(workspace),),
    )
    decision = policy.ScopeGuard().evaluate(scope_request)
    broker = approvals.ApprovalBroker()
    pending = broker.request(scope_request, decision)
    broker.decide(pending.id, approve=True)
    result = execution.GuardedExecutor({"container": backend}, broker=broker).execute(
        "container", request, approval_id={"workspace": pending.id},
    )
    assert result.succeeded


def test_read_only_workspace_cannot_hide_writable_submount(isolated_execution):
    workspace = isolated_execution / "workspace"
    writable = workspace / "writable"
    writable.mkdir(parents=True)
    runner = FakeRunner()
    backend = execution.ContainerBackend(
        runtime="/usr/bin/podman", runner=runner, rootless_verified=True,
    )
    _allow("execute", "container")
    request = execution.ExecutionRequest(
        ("touch", "/write/result"), workspace=str(workspace), workspace_mode="read_only",
        image="base", mounts=(execution.ContainerMount(str(writable), "/write", False),),
        allowed_paths=(str(workspace),),
    )
    with pytest.raises(PermissionError, match="approved authorization"):
        execution.GuardedExecutor({"container": backend}).execute("container", request)
    assert not runner.calls


def test_container_network_requires_destinations_and_blocks_private_targets(isolated_execution):
    workspace = isolated_execution / "workspace"
    workspace.mkdir()
    runner = FakeRunner()
    backend = execution.ContainerBackend(
        runtime="/usr/bin/podman", runner=runner, rootless_verified=True,
    )
    _allow("execute", "container")
    request = execution.ExecutionRequest(
        ("fetch",), workspace=str(workspace), image="base", network=True,
        network_hosts=("http://169.254.169.254/",), allowed_paths=(str(workspace),),
    )
    with pytest.raises(PermissionError):
        execution.GuardedExecutor({"container": backend}).execute("container", request)
    assert runner.calls == []


def test_container_public_network_requires_destination_approval(isolated_execution):
    runner = FakeRunner()
    backend = execution.ContainerBackend(
        runtime="/usr/bin/podman", runner=runner, rootless_verified=True,
    )
    _allow("execute", "container")
    host = "https://example.com"
    request = execution.ExecutionRequest(
        ("fetch",), image="base", network=True, network_hosts=(host,),
    )
    network_request = policy.ScopeRequest(
        "terminal.network.unrestricted", "network",
        "https://public-internet.invalid/",
        arguments={"backend": "container", "argv": ["fetch"],
                   "declared_destinations": [host],
                   "enforcement": "unrestricted-public-egress"},
    )
    decision = policy.ScopeGuard().evaluate(network_request)
    broker = approvals.ApprovalBroker()
    pending = broker.request(network_request, decision)
    broker.decide(pending.id, approve=True)
    result = execution.GuardedExecutor({"container": backend}, broker=broker).execute(
        "container", request, approval_id={"network": pending.id},
    )
    assert result.succeeded
    command = runner.calls[0][0]
    assert command[command.index("--network") + 1] == "slirp4netns"


def test_rootless_docker_command_uses_docker_network_semantics(isolated_execution):
    backend = execution.ContainerBackend(
        runtime="/usr/bin/docker", runner=FakeRunner(), rootless_verified=True,
    )
    command, _ = backend._command(execution.ExecutionRequest(
        ("true",), image="base", network=True, network_hosts=("https://example.com",),
    ))
    assert "--userns=keep-id" not in command
    assert command[command.index("--network") + 1] == "bridge"


def test_container_timeout_attempts_truthful_cleanup(isolated_execution):
    runner = TimeoutRunner()
    backend = execution.ContainerBackend(
        runtime="/usr/bin/podman", runner=runner, rootless_verified=True,
    )
    _allow("execute", "container")
    result = execution.GuardedExecutor({"container": backend}).execute(
        "container", execution.ExecutionRequest(
            ("sleep", "60"), image="base", limits=execution.ResourceLimits(timeout=1),
        ),
    )
    assert result.timed_out and result.state == "failed"
    assert runner.calls[1][0][1:3] == ["rm", "-f"]


def test_container_persistence_and_unrelated_host_access_are_explicit(isolated_execution):
    workspace = isolated_execution / "workspace"
    workspace.mkdir()
    backend = execution.ContainerBackend(
        runtime="/usr/bin/podman", runner=FakeRunner(), rootless_verified=True,
    )
    with pytest.raises(ValueError, match="explicit name"):
        backend._command(execution.ExecutionRequest(
            ("true",), workspace=str(workspace), image="base", persistent=True,
        ))
    outside = isolated_execution / "outside"
    outside.mkdir()
    with pytest.raises(PermissionError, match="sandbox-escape"):
        backend._command(execution.ExecutionRequest(
            ("true",), workspace=str(workspace), image="base",
            mounts=(execution.ContainerMount(str(outside), "/outside"),),
        ))


def test_ssh_backend_uses_registered_target_and_secret_reference(monkeypatch, isolated_execution):
    identity = isolated_execution / "id_ed25519"
    identity.write_text("fixture")
    monkeypatch.setenv("TARS_TEST_SSH_KEY", str(identity))
    runner = FakeRunner(stdout="remote\n")
    target = execution.SSHExecutionTarget(
        "lab", "example.com", "operator", 2222, "env:TARS_TEST_SSH_KEY",
        allowed_commands=("uname",), allowed_paths=("/srv/project",),
    )
    backend = execution.SSHBackend((target,), ssh_binary="/usr/bin/ssh", runner=runner)
    request = execution.ExecutionRequest(("uname", "-a"), target="lab")
    target_url = "ssh://example.com:2222/"
    decision = policy.ScopeGuard().evaluate(policy.ScopeRequest(
        "terminal.run", "remote", target_url,
    ))
    broker = approvals.ApprovalBroker()
    pending = broker.request(
        policy.ScopeRequest("terminal.run", "remote", target_url), decision,
        scope="target",
    )
    broker.decide(pending.id, approve=True)
    result = execution.GuardedExecutor({"ssh": backend}, broker=broker).execute(
        "ssh", request, approval_id=pending.id,
    )
    command = runner.calls[0][0]
    assert result.succeeded and "operator@example.com" in command
    assert any(value.startswith("Hostname=") for value in command)
    assert "HostKeyAlias=example.com" in command
    assert command[-1] == "uname -a" and str(identity) in command
    assert backend.status().support == "experimental"


def test_ssh_uses_the_guarded_dns_snapshot_without_reresolving(
        monkeypatch, isolated_execution):
    resolutions = []

    def rebinding(host, port, **kwargs):
        address = "93.184.216.34" if not resolutions else "127.0.0.1"
        resolutions.append(address)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                 (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", rebinding)
    runner = FakeRunner(stdout="remote\n")
    target = execution.SSHExecutionTarget(
        "lab", "example.com", "operator", 22,
        allowed_commands=("uname",),
    )
    backend = execution.SSHBackend((target,), ssh_binary="/usr/bin/ssh", runner=runner)
    _allow("remote", "example.com")
    result = execution.GuardedExecutor({"ssh": backend}).execute(
        "ssh", execution.ExecutionRequest(("uname",), target="lab"),
    )
    assert result.succeeded and resolutions == ["93.184.216.34"]
    command = runner.calls[0][0]
    assert "Hostname=93.184.216.34" in command
    assert "Hostname=127.0.0.1" not in command


def test_ssh_backend_rejects_target_config_change_after_authorization(isolated_execution):
    original = execution.SSHExecutionTarget(
        "lab", "example.com", "operator", allowed_commands=("id",),
    )
    backend = execution.SSHBackend(
        (original,), ssh_binary="/usr/bin/ssh", runner=FakeRunner(),
    )
    authorization = execution._ExecutionAuthorization(
        ("action",),
        network_destination=execution.tcp_destination(
            original.host, original.port, scheme="ssh", resolve_dns=True,
        ),
        ssh_target=original,
    )
    backend.targets["lab"] = execution.SSHExecutionTarget(
        "lab", "example.com", "root", allowed_commands=("id", "rm"),
    )
    with pytest.raises(PermissionError, match="guarded configuration"):
        backend.execute(
            execution.ExecutionRequest(("id",), target="lab"),
            authorization=authorization,
        )


def test_ssh_unregistered_target_and_missing_container_runtime_are_explicit(isolated_execution):
    ssh = execution.SSHBackend((), ssh_binary="/usr/bin/ssh", runner=FakeRunner())
    with pytest.raises(PermissionError, match="registered"):
        execution.GuardedExecutor({"ssh": ssh}).execute(
            "ssh", execution.ExecutionRequest(("id",), target="unknown"),
        )
    container = execution.ContainerBackend(runtime="", runner=FakeRunner())
    container.runtime = None
    assert not container.status().available


def test_ssh_command_and_working_directory_scope_are_enforced(isolated_execution):
    target = execution.SSHExecutionTarget(
        "lab", "example.com", "operator", allowed_commands=("ls",),
        allowed_paths=("/srv/project",),
    )
    backend = execution.SSHBackend((target,), ssh_binary="/usr/bin/ssh", runner=FakeRunner())
    authorization = execution._ExecutionAuthorization(
        ("fixture",), network_destination=execution.tcp_destination(
            "example.com", 22, scheme="ssh", resolve_dns=True,
        ), ssh_target=target,
    )
    with pytest.raises(PermissionError, match="command"):
        backend.execute(execution.ExecutionRequest(("rm", "-rf", "/"), target="lab"),
                        authorization=authorization)
    with pytest.raises(PermissionError, match="cwd"):
        backend.execute(execution.ExecutionRequest(("ls",), target="lab", cwd="/etc"),
                        authorization=authorization)
