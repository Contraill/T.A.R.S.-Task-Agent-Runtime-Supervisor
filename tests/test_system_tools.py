import subprocess

import pytest

from tars import approvals, policy, state_store, system_tools


class FakeRunner:
    def __init__(self, outputs=()):
        self.outputs = iter(outputs)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        try:
            return next(self.outputs)
        except StopIteration:
            return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture
def system_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")


def _approve(request):
    decision = policy.ScopeGuard().evaluate(request)
    broker = approvals.ApprovalBroker()
    pending = broker.request(request, decision)
    broker.decide(pending.id, approve=True)
    return pending.id


def test_service_change_verifies_real_post_state(system_environment):
    outputs = (
        subprocess.CompletedProcess([], 0, "ActiveState=inactive\nSubState=dead\n", ""),
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 0, "ActiveState=active\nSubState=running\n", ""),
    )
    runner = FakeRunner(outputs)
    tools = system_tools.ServiceTools(runner=runner)
    request = policy.ScopeRequest(
        "service.start", "service", "fixture.service", {"operation": "start"},
    )
    result = tools.mutate("start", "fixture.service", approval_id=_approve(request))
    assert result.succeeded and result.data["verified"]
    assert runner.calls[1] == ["systemctl", "--user", "start", "fixture.service"]
    assert result.evidence_ids


def test_service_failed_post_state_is_not_reported_success(system_environment):
    outputs = (
        subprocess.CompletedProcess([], 0, "ActiveState=inactive\n", ""),
        subprocess.CompletedProcess([], 0, "", ""),
        subprocess.CompletedProcess([], 0, "ActiveState=inactive\n", ""),
    )
    tools = system_tools.ServiceTools(runner=FakeRunner(outputs))
    request = policy.ScopeRequest(
        "service.start", "service", "fixture.service", {"operation": "start"},
    )
    result = tools.mutate("start", "fixture.service", approval_id=_approve(request))
    assert result.state == "failed" and not result.data["verified"]


def test_pacman_install_owns_full_upgrade_semantics_and_mutation_policy(system_environment):
    runner = FakeRunner()
    packages = system_tools.PacmanBackend(binary="/usr/bin/pacman", runner=runner)
    with pytest.raises(PermissionError):
        packages.install(["example"])
    assert runner.calls == []
    policy.add_rule("elevated", "allow", target="pacman")
    result = packages.install(["example"])
    assert result.succeeded
    assert runner.calls[0] == [
        "/usr/bin/pacman", "-Syu", "--needed", "--noconfirm", "example",
    ]


def test_pacman_read_operations_and_unavailable_status(system_environment):
    runner = FakeRunner((subprocess.CompletedProcess([], 0, "pkg 1.0", ""),))
    packages = system_tools.PacmanBackend(binary="/usr/bin/pacman", runner=runner)
    assert packages.installed("pkg").succeeded
    packages.binary = None
    assert not packages.status()["available"]


def test_structured_system_inspection_is_guarded_and_evidenced(system_environment):
    tools = system_tools.SystemInspection()
    info = tools.info()
    hardware = tools.hardware()
    network = tools.network()
    assert info.data["result"]["system"]
    assert hardware.data["result"]["logical_cpus"]
    assert "interfaces" in network.data["result"]
    assert info.evidence_ids and hardware.evidence_ids
