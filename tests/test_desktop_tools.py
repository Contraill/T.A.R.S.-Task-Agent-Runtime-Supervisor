import subprocess

import pytest

from tars import approvals, desktop_tools, policy, state_store


@pytest.fixture
def desktop(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    return tmp_path


def approve(tool, target, effect="write", *, roots=()):
    request = policy.ScopeRequest(tool, effect, str(target), allowed_paths=tuple(map(str, roots)))
    decision = policy.ScopeGuard().evaluate(request)
    broker = approvals.ApprovalBroker()
    pending = broker.request(request, decision, scope="target")
    broker.decide(pending.id, approve=True)
    return pending.id


def test_notification_is_typed_and_reports_delivery(monkeypatch, desktop):
    calls = []
    monkeypatch.setattr(desktop_tools.shutil, "which", lambda name: "/usr/bin/notify-send")
    runner = lambda argv, **kwargs: (calls.append(argv) or subprocess.CompletedProcess(argv, 0, "", ""))
    tools = desktop_tools.NotificationTools(runner=runner)
    result = tools.send("T.A.R.S.", "complete",
                        approval_id=approve("notify.send", "desktop-notification"))
    assert result.succeeded and result.data["delivered"] and calls[0][0].endswith("notify-send")


def test_screen_capture_requires_real_output(monkeypatch, desktop):
    output = desktop / "screen.png"
    monkeypatch.setattr(desktop_tools.shutil, "which",
                        lambda name: "/usr/bin/spectacle" if name == "spectacle" else None)
    def runner(argv, **kwargs):
        output.write_bytes(b"png")
        return subprocess.CompletedProcess(argv, 0, "", "")
    result = desktop_tools.ScreenCaptureTools((desktop,), runner=runner).capture(
        output, approval_id=approve("screen.capture", output, roots=(desktop,)),
    )
    assert result.succeeded and result.data["verified"] and result.evidence_ids
