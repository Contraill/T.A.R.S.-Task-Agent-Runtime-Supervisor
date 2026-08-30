import subprocess

import pytest

from tars import approvals, policy, state_store, workspace_recovery
from tars.cli import build_parser


@pytest.fixture
def recovery(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    root = tmp_path / "workspace"
    root.mkdir()
    tools = workspace_recovery.WorkspaceRecovery(
        (root,), storage_root=tmp_path / "checkpoint-storage",
    )
    return tools, root


def approve_rollback(root):
    request = policy.ScopeRequest(
        "workspace.rollback", "destructive", str(root), allowed_paths=(str(root),),
        destructive=True,
    )
    decision = policy.ScopeGuard().evaluate(request)
    broker = approvals.ApprovalBroker()
    pending = broker.request(request, decision, scope="target")
    broker.decide(pending.id, approve=True)
    return pending.id


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)


def init_repo(root):
    git(root, "init", "-q")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.com")
    (root / "tracked.txt").write_text("base\n")
    git(root, "add", "tracked.txt")
    git(root, "commit", "-qm", "base")


def test_git_checkpoint_preview_and_exact_worktree_index_restore(recovery):
    tools, root = recovery
    init_repo(root)
    (root / "tracked.txt").write_text("staged\n")
    git(root, "add", "tracked.txt")
    (root / "tracked.txt").write_text("staged\nunstaged\n")
    (root / "baseline.txt").write_text("baseline")
    checkpoint = tools.create_git(root)

    (root / "tracked.txt").write_text("later mutation\n")
    (root / "baseline.txt").write_text("changed")
    (root / "new.txt").write_text("new")
    preview = tools.preview(checkpoint.data["checkpoint_id"])
    assert preview.succeeded and preview.data["supported"]
    assert any(change["operation"] == "quarantine_new_untracked"
               for change in preview.data["changes"])

    restored = tools.rollback(
        checkpoint.data["checkpoint_id"], approval_id=approve_rollback(root),
    )
    assert restored.succeeded and (root / "tracked.txt").read_text() == "staged\nunstaged\n"
    assert (root / "baseline.txt").read_text() == "baseline" and not (root / "new.txt").exists()
    assert "staged" in git(root, "diff", "--cached").stdout
    assert "unstaged" in git(root, "diff").stdout
    assert restored.data["safety_checkpoint_id"]
    assert workspace_recovery.load(checkpoint.data["checkpoint_id"]).state == "restored"


def test_git_rollback_refuses_changed_head_without_rewriting_history(recovery):
    tools, root = recovery
    init_repo(root)
    checkpoint = tools.create_git(root)
    (root / "later.txt").write_text("commit")
    git(root, "add", "later.txt")
    git(root, "commit", "-qm", "later")
    head = git(root, "rev-parse", "HEAD").stdout.strip()
    preview = tools.preview(checkpoint.data["checkpoint_id"])
    assert not preview.data["supported"]
    result = tools.rollback(checkpoint.data["checkpoint_id"],
                            approval_id=approve_rollback(root))
    assert result.state == "failed"
    assert git(root, "rev-parse", "HEAD").stdout.strip() == head


def test_bounded_filesystem_snapshot_restore_and_symlink_rejection(recovery, tmp_path):
    tools, root = recovery
    target = root / "note.txt"
    target.write_text("before")
    checkpoint = tools.create_filesystem(root, (target,))
    target.write_text("after")
    result = tools.rollback(checkpoint.data["checkpoint_id"],
                            approval_id=approve_rollback(root))
    assert result.succeeded and target.read_text() == "before"
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = root / "link"
    link.symlink_to(outside)
    with pytest.raises((ValueError, PermissionError)):
        tools.create_filesystem(root, (link,))
    with pytest.raises(FileNotFoundError):
        tools.create_filesystem(root, (root / "missing",))


def test_git_checkpoint_rejects_untracked_symlink(recovery, tmp_path):
    tools, root = recovery
    init_repo(root)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (root / "untracked-link").symlink_to(outside)
    with pytest.raises(ValueError, match="unsupported untracked entry"):
        tools.create_git(root)


def test_rollback_requires_explicit_destructive_approval(recovery):
    tools, root = recovery
    target = root / "note.txt"
    target.write_text("before")
    checkpoint = tools.create_filesystem(root, (target,))
    target.write_text("after")
    with pytest.raises(PermissionError):
        tools.rollback(checkpoint.data["checkpoint_id"])
    assert target.read_text() == "after"


def test_workspace_cli_surfaces_parse_checkpoint_preview_and_rollback():
    parser = build_parser()
    checkpoint = parser.parse_args(["workspace", "checkpoint", "/workspace",
                                    "--path", "/workspace/file"])
    assert checkpoint.path == ["/workspace/file"]
    assert parser.parse_args(["workspace", "preview", "wcp-one"]).checkpoint_id == "wcp-one"
    rollback = parser.parse_args(["workspace", "rollback", "wcp-one",
                                  "--approval", "approval-one"])
    assert rollback.approval == "approval-one"
