import subprocess

import pytest

from tars import approvals, git_tools, policy, state_store


def _run(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture
def repository(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-b", "main")
    _run(repo, "config", "user.name", "Test User")
    _run(repo, "config", "user.email", "test@example.com")
    (repo / "file.txt").write_text("one\n")
    _run(repo, "add", "file.txt")
    _run(repo, "commit", "-m", "initial")
    return git_tools.GitTools(repo), repo


def _approve(tool, effect, target, roots, *, destructive=False):
    request = policy.ScopeRequest(
        tool, effect, str(target), allowed_paths=(str(roots),), destructive=destructive,
    )
    decision = policy.ScopeGuard().evaluate(request)
    broker = approvals.ApprovalBroker()
    pending = broker.request(request, decision, scope="target")
    broker.decide(pending.id, approve=True)
    return pending.id


def test_git_read_tools_return_real_repository_state(repository):
    tools, repo = repository
    status = tools.status()
    assert status.succeeded and status.data["after"]["branch"] == "main"
    assert tools.log().data["stdout"].split("\t")[-1].strip() == "initial"
    assert tools.show().evidence_ids and tools.diff().succeeded


def test_git_commit_reports_sha_and_human_metadata(repository):
    tools, repo = repository
    (repo / "file.txt").write_text("two\n")
    _run(repo, "add", "file.txt")
    approval = _approve("git.commit", "write", repo, repo)
    result = tools.commit("update", approval_id=approval)
    assert result.succeeded and len(result.data["commit"]["sha"]) == 40
    assert result.data["commit"]["author_name"] == "Test User"
    assert not result.data["after"]["dirty"]


def test_git_checkpoint_and_explicit_rollback(repository):
    tools, repo = repository
    head = tools.status().data["after"]["head"]
    (repo / "file.txt").write_text("changed\n")
    checkpoint = tools.checkpoint()
    assert checkpoint.data["head"] == head and checkpoint.data["dirty"]
    rollback = _approve("git.rollback", "destructive", repo, repo, destructive=True)
    result = tools.rollback(head, approval_id=rollback)
    assert result.succeeded and (repo / "file.txt").read_text() == "one\n"


def test_git_push_requires_both_high_impact_and_network_approval(repository):
    tools, repo = repository
    _run(repo, "remote", "add", "origin", "https://example.com/repository.git")
    policy.add_rule("network", "allow", target="example.com")
    with pytest.raises(PermissionError):
        tools.push()


@pytest.mark.parametrize("ref", ["--help", "bad ref", "refs/../main", "main^{tree}"])
def test_git_refs_cannot_inject_cli_options(repository, ref):
    tools, _ = repository
    with pytest.raises(ValueError):
        tools.show(ref)
    with pytest.raises(ValueError):
        tools.branch(ref)


@pytest.mark.parametrize("remote", ["--upload-pack=evil", "bad remote", "../origin"])
def test_git_remote_names_cannot_inject_cli_options(repository, remote):
    tools, _ = repository
    with pytest.raises(ValueError):
        tools.push(remote=remote)
