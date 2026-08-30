import pytest

from tars import approvals, fs_tools, policy, state_store


@pytest.fixture
def filesystem(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")
    root = tmp_path / "workspace"
    root.mkdir()
    return fs_tools.FilesystemTools((root,)), root


def _approval(tool, effect, target, roots, *, destructive=False):
    request = policy.ScopeRequest(
        tool, effect, str(target), allowed_paths=tuple(str(root) for root in roots),
        destructive=destructive,
    )
    decision = policy.ScopeGuard().evaluate(request)
    broker = approvals.ApprovalBroker()
    pending = broker.request(request, decision, scope="target")
    broker.decide(pending.id, approve=True)
    return pending.id


def test_filesystem_read_list_stat_search_and_evidence(filesystem):
    tools, root = filesystem
    path = root / "note.txt"
    path.write_text("alpha\nneedle here\n")
    assert tools.list(root).data["entries"][0]["name"] == "note.txt"
    assert tools.stat(path).data["size"] > 0
    read = tools.read(path, limit=5)
    assert read.data["content"] == "alpha" and read.data["truncated"]
    search = tools.search(root, "needle")
    assert search.data["hits"][0]["line"] == 2
    assert read.evidence_ids and search.evidence_ids


def test_filesystem_mutations_require_matching_approval(filesystem):
    tools, root = filesystem
    target = root / "created.txt"
    with pytest.raises(PermissionError):
        tools.write(target, "one")
    approval = _approval("fs.write", "write", target, (root,))
    assert tools.write(target, "one", approval_id=approval).succeeded
    patch_approval = _approval("fs.patch", "write", target, (root,))
    result = tools.patch(target, [("one", "two")], approval_id=patch_approval)
    assert result.data["replacements"] == 1 and target.read_text() == "two"
    with pytest.raises(ValueError, match="exactly once"):
        second = _approval("fs.patch", "write", target, (root,))
        tools.patch(target, [("missing", "value")], approval_id=second)


def test_copy_move_delete_are_multi_scope_and_destructive(filesystem):
    tools, root = filesystem
    source = root / "source.txt"
    copied = root / "copied.txt"
    moved = root / "moved.txt"
    source.write_text("payload")
    copy_write = _approval("fs.copy", "write", copied, (root,))
    assert tools.copy(source, copied, approval_ids={"write": copy_write}).succeeded
    move_source = _approval("fs.move", "write", copied, (root,))
    move_destination = _approval("fs.move", "write", moved, (root,))
    assert tools.move(copied, moved, approval_ids={
        "source": move_source, "destination": move_destination,
    }).succeeded
    delete = _approval("fs.delete", "destructive", moved, (root,), destructive=True)
    assert tools.delete(moved, approval_id=delete).succeeded and not moved.exists()


def test_filesystem_escape_is_denied_before_mutation(filesystem, tmp_path):
    tools, root = filesystem
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError, match="outside"):
        tools.write(root / "escape" / "bad", "no")
    assert not (outside / "bad").exists()
