import pytest

from tars import memory, state_store


@pytest.fixture
def isolated_memory(monkeypatch, tmp_path):
    root = tmp_path / "memory"
    monkeypatch.setattr(memory, "MEMORY_ROOT", root)
    monkeypatch.setattr(memory, "MEMORY_HISTORY_ROOT", root / ".history")
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index")
    return root


def test_canonical_memory_is_human_readable_searchable_and_rebuildable(isolated_memory):
    entry = memory.remember(
        "The user prefers concise technical explanations.", title="Response style",
        scope="profile", source="explicit-user", tags=("preference", "writing"),
    )
    document = (isolated_memory / "profile" / f"{entry.id}.md").read_text()
    assert document.startswith("---\n") and "concise technical" in document
    hit = memory.search("concise", scope="profile")[0]
    assert hit.entry.id == entry.id
    assert "fts5" in hit.signals and "scope:profile" in hit.signals
    with state_store.transaction(immediate=True) as conn:
        conn.execute("DELETE FROM memory_fts")
        conn.execute("DELETE FROM memory_index")
    assert memory.rebuild_index() == 1
    assert memory.inspect(entry.id).source == "explicit-user"


def test_deduplication_candidate_policy_and_forget_history(isolated_memory):
    first = memory.remember("Always use metric units", scope="profile")
    duplicate = memory.remember("  always USE metric units ", scope="profile")
    assert duplicate.id == first.id
    candidate = memory.stage_candidate(
        "Project Alpha uses Python", kind="projects", scope="project:alpha",
        source="model-proposal", confidence=0.7,
    )
    assert memory.review_candidates()[0]["id"] == candidate
    promoted = memory.decide_candidate(candidate, promote=True, reason="confirmed")
    assert promoted.scope == "project:alpha"
    archive = memory.forget(first.id)
    assert archive.is_file()
    with pytest.raises(KeyError):
        memory.inspect(first.id)
    assert memory.doctor()["ok"]


def test_expired_memory_is_not_recalled(isolated_memory):
    memory.remember("obsolete preference", expiry="2000-01-01T00:00:00+00:00")
    assert memory.search("obsolete") == []


def test_superseded_memory_is_retained_but_not_recalled(isolated_memory):
    old = memory.remember("favorite editor is Vim", scope="profile")
    new = memory.remember("favorite editor is Helix", scope="profile", supersedes=old.id)
    assert memory.inspect(old.id).content.endswith("Vim")
    assert [hit.entry.id for hit in memory.search("favorite editor")] == [new.id]


def test_invalid_candidate_does_not_enter_corpus(isolated_memory):
    with pytest.raises(ValueError):
        memory.stage_candidate("")
    assert memory.doctor()["files"] == 0


def test_candidate_review_claim_is_released_after_promotion_failure(
        monkeypatch, isolated_memory):
    candidate = memory.stage_candidate("retry me", scope="profile")
    original = memory.remember
    monkeypatch.setattr(
        memory, "remember",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected write failure")))
    with pytest.raises(RuntimeError, match="injected write failure"):
        memory.decide_candidate(candidate, promote=True)
    assert memory.review_candidates()[0]["status"] == "staged"
    monkeypatch.setattr(memory, "remember", original)
    assert memory.decide_candidate(candidate, promote=True).content == "retry me"
