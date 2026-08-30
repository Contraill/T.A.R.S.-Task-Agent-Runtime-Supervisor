import uuid

import pytest

from tars import memory, memory_maintenance as maintenance, state_store


@pytest.fixture
def isolated_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index")
    root = tmp_path / "memory"
    monkeypatch.setattr(memory, "MEMORY_ROOT", root)
    monkeypatch.setattr(memory, "MEMORY_HISTORY_ROOT", root / ".history")
    return root


def test_audit_detects_duplicates_index_drift_and_prompt_pressure(isolated_memory):
    entry = memory.remember("Use metric units", scope="profile")
    original = isolated_memory / "profile" / f"{entry.id}.md"
    duplicate_id = "mem-" + uuid.uuid4().hex
    duplicate = isolated_memory / "profile" / f"{duplicate_id}.md"
    duplicate.write_text(original.read_text().replace(entry.id, duplicate_id))
    report = maintenance.audit()
    assert str(duplicate) in report["index_drift"]["unindexed_files"]
    assert report["prompt_pressure"] == {"projections": 0, "average": 0.0, "maximum": 0.0}
    assert memory.rebuild_index() == 2
    report = maintenance.audit()
    assert sorted(report["duplicates"][0]) == sorted([entry.id, duplicate_id])


def test_apply_expires_recoverably_and_repairs_index(isolated_memory):
    expired = memory.remember("Expired fact", expiry="2000-01-01T00:00:00+00:00")
    current = memory.remember("Current fact")
    run = maintenance.run_maintenance(trigger="explicit", apply=True)
    assert run.status == "completed" and run.mode == "apply"
    assert run.actions[0] == {"action": "forget_expired", "memory_id": expired.id}
    assert len(run.rollback_refs) == 1
    assert memory.inspect(current.id).content == "Current fact"
    with pytest.raises(KeyError):
        memory.inspect(expired.id)
    assert maintenance.load_run(run.id) == run


def test_reflection_stages_only_with_model_provenance(isolated_memory):
    with pytest.raises(ValueError, match="provenance"):
        maintenance.stage_reflection([{"content": "lesson"}], model_provenance={})
    run = maintenance.stage_reflection(
        [{"content": "Prefer bounded retries", "kind": "episodic", "scope": "project:x",
          "confidence": 0.7, "tags": ["lesson"]}],
        trigger="context_rollover",
        model_provenance={"model": "local-model", "backend": "llama.cpp", "request_id": "one"},
    )
    assert run.mode == "reflection" and run.model_provenance["request_id"] == "one"
    candidates = memory.review_candidates()
    assert len(candidates) == 1 and candidates[0]["status"] == "staged"
    assert candidates[0]["source"] == f"reflection:{run.id}"
    assert maintenance.audit()["entries"] == 0


def test_failed_reflection_is_inspectable(isolated_memory):
    with pytest.raises(KeyError):
        maintenance.stage_reflection(
            [{"kind": "episodic"}],
            model_provenance={"model": "local-model", "backend": "llama.cpp"},
        )
    run = maintenance.list_runs()[0]
    assert run.status == "failed" and "error" in run.report


def test_maintenance_triggers_are_explicit_and_inspectable(isolated_memory):
    for trigger in maintenance.TRIGGERS:
        assert maintenance.run_maintenance(trigger=trigger).trigger == trigger
    assert len(maintenance.list_runs()) == len(maintenance.TRIGGERS)
    with pytest.raises(ValueError):
        maintenance.run_maintenance(trigger="startup")
    assert state_store.health()["schema_version"] == 13
