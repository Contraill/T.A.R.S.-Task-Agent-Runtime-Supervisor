from types import SimpleNamespace

import pytest

from tars import checkpoints, context, context_engine, context_epochs, conversation, state_store, tasks


@pytest.fixture
def isolated_state(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index")
    monkeypatch.setattr(tasks, "resolve_role_id", lambda value: value)


def test_epoch_rollover_checkpoints_archives_and_protects_truth(isolated_state):
    conv = conversation.create_conversation(title="Long task")
    task = tasks.create_task("finish the migration", "general", conversation_id=conv.id)
    conversation.add_message(conv.id, "user", "old instruction", related_task_id=task.id)
    conversation.add_message(conv.id, "assistant", "old response", related_task_id=task.id)
    conversation.add_message(
        conv.id, "tool", "unresolved result", related_task_id=task.id,
        metadata={"unresolved": True},
    )
    latest = conversation.add_message(
        conv.id, "user", "latest instruction must survive", related_task_id=task.id
    )
    epoch = context_epochs.rollover(task.id)
    assert epoch.epoch == 1 and len(epoch.archived_messages) == 2
    assert tasks.load_task(task.id).epoch == 2
    assert checkpoints.verify_checkpoint(epoch.checkpoint_id)
    active = conversation.list_messages(conv.id, include_sideband=False)
    assert [message.content for message in active] == ["unresolved result", latest.content]
    hits = context_epochs.search_transcript(conv.id, "old instruction")
    assert hits[0]["include_in_context"] == 0
    assert context_epochs.list_epochs(task.id)[0].id == epoch.id
    assert state_store.health()["schema_version"] == 13


def test_rollover_without_eligible_history_is_safe(isolated_state):
    conv = conversation.create_conversation()
    task = tasks.create_task("short", "general", conversation_id=conv.id)
    conversation.add_message(conv.id, "user", "only latest", related_task_id=task.id)
    with pytest.raises(RuntimeError, match="no older context"):
        context_epochs.rollover(task.id)
    assert tasks.load_task(task.id).epoch == 1
    assert checkpoints.latest_checkpoint(task.id) is None


def test_watermarks_and_automatic_rollover(monkeypatch):
    assert context_engine.pressure_for(SimpleNamespace(pressure=0.69)).level == "normal"
    assert context_engine.pressure_for(SimpleNamespace(pressure=0.70)).level == "soft"
    assert context_engine.pressure_for(SimpleNamespace(pressure=0.85)).level == "hard"
    assert context_engine.pressure_for(SimpleNamespace(pressure=0.96)).level == "emergency"
    with pytest.raises(ValueError):
        context_engine.pressure_for(SimpleNamespace(pressure=0.2), soft=.9, hard=.8)

    engine = context_engine.ContextEngine({"context": {}})
    projections = iter([SimpleNamespace(pressure=.9), SimpleNamespace(pressure=.2)])
    engine.manager = SimpleNamespace(build=lambda *args, **kwargs: next(projections))
    monkeypatch.setattr(context_engine, "rollover", lambda task_id, reason: "epoch-record")
    projection, pressure, epoch = engine.prepare("conv", "general", task_id="task", exact=False)
    assert projection.pressure == .2 and pressure.level == "normal" and epoch == "epoch-record"


def test_context_history_marks_controls_and_unresolved_tools_as_protected(monkeypatch):
    records = [
        SimpleNamespace(seq=1, role="assistant", content="old", kind="message",
                        metadata={}, include_in_context=True),
        SimpleNamespace(seq=2, role="tool", content="pending result", kind="message",
                        metadata={"unresolved": True}, include_in_context=True),
        SimpleNamespace(seq=3, role="user", content="redirect", kind="control",
                        metadata={}, include_in_context=True),
    ]
    monkeypatch.setattr(context, "list_messages", lambda *args, **kwargs: records)
    history, through = context._history_messages("conv", limit=10)
    assert through == 3
    assert not history[0]["_protected"]
    assert history[1]["_protected"] and history[2]["_protected"]
