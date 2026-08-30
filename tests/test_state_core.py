from pathlib import Path
from types import SimpleNamespace

import pytest

from tars import activity, conversation, identity, prompt_compiler, projects, role_state, sessions
from tars import state_events, state_store


@pytest.fixture
def isolated_state(monkeypatch, tmp_path):
    db = tmp_path / "state.sqlite3"
    monkeypatch.setattr(state_store, "STATE_DB_PATH", db)
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy-tasks")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index.json")
    return db


def test_schema_session_event_role_and_project_state(monkeypatch, isolated_state, tmp_path):
    monkeypatch.setattr(sessions, "resolve_role_id", lambda value: value)
    monkeypatch.setattr(role_state, "resolve_role_id", lambda value: value)
    conv = conversation.create_conversation(title="Durable")
    session = sessions.create_session(conversation_id=conv.id, role_id="general")
    event = state_events.append_state_event(
        "user_message", "hello", session_id=session.id, conversation_id=conv.id,
        role="general", payload={"message_id": "one"},
    )
    assert state_events.read_state_events(session_id=session.id)[-1] == event
    message = conversation.add_message(conv.id, "assistant", "welcome", session_id=session.id)
    trace = activity.activity_trace(session_id=session.id)
    assert trace[-1]["type"] == "assistant_response"
    assert trace[-1]["payload"]["message_id"] == message.id
    assert sessions.close_session(session.id).state == "closed"
    assert role_state.save_role_state("general", {"reasoning": "summary"})["reasoning"] == "summary"
    (tmp_path / "TARS.md").write_text("native context")
    (tmp_path / "AGENTS.md").write_text("compatibility context")
    context = projects.register_project(tmp_path)
    assert [path.name for path in context.files] == ["TARS.md", "AGENTS.md"]
    health = state_store.health()
    assert health["ok"] and health["schema_version"] == 10
    assert health["counts"]["sessions"] == 1 and health["counts"]["state_events"] == 4


def test_append_event_transaction_rolls_back_on_invalid_reference(isolated_state):
    with pytest.raises(Exception):
        state_events.append_state_event("activity", task_id="missing")
    assert state_store.health()["counts"]["state_events"] == 0


def test_schema_upgrade_preserves_existing_conversation(isolated_state):
    state_store.ensure_state_store()
    conv = conversation.create_conversation(title="before upgrade")
    with state_store.transaction(immediate=True) as conn:
        conn.execute("UPDATE meta SET value='4' WHERE key='schema_version'")
    state_store.ensure_state_store()
    assert conversation.load_conversation(conv.id).title == "before upgrade"
    assert state_store.health()["schema_version"] == 10


def test_identity_inheritance_and_prompt_explain(monkeypatch, tmp_path):
    persona = tmp_path / "persona"
    monkeypatch.setattr(identity, "IDENTITY_PATH", persona / "IDENTITY.md")
    monkeypatch.setattr(identity, "SOUL_PATH", persona / "SOUL.md")
    monkeypatch.setattr(identity, "ROLE_PERSONA_ROOT", persona / "roles")
    identity.ensure_identity_files()
    (persona / "roles/general.md").write_text("Be concise.")
    monkeypatch.setattr(prompt_compiler, "resolve_role_id", lambda value: "general")
    monkeypatch.setattr(prompt_compiler, "get_role", lambda value: SimpleNamespace(
        display_name="General", description="Daily role", capabilities=("conversation",)
    ))
    monkeypatch.setattr(prompt_compiler, "load_identity", identity.load_identity)
    compiled = prompt_compiler.PromptCompiler().compile(
        role_name="general", personal_memory=["prefers concise answers"],
        pending_controls=["do not modify files"],
    )
    explanation = compiled.explain()
    names = [source["name"] for source in explanation["sources"]]
    assert names == ["base_identity", "role_overlay", "capabilities", "personal_memory", "pending_controls"]
    assert explanation["sources"][-1]["protected"]
    assert len([message for message in compiled.messages if message["role"] == "system"]) == 1
    assert "Be concise" in compiled.messages[0]["content"]


def test_reasoning_visibility_never_synthesizes_raw():
    assert activity.reasoning_view("raw", emitted_raw="") == ""
    assert activity.reasoning_view("raw", emitted_raw="backend thought") == "backend thought"
    assert activity.reasoning_view("summary", summary="short") == "short"
    assert activity.reasoning_view("hidden", emitted_raw="secret") == ""
