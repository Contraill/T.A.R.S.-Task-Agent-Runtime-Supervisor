from types import SimpleNamespace

import pytest

from tars import identity, memory, prompt_compiler, state_store, temporary


@pytest.fixture
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index")
    memory_root = tmp_path / "memory"
    monkeypatch.setattr(memory, "MEMORY_ROOT", memory_root)
    monkeypatch.setattr(memory, "MEMORY_HISTORY_ROOT", memory_root / ".history")
    persona = tmp_path / "persona"
    monkeypatch.setattr(identity, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(identity, "IDENTITY_PATH", persona / "IDENTITY.md")
    monkeypatch.setattr(identity, "SOUL_PATH", persona / "SOUL.md")
    monkeypatch.setattr(identity, "ROLE_PERSONA_ROOT", persona / "roles")
    monkeypatch.setattr(prompt_compiler, "load_identity", identity.load_identity)
    monkeypatch.setattr(prompt_compiler, "resolve_role_id", lambda value: value)
    monkeypatch.setattr(prompt_compiler, "get_role", lambda value: SimpleNamespace(
        display_name="General", description="General role", capabilities=("conversation",)
    ))
    monkeypatch.setattr(temporary, "resolve_role_id", lambda value: value)
    monkeypatch.setattr(temporary, "budget_for_role", lambda *args, **kwargs: SimpleNamespace(
        usable_input=4096
    ))
    identity.ensure_identity_files()
    entry = memory.remember("The user prefers metric units", scope="profile", source="user")
    return tmp_path, entry


def test_temporary_multiturn_reads_identity_and_memory_without_writes(isolated_environment):
    tmp_path, memory_entry = isolated_environment
    before = state_store.health()["counts"].copy()
    memory_before = (tmp_path / "memory" / "profile" / f"{memory_entry.id}.md").read_bytes()
    prompts = []

    def complete(cfg, role, messages, max_tokens, thinking):
        prompts.append(messages)
        users = [item for item in messages if item["role"] == "user"]
        return {"choices": [{"message": {"content": f"reply-{len(users)}"}}]}

    session = temporary.TemporarySession({}, "general")
    session.send("Which units do I prefer?", complete=complete)
    session.send("What did I just ask?", complete=complete)
    assert temporary.TEMPORARY_NOTICE in prompts[0][0]["content"]
    assert memory_entry.id in "\n".join(item["content"] for item in prompts[0])
    assert [item["content"] for item in session.turns] == [
        "Which units do I prefer?", "reply-1", "What did I just ask?", "reply-2",
    ]
    session.close()
    assert session.turns == []
    assert state_store.health()["counts"] == before
    assert (tmp_path / "memory" / "profile" / f"{memory_entry.id}.md").read_bytes() == memory_before


def test_temporary_failure_and_close_discard_state(isolated_environment):
    session = temporary.TemporarySession({}, "general")

    def fail(*args, **kwargs):
        raise RuntimeError("backend failed")

    with pytest.raises(RuntimeError, match="backend failed"):
        session.send("do not retain", complete=fail)
    assert session.turns == []
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.send("later", complete=fail)
    counts = state_store.health()["counts"]
    assert counts["sessions"] == counts["tasks"] == counts["checkpoints"] == 0


def test_temporary_stream_is_chunked_coherent_and_ephemeral(isolated_environment):
    before = state_store.health()["counts"].copy()
    prompts = []

    def stream(cfg, role, messages, **kwargs):
        prompts.append(messages)
        yield {"content": "chunk ", "reasoning": "real", "finish_reason": None}
        yield {"content": "two", "reasoning": "", "finish_reason": "stop"}

    session = temporary.TemporarySession({}, "general")
    assert [event["content"] for event in session.stream("hello", stream=stream)] == [
        "chunk ", "two"]
    assert [turn["content"] for turn in session.turns] == ["hello", "chunk two"]
    assert temporary.TEMPORARY_NOTICE in prompts[0][0]["content"]
    session.close()
    assert state_store.health()["counts"] == before


def test_abandoned_temporary_stream_rolls_back_user_turn(isolated_environment):
    def stream(*args, **kwargs):
        yield {"content": "partial", "finish_reason": None}
        yield {"content": "never consumed", "finish_reason": "stop"}

    session = temporary.TemporarySession({}, "general")
    response = session.stream("do not retain partial", stream=stream)
    next(response)
    response.close()
    assert session.turns == []


def test_temporary_context_is_bounded_without_persistent_projection(monkeypatch, isolated_environment):
    monkeypatch.setattr(temporary, "budget_for_role", lambda *args, **kwargs: SimpleNamespace(
        usable_input=180
    ))
    session = temporary.TemporarySession({}, "general")
    session.turns = [
        {"role": "user", "content": "old " * 80},
        {"role": "assistant", "content": "answer " * 80},
        {"role": "user", "content": "latest"},
    ]
    messages = session._messages("latest")
    assert messages[-1]["content"] == "latest"
    assert "old old" not in "\n".join(item["content"] for item in messages)
    assert state_store.health()["counts"]["context_projections"] == 0


def test_temporary_cli_loop_has_unmistakable_banner(monkeypatch, isolated_environment):
    values = iter(["hello", "/temporary"])
    output = []
    monkeypatch.setattr(temporary.TemporarySession, "send", lambda self, value: {
        "choices": [{"message": {"content": "ephemeral"}}]
    })
    assert temporary.run_temporary(
        {}, role_id="general", input_fn=lambda prompt: next(values), output_fn=output.append
    ) == 0
    assert output == ["TEMPORARY · new T.A.R.S. state will not be persisted", "ephemeral"]
