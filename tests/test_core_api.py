import json
import ssl
import threading
import time
from urllib.request import Request, urlopen

import pytest

from tars import cli, conversation, core_api, core_auth, events, scheduler, state_store, tasks
from tars.core_client import CoreClient as NativeCoreClient


@pytest.fixture
def core_state(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy-tasks")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index.json")
    monkeypatch.setattr(tasks, "resolve_role_id", lambda value: value)
    return tmp_path


def paired(*, permissions=core_auth.DEFAULT_PERMISSIONS):
    pairing = core_auth.create_pairing(permissions=permissions)
    return core_auth.exchange_pairing(pairing["code"], "test client")


def test_pairing_token_is_one_time_hashed_and_revocable(core_state):
    pairing = core_auth.create_pairing()
    client, token = core_auth.exchange_pairing(pairing["code"], "laptop")
    assert core_auth.authenticate(token).id == client.id
    with state_store.connect() as conn:
        row = conn.execute("SELECT token_hash,token_salt FROM core_clients WHERE id=?",
                           (client.id,)).fetchone()
        assert token not in (row["token_hash"], row["token_salt"])
        assert row["token_hash"].startswith("v2$")
        assert pairing["code"] not in conn.execute(
            "SELECT code_hash FROM core_pairings").fetchone()[0]
    with pytest.raises(PermissionError, match="consumed"):
        core_auth.exchange_pairing(pairing["code"], "replay")
    assert core_auth.revoke(client.id).state == "revoked"
    with pytest.raises(PermissionError, match="revoked"):
        core_auth.authenticate(token)


def test_authentication_uses_indexed_client_and_one_scrypt(core_state, monkeypatch):
    first, first_token = paired()
    paired()
    paired()
    calls = []
    real_derive = core_auth._derive

    def counted(secret, salt):
        calls.append((secret, salt))
        return real_derive(secret, salt)

    monkeypatch.setattr(core_auth, "_derive", counted)
    assert core_auth.authenticate(first_token).id == first.id
    assert len(calls) == 1
    with pytest.raises(PermissionError, match="invalid"):
        core_auth.authenticate(first_token.rsplit(".", 1)[0] + ".wrong")
    assert len(calls) == 2
    with pytest.raises(PermissionError, match="invalid"):
        core_auth.authenticate("client-does-not-exist.secret")
    assert len(calls) == 3


def test_legacy_unindexed_credentials_are_revoked_on_upgrade(core_state):
    client, _ = paired()
    with state_store.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE core_clients SET token_hash='legacy-hash' WHERE id=?", (client.id,)
        )
    state_store.ensure_state_store()
    assert next(item for item in core_auth.list_clients() if item.id == client.id).state == "revoked"


def test_api_uses_canonical_conversation_task_and_permissions(core_state):
    client, token = paired()
    api = core_api.CoreAPI()
    auth = f"Bearer {token}"
    status, conv = api.dispatch(
        "POST", "/v1/conversations", {"title": "Shared"}, authorization=auth)
    assert status == 201 and conversation.load_conversation(conv["id"]).title == "Shared"
    status, message = api.dispatch(
        "POST", f"/v1/conversations/{conv['id']}/messages", {"content": "hello"},
        authorization=auth)
    assert status == 201 and conversation.list_messages(conv["id"])[0].id == message["id"]
    second, second_token = paired(permissions=("conversation.read",))
    _, shared = api.dispatch("GET", "/v1/conversations", authorization=f"Bearer {second_token}")
    assert shared[0]["id"] == conv["id"] and second.id != client.id
    task = tasks.create_task("continue", "general", conversation_id=conv["id"],
                             make_active=False)
    status, rows = api.dispatch("GET", "/v1/tasks", authorization=auth)
    assert status == 200 and rows[0]["task_id"] == task.id

    read_only, read_token = paired(permissions=("task.read",))
    with pytest.raises(PermissionError, match="task.control"):
        api.dispatch("POST", f"/v1/tasks/{task.id}/control",
                     {"kind": "pause"}, authorization=f"Bearer {read_token}")
    assert read_only.principal_id == client.principal_id == "local-owner"


def test_task_event_stream_resumes_and_hides_internal_events(core_state):
    client, token = paired()
    task = tasks.create_task("stream", "general", make_active=False)
    first = events.append_event(task.id, "progress", "one")
    events.append_event(task.id, "status", "internal", visibility="internal")
    api = core_api.CoreAPI()
    authenticated = api.authenticate_header(f"Bearer {token}")
    stream = api.stream_task_events(task.id, authenticated, after=0, follow=False)
    received = list(stream)
    assert [item["message"] for item in received] == ["Task created with owner general", "one"]
    events.append_event(task.id, "result", "two")
    resumed = list(api.stream_task_events(
        task.id, authenticated, after=first.id, follow=False))
    assert [item["message"] for item in resumed] == ["two"]


def test_open_event_stream_stops_after_client_revocation(core_state):
    client, token = paired()
    task = tasks.create_task("stream", "general", make_active=False)
    api = core_api.CoreAPI()
    authenticated = api.authenticate_header(f"Bearer {token}")
    stream = api.stream_task_events(task.id, authenticated, follow=True, poll_seconds=0.02)
    assert next(stream)["task_id"] == task.id
    core_auth.revoke(client.id)
    with pytest.raises(StopIteration):
        next(stream)


def test_open_http_event_stream_closes_after_client_revocation(core_state):
    client, token = paired()
    task = tasks.create_task("network stream", "general", make_active=False)
    server = core_api.make_server(core_api.CoreServerConfig(port=0))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    response = None
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/tasks/{task.id}/events",
            headers={"Authorization": f"Bearer {token}", "Accept": "text/event-stream"})
        response = urlopen(request, timeout=2)
        while response.readline() not in {b"\n", b""}:
            pass
        core_auth.revoke(client.id)
        closed = []

        def read_to_close():
            closed.append(response.read())

        reader = threading.Thread(target=read_to_close, daemon=True)
        reader.start()
        reader.join(timeout=2)
        assert not reader.is_alive() and closed == [b""]
    finally:
        if response is not None:
            response.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


def test_sse_denial_and_missing_task_happen_before_success_headers(core_state):
    _, no_read_token = paired(permissions=("status.read",))
    _, read_token = paired(permissions=("task.read",))
    task = tasks.create_task("protected stream", "general", make_active=False)
    server = core_api.make_server(core_api.CoreServerConfig(port=0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def status(path, token):
            request = Request(
                f"http://127.0.0.1:{server.server_port}{path}",
                headers={"Authorization": f"Bearer {token}",
                         "Accept": "text/event-stream"})
            try:
                urlopen(request, timeout=2)
            except Exception as exc:
                return exc.code
            raise AssertionError("SSE request unexpectedly succeeded")

        assert status(f"/v1/tasks/{task.id}/events?follow=0", no_read_token) == 403
        assert status("/v1/tasks/task-missing/events?follow=0", read_token) == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_network_defaults_require_explicit_tls_for_remote(core_state):
    core_api.CoreServerConfig().validate()
    with pytest.raises(PermissionError, match="allow_remote"):
        core_api.CoreServerConfig(host="0.0.0.0").validate()
    with pytest.raises(PermissionError, match="TLS"):
        core_api.CoreServerConfig(host="0.0.0.0", allow_remote=True).validate()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    core_api.CoreServerConfig(
        host="192.0.2.1", allow_remote=True, ssl_context=context).validate()


def test_core_serve_options_use_config_unless_overridden(core_state):
    parser = cli.build_parser()
    configured = {"core": {"host": "127.0.0.2", "port": 9123, "allow_remote": True}}
    defaults = parser.parse_args(["core", "serve"])
    assert cli._core_server_options(configured, defaults) == ("127.0.0.2", 9123, True)
    explicit = parser.parse_args([
        "core", "serve", "--host", "127.0.0.3", "--port", "9456",
        "--no-allow-remote",
    ])
    assert cli._core_server_options(configured, explicit) == ("127.0.0.3", 9456, False)


def test_loopback_http_and_native_client_round_trip(core_state):
    client, token = paired()
    server = core_api.make_server(core_api.CoreServerConfig(port=0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        native = NativeCoreClient(f"http://127.0.0.1:{server.server_port}", token)
        assert native.status()["client_id"] == client.id
        task = tasks.create_task("remote view", "general", make_active=False)
        seen = list(native.stream_events(task.id, follow=False))
        assert seen[0]["task_id"] == task.id
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_pair_exchange_is_loopback_by_default(core_state):
    pairing = core_auth.create_pairing()
    with pytest.raises(PermissionError, match="loopback"):
        core_api.CoreAPI().dispatch(
            "POST", "/v1/pair/exchange", {"code": pairing["code"], "name": "remote"},
            remote_addr="198.51.100.2")


def test_core_schedule_mutation_wakes_scheduler_service(core_state):
    client, token = paired()
    auth = f"Bearer {token}"
    api = core_api.CoreAPI()
    task = tasks.create_task("Core scheduled", "general", make_active=False)
    engine = scheduler.Scheduler()
    stop = threading.Event()
    executed = []

    def execute(run):
        executed.append(run.id)
        stop.set()
        engine.wake()
        return {"ok": True}

    thread = threading.Thread(
        target=engine.run_forever, args=(execute,), kwargs={"stop": stop}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while not engine._wake_socket and time.monotonic() < deadline:
        time.sleep(0.01)
    assert engine._wake_socket is not None
    status, record = api.dispatch(
        "POST", "/v1/schedules",
        {"task_id": task.id, "kind": "one-shot",
         "expression": state_store.now_utc()}, authorization=auth)
    assert status == 201 and record["task_id"] == task.id
    thread.join(timeout=2)
    assert executed and not thread.is_alive()


def test_core_rejects_unconfigured_condition_schedule(core_state):
    client, token = paired()
    task = tasks.create_task("watch", "general", make_active=False)
    with pytest.raises(ValueError, match="not configured"):
        core_api.CoreAPI().dispatch(
            "POST", "/v1/schedules",
            {"task_id": task.id, "kind": "condition",
             "expression": "missing@every 1m"},
            authorization=f"Bearer {token}")
