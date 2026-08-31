from __future__ import annotations

from dataclasses import asdict, dataclass
import ipaddress
import json
import ssl
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import __version__
from .agent_loop import submit_task_control
from .conversation import add_message, create_conversation, list_conversations, list_messages
from .core_auth import CoreClient, authenticate, exchange_pairing, list_clients, revoke
from .events import read_events_since
from .scheduler import (create_schedule, edit_schedule, list_schedules,
                        load_schedule, remove_schedule, set_enabled)
from .state_store import health as state_health
from .tasks import canonical_task_state, list_tasks, load_task


def _records(items):
    return [asdict(item) for item in items]


def _loopback(address: str) -> bool:
    try:
        return ipaddress.ip_address(address.split("%", 1)[0]).is_loopback
    except ValueError:
        return address.casefold() == "localhost"


@dataclass(frozen=True)
class CoreServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    allow_remote: bool = False
    ssl_context: ssl.SSLContext | None = None

    def validate(self):
        if not 0 <= int(self.port) <= 65535:
            raise ValueError("Core API port is invalid")
        if not _loopback(self.host):
            if not self.allow_remote:
                raise PermissionError("non-loopback Core binding requires explicit allow_remote")
            if self.ssl_context is None:
                raise PermissionError("direct remote Core binding requires TLS")
        return self


class CoreAPI:
    """Authenticated interface over the one canonical local state model."""

    def __init__(self, *, allow_remote_pairing=False, conditions=None):
        self.allow_remote_pairing = bool(allow_remote_pairing)
        self.conditions = dict(conditions or {})

    def authenticate_header(self, authorization: str) -> CoreClient:
        scheme, _, token = str(authorization).partition(" ")
        if scheme.casefold() != "bearer" or not token:
            raise PermissionError("Bearer client authentication is required")
        return authenticate(token)

    def dispatch(self, method: str, raw_path: str, body=None, *, authorization="",
                 remote_addr="127.0.0.1") -> tuple[int, dict | list]:
        method = method.upper()
        parsed = urlsplit(raw_path)
        path = [part for part in parsed.path.split("/") if part]
        query = parse_qs(parsed.query)
        body = body or {}
        if method == "POST" and path == ["v1", "pair", "exchange"]:
            if not _loopback(remote_addr) and not self.allow_remote_pairing:
                raise PermissionError("pairing exchange is loopback-only")
            client, token = exchange_pairing(body.get("code", ""), body.get("name", ""),
                                             metadata=body.get("metadata"))
            return 201, {"client": asdict(client), "token": token}

        client = self.authenticate_header(authorization)
        if method == "GET" and path == ["v1", "status"]:
            client.require("status.read")
            health = state_health()
            return 200, {"version": __version__, "state_ok": health["ok"],
                         "schema_version": health["schema_version"],
                         "client_id": client.id, "principal_id": client.principal_id}
        if method == "GET" and path == ["v1", "conversations"]:
            client.require("conversation.read")
            return 200, _records(list_conversations(limit=int(query.get("limit", [50])[0])))
        if method == "POST" and path == ["v1", "conversations"]:
            client.require("conversation.write")
            record = create_conversation(title=body.get("title", ""), source=f"client:{client.id}",
                                         metadata={"client_id": client.id,
                                                   "principal_id": client.principal_id},
                                         make_active=False)
            return 201, asdict(record)
        if method == "GET" and path == ["v1", "clients"]:
            client.require("client.admin")
            return 200, _records(list_clients())
        if (method == "POST" and len(path) == 4 and path[:2] == ["v1", "clients"]
                and path[3] == "revoke"):
            client.require("client.admin")
            if path[2] == client.id:
                raise PermissionError("a client cannot revoke its current token")
            return 200, asdict(revoke(path[2]))
        if len(path) == 4 and path[:2] == ["v1", "conversations"] and path[3] == "messages":
            conversation_id = path[2]
            if method == "GET":
                client.require("conversation.read")
                return 200, _records(list_messages(
                    conversation_id, limit=int(query.get("limit", [200])[0])))
            if method == "POST":
                client.require("conversation.write")
                content = str(body.get("content", "")).strip()
                if not content:
                    raise ValueError("message content is required")
                record = add_message(
                    conversation_id, "user", content,
                    metadata={"client_id": client.id, "principal_id": client.principal_id})
                return 201, asdict(record)
        if method == "GET" and path == ["v1", "tasks"]:
            client.require("task.read")
            return 200, [canonical_task_state(task.id) for task in list_tasks(
                limit=int(query.get("limit", [50])[0]))]
        if method == "GET" and path == ["v1", "schedules"]:
            client.require("task.read")
            return 200, _records(list_schedules(limit=int(query.get("limit", [100])[0])))
        if method == "POST" and path == ["v1", "schedules"]:
            client.require("task.control")
            kind = str(body.get("kind", ""))
            expression = str(body.get("expression", ""))
            if kind == "condition":
                name = expression.split("@", 1)[0].strip()
                if name not in self.conditions:
                    raise ValueError(f"condition is not configured: {name}")
            record = create_schedule(
                str(body.get("task_id", "")), kind, expression,
                next_run_at=body.get("next_run_at"),
                missed_policy=body.get("missed_policy", "run-once"),
                max_catch_up=int(body.get("max_catch_up", 1)),
                concurrency_key=body.get("concurrency_key", "default"),
                max_concurrency=int(body.get("max_concurrency", 1)),
                delivery_target=body.get("delivery_target", ""))
            return 201, asdict(record)
        if len(path) >= 3 and path[:2] == ["v1", "schedules"]:
            schedule_id = path[2]
            load_schedule(schedule_id)
            if method == "GET" and len(path) == 3:
                client.require("task.read")
                return 200, asdict(load_schedule(schedule_id))
            if method == "POST" and len(path) == 4:
                client.require("task.control")
                action = path[3]
                if action == "edit":
                    current = load_schedule(schedule_id)
                    expression = body.get("expression")
                    if current.kind == "condition" and expression is not None:
                        name = str(expression).split("@", 1)[0].strip()
                        if name not in self.conditions:
                            raise ValueError(f"condition is not configured: {name}")
                    record = edit_schedule(
                        schedule_id, expression=body.get("expression"),
                        next_run_at=body.get("next_run_at"),
                        missed_policy=body.get("missed_policy"),
                        max_catch_up=body.get("max_catch_up"),
                        concurrency_key=body.get("concurrency_key"),
                        max_concurrency=body.get("max_concurrency"),
                        delivery_target=body.get("delivery_target"))
                    return 200, asdict(record)
                if action in {"pause", "resume"}:
                    return 200, asdict(set_enabled(schedule_id, action == "resume"))
                if action == "remove":
                    remove_schedule(schedule_id)
                    return 200, {"id": schedule_id, "removed": True}
        if len(path) >= 3 and path[:2] == ["v1", "tasks"]:
            task_id = path[2]
            load_task(task_id)
            if method == "GET" and len(path) == 3:
                client.require("task.read")
                return 200, canonical_task_state(task_id)
            if method == "GET" and len(path) == 4 and path[3] == "events":
                client.require("task.read")
                after = int(query.get("after", [0])[0])
                events = [event for event in read_events_since(task_id, after, 500)
                          if event["visibility"] != "internal"]
                return 200, events
            if method == "POST" and len(path) == 4 and path[3] == "control":
                client.require("task.control")
                control, feedback = submit_task_control(
                    task_id, str(body.get("kind", "")), str(body.get("message", "")),
                    payload={**dict(body.get("payload") or {}),
                             "client_id": client.id,
                             "principal_id": client.principal_id})
                return 202, {"control_id": control.id, "feedback": feedback}
        raise KeyError(f"unknown Core API route: {method} {parsed.path}")

    def stream_task_events(self, task_id: str, client: CoreClient, *, after=0,
                           follow=True, stop=None, poll_seconds=0.1):
        client.require("task.read")
        load_task(task_id)
        cursor = int(after)
        while True:
            events = [event for event in read_events_since(task_id, cursor, 200)
                      if event["visibility"] != "internal"]
            for event in events:
                cursor = max(cursor, event["id"])
                yield event
            if not follow or (stop is not None and stop.is_set()):
                return
            time.sleep(max(0.02, float(poll_seconds)))


class _CoreHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, api):
        self.api = api
        super().__init__(address, handler)


class CoreRequestHandler(BaseHTTPRequestHandler):
    server_version = "TARS-Core"

    def log_message(self, format, *args):
        return

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length < 0 or length > 1_048_576:
            raise ValueError("request body exceeds 1 MiB")
        body = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    def _json(self, status, value):
        data = json.dumps(value, separators=(",", ":"), default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self):
        try:
            parsed = urlsplit(self.path)
            parts = [part for part in parsed.path.split("/") if part]
            if (self.command == "GET" and len(parts) == 4 and
                    parts[:2] == ["v1", "tasks"] and parts[3] == "events" and
                    "text/event-stream" in self.headers.get("Accept", "")):
                return self._stream_events(parts[2], parsed)
            status, value = self.server.api.dispatch(
                self.command, self.path, self._body() if self.command == "POST" else {},
                authorization=self.headers.get("Authorization", ""),
                remote_addr=self.client_address[0])
            self._json(status, value)
        except PermissionError as exc:
            self._json(403, {"error": str(exc)})
        except KeyError as exc:
            self._json(404, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": f"Core request failed: {exc}"})

    do_GET = _dispatch
    do_POST = _dispatch

    def _stream_events(self, task_id, parsed):
        client = self.server.api.authenticate_header(self.headers.get("Authorization", ""))
        query = parse_qs(parsed.query)
        after = int(query.get("after", [self.headers.get("Last-Event-ID", "0")])[0])
        follow = query.get("follow", ["1"])[0] not in {"0", "false"}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive" if follow else "close")
        self.end_headers()
        try:
            for event in self.server.api.stream_task_events(
                    task_id, client, after=after, follow=follow):
                payload = json.dumps(event, separators=(",", ":"), default=str)
                self.wfile.write(
                    f"id: {event['id']}\nevent: {event['type']}\ndata: {payload}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            if not follow:
                self.close_connection = True


def make_server(config: CoreServerConfig, api=None):
    config.validate()
    server = _CoreHTTPServer(
        (config.host, int(config.port)), CoreRequestHandler,
        api or CoreAPI(allow_remote_pairing=config.allow_remote and config.ssl_context is not None))
    if config.ssl_context is not None:
        server.socket = config.ssl_context.wrap_socket(server.socket, server_side=True)
    return server
