from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from .secret_store import SecretStore, parse_reference


class CoreClient:
    def __init__(self, base_url: str, token: str | None = None, *, token_ref=None,
                 secret_store=None, transport=None):
        if bool(token) == bool(token_ref):
            raise ValueError("provide exactly one Core token or token reference")
        if token_ref:
            parse_reference(token_ref)
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.token_ref = token_ref
        self.secret_store = secret_store or SecretStore()
        self.transport = transport or urlopen

    def _authorization(self):
        if self._token is not None:
            return "Bearer " + self._token
        with self.secret_store.resolve(self.token_ref, consumer="core:client") as token:
            return "Bearer " + token

    @classmethod
    def pair(cls, base_url: str, code: str, name: str, *, metadata=None, transport=None):
        opener = transport or urlopen
        request = Request(
            base_url.rstrip("/") + "/v1/pair/exchange",
            data=json.dumps({"code": code, "name": name,
                             "metadata": metadata or {}}).encode(), method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"})
        with opener(request, timeout=30) as response:
            result = json.loads(response.read())
        return cls(base_url, result["token"], transport=transport), result["client"]

    def request(self, method: str, path: str, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(
            self.base_url + path, data=data, method=method,
            headers={"Authorization": self._authorization(),
                     "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self.transport(request, timeout=30) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", str(exc))
            except Exception:
                detail = str(exc)
            raise RuntimeError(f"Core API {exc.code}: {detail}") from exc

    def status(self):
        return self.request("GET", "/v1/status")

    def conversations(self):
        return self.request("GET", "/v1/conversations")

    def messages(self, conversation_id):
        return self.request("GET", f"/v1/conversations/{conversation_id}/messages")

    def send_message(self, conversation_id, content):
        return self.request("POST", f"/v1/conversations/{conversation_id}/messages",
                            {"content": content})

    def tasks(self):
        return self.request("GET", "/v1/tasks")

    def schedules(self):
        return self.request("GET", "/v1/schedules")

    def add_schedule(self, task_id, kind, expression, **options):
        return self.request("POST", "/v1/schedules",
                            {"task_id": task_id, "kind": kind,
                             "expression": expression, **options})

    def schedule_action(self, schedule_id, action, **changes):
        return self.request("POST", f"/v1/schedules/{schedule_id}/{action}", changes)

    def task_events(self, task_id, *, after=0):
        return self.request("GET", f"/v1/tasks/{task_id}/events?after={int(after)}")

    def stream_events(self, task_id, *, after=0, follow=True):
        request = Request(
            self.base_url + f"/v1/tasks/{task_id}/events?after={int(after)}&follow={1 if follow else 0}",
            headers={"Authorization": self._authorization(),
                     "Accept": "text/event-stream"})
        with self.transport(request, timeout=None) as response:
            data = []
            for raw in response:
                line = raw.decode().rstrip("\r\n")
                if line.startswith("data: "):
                    data.append(line[6:])
                elif not line and data:
                    event = json.loads("\n".join(data))
                    after = max(after, int(event["id"]))
                    data.clear()
                    yield event

    def control(self, task_id, kind, message="", payload=None):
        return self.request("POST", f"/v1/tasks/{task_id}/control",
                            {"kind": kind, "message": message, "payload": payload or {}})
