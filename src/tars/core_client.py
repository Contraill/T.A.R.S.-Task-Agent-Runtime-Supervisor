from __future__ import annotations

from ipaddress import ip_address
import json
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request

from .network import network_destination, open_bound
from .secret_store import SecretStore, parse_reference


def _core_destination(url, *, resolve_dns=True):
    destination = network_destination(
        url, resolve_dns=resolve_dns, allow_loopback=True,
    )
    if (resolve_dns and destination.scheme == "http"
            and not all(ip_address(value).is_loopback for value in destination.addresses)):
        raise PermissionError(
            "authenticated non-loopback Core endpoints require HTTPS"
        )
    return destination


class CoreClient:
    def __init__(self, base_url: str, token: str | None = None, *, token_ref=None,
                 secret_store=None, transport=None):
        if bool(token) == bool(token_ref):
            raise ValueError("provide exactly one Core token or token reference")
        if token_ref:
            parse_reference(token_ref)
        configured = _core_destination(base_url, resolve_dns=False)
        if urlsplit(configured.request_url).query:
            raise ValueError("Core base URLs cannot contain query parameters")
        self.base_url = configured.request_url.rstrip("/")
        self._token = token
        self.token_ref = token_ref
        self.secret_store = secret_store or SecretStore()
        self.transport = transport

    def _authorization(self):
        if self._token is not None:
            return "Bearer " + self._token
        with self.secret_store.resolve(self.token_ref, consumer="core:client") as token:
            return "Bearer " + token

    @classmethod
    def pair(cls, base_url: str, code: str, name: str, *, metadata=None, transport=None):
        configured = _core_destination(base_url, resolve_dns=False)
        if urlsplit(configured.request_url).query:
            raise ValueError("Core base URLs cannot contain query parameters")
        destination = _core_destination(
            configured.request_url.rstrip("/") + "/v1/pair/exchange",
        )
        data = json.dumps({"code": code, "name": name,
                           "metadata": metadata or {}}).encode()
        request = Request(
            destination.request_url,
            data=data, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json"})
        response = (transport(request, timeout=30) if transport is not None else
                    open_bound(
                        destination, method="POST", headers=dict(request.headers),
                        body=data, timeout=30,
                    ))
        with response:
            cls._verify_response_destination(response, destination)
            result = json.loads(response.read())
        return cls(configured.request_url, result["token"], transport=transport), result["client"]

    @staticmethod
    def _verify_response_destination(response, destination):
        status = getattr(response, "status", None)
        status = response.getcode() if status is None and hasattr(response, "getcode") else status
        if status in {301, 302, 303, 307, 308}:
            raise PermissionError("Core redirects require separate origin authorization")
        if hasattr(response, "geturl"):
            actual = _core_destination(response.geturl(), resolve_dns=False)
            if actual.request_url != destination.request_url:
                raise PermissionError("Core response crossed its authorized origin")

    def _open(self, request, destination, *, timeout):
        if self.transport is not None:
            return self.transport(request, timeout=timeout)
        return open_bound(
            destination, method=request.get_method(), headers=dict(request.headers),
            body=request.data, timeout=timeout,
        )

    def request(self, method: str, path: str, body=None):
        if not str(path).startswith("/") or str(path).startswith("//"):
            raise ValueError("Core request path must be origin-relative")
        data = None if body is None else json.dumps(body).encode()
        destination = _core_destination(self.base_url + path)
        request = Request(
            destination.request_url, data=data, method=method,
            headers={"Authorization": self._authorization(),
                     "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self._open(request, destination, timeout=30) as response:
                self._verify_response_destination(response, destination)
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
        destination = _core_destination(
            self.base_url
            + f"/v1/tasks/{task_id}/events?after={int(after)}&follow={1 if follow else 0}"
        )
        request = Request(
            destination.request_url,
            headers={"Authorization": self._authorization(),
                     "Accept": "text/event-stream"})
        with self._open(request, destination, timeout=None) as response:
            self._verify_response_destination(response, destination)
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
