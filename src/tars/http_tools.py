from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import urllib.request

from .network import NetworkDestination, network_destination, open_bound
from .policy import ScopeRequest, canonical_path, redact
from .secure_paths import AnchoredRoot, select_anchor
from .tool_core import ToolResult, ToolRuntime


@dataclass(frozen=True)
class HTTPResponse:
    url: str
    status: int
    headers: dict
    body: bytes
    truncated: bool
    peer_ip: str = ""


class UrllibHTTPTransport:
    def __call__(self, method, url, *, headers, body, timeout, max_bytes,
                 destination=None):
        bound = destination or network_destination(url, resolve_dns=True)
        if not isinstance(bound, NetworkDestination) or bound.request_url != url:
            raise PermissionError("HTTP request differs from its authorized destination")
        with open_bound(
            bound, method=method, headers=headers, body=body, timeout=timeout,
        ) as response:
            payload = b"" if method == "HEAD" else response.read(max_bytes + 1)
            return HTTPResponse(
                response.geturl(), response.status, dict(response.headers.items()),
                payload[:max_bytes], len(payload) > max_bytes, response.peer_ip,
            )


class HTTPTools:
    def __init__(self, *, runtime=None, transport=None, timeout=30, max_bytes=2_000_000,
                 max_redirects=5):
        self.runtime = runtime or ToolRuntime()
        self.transport = transport or UrllibHTTPTransport()
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.cache = {}

    @staticmethod
    def _validate(url):
        return network_destination(url, resolve_dns=True)

    def request(self, method, url, *, headers=None, body=None, approval_ids=None,
                task_id=None, session_id=None, use_cache=True, output=None,
                allowed_paths=(), sensitive_values=()):
        method = method.upper()
        if method not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError(f"unsupported HTTP method: {method}")
        destination = self._validate(url)
        current = destination.request_url
        approved_origin = destination.origin
        safe_headers = {
            str(key): str(value) for key, value in dict(headers or {}).items()
        }
        if body is not None and not isinstance(body, (bytes, bytearray, memoryview)):
            raise TypeError("HTTP request bodies must be bytes")
        body = None if body is None else bytes(body)
        cached = (self.cache.get(destination.policy_url)
                  if use_cache and method in {"GET", "HEAD"} else None)
        if cached:
            if cached.headers.get("ETag"):
                safe_headers.setdefault("If-None-Match", cached.headers["ETag"])
            if cached.headers.get("Last-Modified"):
                safe_headers.setdefault("If-Modified-Since", cached.headers["Last-Modified"])
        requests = [("network", ScopeRequest(
            "http.request", "network", destination.policy_url,
            {"method": method, "headers": dict(safe_headers), "body": body or b"",
             "origin": approved_origin, "url_sha256": destination.url_sha256},
            task_id=task_id, session_id=session_id, allowed_hosts=(approved_origin,),
        ))]
        if method not in {"GET", "HEAD"}:
            requests.append(("write", ScopeRequest(
                "http.request", "write", destination.policy_url,
                {"method": method, "headers": dict(safe_headers), "body": body or b"",
                 "origin": approved_origin, "url_sha256": destination.url_sha256},
                task_id=task_id, session_id=session_id,
            )))
        output_path = None
        output_anchors = ()
        if output is not None:
            if method != "GET":
                raise ValueError("HTTP output files are supported only for GET")
            output_path = os.path.abspath(os.fspath(output))
            output_roots = tuple(canonical_path(path) for path in allowed_paths)
            output_anchors = tuple(AnchoredRoot(path) for path in output_roots)
            requests.append(("output", ScopeRequest(
                "http.download", "write", output_path,
                {"url": destination.policy_url, "url_sha256": destination.url_sha256},
                task_id=task_id, session_id=session_id,
                allowed_paths=output_roots,
            )))
        try:
            actions = self.runtime.authorize(tuple(requests), approval_ids)
        except Exception:
            for anchor in output_anchors:
                anchor.close()
            raise
        try:
            redirects = []
            for _ in range(self.max_redirects + 1):
                response = self.transport(
                    method, current, headers=safe_headers, body=body,
                    timeout=self.timeout, max_bytes=self.max_bytes,
                    destination=destination,
                )
                response_destination = network_destination(
                    response.url, resolve_dns=False,
                )
                if response_destination.request_url != destination.request_url:
                    raise PermissionError(
                        "HTTP transport returned a different destination than it received"
                    )
                if response.status == 304 and cached:
                    response = cached
                    break
                if response.status not in {301, 302, 303, 307, 308}:
                    break
                location = response.headers.get("Location")
                if not location:
                    break
                if method not in {"GET", "HEAD"}:
                    raise PermissionError(
                        "mutating HTTP redirects require separate authorization"
                    )
                next_url = urllib.request.urljoin(current, location)
                destination = self._validate(next_url)
                if destination.origin != approved_origin:
                    raise PermissionError("cross-origin redirects require a separate request")
                current = destination.request_url
                redirects.append(destination.policy_url)
            else:
                raise RuntimeError("HTTP redirect limit exceeded")
            content_type = response.headers.get("Content-Type", "application/octet-stream")
            textual = content_type.startswith("text/") or any(
                marker in content_type for marker in ("json", "xml", "javascript")
            )
            content = response.body.decode("utf-8", errors="replace") if textual else ""
            for sensitive in sensitive_values:
                if sensitive:
                    content = content.replace(str(sensitive), "[REDACTED]")
            response_headers = redact(response.headers)
            data = {"url": response.url, "status": response.status,
                    "headers": response_headers, "content_type": content_type,
                    "content": content, "bytes": len(response.body),
                    "truncated": response.truncated, "redirects": redirects,
                    "peer_ip": response.peer_ip}
            state = "succeeded" if 200 <= response.status < 400 else "failed"
            if output_path and state == "succeeded":
                if response.truncated:
                    raise RuntimeError("download exceeded maximum response size")
                anchor, parts, target = select_anchor(output_anchors, output_path)
                anchor.atomic_write(parts, response.body)
                data.update({"output": output_path,
                             "sha256": hashlib.sha256(response.body).hexdigest(),
                             "verified_bytes": anchor.stat(parts).st_size})
            if method in {"GET", "HEAD"} and state == "succeeded":
                self.cache[destination.policy_url] = response
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        finally:
            for anchor in output_anchors:
                anchor.close()
        self.runtime.finish(actions, state=state, result=data)
        evidence_payload = content.encode() if textual else response.body
        evidence = self.runtime.evidence(
            "http", response.url, evidence_payload, task_id=task_id,
            event_uuid=actions[0].event_uuid,
            metadata={"status": response.status, "content_type": content_type,
                      "truncated": response.truncated,
                      "relevant_chunk": content[:4096] if textual else ""},
        )
        return ToolResult("http.request", state, data,
                          error="" if state == "succeeded" else f"HTTP {response.status}",
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def get(self, url, **kwargs): return self.request("GET", url, **kwargs)
    def head(self, url, **kwargs): return self.request("HEAD", url, **kwargs)
    def download(self, url, output, **kwargs):
        return self.request("GET", url, output=output, **kwargs)
