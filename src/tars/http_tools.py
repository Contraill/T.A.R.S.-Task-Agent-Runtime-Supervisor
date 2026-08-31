from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import urllib.error
import urllib.request

from .policy import ScopeRequest, canonical_path, normalize_network_target, redact
from .tool_core import ToolResult, ToolRuntime


@dataclass(frozen=True)
class HTTPResponse:
    url: str
    status: int
    headers: dict
    body: bytes
    truncated: bool


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class UrllibHTTPTransport:
    def __init__(self):
        self.opener = urllib.request.build_opener(_NoRedirect)

    def __call__(self, method, url, *, headers, body, timeout, max_bytes):
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            response = self.opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            payload = b"" if method == "HEAD" else response.read(max_bytes + 1)
            return HTTPResponse(
                response.geturl(), response.status, dict(response.headers.items()),
                payload[:max_bytes], len(payload) > max_bytes,
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
        return normalize_network_target(url, resolve_dns=True)

    def request(self, method, url, *, headers=None, body=None, approval_ids=None,
                task_id=None, session_id=None, use_cache=True, output=None,
                allowed_paths=(), sensitive_values=()):
        method = method.upper()
        if method not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError(f"unsupported HTTP method: {method}")
        current, approved_host = self._validate(url)
        safe_headers = dict(headers or {})
        cached = self.cache.get(current) if use_cache and method in {"GET", "HEAD"} else None
        if cached:
            if cached.headers.get("ETag"):
                safe_headers.setdefault("If-None-Match", cached.headers["ETag"])
            if cached.headers.get("Last-Modified"):
                safe_headers.setdefault("If-Modified-Since", cached.headers["Last-Modified"])
        requests = [("network", ScopeRequest(
            "http.request", "network", current,
            {"method": method, "headers": safe_headers, "body": body or b""},
            task_id=task_id, session_id=session_id, allowed_hosts=(approved_host,),
        ))]
        if method not in {"GET", "HEAD"}:
            requests.append(("write", ScopeRequest(
                "http.request", "write", current,
                {"method": method, "headers": safe_headers, "body": body or b""},
                task_id=task_id, session_id=session_id,
            )))
        output_path = None
        if output is not None:
            if method != "GET":
                raise ValueError("HTTP output files are supported only for GET")
            output_path = canonical_path(output)
            requests.append(("output", ScopeRequest(
                "http.download", "write", output_path, {"url": current},
                task_id=task_id, session_id=session_id,
                allowed_paths=tuple(allowed_paths),
            )))
        actions = self.runtime.authorize(tuple(requests), approval_ids)
        try:
            redirects = []
            for _ in range(self.max_redirects + 1):
                response = self.transport(
                    method, current, headers=safe_headers, body=body,
                    timeout=self.timeout, max_bytes=self.max_bytes,
                )
                if response.status == 304 and cached:
                    response = cached
                    break
                if response.status not in {301, 302, 303, 307, 308}:
                    break
                location = response.headers.get("Location")
                if not location:
                    break
                next_url = urllib.request.urljoin(current, location)
                current, redirect_host = self._validate(next_url)
                if redirect_host != approved_host:
                    raise PermissionError("cross-host redirects require a separate request")
                redirects.append(current)
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
                    "truncated": response.truncated, "redirects": redirects}
            state = "succeeded" if 200 <= response.status < 400 else "failed"
            if output_path and state == "succeeded":
                if response.truncated:
                    raise RuntimeError("download exceeded maximum response size")
                target = Path(output_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(response.body)
                data.update({"output": output_path,
                             "sha256": hashlib.sha256(response.body).hexdigest(),
                             "verified_bytes": target.stat().st_size})
            if method in {"GET", "HEAD"} and state == "succeeded":
                self.cache[current] = response
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
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
