from __future__ import annotations

import json
from ipaddress import ip_address

from .network import network_destination, open_bound


def local_runtime_destination(url):
    """Bind one runtime URL to a literal loopback peer without proxy or DNS use."""
    destination = network_destination(
        str(url), resolve_dns=False, allow_loopback=True)
    if not destination.addresses or not all(
            ip_address(address).is_loopback for address in destination.addresses):
        raise ValueError("local runtime request must target a literal loopback address")
    return destination


def request_json(method, url, *, payload=None, timeout=30):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    with open_bound(
        local_runtime_destination(url), method=method, headers=headers,
        body=body, timeout=timeout,
    ) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(
                f"local runtime returned HTTP status {response.status}")
        return json.loads(response.read().decode("utf-8"))


def request_sse(url, *, payload, timeout=1200):
    body = json.dumps(payload).encode("utf-8")
    with open_bound(
        local_runtime_destination(url), method="POST",
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"},
        body=body, timeout=timeout,
    ) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(
                f"local runtime returned HTTP status {response.status}")
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value
