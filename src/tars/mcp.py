from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import threading
import select
import re
from urllib.request import Request
import urllib.error
import urllib.request
import uuid

from . import __version__
from .policy import ScopeRequest, normalize_network_target, redact
from .state_store import (connect, ensure_state_store, json_dumps, json_loads,
                          now_utc, transaction)
from .tool_core import ToolResult, ToolRuntime
from .secret_store import SecretStore, parse_reference


PROTOCOL_VERSION = "2025-03-26"
VALID_EFFECTS = {"read", "write", "execute", "network", "service", "remote",
                 "secret", "elevated", "destructive", "sandbox_escape"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class MCPServerRecord:
    name: str
    transport: str
    config: dict
    enabled: bool
    tool_filter: dict
    effect_policy: dict
    created_at: str
    updated_at: str


def _record(row):
    return MCPServerRecord(row["name"], row["transport"],
                           json_loads(row["config_json"], {}), bool(row["enabled"]),
                           json_loads(row["tool_filter_json"], {}),
                           json_loads(row["effect_policy_json"], {}),
                           row["created_at"], row["updated_at"])


def register(name, transport, config, *, enabled=True, tool_filter=None,
             effect_policy=None):
    name = str(name).strip()
    if not NAME_RE.fullmatch(name) or transport not in {"stdio", "streamable-http"}:
        raise ValueError("invalid MCP server name or transport")
    config = dict(config or {})
    if transport == "stdio":
        if not isinstance(config.get("argv"), list) or not config["argv"]:
            raise ValueError("stdio MCP config requires a non-empty argv list")
        sensitive = re.compile(
            r"(?i)^--?(?:api[-_]?key|authorization|password|secret|token)(?:=|$)")
        if any(sensitive.search(str(value)) for value in config["argv"]):
            raise ValueError("MCP credentials must be injected through secret references, not argv")
        if config.get("env"):
            for value in config["env"].values():
                parse_reference(value)
    else:
        if not str(config.get("url", "")).startswith(("http://", "https://")):
            raise ValueError("streamable HTTP MCP config requires an HTTP(S) URL")
        config["url"] = normalize_network_target(config["url"], resolve_dns=True)[0]
        if config.get("authorization_ref"):
            parse_reference(config["authorization_ref"])
    effects = dict(effect_policy or {})
    if any(value not in VALID_EFFECTS for value in effects.values()):
        raise ValueError("invalid MCP effect policy")
    stamp = now_utc()
    ensure_state_store()
    with transaction(immediate=True) as conn:
        conn.execute(
            "INSERT INTO mcp_servers VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "transport=excluded.transport,config_json=excluded.config_json,enabled=excluded.enabled,"
            "tool_filter_json=excluded.tool_filter_json,effect_policy_json=excluded.effect_policy_json,"
            "updated_at=excluded.updated_at",
            (name, transport, json_dumps(config), int(enabled),
             json_dumps(tool_filter or {}), json_dumps(effects), stamp, stamp),
        )
    return load_server(name)


def load_server(name):
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM mcp_servers WHERE name=?", (name,)).fetchone()
        if not row:
            raise KeyError(f"unknown MCP server: {name}")
        return _record(row)
    finally:
        conn.close()


def list_servers():
    ensure_state_store()
    conn = connect()
    try:
        return [_record(row) for row in conn.execute(
            "SELECT * FROM mcp_servers ORDER BY name").fetchall()]
    finally:
        conn.close()


def set_enabled(name, enabled):
    with transaction(immediate=True) as conn:
        if not conn.execute("UPDATE mcp_servers SET enabled=?,updated_at=? WHERE name=?",
                            (int(enabled), now_utc(), name)).rowcount:
            raise KeyError(f"unknown MCP server: {name}")
    return load_server(name)


def _redact_secret_values(value, secrets):
    if isinstance(value, dict):
        return {key: _redact_secret_values(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_secret_values(item, secrets) for item in value]
    if isinstance(value, str):
        result = value
        for secret in secrets:
            if secret:
                result = result.replace(secret, "[REDACTED]")
        return result
    return value


class StdioTransport:
    def __init__(self, config, *, popen=subprocess.Popen, secret_store=None,
                 consumer="mcp:stdio"):
        store = secret_store or SecretStore()
        env = os.environ.copy()
        resolved = store.resolve_many(config.get("env", {}), consumer=consumer)
        env.update(resolved)
        self.secret_values = tuple(resolved.values())
        self.process = popen(config["argv"], cwd=config.get("cwd") or None, env=env,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, bufsize=1)
        self.lock = threading.Lock()
        self.timeout = float(config.get("timeout", 30))

    def request(self, payload):
        with self.lock:
            self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            ready, _, _ = select.select([self.process.stdout], [], [], self.timeout)
            if not ready:
                raise TimeoutError("MCP stdio request timed out")
            line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read(4096)
            raise RuntimeError("MCP stdio server closed" + (f": {error}" if error else ""))
        return json.loads(line)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


class StreamableHTTPTransport:
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    def __init__(self, config, *, opener=None, secret_store=None,
                 consumer="mcp:http"):
        self.url = config["url"]
        self.timeout = float(config.get("timeout", 30))
        self.authorization_ref = config.get("authorization_ref")
        self.secret_store = secret_store or SecretStore()
        self.consumer = consumer
        self.opener = opener or urllib.request.build_opener(self._NoRedirect).open
        self.session_id = None
        self.secret_values = ()

    def request(self, payload):
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.authorization_ref:
            with self.secret_store.resolve(
                    self.authorization_ref, consumer=self.consumer) as secret:
                self.secret_values = (secret,)
                headers["Authorization"] = "Bearer " + secret
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        current, approved_host = normalize_network_target(self.url, resolve_dns=True)
        for _ in range(6):
            request = Request(current, data=json.dumps(payload).encode(), headers=headers,
                              method="POST")
            try:
                response = self.opener(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                response = exc
            status = getattr(response, "status", None)
            status = response.getcode() if status is None else status
            if status not in {301, 302, 303, 307, 308}:
                break
            location = response.headers.get("Location")
            if not location:
                break
            next_url = urllib.request.urljoin(current, location)
            current, redirect_host = normalize_network_target(next_url, resolve_dns=True)
            if redirect_host != approved_host:
                raise PermissionError("cross-host MCP redirects require separate authorization")
        else:
            raise RuntimeError("MCP HTTP redirect limit exceeded")
        self.session_id = response.headers.get("Mcp-Session-Id", self.session_id)
        body = response.read(8_000_001)
        if len(body) > 8_000_000:
            raise RuntimeError("MCP HTTP response exceeded size limit")
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            events = [line[6:] for line in body.decode().splitlines() if line.startswith("data: ")]
            if not events:
                raise RuntimeError("MCP HTTP response contained no data event")
            return json.loads(events[-1])
        return json.loads(body)

    def close(self):
        return None


def normalize_tool(server, tool, effect_policy):
    name = str(tool.get("name", "")).strip()
    schema = tool.get("inputSchema", {"type": "object", "properties": {}})
    if not NAME_RE.fullmatch(name) or not isinstance(schema, dict) or schema.get("type", "object") != "object":
        raise ValueError("invalid MCP tool schema")
    effect = effect_policy.get(name, "elevated")
    return {"name": f"mcp.{server}.{name}", "server": server, "remote_name": name,
            "description": str(tool.get("description", "")), "inputSchema": schema,
            "effect": effect, "trusted": False, "capability": f"mcp.{server}.{name}"}


def _included(name, filters):
    include = set(filters.get("include", ()))
    exclude = set(filters.get("exclude", ()))
    return name not in exclude and (not include or name in include)


class MCPClient:
    def __init__(self, server, *, transport=None, runtime=None,
                 connection_approval_id=None, secret_store=None):
        self.server = load_server(server) if isinstance(server, str) else server
        if not self.server.enabled:
            raise RuntimeError(f"MCP server is disabled: {self.server.name}")
        self.runtime = runtime or ToolRuntime()
        self.state = "connecting"
        if transport is None:
            effect = "execute" if self.server.transport == "stdio" else "network"
            target = (self.server.config["argv"][0] if effect == "execute"
                      else self.server.config["url"])
            request = ScopeRequest(f"mcp.{self.server.name}.connect", effect, target,
                                   {"transport": self.server.transport})
            actions = self.runtime.authorize(
                (("connect", request),), {"connect": connection_approval_id})
            transport_cls = StdioTransport if self.server.transport == "stdio" else StreamableHTTPTransport
            try:
                transport = transport_cls(
                    self.server.config, secret_store=secret_store,
                    consumer=f"mcp:{self.server.name}")
            except Exception as exc:
                self.runtime.finish(actions, state="failed", result={"error": str(exc)})
                self.state = "error"
                raise
            self.runtime.finish(actions, state="succeeded",
                                result={"transport": self.server.transport})
        self.transport = transport
        self._next_id = 0
        self._initialized = False
        self.state = "connected"

    def _request(self, method, params=None):
        self._next_id += 1
        response = self.transport.request({"jsonrpc": "2.0", "id": self._next_id,
                                           "method": method, "params": params or {}})
        secrets = getattr(self.transport, "secret_values", ())
        response = _redact_secret_values(response, secrets)
        if response.get("error"):
            raise RuntimeError(str(response["error"].get("message", "MCP request failed")))
        return response.get("result", {})

    def initialize(self):
        result = self._request("initialize", {"protocolVersion": PROTOCOL_VERSION,
                               "capabilities": {}, "clientInfo": {"name": "tars", "version": __version__}})
        self._initialized = True
        return redact(result)

    def discover_tools(self):
        if not self._initialized:
            self.initialize()
        result = self._request("tools/list")
        tools = []
        names = set()
        for raw in result.get("tools", ()):
            if _included(str(raw.get("name", "")), self.server.tool_filter):
                tool = normalize_tool(self.server.name, raw, self.server.effect_policy)
                if tool["name"] in names:
                    raise ValueError(f"duplicate MCP tool name: {tool['name']}")
                names.add(tool["name"])
                tools.append(tool)
        return tools

    def tool_summaries(self):
        return [{key: tool[key] for key in ("name", "description", "effect", "trusted")}
                for tool in self.discover_tools()]

    def tool_schema(self, name):
        tools = {tool["remote_name"]: tool for tool in self.discover_tools()}
        if name not in tools:
            raise KeyError(f"MCP tool is unavailable or filtered: {name}")
        return tools[name]

    def call_tool(self, name, arguments, *, target="", approval_id=None, task_id=None,
                  session_id=None, allowed_paths=(), allowed_hosts=()):
        tools = {item["remote_name"]: item for item in self.discover_tools()}
        if name not in tools:
            raise KeyError(f"MCP tool is unavailable or filtered: {name}")
        descriptor = tools[name]
        request = ScopeRequest(
            descriptor["name"], descriptor["effect"], str(target), dict(arguments),
            task_id=task_id, session_id=session_id, allowed_paths=tuple(allowed_paths),
            allowed_hosts=tuple(allowed_hosts),
            destructive=descriptor["effect"] == "destructive",
            elevated=descriptor["effect"] == "elevated",
            sandbox_escape=descriptor["effect"] == "sandbox_escape",
        )
        actions = self.runtime.authorize((("action", request),), {"action": approval_id})
        try:
            result = redact(self._request("tools/call", {"name": name, "arguments": arguments}))
            failed = bool(result.get("isError"))
            state = "failed" if failed else "succeeded"
            self.runtime.finish(actions, state=state, result=result)
            evidence = self.runtime.evidence("mcp_result", descriptor["name"], repr(result),
                                             task_id=task_id,
                                             event_uuid=actions[0].event_uuid,
                                             result_ref=self.server.name)
            return ToolResult(descriptor["name"], state, result,
                              error="MCP tool returned an error" if failed else "",
                              action_ids=tuple(item.id for item in actions),
                              evidence_ids=(evidence.id,))
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise

    def close(self):
        self.transport.close()
        self.state = "closed"


class ControlledMCPServer:
    """Minimal MCP server adapter; every exposed handler owns its normal guard path."""
    def __init__(self, *, runtime=None):
        self.tools = {}
        self.runtime = runtime or ToolRuntime()

    def register(self, name, description, input_schema, handler, *, effect="read",
                 target_arg=None, allowed_paths=(), allowed_hosts=()):
        if not callable(handler):
            raise TypeError("MCP handler must be callable")
        normalized = normalize_tool("server", {"name": name, "description": description,
                                    "inputSchema": input_schema}, {name: "elevated"})
        if effect not in VALID_EFFECTS:
            raise ValueError("invalid MCP server tool effect")
        self.tools[name] = (normalized, handler, effect, target_arg,
                            tuple(allowed_paths), tuple(allowed_hosts))
        return self

    def handle(self, request):
        request_id = request.get("id")
        method = request.get("method")
        try:
            if method == "initialize":
                result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                          "serverInfo": {"name": "tars-controlled", "version": "0.7.6"}}
            elif method == "tools/list":
                result = {"tools": [{"name": name, "description": item[0]["description"],
                                      "inputSchema": item[0]["inputSchema"]}
                                     for name, item in sorted(self.tools.items())]}
            elif method == "tools/call":
                params = request.get("params", {})
                if params.get("name") not in self.tools:
                    raise KeyError("tool is not exposed")
                descriptor, handler, effect, target_arg, allowed_paths, allowed_hosts = self.tools[params["name"]]
                arguments = dict(params.get("arguments", {}))
                target = str(arguments.get(target_arg, "")) if target_arg else ""
                scope = ScopeRequest(f"mcp.server.{params['name']}", effect, target,
                                     arguments, allowed_paths=allowed_paths,
                                     allowed_hosts=allowed_hosts,
                                     destructive=effect == "destructive",
                                     elevated=effect == "elevated",
                                     sandbox_escape=effect == "sandbox_escape")
                actions = self.runtime.authorize((("action", scope),), {})
                try:
                    value = handler(arguments)
                except Exception as exc:
                    self.runtime.finish(actions, state="failed", result={"error": str(exc)})
                    raise
                if not isinstance(value, ToolResult):
                    self.runtime.finish(actions, state="failed",
                                        result={"error": "handler returned no ToolResult"})
                    raise TypeError("controlled MCP handlers must return a real ToolResult")
                self.runtime.finish(actions, state=value.state, result=value.data)
                result = {"content": [{"type": "text", "text": json.dumps(redact(value.data))}],
                          "isError": not value.succeeded}
            else:
                raise KeyError("method is not supported")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": request_id,
                    "error": {"code": -32000, "message": str(exc)}}

    def serve_stdio(self, input_stream, output_stream):
        for line in input_stream:
            response = self.handle(json.loads(line))
            output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
            output_stream.flush()
