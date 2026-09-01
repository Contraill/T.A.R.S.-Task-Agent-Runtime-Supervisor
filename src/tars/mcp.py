from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import subprocess
import threading
import select
import re
from urllib.request import Request
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid

from . import __version__
from .policy import ScopeRequest, canonical_path, normalize_network_target, redact
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


@dataclass(frozen=True)
class MCPPreparedCall:
    remote_name: str
    argument_json: str
    requests: tuple[tuple[str, ScopeRequest], ...]
    descriptor: dict


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
    effects = _normalize_effect_policy(name, effect_policy or {})
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


def _canonical_json(value, *, label):
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
            allow_nan=False,
        )
        snapshot = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be finite JSON data") from exc
    return snapshot, encoded


def _sha256_json(value):
    _, encoded = _canonical_json(value, label="MCP authority metadata")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_pointer(value, pointer):
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("MCP authority target must be an RFC 6901 JSON pointer")
    current = value
    for raw in pointer[1:].split("/"):
        if re.search(r"~(?![01])", raw):
            raise ValueError("MCP authority target contains an invalid JSON pointer escape")
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise ValueError(f"MCP authority target is missing argument {pointer}")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or int(token) >= len(current):
                raise ValueError(f"MCP authority target is missing argument {pointer}")
            current = current[int(token)]
        else:
            raise ValueError(f"MCP authority target cannot traverse argument {pointer}")
    return current


_SCHEMA_ANNOTATIONS = {
    "$id", "$schema", "title", "description", "default", "examples", "deprecated",
    "readOnly", "writeOnly", "format",
}
_SCHEMA_KEYWORDS = {
    "type", "enum", "const", "properties", "required", "additionalProperties",
    "minProperties", "maxProperties", "items", "minItems", "maxItems", "uniqueItems",
    "minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "multipleOf", "allOf", "anyOf", "oneOf", "not",
}


def _json_equal(left, right):
    return type(left) is type(right) and left == right


def _schema_type_matches(value, expected):
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, False)


def _validate_schema_definition(schema, *, path="$", depth=0):
    if depth > 64:
        raise ValueError("MCP input schema nesting exceeds the validation limit")
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise ValueError(f"invalid MCP input schema at {path}")
    unsupported = [
        key for key in schema
        if key not in _SCHEMA_ANNOTATIONS and key not in _SCHEMA_KEYWORDS
        and not str(key).startswith("x-")
    ]
    if unsupported:
        raise ValueError("unsupported MCP input schema keyword: " + str(unsupported[0]))
    expected = schema.get("type")
    if expected is not None:
        expected_types = [expected] if isinstance(expected, str) else expected
        if (not isinstance(expected_types, list) or not expected_types
                or any(item not in {"object", "array", "string", "integer", "number",
                                    "boolean", "null"} for item in expected_types)):
            raise ValueError(f"invalid MCP input schema type at {path}")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise ValueError(f"invalid MCP enum at {path}")
    for keyword in ("allOf", "anyOf", "oneOf"):
        if keyword not in schema:
            continue
        alternatives = schema[keyword]
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError(f"invalid MCP input schema {keyword} at {path}")
        for index, alternative in enumerate(alternatives):
            _validate_schema_definition(
                alternative, path=f"{path}.{keyword}[{index}]", depth=depth + 1)
    if "not" in schema:
        _validate_schema_definition(schema["not"], path=f"{path}.not", depth=depth + 1)
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    additional = schema.get("additionalProperties", True)
    if not isinstance(properties, dict) or not isinstance(required, list) or any(
            not isinstance(item, str) for item in required):
        raise ValueError(f"invalid MCP object schema at {path}")
    if not isinstance(additional, (bool, dict)):
        raise ValueError(f"invalid MCP additionalProperties at {path}")
    for key, child in properties.items():
        _validate_schema_definition(child, path=f"{path}.properties.{key}", depth=depth + 1)
    if isinstance(additional, dict):
        _validate_schema_definition(
            additional, path=f"{path}.additionalProperties", depth=depth + 1)
    items = schema.get("items", True)
    _validate_schema_definition(items, path=f"{path}.items", depth=depth + 1)
    for keyword in ("minProperties", "maxProperties", "minItems", "maxItems",
                    "minLength", "maxLength"):
        if keyword in schema and (not isinstance(schema[keyword], int)
                                  or isinstance(schema[keyword], bool)
                                  or schema[keyword] < 0):
            raise ValueError(f"invalid MCP {keyword} at {path}")
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise ValueError(f"invalid MCP uniqueItems at {path}")
    if "pattern" in schema:
        if not isinstance(schema["pattern"], str):
            raise ValueError(f"invalid MCP input schema pattern at {path}")
        try:
            re.compile(schema["pattern"])
        except re.error as exc:
            raise ValueError(f"invalid MCP input schema pattern at {path}") from exc
    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        value = schema.get(keyword)
        if keyword in schema and (not isinstance(value, (int, float))
                                  or isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError(f"invalid MCP {keyword} at {path}")
    if "multipleOf" in schema:
        divisor = schema["multipleOf"]
        if (not isinstance(divisor, (int, float)) or isinstance(divisor, bool)
                or not math.isfinite(divisor) or divisor <= 0):
            raise ValueError(f"invalid MCP multipleOf at {path}")


def _validate_arguments(value, schema, *, path="$", depth=0):
    if depth > 64:
        raise ValueError("MCP input schema nesting exceeds the validation limit")
    if isinstance(schema, bool):
        if not schema:
            raise ValueError(f"MCP tool arguments are forbidden at {path}")
        return
    if not isinstance(schema, dict):
        raise ValueError(f"invalid MCP input schema at {path}")
    unsupported = [
        key for key in schema
        if key not in _SCHEMA_ANNOTATIONS and key not in _SCHEMA_KEYWORDS
        and not str(key).startswith("x-")
    ]
    if unsupported:
        raise ValueError("unsupported MCP input schema keyword: " + str(unsupported[0]))
    expected = schema.get("type")
    if expected is not None:
        expected_types = [expected] if isinstance(expected, str) else expected
        if (not isinstance(expected_types, list) or not expected_types
                or any(item not in {"object", "array", "string", "integer", "number",
                                    "boolean", "null"} for item in expected_types)):
            raise ValueError(f"invalid MCP input schema type at {path}")
        if not any(_schema_type_matches(value, item) for item in expected_types):
            raise ValueError(f"MCP tool argument {path} has the wrong type")
    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, list) or not any(_json_equal(value, item) for item in choices):
            raise ValueError(f"MCP tool argument {path} is outside its enum")
    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ValueError(f"MCP tool argument {path} does not match its const value")
    for keyword in ("allOf", "anyOf", "oneOf"):
        if keyword not in schema:
            continue
        alternatives = schema[keyword]
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError(f"invalid MCP input schema {keyword} at {path}")
        matches = 0
        for alternative in alternatives:
            try:
                _validate_arguments(value, alternative, path=path, depth=depth + 1)
            except ValueError:
                continue
            matches += 1
        if keyword == "allOf" and matches != len(alternatives):
            raise ValueError(f"MCP tool argument {path} does not satisfy allOf")
        if keyword == "anyOf" and not matches:
            raise ValueError(f"MCP tool argument {path} does not satisfy anyOf")
        if keyword == "oneOf" and matches != 1:
            raise ValueError(f"MCP tool argument {path} does not satisfy oneOf")
    if "not" in schema:
        try:
            _validate_arguments(value, schema["not"], path=path, depth=depth + 1)
        except ValueError:
            pass
        else:
            raise ValueError(f"MCP tool argument {path} matches a forbidden schema")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional = schema.get("additionalProperties", True)
        if not isinstance(properties, dict) or not isinstance(required, list) or any(
                not isinstance(item, str) for item in required):
            raise ValueError(f"invalid MCP object schema at {path}")
        if not isinstance(additional, (bool, dict)):
            raise ValueError(f"invalid MCP additionalProperties at {path}")
        missing = [item for item in required if item not in value]
        if missing:
            raise ValueError(f"MCP tool argument {path} is missing {missing[0]}")
        if len(value) < int(schema.get("minProperties", 0)):
            raise ValueError(f"MCP tool argument {path} has too few properties")
        maximum = schema.get("maxProperties")
        if maximum is not None and len(value) > int(maximum):
            raise ValueError(f"MCP tool argument {path} has too many properties")
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in properties:
                _validate_arguments(item, properties[key], path=child, depth=depth + 1)
            elif additional is False:
                raise ValueError(f"MCP tool argument {child} is not permitted")
            elif isinstance(additional, dict):
                _validate_arguments(item, additional, path=child, depth=depth + 1)
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise ValueError(f"MCP tool argument {path} has too few items")
        maximum = schema.get("maxItems")
        if maximum is not None and len(value) > int(maximum):
            raise ValueError(f"MCP tool argument {path} has too many items")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value]
            if len(set(encoded)) != len(encoded):
                raise ValueError(f"MCP tool argument {path} has duplicate items")
        items = schema.get("items", True)
        for index, item in enumerate(value):
            _validate_arguments(item, items, path=f"{path}[{index}]", depth=depth + 1)
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise ValueError(f"MCP tool argument {path} is too short")
        maximum = schema.get("maxLength")
        if maximum is not None and len(value) > int(maximum):
            raise ValueError(f"MCP tool argument {path} is too long")
        if "pattern" in schema:
            try:
                matches = re.search(str(schema["pattern"]), value)
            except re.error as exc:
                raise ValueError(f"invalid MCP input schema pattern at {path}") from exc
            if not matches:
                raise ValueError(f"MCP tool argument {path} does not match its pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError(f"MCP tool argument {path} must be finite")
        comparisons = (
            ("minimum", lambda item: value >= item),
            ("maximum", lambda item: value <= item),
            ("exclusiveMinimum", lambda item: value > item),
            ("exclusiveMaximum", lambda item: value < item),
        )
        for keyword, predicate in comparisons:
            if keyword in schema and not predicate(schema[keyword]):
                raise ValueError(f"MCP tool argument {path} violates {keyword}")
        if "multipleOf" in schema:
            divisor = schema["multipleOf"]
            if not isinstance(divisor, (int, float)) or divisor <= 0:
                raise ValueError(f"invalid MCP multipleOf at {path}")
            quotient = value / divisor
            if not math.isclose(quotient, round(quotient), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"MCP tool argument {path} violates multipleOf")


def _normalize_scope(server, tool, raw, index):
    if not isinstance(raw, dict):
        raise ValueError(f"invalid MCP authority scope for {tool}")
    allowed_keys = {
        "name", "effect", "target", "target_kind", "allowed_paths", "allowed_hosts",
        "destructive", "elevated", "sandbox_escape",
    }
    if set(raw) - allowed_keys:
        raise ValueError(f"unknown MCP authority scope field for {tool}")
    effect = raw.get("effect")
    if effect not in VALID_EFFECTS:
        raise ValueError(f"invalid MCP effect policy for {tool}")
    name = str(raw.get("name", f"scope-{index}"))
    if not NAME_RE.fullmatch(name):
        raise ValueError(f"invalid MCP authority scope name for {tool}")
    pointer = raw.get("target")
    kind = raw.get("target_kind", "value" if pointer is not None else "opaque")
    if kind not in {"opaque", "value", "path", "network"}:
        raise ValueError(f"invalid MCP authority target kind for {tool}")
    if pointer is not None and (not isinstance(pointer, str) or not pointer.startswith("/")):
        raise ValueError(f"invalid MCP authority target pointer for {tool}")
    if kind == "opaque" and pointer is not None:
        raise ValueError(f"opaque MCP authority scopes cannot extract a target for {tool}")
    if kind != "opaque" and pointer is None:
        raise ValueError(f"MCP authority scope requires a target pointer for {tool}")
    paths = raw.get("allowed_paths", [])
    hosts = raw.get("allowed_hosts", [])
    if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
        raise ValueError(f"invalid MCP allowed path scope for {tool}")
    if not isinstance(hosts, list) or not all(isinstance(item, str) for item in hosts):
        raise ValueError(f"invalid MCP allowed host scope for {tool}")
    if kind == "path":
        if not paths or hosts:
            raise ValueError(f"path MCP authority requires allowed_paths only for {tool}")
        paths = [canonical_path(item) for item in paths]
    elif paths:
        raise ValueError(f"allowed_paths requires a path MCP target for {tool}")
    if kind == "network":
        if effect not in {"network", "remote"} or not hosts:
            raise ValueError(f"network MCP authority requires network effect and allowed_hosts for {tool}")
        hosts = [_network_origin(item) for item in hosts]
    elif hosts:
        raise ValueError(f"allowed_hosts requires a network MCP target for {tool}")
    if effect in {"network", "remote"} and kind != "network":
        raise ValueError(f"network MCP effects require a network target for {tool}")
    flags = {}
    for flag in ("destructive", "elevated", "sandbox_escape"):
        value = raw.get(flag, effect == flag)
        if not isinstance(value, bool):
            raise ValueError(f"invalid MCP authority flag for {tool}")
        flags[flag] = value
    return {
        "name": name, "effect": effect, "target": pointer, "target_kind": kind,
        "allowed_paths": sorted(set(paths)), "allowed_hosts": sorted(set(hosts)),
        **flags,
    }


def _normalize_contract(server, tool, raw):
    if isinstance(raw, str):
        raw = {"scopes": [{"effect": raw}]}
    if not isinstance(raw, dict) or set(raw) != {"scopes"}:
        raise ValueError(f"invalid MCP authority contract for {tool}")
    scopes = raw["scopes"]
    if not isinstance(scopes, list) or not scopes:
        raise ValueError(f"MCP authority contract requires scopes for {tool}")
    normalized = [_normalize_scope(server, tool, item, index)
                  for index, item in enumerate(scopes)]
    names = [item["name"] for item in normalized]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate MCP authority scope name for {tool}")
    return {"scopes": normalized}


def _normalize_effect_policy(server, policy):
    if not isinstance(policy, dict):
        raise ValueError("MCP effect policy must be an object")
    result = {}
    for tool, raw in policy.items():
        name = str(tool).strip()
        if not NAME_RE.fullmatch(name):
            raise ValueError("invalid MCP tool name in effect policy")
        result[name] = _normalize_contract(server, name, raw)
    return result


def _server_identity(server):
    return _sha256_json({
        "name": server.name, "transport": server.transport, "config": server.config,
        "tool_filter": server.tool_filter, "effect_policy": server.effect_policy,
        "created_at": server.created_at, "updated_at": server.updated_at,
    })


def _snapshot_server(server):
    config, _ = _canonical_json(server.config, label="MCP server config")
    filters, _ = _canonical_json(server.tool_filter, label="MCP tool filter")
    effects, _ = _canonical_json(server.effect_policy, label="MCP effect policy")
    if not isinstance(config, dict) or not isinstance(filters, dict) or not isinstance(effects, dict):
        raise ValueError("MCP server configuration must contain JSON objects")
    return MCPServerRecord(
        str(server.name), str(server.transport), config, bool(server.enabled),
        filters, effects, str(server.created_at), str(server.updated_at),
    )


def _network_origin(value):
    normalized, _ = normalize_network_target(value)
    parsed = urlsplit(normalized)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.scheme}://{parsed.hostname}:{port}"


def _target_values(arguments, scope, *, server, tool):
    if scope["target_kind"] == "opaque":
        return [f"mcp://{server}/{tool}"]
    value = _json_pointer(arguments, scope["target"])
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError(f"MCP authority target {scope['target']} is empty")
    result = []
    for item in values:
        if not isinstance(item, (str, int, float)) or isinstance(item, bool):
            raise ValueError(f"MCP authority target {scope['target']} must be scalar or an array of scalars")
        target = str(item)
        if not target:
            raise ValueError(f"MCP authority target {scope['target']} is empty")
        if scope["target_kind"] == "network":
            if _network_origin(target) not in scope["allowed_hosts"]:
                raise PermissionError(
                    "MCP network target is outside the configured origin scope")
        result.append(target)
    return result


def _scope_requests(server, descriptor, arguments, *, task_id=None, session_id=None,
                    server_identity=None):
    contract = descriptor["authority"]
    authority_sha = descriptor["authority_sha256"]
    base_arguments = {
        "remote_name": descriptor["remote_name"],
        "remote_arguments": arguments,
        "server_identity_sha256": server_identity or _sha256_json({"name": server}),
        "tool_contract_sha256": authority_sha,
    }
    result = []
    for scope in contract["scopes"]:
        targets = _target_values(arguments, scope, server=server,
                                 tool=descriptor["remote_name"])
        for target_index, target in enumerate(targets):
            key = scope["name"] if len(targets) == 1 else f"{scope['name']}:{target_index}"
            scope_arguments = base_arguments | {
                "authority_scope": scope["name"], "authority_target_index": target_index,
            }
            result.append((key, ScopeRequest(
                descriptor["name"], scope["effect"], target, scope_arguments,
                task_id=task_id, session_id=session_id,
                allowed_paths=tuple(scope["allowed_paths"]),
                allowed_hosts=tuple(scope["allowed_hosts"]),
                destructive=scope["destructive"], elevated=scope["elevated"],
                sandbox_escape=scope["sandbox_escape"],
            )))
    return tuple(result)


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
    _validate_schema_definition(schema)
    contract = _normalize_contract(
        server, name,
        effect_policy.get(name, {"scopes": [{"effect": "elevated"}]}),
    )
    effects = tuple(dict.fromkeys(scope["effect"] for scope in contract["scopes"]))
    authority_sha = _sha256_json({
        "server": server, "remote_name": name, "inputSchema": schema,
        "authority": contract,
    })
    return {"name": f"mcp.{server}.{name}", "server": server, "remote_name": name,
            "description": str(tool.get("description", "")), "inputSchema": schema,
            "effect": effects[0] if len(effects) == 1 else "compound",
            "effects": effects, "authority": contract,
            "authority_sha256": authority_sha,
            "trusted": False, "capability": f"mcp.{server}.{name}"}


def _included(name, filters):
    include = set(filters.get("include", ()))
    exclude = set(filters.get("exclude", ()))
    return name not in exclude and (not include or name in include)


class MCPClient:
    def __init__(self, server, *, transport=None, runtime=None,
                 connection_approval_id=None, secret_store=None):
        loaded = load_server(server) if isinstance(server, str) else server
        self.server = _snapshot_server(loaded)
        if not self.server.enabled:
            raise RuntimeError(f"MCP server is disabled: {self.server.name}")
        self.runtime = runtime or ToolRuntime()
        self.state = "connecting"
        self._server_identity = _server_identity(self.server)
        if transport is None:
            effect = "execute" if self.server.transport == "stdio" else "network"
            target = (self.server.config["argv"][0] if effect == "execute"
                      else self.server.config["url"])
            request = ScopeRequest(f"mcp.{self.server.name}.connect", effect, target,
                                   {"transport": self.server.transport,
                                    "config": self.server.config,
                                    "server_identity_sha256": self._server_identity})
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
        return [{key: tool[key] for key in ("name", "description", "effect", "effects", "trusted")}
                for tool in self.discover_tools()]

    def tool_schema(self, name):
        tools = {tool["remote_name"]: tool for tool in self.discover_tools()}
        if name not in tools:
            raise KeyError(f"MCP tool is unavailable or filtered: {name}")
        return tools[name]

    def prepare_call(self, name, arguments, *, task_id=None, session_id=None):
        tools = {item["remote_name"]: item for item in self.discover_tools()}
        if name not in tools:
            raise KeyError(f"MCP tool is unavailable or filtered: {name}")
        descriptor = tools[name]
        if not isinstance(arguments, dict):
            raise ValueError("MCP tool arguments must be an object")
        snapshot, argument_json = _canonical_json(arguments, label="MCP tool arguments")
        _validate_arguments(snapshot, descriptor["inputSchema"])
        requests = _scope_requests(
            self.server.name, descriptor, snapshot, task_id=task_id,
            session_id=session_id, server_identity=self._server_identity,
        )
        return MCPPreparedCall(name, argument_json, requests, descriptor)

    def call_tool(self, name, arguments, *, approval_id=None, approval_ids=None,
                  task_id=None, session_id=None):
        prepared = self.prepare_call(name, arguments, task_id=task_id,
                                     session_id=session_id)
        if approval_id is not None and approval_ids is not None:
            raise ValueError("use either approval_id or approval_ids, not both")
        if approval_id is not None:
            if len(prepared.requests) != 1:
                raise ValueError("compound MCP calls require one approval per authority scope")
            approvals = {prepared.requests[0][0]: approval_id}
        else:
            approvals = dict(approval_ids or {})
            unknown = set(approvals) - {key for key, _ in prepared.requests}
            if unknown:
                raise ValueError("approval mapping contains an unknown MCP authority scope")
        actions = self.runtime.authorize(prepared.requests, approvals)
        try:
            actual_arguments = json.loads(prepared.argument_json)
            result = redact(self._request(
                "tools/call", {"name": prepared.remote_name,
                               "arguments": actual_arguments},
            ))
            failed = bool(result.get("isError"))
            state = "failed" if failed else "succeeded"
            self.runtime.finish(actions, state=state, result=result)
            evidence = self.runtime.evidence("mcp_result", prepared.descriptor["name"], repr(result),
                                             task_id=task_id,
                                             event_uuid=actions[0].event_uuid,
                                             result_ref=self.server.name)
            return ToolResult(prepared.descriptor["name"], state, result,
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
        if effect not in VALID_EFFECTS:
            raise ValueError("invalid MCP server tool effect")
        if target_arg is None and (allowed_paths or allowed_hosts):
            raise ValueError("controlled MCP path/host scopes require a target argument")
        if allowed_paths and allowed_hosts:
            raise ValueError("controlled MCP target cannot be both path and network")
        if allowed_hosts and effect not in {"network", "remote"}:
            raise ValueError("controlled MCP host scope requires a network effect")
        scope = {"effect": effect}
        if target_arg is not None:
            token = str(target_arg).replace("~", "~0").replace("/", "~1")
            scope["target"] = f"/{token}"
            if allowed_paths:
                scope["target_kind"] = "path"
                scope["allowed_paths"] = list(allowed_paths)
            elif effect in {"network", "remote"}:
                scope["target_kind"] = "network"
                scope["allowed_hosts"] = list(allowed_hosts)
        contract = {name: {"scopes": [scope]}}
        normalized = normalize_tool(
            "server", {"name": name, "description": description,
                       "inputSchema": input_schema}, contract,
        )
        self.tools[name] = (normalized, handler)
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
                descriptor, handler = self.tools[params["name"]]
                arguments = params.get("arguments", {})
                if not isinstance(arguments, dict):
                    raise ValueError("MCP tool arguments must be an object")
                snapshot, argument_json = _canonical_json(
                    arguments, label="MCP tool arguments")
                _validate_arguments(snapshot, descriptor["inputSchema"])
                scopes = _scope_requests("server", descriptor, snapshot)
                actions = self.runtime.authorize(scopes, {})
                try:
                    value = handler(json.loads(argument_json))
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
