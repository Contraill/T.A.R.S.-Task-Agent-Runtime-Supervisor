from __future__ import annotations

from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
import socket
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import uuid

from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction

EFFECTS = {"read", "write", "execute", "network", "service", "remote", "secret",
           "elevated", "destructive", "sandbox_escape"}
RISK_CLASSES = {"safe/read", "write", "execute", "network", "elevated",
                "destructive/high-impact"}
POLICY_ACTIONS = {"allow", "deny", "ask"}

DEFAULT_ACTIONS = {
    "read": "allow",
    "write": "ask",
    "execute": "ask",
    "network": "ask",
    "service": "ask",
    "remote": "ask",
    "secret": "ask",
    "elevated": "deny",
    "destructive": "ask",
    "sandbox_escape": "deny",
}

RISK_BY_EFFECT = {
    "read": "safe/read", "write": "write", "execute": "execute",
    "network": "network", "service": "elevated", "remote": "network",
    "secret": "elevated", "elevated": "elevated",
    "destructive": "destructive/high-impact", "sandbox_escape": "elevated",
}

SENSITIVE_KEYS = {
    "authorization", "cookie", "cookies", "password", "passwd", "secret", "token",
    "api_key", "apikey", "access_key", "private_key", "credential", "credentials",
}


@dataclass(frozen=True)
class ScopeRequest:
    tool: str
    effect: str
    target: str = ""
    arguments: dict = field(default_factory=dict)
    task_id: str | None = None
    session_id: str | None = None
    allowed_paths: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ()
    destructive: bool = False
    elevated: bool = False
    sandbox_escape: bool = False


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    risk_class: str
    effect: str
    target: str
    reason: str
    rule_id: str | None = None
    normalized_arguments: dict = field(default_factory=dict)

    @property
    def allowed(self):
        return self.action == "allow"


def _sensitive_key(value):
    normalized = str(value).casefold().replace("-", "_")
    return normalized in SENSITIVE_KEYS or any(
        marker in normalized
        for marker in ("authorization", "credential", "password", "private_key", "secret", "token", "api_key")
    )


def redact(value):
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _sensitive_key(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def canonical_path(value: str) -> str:
    if not value:
        raise ValueError("filesystem target is required")
    return str(Path(value).expanduser().resolve(strict=False))


def _within(path: str, roots: tuple[str, ...]) -> bool:
    candidate = Path(path)
    for raw in roots:
        root = Path(canonical_path(raw))
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _path_request(request: ScopeRequest) -> bool:
    return bool(request.allowed_paths) or request.tool.startswith("fs.")


def _unsafe_ip(value: str) -> bool:
    address = ip_address(value)
    return not address.is_global


def normalize_network_target(value: str, *, resolve_dns=False) -> tuple[str, str]:
    parsed = urlsplit(value if "://" in value else "https://" + value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("network target must be an HTTP(S) URL or host")
    if parsed.username or parsed.password:
        raise ValueError("credentials in network targets are forbidden")
    host = parsed.hostname.rstrip(".").casefold()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("loopback network targets are denied")
    try:
        if _unsafe_ip(host):
            raise ValueError("private, loopback, link-local and reserved targets are denied")
    except ValueError as exc:
        if "denied" in str(exc):
            raise
    try:
        legacy = socket.inet_aton(host)
    except OSError:
        legacy = None
    if legacy is not None and _unsafe_ip(socket.inet_ntoa(legacy)):
        raise ValueError("non-canonical private or loopback targets are denied")
    if resolve_dns:
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443)}
        except socket.gaierror as exc:
            raise ValueError(f"network target cannot be resolved: {host}") from exc
        if any(_unsafe_ip(item) for item in addresses):
            raise ValueError("network target resolves to a non-public address")
    port = parsed.port
    authority = f"{host}:{port}" if port else host
    safe_query = urlencode([
        (key, "[REDACTED]" if _sensitive_key(key) else value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    ])
    normalized = urlunsplit((parsed.scheme, authority, parsed.path or "/", safe_query, ""))
    return normalized, host


def add_rule(effect: str, action: str, *, target="", scope="persistent", expires_at=None,
             metadata=None, target_kind=None):
    if effect not in EFFECTS or action not in POLICY_ACTIONS:
        raise ValueError("invalid policy effect or action")
    rule_id = "rule-" + uuid.uuid4().hex
    rule_metadata = dict(metadata or {})
    if target_kind:
        rule_metadata["target_kind"] = target_kind
    normalized_target = target
    if rule_metadata.get("target_kind") == "path" and target:
        normalized_target = canonical_path(target)
    elif effect in {"network", "remote"} and target:
        parsed = urlsplit(target if "://" in target else "https://" + target)
        normalized_target = (parsed.hostname or target).rstrip(".").casefold()
    ensure_state_store()
    with transaction(immediate=True) as conn:
        conn.execute(
            """INSERT INTO policy_rules(id,effect,action,target,scope,created_at,expires_at,metadata_json)
               VALUES(?,?,?,?,?,?,?,?)""",
            (rule_id, effect, action, normalized_target, scope, now_utc(), expires_at,
             json_dumps(redact(rule_metadata))),
        )
    return rule_id


def list_rules():
    ensure_state_store()
    conn = connect()
    try:
        return [dict(row) | {"metadata": json_loads(row["metadata_json"], {})}
                for row in conn.execute(
                    "SELECT * FROM policy_rules ORDER BY created_at DESC"
                ).fetchall()]
    finally:
        conn.close()


class ScopeGuard:
    def evaluate(self, request: ScopeRequest) -> PolicyDecision:
        if request.effect not in EFFECTS:
            return PolicyDecision("deny", "elevated", request.effect, request.target,
                                  "unknown effects are denied", normalized_arguments=redact(request.arguments))
        effect = request.effect
        if request.sandbox_escape:
            effect = "sandbox_escape"
        elif request.elevated:
            effect = "elevated"
        elif request.destructive:
            effect = "destructive"
        target = request.target
        reason = "default policy"
        missing_path_scope = False
        if _path_request(request) and target:
            try:
                target = canonical_path(target)
            except (OSError, ValueError) as exc:
                return PolicyDecision("deny", RISK_BY_EFFECT[effect], effect, target,
                                      f"invalid filesystem target: {exc}",
                                      normalized_arguments=redact(request.arguments))
            if request.allowed_paths and not _within(target, request.allowed_paths):
                return PolicyDecision("deny", RISK_BY_EFFECT[effect], effect, target,
                                      "canonical target is outside authorized filesystem scope",
                                      normalized_arguments=redact(request.arguments))
            missing_path_scope = not request.allowed_paths
        if request.effect in {"network", "remote"}:
            try:
                target, host = normalize_network_target(target)
            except ValueError as exc:
                return PolicyDecision("deny", RISK_BY_EFFECT[effect], effect, target, str(exc),
                                      normalized_arguments=redact(request.arguments))
            if request.allowed_hosts and host not in {
                item.rstrip(".").casefold() for item in request.allowed_hosts
            }:
                return PolicyDecision("deny", RISK_BY_EFFECT[effect], effect, target,
                                      "destination is outside authorized network scope",
                                      normalized_arguments=redact(request.arguments))
        rule = self._matching_rule(effect, target)
        if missing_path_scope and not rule:
            return PolicyDecision("deny", RISK_BY_EFFECT[effect], effect, target,
                                  "filesystem tools require an authorized path scope",
                                  normalized_arguments=redact(request.arguments))
        action = rule["action"] if rule else DEFAULT_ACTIONS[effect]
        if rule:
            reason = f"matched {rule['scope']} policy rule"
        return PolicyDecision(action, RISK_BY_EFFECT[effect], effect, target, reason,
                              rule["id"] if rule else None, redact(request.arguments))

    @staticmethod
    def _matching_rule(effect, target):
        ensure_state_store()
        conn = connect()
        try:
            rows = conn.execute(
                """SELECT * FROM policy_rules WHERE effect=?
                   AND (expires_at IS NULL OR expires_at>?)
                   ORDER BY CASE WHEN target='' THEN 1 ELSE 0 END, created_at DESC""",
                (effect, now_utc()),
            ).fetchall()
            for row in rows:
                configured = row["target"]
                if not configured:
                    return row
                if row["metadata_json"] and json_loads(row["metadata_json"], {}).get("target_kind") == "path":
                    if _within(target, (configured,)):
                        return row
                elif effect in {"network", "remote"}:
                    try:
                        host = urlsplit(target).hostname or target
                    except ValueError:
                        host = target
                    if host == configured or host.endswith("." + configured):
                        return row
                elif target == configured:
                    return row
            return None
        finally:
            conn.close()
