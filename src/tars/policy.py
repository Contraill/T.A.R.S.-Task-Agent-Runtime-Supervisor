from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from urllib.parse import urlsplit
import uuid

from .network import network_destination, tcp_destination
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
    "sandbox_escape": "ask",
}

RISK_BY_EFFECT = {
    "read": "safe/read", "write": "write", "execute": "execute",
    "network": "network", "service": "elevated", "remote": "network",
    "secret": "elevated", "elevated": "elevated",
    "destructive": "destructive/high-impact", "sandbox_escape": "elevated",
}

SENSITIVE_KEYS = {
    "authorization", "cookie", "cookies", "password", "passwd", "secret", "token",
    "set_cookie", "api_key", "apikey", "access_key", "private_key", "credential",
    "credentials",
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
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    if isinstance(value, Path):
        return str(value)
    return value


def redact_arguments(arguments):
    safe = redact(arguments)
    argv = safe.get("argv") if isinstance(safe, dict) else None
    if not isinstance(argv, list):
        return safe
    flag = re.compile(
        r"^--?(?:api[-_]?key|authorization|credential|password|private[-_]?key|secret|token)$",
        re.IGNORECASE,
    )
    assignment = re.compile(
        r"^(--?(?:api[-_]?key|authorization|credential|password|private[-_]?key|secret|token)=).+$",
        re.IGNORECASE,
    )
    inline = re.compile(
        r"(?i)(--?(?:api[-_]?key|authorization|credential|password|private[-_]?key|secret|token)(?:=|\s+))([^\s'\"]+)"
    )
    result = []
    hide_next = False
    for value in argv:
        text = str(value)
        if hide_next:
            result.append("[REDACTED]")
            hide_next = False
        elif flag.match(text):
            result.append(text)
            hide_next = True
        elif assignment.match(text):
            result.append(assignment.sub(r"\1[REDACTED]", text))
        else:
            result.append(inline.sub(r"\1[REDACTED]", text))
    safe["argv"] = result
    return safe


def _intent_value(value, *, sensitive=False):
    if sensitive:
        encoded = json_dumps(value).encode("utf-8") if not isinstance(value, bytes) else value
        return {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded),
                "protected": True}
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _intent_value(item, sensitive=_sensitive_key(key))
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_intent_value(item) for item in value]
    if isinstance(value, str) and len(value.encode("utf-8")) > 1024:
        encoded = value.encode("utf-8")
        return {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_intent(request: ScopeRequest, decision: PolicyDecision) -> dict:
    paths = sorted({canonical_path(path) for path in request.allowed_paths})
    hosts = sorted({normalize_network_target(host)[1] for host in request.allowed_hosts})
    intent = {
        "tool": request.tool, "effect": decision.effect, "risk_class": decision.risk_class,
        "target": decision.target, "task_id": request.task_id, "session_id": request.session_id,
        "allowed_paths": paths, "allowed_hosts": hosts,
        "destructive": bool(request.destructive), "elevated": bool(request.elevated),
        "sandbox_escape": bool(request.sandbox_escape),
        "arguments": _intent_value(request.arguments),
    }
    encoded = json_dumps(intent).encode("utf-8")
    return {"version": 1, "sha256": hashlib.sha256(encoded).hexdigest(), "value": intent}


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


def normalize_network_target(value: str, *, resolve_dns=False) -> tuple[str, str]:
    destination = network_destination(value, resolve_dns=resolve_dns)
    return destination.policy_url, destination.host


def normalize_remote_target(value: str, *, resolve_dns=False) -> tuple[str, str]:
    parsed = urlsplit(str(value))
    if parsed.scheme == "ssh" and parsed.hostname:
        if (parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}):
            raise ValueError("SSH authority targets contain only host and port")
        destination = tcp_destination(
            parsed.hostname, parsed.port or 22, scheme="ssh",
            resolve_dns=resolve_dns,
        )
        return destination.policy_url, destination.host
    return normalize_network_target(value, resolve_dns=resolve_dns)


def add_rule_in_transaction(conn, effect: str, action: str, *, target="",
                            scope="persistent", expires_at=None, metadata=None,
                            target_kind=None):
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
    conn.execute(
        """INSERT INTO policy_rules(id,effect,action,target,scope,created_at,expires_at,metadata_json)
           VALUES(?,?,?,?,?,?,?,?)""",
        (rule_id, effect, action, normalized_target, scope, now_utc(), expires_at,
         json_dumps(redact(rule_metadata))),
    )
    return rule_id


def add_rule(effect: str, action: str, *, target="", scope="persistent", expires_at=None,
             metadata=None, target_kind=None):
    ensure_state_store()
    with transaction(immediate=True) as conn:
        return add_rule_in_transaction(
            conn, effect, action, target=target, scope=scope,
            expires_at=expires_at, metadata=metadata, target_kind=target_kind)


def list_rules():
    ensure_state_store()
    conn = connect()
    try:
        current = now_utc()
        result = []
        for row in conn.execute(
                "SELECT * FROM policy_rules ORDER BY created_at DESC").fetchall():
            loaded_metadata = json_loads(row["metadata_json"], None)
            metadata = loaded_metadata if isinstance(loaded_metadata, dict) else {}
            expired = bool(row["expires_at"] and row["expires_at"] <= current)
            authority = metadata.get("authority_intent")
            valid = isinstance(loaded_metadata, dict) and (
                not metadata.get("approval_id") or (
                    isinstance(authority, dict) and bool(authority.get("sha256"))))
            result.append(dict(row) | {
                "metadata": metadata, "expired": expired, "valid": valid,
                "active": not expired and valid,
            })
        return result
    finally:
        conn.close()


class ScopeGuard:
    def evaluate(self, request: ScopeRequest) -> PolicyDecision:
        if request.effect not in EFFECTS:
            return PolicyDecision("deny", "elevated", request.effect, request.target,
                                  "unknown effects are denied", normalized_arguments=redact_arguments(request.arguments))
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
                                      normalized_arguments=redact_arguments(request.arguments))
            if request.allowed_paths and not _within(target, request.allowed_paths):
                return PolicyDecision("deny", RISK_BY_EFFECT[effect], effect, target,
                                      "canonical target is outside authorized filesystem scope",
                                      normalized_arguments=redact_arguments(request.arguments))
            missing_path_scope = not request.allowed_paths
        if request.effect in {"network", "remote"}:
            try:
                normalizer = (
                    normalize_remote_target if request.effect == "remote"
                    else normalize_network_target
                )
                target, host = normalizer(target)
            except ValueError as exc:
                return PolicyDecision("deny", RISK_BY_EFFECT[effect], effect, target, str(exc),
                                      normalized_arguments=redact_arguments(request.arguments))
            allowed_hosts = set()
            for item in request.allowed_hosts:
                parsed_allowed = urlsplit(item if "://" in item else "https://" + item)
                allowed_hosts.add((parsed_allowed.hostname or item).rstrip(".").casefold())
            if allowed_hosts and host not in allowed_hosts:
                return PolicyDecision("deny", RISK_BY_EFFECT[effect], effect, target,
                                      "destination is outside authorized network scope",
                                      normalized_arguments=redact_arguments(request.arguments))
        rule = self._matching_rule(effect, target, request=request)
        if missing_path_scope and not rule:
            return PolicyDecision("deny", RISK_BY_EFFECT[effect], effect, target,
                                  "filesystem tools require an authorized path scope",
                                  normalized_arguments=redact_arguments(request.arguments))
        action = rule["action"] if rule else DEFAULT_ACTIONS[effect]
        if rule:
            reason = f"matched {rule['scope']} policy rule"
        return PolicyDecision(action, RISK_BY_EFFECT[effect], effect, target, reason,
                              rule["id"] if rule else None, redact_arguments(request.arguments))

    @staticmethod
    def _matching_rule(effect, target, *, request=None):
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
                metadata = json_loads(row["metadata_json"], None)
                if not isinstance(metadata, dict):
                    continue
                target_matches = not configured
                if metadata.get("target_kind") == "path":
                    target_matches = bool(configured) and _within(target, (configured,))
                elif effect in {"network", "remote"} and configured:
                    try:
                        host = urlsplit(target).hostname or target
                    except ValueError:
                        host = target
                    target_matches = host == configured or host.endswith("." + configured)
                elif configured:
                    target_matches = target == configured
                if not target_matches:
                    continue
                authority = metadata.get("authority_intent")
                if metadata.get("approval_id") and not (
                        isinstance(authority, dict) and authority.get("sha256")):
                    continue
                if authority is not None:
                    if not isinstance(authority, dict) or not authority.get("sha256"):
                        continue
                    if request is None:
                        continue
                    candidate = PolicyDecision(
                        row["action"], RISK_BY_EFFECT[effect], effect, target,
                        "persistent authority candidate",
                        normalized_arguments=redact_arguments(request.arguments),
                    )
                    if canonical_intent(request, candidate)["sha256"] != authority["sha256"]:
                        continue
                return row
            return None
        finally:
            conn.close()
