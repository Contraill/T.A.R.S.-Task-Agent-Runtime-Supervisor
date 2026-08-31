from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
import uuid

from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction


PERMISSIONS = {
    "status.read", "conversation.read", "conversation.write",
    "task.read", "task.control", "client.admin",
}
DEFAULT_PERMISSIONS = ("status.read", "conversation.read", "conversation.write",
                       "task.read", "task.control")


def _derive(token: str, salt: bytes) -> str:
    return hashlib.scrypt(token.encode(), salt=salt, n=2 ** 14, r=8, p=1).hex()


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _permissions(values) -> tuple[str, ...]:
    result = tuple(sorted(set(map(str, values))))
    unknown = set(result) - PERMISSIONS
    if unknown:
        raise ValueError(f"unknown client permissions: {', '.join(sorted(unknown))}")
    return result


@dataclass(frozen=True)
class CoreClient:
    id: str
    name: str
    principal_id: str
    permissions: tuple[str, ...]
    state: str
    created_at: str
    last_seen_at: str | None
    revoked_at: str | None
    metadata: dict

    def require(self, permission: str) -> None:
        if self.state != "active" or permission not in self.permissions:
            raise PermissionError(f"client lacks permission: {permission}")


def _client(row) -> CoreClient:
    return CoreClient(
        row["id"], row["name"], row["principal_id"],
        tuple(json_loads(row["permissions_json"], [])), row["state"], row["created_at"],
        row["last_seen_at"], row["revoked_at"], json_loads(row["metadata_json"], {}))


def create_pairing(*, permissions=DEFAULT_PERMISSIONS, principal_id="local-owner",
                   ttl_seconds=300) -> dict:
    ensure_state_store()
    permissions = _permissions(permissions)
    if not principal_id.strip():
        raise ValueError("principal id is required")
    code = secrets.token_urlsafe(32)
    pair_id = "pair-" + uuid.uuid4().hex
    created = datetime.now(timezone.utc)
    expires = created + timedelta(seconds=max(30, int(ttl_seconds)))
    with transaction(immediate=True) as conn:
        conn.execute(
            "INSERT INTO core_pairings(id,code_hash,permissions_json,principal_id,created_at,expires_at) VALUES(?,?,?,?,?,?)",
            (pair_id, hashlib.sha256(code.encode()).hexdigest(), json_dumps(permissions),
             principal_id, created.isoformat().replace("+00:00", "Z"),
             expires.isoformat().replace("+00:00", "Z")))
    return {"pairing_id": pair_id, "code": code,
            "expires_at": expires.isoformat().replace("+00:00", "Z"),
            "permissions": list(permissions)}


def exchange_pairing(code: str, name: str, *, metadata=None) -> tuple[CoreClient, str]:
    ensure_state_store()
    digest = hashlib.sha256(str(code).encode()).hexdigest()
    now = datetime.now(timezone.utc)
    with transaction(immediate=True) as conn:
        row = conn.execute("SELECT * FROM core_pairings WHERE code_hash=?", (digest,)).fetchone()
        if not row or row["consumed_at"] or _time(row["expires_at"]) <= now:
            raise PermissionError("pairing code is invalid, expired or already consumed")
        token = secrets.token_urlsafe(48)
        salt = secrets.token_bytes(16)
        client_id = "client-" + uuid.uuid4().hex
        stamp = now.isoformat().replace("+00:00", "Z")
        conn.execute(
            """INSERT INTO core_clients(id,name,principal_id,token_salt,token_hash,
               permissions_json,state,created_at,metadata_json)
               VALUES(?,?,?,?,?,?,'active',?,?)""",
            (client_id, str(name).strip() or "Unnamed client", row["principal_id"],
             salt.hex(), _derive(token, salt), row["permissions_json"], stamp,
             json_dumps(metadata or {})))
        conn.execute("UPDATE core_pairings SET consumed_at=?,client_id=? WHERE id=?",
                     (stamp, client_id, row["id"]))
        client_row = conn.execute("SELECT * FROM core_clients WHERE id=?", (client_id,)).fetchone()
    return _client(client_row), token


def authenticate(token: str) -> CoreClient:
    ensure_state_store()
    if not token:
        raise PermissionError("missing client token")
    with connect() as conn:
        rows = conn.execute("SELECT * FROM core_clients WHERE state='active'").fetchall()
    matched = None
    for row in rows:
        salt = bytes.fromhex(row["token_salt"])
        if hmac.compare_digest(_derive(token, salt), row["token_hash"]):
            matched = row
    if matched is None:
        raise PermissionError("invalid or revoked client token")
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        conn.execute("UPDATE core_clients SET last_seen_at=? WHERE id=?", (stamp, matched["id"]))
        row = conn.execute("SELECT * FROM core_clients WHERE id=?", (matched["id"],)).fetchone()
    return _client(row)


def list_clients(*, include_revoked=True) -> list[CoreClient]:
    ensure_state_store()
    sql = "SELECT * FROM core_clients"
    if not include_revoked:
        sql += " WHERE state='active'"
    sql += " ORDER BY created_at"
    with connect() as conn:
        return [_client(row) for row in conn.execute(sql).fetchall()]


def revoke(client_id: str) -> CoreClient:
    ensure_state_store()
    with transaction(immediate=True) as conn:
        changed = conn.execute(
            "UPDATE core_clients SET state='revoked',revoked_at=? WHERE id=? AND state='active'",
            (now_utc(), client_id)).rowcount
        row = conn.execute("SELECT * FROM core_clients WHERE id=?", (client_id,)).fetchone()
    if not row:
        raise KeyError(f"unknown client: {client_id}")
    if not changed and row["state"] != "revoked":
        raise RuntimeError("client revocation failed")
    return _client(row)
