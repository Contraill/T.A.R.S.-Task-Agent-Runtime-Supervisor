from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import threading
import uuid

from .state_store import (connect, current_state_db_path, ensure_state_store, json_dumps,
                          now_utc, state_db_path_scope, transaction)


_CURRENT_OWNER = ContextVar("tars_current_owner", default=None)
_MODEL_EXECUTION_OWNER = ContextVar("tars_model_execution_owner", default=None)

MODEL_EXECUTION_RESOURCE = ("gpu-slot", "local-inference:0")


def _read_process_start(pid: int) -> str:
    value = Path(f"/proc/{int(pid)}/stat").read_text()
    fields = value[value.rfind(")") + 2:].split()
    return fields[19]


def process_start(pid: int) -> str:
    try:
        return _read_process_start(pid)
    except (OSError, IndexError):
        return ""


@dataclass(frozen=True)
class Owner:
    token: str
    pid: int
    process_start: str

    @classmethod
    def create(cls, prefix="owner"):
        pid = os.getpid()
        return cls(f"{prefix}-{uuid.uuid4().hex}", pid, process_start(pid))


def current_owner() -> Owner | None:
    return _CURRENT_OWNER.get()


def model_execution_owner() -> Owner | None:
    return _MODEL_EXECUTION_OWNER.get()


@contextmanager
def owner_scope(owner: Owner):
    token = _CURRENT_OWNER.set(owner)
    try:
        yield owner
    finally:
        _CURRENT_OWNER.reset(token)


def owner_alive(pid: int, expected_start: str) -> bool:
    return bool(expected_start and process_start(int(pid)) == expected_start)


def owner_gone(pid: int, expected_start: str) -> bool:
    """Return true only when PID absence/reuse proves this identity cannot act."""
    if not expected_start:
        return False
    try:
        return _read_process_start(int(pid)) != expected_start
    except (FileNotFoundError, ProcessLookupError):
        return True
    except (OSError, IndexError):
        return False


def _same_owner(row, owner: Owner) -> bool:
    return bool(row and row["owner_token"] == owner.token
                and row["owner_pid"] == owner.pid
                and row["owner_start"] == owner.process_start)


def _expiry(seconds: float, *, now=None) -> str:
    current = now or datetime.now(timezone.utc)
    return (current + timedelta(seconds=max(1.0, float(seconds)))).isoformat()


def claim_in_transaction(conn, resource_type, resource_key, owner: Owner, *,
                         lease_seconds=30.0, metadata=None, now=None) -> bool:
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    expires = _expiry(lease_seconds, now=now)
    row = conn.execute(
        "SELECT * FROM resource_leases WHERE resource_type=? AND resource_key=?",
        (resource_type, resource_key),
    ).fetchone()
    if resource_type == "task-execution":
        fence = conn.execute(
            "SELECT 1 FROM resource_leases "
            "WHERE resource_type='task-execution-fence' AND resource_key=?",
            (resource_key,),
        ).fetchone()
        task = conn.execute(
            "SELECT phase FROM tasks WHERE id=?", (resource_key,),
        ).fetchone()
        if fence or (task and task["phase"] == "cancellation-recovery-required"):
            return False
    if row and not _same_owner(row, owner) and not owner_gone(
            row["owner_pid"], row["owner_start"]):
        return False
    if row:
        changed = conn.execute(
            """UPDATE resource_leases SET owner_token=?,owner_pid=?,owner_start=?,
               acquired_at=?,heartbeat_at=?,expires_at=?,metadata_json=?
               WHERE resource_type=? AND resource_key=? AND owner_token=?
               AND owner_pid=? AND owner_start=? AND expires_at=?""",
            (owner.token, owner.pid, owner.process_start, stamp, stamp, expires,
             json_dumps(metadata or {}), resource_type, resource_key,
             row["owner_token"], row["owner_pid"], row["owner_start"],
             row["expires_at"]),
        ).rowcount
        return changed == 1
    try:
        conn.execute(
            """INSERT INTO resource_leases(resource_type,resource_key,owner_token,
               owner_pid,owner_start,acquired_at,heartbeat_at,expires_at,metadata_json)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (resource_type, resource_key, owner.token, owner.pid, owner.process_start,
             stamp, stamp, expires, json_dumps(metadata or {})),
        )
    except Exception as exc:
        if "UNIQUE" in str(exc):
            return False
        raise
    return True


def claim(resource_type, resource_key, owner: Owner, *, lease_seconds=30.0,
          metadata=None) -> bool:
    ensure_state_store()
    with transaction(immediate=True) as conn:
        return claim_in_transaction(conn, resource_type, resource_key, owner,
                                    lease_seconds=lease_seconds, metadata=metadata)


def claim_workspace(resource_key, owner: Owner, *, lease_seconds=30.0,
                    metadata=None) -> bool:
    key = str(Path(resource_key).resolve(strict=False))
    with transaction(immediate=True) as conn:
        rows = conn.execute(
            "SELECT * FROM resource_leases WHERE resource_type='workspace'"
        ).fetchall()
        for row in rows:
            other = row["resource_key"]
            overlaps = (key == other or key.startswith(other.rstrip("/") + "/")
                        or other.startswith(key.rstrip("/") + "/"))
            if not overlaps or _same_owner(row, owner):
                continue
            if not owner_gone(row["owner_pid"], row["owner_start"]):
                return False
            conn.execute(
                "DELETE FROM resource_leases WHERE resource_type='workspace' "
                "AND resource_key=? AND owner_token=? AND owner_pid=? AND owner_start=?",
                (other, row["owner_token"], row["owner_pid"], row["owner_start"]),
            )
        return claim_in_transaction(
            conn, "workspace", key, owner, lease_seconds=lease_seconds,
            metadata=metadata,
        )


def heartbeat(resource_type, resource_key, owner: Owner, *, lease_seconds=30.0) -> bool:
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        changed = conn.execute(
            """UPDATE resource_leases SET heartbeat_at=?,expires_at=?
               WHERE resource_type=? AND resource_key=? AND owner_token=?
               AND owner_pid=? AND owner_start=?""",
            (stamp, _expiry(lease_seconds), resource_type, resource_key, owner.token,
             owner.pid, owner.process_start),
        ).rowcount
    return changed == 1


def release(resource_type, resource_key, owner: Owner) -> bool:
    with transaction(immediate=True) as conn:
        changed = conn.execute(
            "DELETE FROM resource_leases WHERE resource_type=? AND resource_key=? "
            "AND owner_token=? AND owner_pid=? AND owner_start=?",
            (resource_type, resource_key, owner.token, owner.pid, owner.process_start),
        ).rowcount
    return changed == 1


def held_by(resource_type, resource_key, owner: Owner) -> bool:
    ensure_state_store()
    with connect() as conn:
        return held_by_in_transaction(conn, resource_type, resource_key, owner)


def held_by_in_transaction(conn, resource_type, resource_key, owner: Owner, *,
                           require_unexpired=True) -> bool:
    row = conn.execute(
        "SELECT owner_token,owner_pid,owner_start,expires_at FROM resource_leases "
        "WHERE resource_type=? AND resource_key=?", (resource_type, resource_key),
    ).fetchone()
    return bool(_same_owner(row, owner)
                and (not require_unexpired or row["expires_at"] > now_utc())
                and owner_alive(row["owner_pid"], row["owner_start"]))


def active(resource_type, resource_key) -> bool:
    ensure_state_store()
    with connect() as conn:
        return active_in_transaction(conn, resource_type, resource_key)


def active_in_transaction(conn, resource_type, resource_key) -> bool:
    row = conn.execute(
        "SELECT owner_pid,owner_start FROM resource_leases "
        "WHERE resource_type=? AND resource_key=?", (resource_type, resource_key),
    ).fetchone()
    return bool(row and not owner_gone(row["owner_pid"], row["owner_start"]))


def active_metadata(resource_type, resource_key) -> dict | None:
    ensure_state_store()
    with connect() as conn:
        row = conn.execute(
            "SELECT owner_pid,owner_start,metadata_json FROM resource_leases "
            "WHERE resource_type=? AND resource_key=?",
            (resource_type, resource_key),
        ).fetchone()
    if not row or owner_gone(row["owner_pid"], row["owner_start"]):
        return None
    from .state_store import json_loads
    return json_loads(row["metadata_json"], {})


def task_execution_fenced(task_id) -> bool:
    ensure_state_store()
    with connect() as conn:
        fence = conn.execute(
            "SELECT 1 FROM resource_leases "
            "WHERE resource_type='task-execution-fence' AND resource_key=?",
            (task_id,),
        ).fetchone()
        task = conn.execute("SELECT phase FROM tasks WHERE id=?", (task_id,)).fetchone()
    return bool(fence or (task and task["phase"] == "cancellation-recovery-required"))


@contextmanager
def model_execution_scope(*, operation, lease_seconds=30.0, timeout=30.0,
                          metadata=None):
    """Own every managed-runtime touch, reusing only this exact lexical owner."""
    existing = model_execution_owner()
    resource_type, resource_key = MODEL_EXECUTION_RESOURCE
    if existing is not None:
        if not held_by(resource_type, resource_key, existing):
            raise RuntimeError("model execution ownership was lost")
        yield existing
        return

    owner = Owner.create("model-execution")
    details = {"operation": str(operation)} | dict(metadata or {})
    if not acquire(
        resource_type, resource_key, owner,
        lease_seconds=lease_seconds, timeout=timeout, metadata=details,
    ):
        raise RuntimeError("local inference slot is busy")
    token = _MODEL_EXECUTION_OWNER.set(owner)
    try:
        with Heartbeat(
            resource_type, resource_key, owner, lease_seconds=lease_seconds,
        ):
            yield owner
    finally:
        _MODEL_EXECUTION_OWNER.reset(token)
        release(resource_type, resource_key, owner)


@contextmanager
def task_execution_scope(task_id, *, engine, owner=None, lease_seconds=30.0,
                         metadata=None):
    """Own one task across execution engines, borrowing an exact outer owner token."""
    selected = owner or current_owner() or Owner.create(str(engine))
    # Recovery never invokes the external callback. It resolves an unattempted
    # dead owner or atomically places an attempted outcome into fail-closed state.
    from .control_queue import recover_cancellations
    recover_cancellations(task_id, selected, lease_seconds=lease_seconds)
    with connect() as conn:
        task = conn.execute("SELECT phase FROM tasks WHERE id=?", (task_id,)).fetchone()
    if task and task["phase"] == "cancellation-recovery-required":
        raise RuntimeError(
            f"task {task_id} requires explicit recovery after ambiguous cancellation")
    scoped = current_owner()
    borrowed = bool(
        scoped and scoped.token == selected.token
        and held_by("task-execution", task_id, selected)
    )
    details = {"engine": str(engine)} | dict(metadata or {})
    if not claim(
        "task-execution", task_id, selected,
        lease_seconds=lease_seconds, metadata=details,
    ):
        raise RuntimeError(f"task {task_id} already has a live execution owner")
    try:
        with owner_scope(selected):
            if borrowed:
                yield selected
            else:
                with Heartbeat(
                    "task-execution", task_id, selected,
                    lease_seconds=lease_seconds,
                ):
                    yield selected
    finally:
        if not borrowed:
            release("task-execution", task_id, selected)


def acquire(resource_type, resource_key, owner: Owner, *, lease_seconds=30.0,
            timeout=0.0, cancel_event=None, metadata=None) -> bool:
    deadline = datetime.now(timezone.utc).timestamp() + max(0.0, float(timeout))
    while True:
        if claim(resource_type, resource_key, owner, lease_seconds=lease_seconds,
                 metadata=metadata):
            return True
        if cancel_event is not None and cancel_event.is_set():
            return False
        remaining = deadline - datetime.now(timezone.utc).timestamp()
        if remaining <= 0:
            return False
        if cancel_event is not None:
            cancel_event.wait(min(0.1, remaining))
        else:
            threading.Event().wait(min(0.1, remaining))


class Heartbeat:
    def __init__(self, resource_type, resource_key, owner, *, lease_seconds=30.0):
        self.resource_type = resource_type
        self.resource_key = resource_key
        self.owner = owner
        self.lease_seconds = float(lease_seconds)
        self.stop_event = threading.Event()
        self.error = None
        self.lost = False
        self.state_db_path = current_state_db_path()
        self.thread = threading.Thread(target=self._run, name="tars-lease-heartbeat", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, *_):
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, self.lease_seconds))
        if exc_type is None:
            if self.error is not None:
                raise RuntimeError("durable lease heartbeat failed") from self.error
            if self.lost:
                raise RuntimeError("durable lease ownership was lost")

    def _run(self):
        with state_db_path_scope(self.state_db_path):
            interval = max(0.5, self.lease_seconds / 3)
            while not self.stop_event.wait(interval):
                try:
                    renewed = heartbeat(
                        self.resource_type, self.resource_key, self.owner,
                        lease_seconds=self.lease_seconds)
                except Exception as exc:
                    self.error = exc
                    return
                if not renewed:
                    self.lost = True
                    return
