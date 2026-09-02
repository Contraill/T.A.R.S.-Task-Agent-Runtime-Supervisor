from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import threading

from .secure_paths import AnchoredRoot


INSTALLATION_LOCK_NAME = ".tars-restore.lock"
_guard = threading.Lock()
_thread_locks: dict[str, threading.RLock] = {}
_local = threading.local()


def _reset_after_fork() -> None:
    global _guard, _thread_locks, _local
    _guard = threading.Lock()
    _thread_locks = {}
    _local = threading.local()


os.register_at_fork(after_in_child=_reset_after_fork)


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _local_lock(key: str) -> threading.RLock:
    with _guard:
        return _thread_locks.setdefault(key, threading.RLock())


def _assert_lock_binding(anchor: AnchoredRoot, target: Path, identity) -> None:
    try:
        with AnchoredRoot(target.parent) as current:
            bound_parent = os.fstat(anchor.fd)
            current_parent = os.fstat(current.fd)
            if (bound_parent.st_dev, bound_parent.st_ino) != (
                    current_parent.st_dev, current_parent.st_ino):
                raise RuntimeError(
                    "transaction directory changed during the operation"
                )
        current_lock = anchor.lstat((target.name,))
    except FileNotFoundError as exc:
        raise RuntimeError("transaction path changed during the operation") from exc
    if (
        not stat.S_ISREG(current_lock.st_mode)
        or current_lock.st_nlink != 1
        or (current_lock.st_dev, current_lock.st_ino)
        != (identity.st_dev, identity.st_ino)
    ):
        raise RuntimeError("transaction lock changed during the operation")


@contextmanager
def exclusive_file_lock(path: str | Path):
    """Hold a stable lock and yield its descriptor-bound parent directory."""
    target = _absolute(path)
    key = os.fspath(target)
    thread_lock = _local_lock(key)
    with thread_lock:
        active = getattr(_local, "active", None)
        if active is None:
            active = _local.active = {}
        entry = active.get(key)
        if entry is not None:
            entry["depth"] += 1
            try:
                yield entry["anchor"]
            finally:
                entry["depth"] -= 1
            return

        anchor = AnchoredRoot.open_or_create(target.parent)
        lock_fd = -1
        try:
            lock_fd = anchor.open(
                (target.name,), os.O_RDWR | os.O_CREAT, mode=0o600
            )
            os.fchmod(lock_fd, 0o600)
            value = os.fstat(lock_fd)
            if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
                raise ValueError("transaction lock must be a singly linked regular file")
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            _assert_lock_binding(anchor, target, value)
            active[key] = {"anchor": anchor, "depth": 1}
            try:
                try:
                    yield anchor
                except BaseException:
                    raise
                else:
                    _assert_lock_binding(anchor, target, value)
            finally:
                active.pop(key, None)
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            if lock_fd >= 0:
                os.close(lock_fd)
            anchor.close()


@contextmanager
def installation_transaction(state_root: str | Path):
    """Exclude backup/restore while reading or changing live configuration."""
    lock_path = _absolute(state_root).parent / INSTALLATION_LOCK_NAME
    with exclusive_file_lock(lock_path) as anchor:
        yield anchor


def _parts(value: str | tuple[str, ...]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


def regular_file_exists(
        anchor: AnchoredRoot, name: str | tuple[str, ...]) -> bool:
    parts = _parts(name)
    try:
        value = anchor.lstat(parts)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(
            f"mutable state path is not a regular file: {anchor.path.joinpath(*parts)}"
        )
    return True


def read_anchored_text(
        anchor: AnchoredRoot, name: str | tuple[str, ...]) -> str:
    with anchor.reader(_parts(name), text=True) as handle:
        return handle.read()


def atomic_write_anchored_text(
        anchor: AnchoredRoot, name: str | tuple[str, ...], payload: str) -> None:
    anchor.atomic_write(_parts(name), payload.encode("utf-8"))


def durable_unlink_anchored(anchor: AnchoredRoot, name: str) -> None:
    try:
        anchor.delete((name,))
    except FileNotFoundError:
        return
    os.fsync(anchor.fd)


def atomic_write_text(path: str | Path, payload: str) -> None:
    target = _absolute(path)
    with AnchoredRoot.open_or_create(target.parent) as anchor:
        atomic_write_anchored_text(anchor, target.name, payload)
