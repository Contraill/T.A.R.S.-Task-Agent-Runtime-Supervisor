from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .config import IDENTITY_PATH, ROLE_PERSONA_ROOT, SOUL_PATH, STATE_ROOT
from .file_transactions import (
    atomic_write_anchored_text,
    exclusive_file_lock,
    installation_transaction,
    read_anchored_text,
    regular_file_exists,
)


@dataclass(frozen=True)
class IdentityBundle:
    identity: str
    soul: str
    role_overlay: str
    sources: tuple[str, ...]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _identity_lock_path() -> Path:
    return IDENTITY_PATH.parent.parent / ".tars-identity.lock"


@contextmanager
def _identity_transaction():
    with installation_transaction(STATE_ROOT):
        with exclusive_file_lock(_identity_lock_path()) as anchor:
            yield anchor


def _ensure_identity_files_unlocked(anchor) -> None:
    if SOUL_PATH.parent != IDENTITY_PATH.parent:
        raise ValueError("identity and soul files must share one directory")
    if ROLE_PERSONA_ROOT.parent != IDENTITY_PATH.parent:
        raise ValueError("Role persona root must be inside the identity directory")
    identity_parts = anchor.relative(IDENTITY_PATH)
    soul_parts = anchor.relative(SOUL_PATH)
    role_root_parts = anchor.relative(ROLE_PERSONA_ROOT)
    anchor.makedirs(role_root_parts)
    if not regular_file_exists(anchor, identity_parts):
        atomic_write_anchored_text(
            anchor,
            identity_parts,
            "# Identity\n\nT.A.R.S. is a local personal agent.\n",
        )
    if not regular_file_exists(anchor, soul_parts):
        atomic_write_anchored_text(
            anchor,
            soul_parts,
            "# Operating principles\n\nPreserve truth, continuity, and user control.\n",
        )


def ensure_identity_files() -> None:
    with _identity_transaction() as anchor:
        _ensure_identity_files_unlocked(anchor)


def _read_identity_bundle(anchor, role_id: str) -> IdentityBundle:
    overlay = ROLE_PERSONA_ROOT / f"{role_id}.md"
    overlay_parts = anchor.relative(overlay)
    sources = [str(IDENTITY_PATH), str(SOUL_PATH)]
    if regular_file_exists(anchor, overlay_parts):
        sources.append(str(overlay))
    values = []
    for path in (IDENTITY_PATH, SOUL_PATH, overlay):
        parts = anchor.relative(path)
        try:
            values.append(read_anchored_text(anchor, parts).strip())
        except FileNotFoundError:
            values.append("")
    return IdentityBundle(*values, tuple(sources))


def load_identity(role_id: str, *, create=True) -> IdentityBundle:
    if create:
        with _identity_transaction() as anchor:
            _ensure_identity_files_unlocked(anchor)
            return _read_identity_bundle(anchor, role_id)
    overlay = ROLE_PERSONA_ROOT / f"{role_id}.md"
    sources = [str(IDENTITY_PATH), str(SOUL_PATH)]
    if overlay.is_file():
        sources.append(str(overlay))
    return IdentityBundle(_read(IDENTITY_PATH), _read(SOUL_PATH), _read(overlay), tuple(sources))
