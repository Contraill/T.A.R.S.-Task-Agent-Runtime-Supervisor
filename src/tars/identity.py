from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import IDENTITY_PATH, ROLE_PERSONA_ROOT, SOUL_PATH


@dataclass(frozen=True)
class IdentityBundle:
    identity: str
    soul: str
    role_overlay: str
    sources: tuple[str, ...]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def ensure_identity_files() -> None:
    IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROLE_PERSONA_ROOT.mkdir(parents=True, exist_ok=True)
    if not IDENTITY_PATH.exists():
        IDENTITY_PATH.write_text("# Identity\n\nT.A.R.S. is a local personal agent.\n", encoding="utf-8")
    if not SOUL_PATH.exists():
        SOUL_PATH.write_text("# Operating principles\n\nPreserve truth, continuity, and user control.\n", encoding="utf-8")


def load_identity(role_id: str) -> IdentityBundle:
    ensure_identity_files()
    overlay = ROLE_PERSONA_ROOT / f"{role_id}.md"
    sources = [str(IDENTITY_PATH), str(SOUL_PATH)]
    if overlay.is_file():
        sources.append(str(overlay))
    return IdentityBundle(_read(IDENTITY_PATH), _read(SOUL_PATH), _read(overlay), tuple(sources))
