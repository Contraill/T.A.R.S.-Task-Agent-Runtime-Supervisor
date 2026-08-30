from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction

NATIVE_CONTEXT = ("TARS.md", ".tars.md")
COMPAT_CONTEXT = ("AGENTS.md", "CLAUDE.md", "README.md", "README", "pyproject.toml", "package.json", "Cargo.toml")


@dataclass(frozen=True)
class ProjectContext:
    root: Path
    files: tuple[Path, ...]
    content: str


def discover_project_context(path: str | Path, *, max_bytes=131072) -> ProjectContext:
    root = Path(path).expanduser().resolve()
    files = []
    remaining = max(0, int(max_bytes))
    sections = []
    for name in (*NATIVE_CONTEXT, *COMPAT_CONTEXT):
        candidate = root / name
        if not candidate.is_file() or candidate in files or remaining <= 0:
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")[:remaining]
        remaining -= len(text.encode("utf-8"))
        files.append(candidate)
        sections.append(f"## {name}\n\n{text.strip()}")
    return ProjectContext(root, tuple(files), "\n\n".join(sections))


def register_project(path: str | Path):
    context = discover_project_context(path)
    ensure_state_store()
    stamp = now_utc()
    project_id = "prj-" + uuid.uuid5(uuid.NAMESPACE_URL, str(context.root)).hex
    with transaction(immediate=True) as conn:
        conn.execute(
            """INSERT INTO project_refs(id,canonical_path,display_name,context_files_json,
               metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(canonical_path) DO UPDATE SET display_name=excluded.display_name,
               context_files_json=excluded.context_files_json,updated_at=excluded.updated_at""",
            (project_id, str(context.root), context.root.name,
             json_dumps([str(p) for p in context.files]), "{}", stamp, stamp),
        )
    return context
