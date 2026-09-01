from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import uuid

from .config import MEMORY_HISTORY_ROOT, MEMORY_ROOT
from .ownership import Owner, claim_in_transaction
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction

MEMORY_KINDS = {"system", "profile", "projects", "episodic", "reference"}
ID_RE = re.compile(r"^mem-[a-f0-9]{32}$")


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    kind: str
    scope: str
    title: str
    content: str
    source: str
    created_at: str
    updated_at: str
    confidence: float
    supersedes: str | None
    expiry: str | None
    tags: tuple[str, ...]
    path: str


@dataclass(frozen=True)
class MemoryHit:
    entry: MemoryEntry
    score: float
    signals: tuple[str, ...]


def _validate_kind(kind):
    if kind not in MEMORY_KINDS:
        raise ValueError(f"invalid memory kind: {kind}")


def _entry_path(entry_id, kind):
    _validate_kind(kind)
    if not ID_RE.match(entry_id):
        raise ValueError("invalid memory id")
    return MEMORY_ROOT / kind / f"{entry_id}.md"


def _serialize(entry: MemoryEntry):
    metadata = asdict(entry)
    metadata.pop("content")
    metadata.pop("path")
    metadata["tags"] = list(entry.tags)
    return "---\n" + "\n".join(
        f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in metadata.items()
    ) + "\n---\n\n" + entry.content.strip() + "\n"


def _parse(path: Path):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError(f"invalid memory document: {path}")
    header, content = text[4:].split("\n---\n", 1)
    metadata = {}
    for line in header.splitlines():
        key, value = line.split(":", 1)
        metadata[key] = json.loads(value.strip())
    return MemoryEntry(
        id=metadata["id"], kind=metadata["kind"], scope=metadata["scope"],
        title=metadata.get("title", ""), content=content.strip(), source=metadata["source"],
        created_at=metadata["created_at"], updated_at=metadata["updated_at"],
        confidence=float(metadata["confidence"]), supersedes=metadata.get("supersedes"),
        expiry=metadata.get("expiry"), tags=tuple(metadata.get("tags", [])), path=str(path),
    )


def _atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".memory-", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _archive(path: Path, action="update"):
    if not path.exists():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    target = MEMORY_HISTORY_ROOT / path.stem / f"{stamp}-{action}-{digest}.md"
    _atomic_write(target, path.read_text(encoding="utf-8"))
    return target


def _index_entry(entry: MemoryEntry, conn):
    conn.execute(
        """INSERT INTO memory_index(id,kind,scope,title,content,source,confidence,
           created_at,updated_at,supersedes,expiry,tags_json,path) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET kind=excluded.kind,scope=excluded.scope,
           title=excluded.title,content=excluded.content,source=excluded.source,
           confidence=excluded.confidence,updated_at=excluded.updated_at,
           supersedes=excluded.supersedes,
           expiry=excluded.expiry,tags_json=excluded.tags_json,path=excluded.path""",
        (entry.id, entry.kind, entry.scope, entry.title, entry.content, entry.source,
         entry.confidence, entry.created_at, entry.updated_at, entry.supersedes, entry.expiry,
         json_dumps(list(entry.tags)), entry.path),
    )
    conn.execute("DELETE FROM memory_fts WHERE id=?", (entry.id,))
    conn.execute(
        "INSERT INTO memory_fts(id,title,content,tags) VALUES(?,?,?,?)",
        (entry.id, entry.title, entry.content, " ".join(entry.tags)),
    )


def remember(content, *, kind="profile", scope="global", title="", source="user",
             confidence=1.0, supersedes=None, expiry=None, tags=(), _entry_id=None):
    ensure_state_store()
    _validate_kind(kind)
    content = str(content).strip()
    if not content:
        raise ValueError("memory content cannot be empty")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if not str(scope).strip():
        raise ValueError("memory scope cannot be empty")
    if supersedes is not None:
        inspect(supersedes)
    duplicate = find_duplicate(content, scope=scope)
    if duplicate:
        return duplicate
    stamp = now_utc()
    entry_id = _entry_id or "mem-" + uuid.uuid4().hex
    path = _entry_path(entry_id, kind)
    entry = MemoryEntry(entry_id, kind, str(scope), str(title), content, str(source),
                        stamp, stamp, confidence, supersedes, expiry,
                        tuple(dict.fromkeys(map(str, tags))), str(path))
    _atomic_write(path, _serialize(entry))
    with transaction(immediate=True) as conn:
        _index_entry(entry, conn)
    return entry


def inspect(entry_id):
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT path FROM memory_index WHERE id=?", (entry_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        raise KeyError(f"unknown memory: {entry_id}")
    return _parse(Path(row["path"]))


def find_duplicate(content, *, scope):
    ensure_state_store()
    normalized = " ".join(str(content).casefold().split())
    conn = connect()
    try:
        rows = conn.execute("SELECT id,content FROM memory_index WHERE scope=?", (scope,)).fetchall()
    finally:
        conn.close()
    for row in rows:
        if " ".join(row["content"].casefold().split()) == normalized:
            return inspect(row["id"])
    return None


def search(query, *, scope=None, kind=None, limit=10, initialize=True):
    readonly_path = None
    if initialize:
        ensure_state_store()
    else:
        from . import state_store
        database = state_store.current_state_db_path()
        if not database.is_file():
            return []
        readonly_path = database
    query = str(query).strip()
    if not query:
        return []
    terms = re.findall(r"[\w-]+", query, flags=re.UNICODE)
    if not terms:
        return []
    fts_query = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    clauses = ["memory_fts MATCH ?"]
    params = [fts_query]
    if scope:
        clauses.append("m.scope=?")
        params.append(scope)
    if kind:
        _validate_kind(kind)
        clauses.append("m.kind=?")
        params.append(kind)
    clauses.append("(m.expiry IS NULL OR m.expiry > ?)")
    clauses.append("NOT EXISTS (SELECT 1 FROM memory_index newer WHERE newer.supersedes=m.id)")
    params.extend([now_utc(), int(limit)])
    if readonly_path is None:
        conn = connect()
    else:
        conn = sqlite3.connect(f"file:{readonly_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""SELECT m.*, bm25(memory_fts) AS rank FROM memory_fts
                JOIN memory_index m ON m.id=memory_fts.id
                WHERE {' AND '.join(clauses)} ORDER BY rank, m.updated_at DESC LIMIT ?""",
            params,
        ).fetchall()
    finally:
        conn.close()
    hits = []
    for row in rows:
        entry = _parse(Path(row["path"]))
        signals = ["fts5", f"scope:{entry.scope}", f"confidence:{entry.confidence:g}", "recency"]
        hits.append(MemoryHit(entry, float(-row["rank"]), tuple(signals)))
    return hits


def forget(entry_id):
    entry = inspect(entry_id)
    path = Path(entry.path)
    archived = _archive(path, "delete")
    path.unlink()
    with transaction(immediate=True) as conn:
        conn.execute("DELETE FROM memory_fts WHERE id=?", (entry_id,))
        conn.execute("DELETE FROM memory_index WHERE id=?", (entry_id,))
    return archived


def rebuild_index(*, memory_root=None, state_db_path=None):
    root = Path(memory_root) if memory_root is not None else MEMORY_ROOT
    if state_db_path is None:
        ensure_state_store()
    entries = []
    for kind in sorted(MEMORY_KINDS):
        for path in sorted((root / kind).glob("mem-*.md")):
            entries.append(_parse(path))
    if state_db_path is None:
        context = transaction(immediate=True)
    else:
        from contextlib import contextmanager
        @contextmanager
        def local_transaction():
            conn = sqlite3.connect(state_db_path)
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        context = local_transaction()
    with context as conn:
        conn.execute("DELETE FROM memory_fts")
        conn.execute("DELETE FROM memory_index")
        for entry in entries:
            _index_entry(entry, conn)
    return len(entries)


def stage_candidate(content, *, kind="profile", scope="global", title="", source="model",
                    confidence=0.5, tags=()):
    ensure_state_store()
    _validate_kind(kind)
    if not str(content).strip():
        raise ValueError("memory candidate cannot be empty")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if not str(scope).strip():
        raise ValueError("memory scope cannot be empty")
    candidate_id = "cand-" + uuid.uuid4().hex
    with transaction(immediate=True) as conn:
        conn.execute(
            """INSERT INTO memory_candidates(id,kind,scope,title,content,source,
               confidence,tags_json,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (candidate_id, kind, scope, title, str(content).strip(), source,
             confidence, json_dumps(list(tags)), "staged", now_utc()),
        )
    return candidate_id


def review_candidates(*, status="staged"):
    ensure_state_store()
    conn = connect()
    try:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM memory_candidates WHERE status=? ORDER BY created_at", (status,)
        ).fetchall()]
    finally:
        conn.close()


def decide_candidate(candidate_id, *, promote: bool, reason=""):
    ensure_state_store()
    owner = Owner.create("memory-review")
    with transaction(immediate=True) as conn:
        row = conn.execute(
            "SELECT * FROM memory_candidates WHERE id=? AND status IN ('staged','reviewing')",
            (candidate_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"unknown staged memory candidate: {candidate_id}")
        if not claim_in_transaction(
            conn, "memory-candidate-review", candidate_id, owner, lease_seconds=30,
            metadata={"promote": bool(promote)},
        ):
            raise RuntimeError(f"memory candidate {candidate_id} has a live reviewer")
        changed = conn.execute(
            "UPDATE memory_candidates SET status='reviewing',reason=?,reviewed_at=? "
            "WHERE id=? AND status IN ('staged','reviewing')",
            (reason, now_utc(), candidate_id)).rowcount
        if changed != 1:
            raise RuntimeError(f"memory candidate {candidate_id} changed concurrently")
    if not promote:
        stamp = now_utc()
        with transaction(immediate=True) as conn:
            changed = conn.execute(
                "UPDATE memory_candidates SET status='rejected',reason=?,reviewed_at=? "
                "WHERE id=? AND status='reviewing' AND EXISTS (SELECT 1 FROM resource_leases "
                "WHERE resource_type='memory-candidate-review' AND resource_key=? "
                "AND owner_token=? AND expires_at>?)",
                (reason, stamp, candidate_id, candidate_id, owner.token, stamp),
            ).rowcount
            if changed != 1:
                raise RuntimeError(f"memory candidate {candidate_id} lost review ownership")
            conn.execute(
                "DELETE FROM resource_leases WHERE resource_type='memory-candidate-review' "
                "AND resource_key=? AND owner_token=?", (candidate_id, owner.token),
            )
        return None
    entry_id = "mem-" + candidate_id.removeprefix("cand-")
    try:
        entry = remember(
            row["content"], kind=row["kind"], scope=row["scope"], title=row["title"],
            source=row["source"], confidence=row["confidence"],
            tags=json_loads(row["tags_json"], []),
            _entry_id=entry_id,
        )
        stamp = now_utc()
        with transaction(immediate=True) as conn:
            changed = conn.execute(
                "UPDATE memory_candidates SET status='promoted',reason=?,reviewed_at=? "
                "WHERE id=? AND status='reviewing' AND EXISTS (SELECT 1 FROM resource_leases "
                "WHERE resource_type='memory-candidate-review' AND resource_key=? "
                "AND owner_token=? AND expires_at>?)",
                (reason, stamp, candidate_id, candidate_id, owner.token, stamp)).rowcount
            if changed != 1:
                raise RuntimeError(f"memory candidate {candidate_id} lost review ownership")
            conn.execute(
                "DELETE FROM resource_leases WHERE resource_type='memory-candidate-review' "
                "AND resource_key=? AND owner_token=?", (candidate_id, owner.token),
            )
    except Exception:
        with transaction(immediate=True) as conn:
            changed = conn.execute(
                "UPDATE memory_candidates SET status='staged',reviewed_at=NULL "
                "WHERE id=? AND status='reviewing' AND EXISTS (SELECT 1 FROM resource_leases "
                "WHERE resource_type='memory-candidate-review' AND resource_key=? "
                "AND owner_token=?)", (candidate_id, candidate_id, owner.token),
            ).rowcount
            if changed:
                conn.execute(
                    "DELETE FROM resource_leases WHERE resource_type='memory-candidate-review' "
                    "AND resource_key=? AND owner_token=?", (candidate_id, owner.token),
                )
        raise
    return entry


def doctor():
    errors = []
    files = 0
    for kind in sorted(MEMORY_KINDS):
        for path in sorted((MEMORY_ROOT / kind).glob("*.md")):
            files += 1
            try:
                _parse(path)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
    indexed = rebuild_index() if not errors else 0
    return {"ok": not errors, "files": files, "indexed": indexed, "errors": errors}


def status():
    ensure_state_store()
    conn = connect()
    try:
        indexed = conn.execute("SELECT COUNT(*) FROM memory_index").fetchone()[0]
        staged = conn.execute(
            "SELECT COUNT(*) FROM memory_candidates WHERE status='staged'"
        ).fetchone()[0]
    finally:
        conn.close()
    files = sum(1 for kind in MEMORY_KINDS for _ in (MEMORY_ROOT / kind).glob("mem-*.md"))
    return {"ok": files == indexed, "files": files, "indexed": indexed, "staged": staged}
