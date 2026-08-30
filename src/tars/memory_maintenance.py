from __future__ import annotations

from dataclasses import dataclass
import uuid

from . import memory
from .memory import MEMORY_KINDS, doctor as memory_doctor, forget, stage_candidate
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction

TRIGGERS = {"explicit", "session_close", "context_rollover", "scheduled"}


@dataclass(frozen=True)
class MaintenanceRun:
    id: str
    trigger: str
    mode: str
    status: str
    report: dict
    actions: tuple[dict, ...]
    rollback_refs: tuple[str, ...]
    model_provenance: dict
    created_at: str
    completed_at: str | None


def _from_row(row):
    return MaintenanceRun(
        id=row["id"], trigger=row["trigger"], mode=row["mode"], status=row["status"],
        report=json_loads(row["report_json"], {}),
        actions=tuple(json_loads(row["actions_json"], [])),
        rollback_refs=tuple(json_loads(row["rollback_refs_json"], [])),
        model_provenance=json_loads(row["model_provenance_json"], {}),
        created_at=row["created_at"], completed_at=row["completed_at"],
    )


def _normalized(value):
    return " ".join(str(value).casefold().split())


def audit():
    ensure_state_store()
    conn = connect()
    try:
        rows = conn.execute("SELECT * FROM memory_index ORDER BY created_at,id").fetchall()
        candidates = conn.execute(
            "SELECT COUNT(*) FROM memory_candidates WHERE status='staged'"
        ).fetchone()[0]
        projections = conn.execute(
            """SELECT COUNT(*) AS count, COALESCE(AVG(CAST(token_count AS REAL)/usable_input),0) AS average,
               COALESCE(MAX(CAST(token_count AS REAL)/usable_input),0) AS maximum
               FROM context_projections WHERE usable_input>0"""
        ).fetchone()
    finally:
        conn.close()
    by_content = {}
    ids = set()
    superseded = set()
    expired = []
    stamp = now_utc()
    indexed_paths = set()
    for row in rows:
        ids.add(row["id"])
        indexed_paths.add(row["path"])
        by_content.setdefault((row["scope"], _normalized(row["content"])), []).append(row["id"])
        if row["supersedes"]:
            superseded.add(row["supersedes"])
        if row["expiry"] and row["expiry"] <= stamp:
            expired.append(row["id"])
    duplicates = [group for group in by_content.values() if len(group) > 1]
    orphan_supersedes = sorted(superseded - ids)
    corpus_paths = {
        str(path) for kind in MEMORY_KINDS
        for path in (memory.MEMORY_ROOT / kind).glob("mem-*.md")
    }
    return {
        "entries": len(rows),
        "staged_candidates": int(candidates),
        "duplicates": duplicates,
        "expired": sorted(expired),
        "superseded": sorted(superseded & ids),
        "orphan_supersedes": orphan_supersedes,
        "index_drift": {
            "unindexed_files": sorted(corpus_paths - indexed_paths),
            "missing_files": sorted(indexed_paths - corpus_paths),
        },
        "prompt_pressure": {
            "projections": int(projections["count"]),
            "average": float(projections["average"]),
            "maximum": float(projections["maximum"]),
        },
    }


def run_maintenance(*, trigger="explicit", apply=False):
    if trigger not in TRIGGERS:
        raise ValueError(f"invalid maintenance trigger: {trigger}")
    ensure_state_store()
    run_id = "maint-" + uuid.uuid4().hex
    created = now_utc()
    with transaction(immediate=True) as conn:
        conn.execute(
            """INSERT INTO memory_maintenance_runs(id,trigger,mode,status,created_at)
               VALUES(?,?,?,?,?)""",
            (run_id, trigger, "apply" if apply else "audit", "running", created),
        )
    before = audit()
    actions = []
    rollback_refs = []
    status = "completed"
    try:
        if apply:
            for entry_id in before["expired"]:
                archive = forget(entry_id)
                actions.append({"action": "forget_expired", "memory_id": entry_id})
                if archive:
                    rollback_refs.append(str(archive))
            repair = memory_doctor()
            actions.append({"action": "rebuild_index", "indexed": repair["indexed"]})
            if not repair["ok"]:
                status = "failed"
        report = {"before": before, "after": audit() if apply else before}
    except Exception as exc:
        status = "failed"
        report = {"before": before, "error": str(exc)}
        with transaction(immediate=True) as conn:
            conn.execute(
                """UPDATE memory_maintenance_runs SET status=?,report_json=?,actions_json=?,
                   rollback_refs_json=?,completed_at=? WHERE id=?""",
                (status, json_dumps(report), json_dumps(actions), json_dumps(rollback_refs),
                 now_utc(), run_id),
            )
        raise
    with transaction(immediate=True) as conn:
        conn.execute(
            """UPDATE memory_maintenance_runs SET status=?,report_json=?,actions_json=?,
               rollback_refs_json=?,completed_at=? WHERE id=?""",
            (status, json_dumps(report), json_dumps(actions), json_dumps(rollback_refs),
             now_utc(), run_id),
        )
    return load_run(run_id)


def stage_reflection(proposals, *, trigger="explicit", model_provenance=None):
    if trigger not in TRIGGERS:
        raise ValueError(f"invalid maintenance trigger: {trigger}")
    provenance = dict(model_provenance or {})
    if not provenance.get("model") or not provenance.get("backend"):
        raise ValueError("model-assisted reflection requires model and backend provenance")
    ensure_state_store()
    run_id = "maint-" + uuid.uuid4().hex
    created = now_utc()
    with transaction(immediate=True) as conn:
        conn.execute(
            """INSERT INTO memory_maintenance_runs(id,trigger,mode,status,
               model_provenance_json,created_at) VALUES(?,?,?,?,?,?)""",
            (run_id, trigger, "reflection", "running", json_dumps(provenance), created),
        )
    candidate_ids = []
    try:
        for proposal in proposals:
            candidate_ids.append(stage_candidate(
                proposal["content"], kind=proposal.get("kind", "episodic"),
                scope=proposal.get("scope", "global"), title=proposal.get("title", ""),
                source=f"reflection:{run_id}", confidence=float(proposal.get("confidence", 0.5)),
                tags=proposal.get("tags", ()),
            ))
    except Exception as exc:
        with transaction(immediate=True) as conn:
            conn.execute(
                """UPDATE memory_maintenance_runs SET status='failed',report_json=?,
                   actions_json=?,completed_at=? WHERE id=?""",
                (json_dumps({"error": str(exc), "candidate_ids": candidate_ids}),
                 json_dumps([
                     {"action": "stage_candidate", "candidate_id": value}
                     for value in candidate_ids
                 ]), now_utc(), run_id),
            )
        raise
    report = {"proposals": len(candidate_ids), "candidate_ids": candidate_ids}
    with transaction(immediate=True) as conn:
        conn.execute(
            """UPDATE memory_maintenance_runs SET status='completed',report_json=?,
               actions_json=?,completed_at=? WHERE id=?""",
            (json_dumps(report), json_dumps([
                {"action": "stage_candidate", "candidate_id": value}
                for value in candidate_ids
            ]), now_utc(), run_id),
        )
    return load_run(run_id)


def load_run(run_id):
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM memory_maintenance_runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown memory maintenance run: {run_id}")
        return _from_row(row)
    finally:
        conn.close()


def list_runs(limit=50):
    ensure_state_store()
    conn = connect()
    try:
        return [_from_row(row) for row in conn.execute(
            "SELECT * FROM memory_maintenance_runs ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()]
    finally:
        conn.close()
