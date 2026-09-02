from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import uuid

from .policy import redact
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    task_id: str | None
    event_uuid: str | None
    evidence_type: str
    source: str
    content_sha256: str
    result_ref: str
    metadata: dict
    created_at: str


def _from_row(row):
    return EvidenceRecord(
        id=row["id"], task_id=row["task_id"], event_uuid=row["event_uuid"],
        evidence_type=row["evidence_type"], source=row["source"],
        content_sha256=row["content_sha256"], result_ref=row["result_ref"],
        metadata=json_loads(row["metadata_json"], {}), created_at=row["created_at"],
    )


def record(evidence_type, source, content, *, task_id=None, event_uuid=None,
           result_ref="", metadata=None):
    payload = content if isinstance(content, bytes) else str(content).encode("utf-8")
    evidence_id = "evidence-" + uuid.uuid4().hex
    ensure_state_store()
    with transaction(immediate=True) as conn:
        conn.execute(
            """INSERT INTO evidence_records(id,task_id,event_uuid,evidence_type,source,
               content_sha256,result_ref,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)""",
            (evidence_id, task_id, event_uuid, evidence_type, str(source),
             hashlib.sha256(payload).hexdigest(), str(result_ref),
             json_dumps(redact(metadata or {})), now_utc()),
        )
    return load(evidence_id)


def load(evidence_id):
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM evidence_records WHERE id=?", (evidence_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown evidence: {evidence_id}")
        return _from_row(row)
    finally:
        conn.close()


def list_records(*, task_id=None, evidence_type=None, limit=50):
    ensure_state_store()
    clauses, params = [], []
    if task_id:
        clauses.append("task_id=?")
        params.append(task_id)
    if evidence_type:
        clauses.append("evidence_type=?")
        params.append(evidence_type)
    sql = "SELECT * FROM evidence_records"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(int(limit))
    conn = connect()
    try:
        return [_from_row(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def verify_artifact(path, claims, *, task_id=None, event_uuid=None, content=None):
    """Verify explicit artifact claims against retained evidence chunks.

    Each claim supplies ``text``, ``evidence_id`` and ``supporting_text``. This
    deliberately verifies provenance links and exact support, not semantic truth.
    """
    artifact = str(path)
    if content is None:
        content = Path(artifact).read_text(encoding="utf-8", errors="replace")
    results = []
    for claim in claims:
        text = str(claim["text"])
        supporting = str(claim["supporting_text"])
        source = load(str(claim["evidence_id"]))
        chunk = str(source.metadata.get("relevant_chunk", ""))
        in_artifact = text in content
        in_evidence = bool(supporting) and supporting in chunk
        results.append({"text": text, "evidence_id": source.id,
                        "in_artifact": in_artifact, "supported": in_evidence,
                        "verified": in_artifact and in_evidence})
    verified = bool(results) and all(item["verified"] for item in results)
    result = {"artifact": artifact, "claims": results, "verified": verified,
              "artifact_sha256": hashlib.sha256(content.encode()).hexdigest()}
    verification = record(
        "artifact_verification", artifact, content, task_id=task_id,
        event_uuid=event_uuid, result_ref=artifact,
        metadata={"verified": verified,
                  "source_evidence_ids": [item["evidence_id"] for item in results]},
    )
    return result | {"evidence_id": verification.id}
