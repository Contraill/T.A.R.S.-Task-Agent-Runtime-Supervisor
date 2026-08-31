from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from .config import STATE_ROOT
from .policy import ScopeRequest, canonical_path
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction
from .tool_core import ToolResult, ToolRuntime


@dataclass(frozen=True)
class WorkspaceCheckpoint:
    id: str
    task_id: str | None
    kind: str
    root: str
    state: str
    storage_ref: str
    metadata: dict
    created_at: str
    restored_at: str | None


def _from_row(row):
    return WorkspaceCheckpoint(
        row["id"], row["task_id"], row["kind"], row["root"], row["state"],
        row["storage_ref"], json_loads(row["metadata_json"], {}), row["created_at"],
        row["restored_at"],
    )


def load(checkpoint_id):
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM workspace_checkpoints WHERE id=?",
                           (checkpoint_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown workspace checkpoint: {checkpoint_id}")
        return _from_row(row)
    finally:
        conn.close()


def list_checkpoints(*, task_id=None, limit=50):
    ensure_state_store()
    sql = "SELECT * FROM workspace_checkpoints"
    params = []
    if task_id:
        sql += " WHERE task_id=?"
        params.append(task_id)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(int(limit))
    conn = connect()
    try:
        return [_from_row(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WorkspaceRecovery:
    def __init__(self, roots, *, runtime=None, storage_root=None, runner=subprocess.run,
                 max_files=10_000, max_bytes=1_000_000_000):
        self.roots = tuple(canonical_path(root) for root in roots)
        if not self.roots:
            raise ValueError("workspace recovery requires at least one scope root")
        self.runtime = runtime or ToolRuntime()
        self.storage_root = Path(storage_root or (STATE_ROOT / "workspace-checkpoints"))
        self.runner = runner
        self.max_files = int(max_files)
        self.max_bytes = int(max_bytes)

    def _authorize(self, tool, effect, target, *, approval_id=None, task_id=None,
                   destructive=False, arguments=None):
        request = ScopeRequest(
            tool, effect, str(target), arguments or {}, task_id=task_id,
            allowed_paths=self.roots,
            destructive=destructive,
        )
        return self.runtime.authorize((("action", request),), {"action": approval_id})

    def _storage(self, checkpoint_id):
        path = self.storage_root / checkpoint_id
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
        path.chmod(0o700)
        return path

    @staticmethod
    def _safe_relative(value):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe checkpoint path: {value}")
        return path

    def _insert(self, checkpoint_id, task_id, kind, root, storage, metadata):
        with transaction(immediate=True) as conn:
            conn.execute(
                """INSERT INTO workspace_checkpoints(id,task_id,kind,root,state,storage_ref,
                   metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?)""",
                (checkpoint_id, task_id, kind, str(root), "ready", str(storage),
                 json_dumps(metadata), now_utc()),
            )
        return load(checkpoint_id)

    def create_git(self, repository, *, task_id=None):
        root = Path(canonical_path(repository))
        actions = self._authorize("workspace.checkpoint", "read", root, task_id=task_id)
        checkpoint_id = "wcp-" + uuid.uuid4().hex
        storage = self._storage(checkpoint_id)
        try:
            def git(arguments):
                return self.runner(["git", *arguments], cwd=root, capture_output=True,
                                   text=True, check=False)
            top = git(["rev-parse", "--show-toplevel"])
            if top.returncode or Path(canonical_path(top.stdout.strip())) != root:
                raise ValueError("checkpoint root must be the Git worktree root")
            head = git(["rev-parse", "HEAD"])
            if head.returncode:
                raise RuntimeError(head.stderr or "cannot resolve Git HEAD")
            branch = git(["branch", "--show-current"]).stdout.strip()
            staged = git(["diff", "--cached", "--binary", "HEAD"])
            unstaged = git(["diff", "--binary"])
            if staged.returncode or unstaged.returncode:
                raise RuntimeError(staged.stderr or unstaged.stderr or "cannot capture Git diff")
            (storage / "staged.patch").write_text(staged.stdout)
            (storage / "unstaged.patch").write_text(unstaged.stdout)
            untracked_output = git(["ls-files", "--others", "--exclude-standard", "-z"])
            if untracked_output.returncode:
                raise RuntimeError(untracked_output.stderr or "cannot list untracked files")
            untracked = [item for item in untracked_output.stdout.split("\0") if item]
            files = []
            total = 0
            snapshot = storage / "untracked"
            for name in untracked:
                relative = self._safe_relative(name)
                entry = root / relative
                if entry.is_symlink() or not entry.is_file():
                    raise ValueError(f"unsupported untracked entry: {name}")
                source = entry.resolve(strict=True)
                source.relative_to(root)
                size = source.stat().st_size
                total += size
                if len(files) >= self.max_files or total > self.max_bytes:
                    raise ValueError("workspace checkpoint bounds exceeded")
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                files.append({"path": relative.as_posix(), "bytes": size,
                              "sha256": _hash(source)})
            metadata = {"head": head.stdout.strip(), "branch": branch,
                        "staged_patch_sha256": hashlib.sha256(staged.stdout.encode()).hexdigest(),
                        "unstaged_patch_sha256": hashlib.sha256(unstaged.stdout.encode()).hexdigest(),
                        "untracked": files, "reversible_scope": "working-tree-only",
                        "external_effects_reversible": False}
            checkpoint = self._insert(checkpoint_id, task_id, "git", root, storage, metadata)
            data = {"checkpoint_id": checkpoint.id, "kind": "git", "root": str(root),
                    "head": metadata["head"], "dirty": bool(staged.stdout or unstaged.stdout or files),
                    "files": len(files), "bytes": total,
                    "external_effects_reversible": False}
        except Exception as exc:
            shutil.rmtree(storage, ignore_errors=True)
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=data)
        evidence = self.runtime.evidence("workspace_checkpoint", root, repr(data),
                                         task_id=task_id, event_uuid=actions[0].event_uuid,
                                         result_ref=checkpoint.id)
        return ToolResult("workspace.checkpoint", "succeeded", data,
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def create_filesystem(self, root, paths, *, task_id=None):
        workspace = Path(canonical_path(root))
        actions = self._authorize("workspace.checkpoint", "read", workspace, task_id=task_id)
        checkpoint_id = "wcp-" + uuid.uuid4().hex
        storage = self._storage(checkpoint_id)
        try:
            files = []
            total = 0
            for value in paths:
                entry = Path(value).absolute()
                if entry.is_symlink():
                    raise ValueError(f"filesystem snapshots do not follow symlinks: {entry}")
                if not entry.exists():
                    raise FileNotFoundError(f"checkpoint path does not exist: {entry}")
                source = Path(canonical_path(entry))
                source.relative_to(workspace)
                candidates = [source] if source.is_file() else sorted(source.rglob("*"))
                for candidate in candidates:
                    if candidate.is_symlink():
                        raise ValueError(f"filesystem snapshots do not follow symlinks: {candidate}")
                    if not candidate.is_file():
                        continue
                    item_relative = candidate.relative_to(workspace)
                    size = candidate.stat().st_size
                    total += size
                    if len(files) >= self.max_files or total > self.max_bytes:
                        raise ValueError("workspace checkpoint bounds exceeded")
                    destination = storage / "files" / item_relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, destination)
                    files.append({"path": item_relative.as_posix(), "bytes": size,
                                  "sha256": _hash(candidate)})
            metadata = {"files": files, "reversible_scope": "captured-files-only",
                        "external_effects_reversible": False}
            checkpoint = self._insert(checkpoint_id, task_id, "filesystem", workspace,
                                      storage, metadata)
            data = {"checkpoint_id": checkpoint.id, "kind": "filesystem",
                    "root": str(workspace), "files": len(files), "bytes": total,
                    "external_effects_reversible": False}
        except Exception as exc:
            shutil.rmtree(storage, ignore_errors=True)
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=data)
        evidence = self.runtime.evidence("workspace_checkpoint", workspace, repr(data),
                                         task_id=task_id, event_uuid=actions[0].event_uuid,
                                         result_ref=checkpoint.id)
        return ToolResult("workspace.checkpoint", "succeeded", data,
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def preview(self, checkpoint_id, *, task_id=None):
        checkpoint = load(checkpoint_id)
        root = Path(checkpoint.root)
        actions = self._authorize("workspace.rollback_preview", "read", root, task_id=task_id)
        changes = []
        if checkpoint.kind == "git":
            current_head = self.runner(["git", "rev-parse", "HEAD"], cwd=root,
                                       capture_output=True, text=True, check=False).stdout.strip()
            changes.append({"operation": "restore_tracked_patch",
                            "supported": current_head == checkpoint.metadata["head"],
                            "reason": "same HEAD required"})
            baseline = {item["path"]: item for item in checkpoint.metadata["untracked"]}
            current_untracked = self.runner(
                ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root,
                capture_output=True, text=True, check=False,
            )
            if current_untracked.returncode:
                changes.append({"operation": "inspect_untracked", "supported": False,
                                "reason": current_untracked.stderr or
                                "cannot inspect untracked files"})
            for name in [item for item in current_untracked.stdout.split("\0")
                         if item and item not in baseline]:
                changes.append({"operation": "quarantine_new_untracked",
                                "path": str(root / self._safe_relative(name)),
                                "supported": True})
        else:
            baseline = {item["path"]: item for item in checkpoint.metadata["files"]}
        for name, item in baseline.items():
            target = root / self._safe_relative(name)
            current = _hash(target) if target.is_file() else None
            if current != item["sha256"]:
                changes.append({"operation": "restore_file", "path": str(target),
                                "before_sha256": current, "after_sha256": item["sha256"],
                                "supported": True})
        data = {"checkpoint_id": checkpoint.id, "kind": checkpoint.kind,
                "root": checkpoint.root, "changes": changes,
                "external_effects_reversible": False,
                "supported": all(item.get("supported", True) for item in changes)}
        self.runtime.finish(actions, state="succeeded", result=data)
        return ToolResult("workspace.rollback_preview", "succeeded", data,
                          action_ids=tuple(a.id for a in actions))

    def rollback(self, checkpoint_id, *, approval_id=None, task_id=None):
        checkpoint = load(checkpoint_id)
        root = Path(checkpoint.root)
        actions = self._authorize("workspace.rollback", "destructive", root,
                                  approval_id=approval_id, task_id=task_id, destructive=True,
                                  arguments={"checkpoint_id": checkpoint_id})
        preview = self.preview(checkpoint_id, task_id=task_id)
        if not preview.data["supported"]:
            data = preview.data | {"restored": False}
            self.runtime.finish(actions, state="failed", result=data)
            return ToolResult("workspace.rollback", "failed", data,
                              error="rollback is not supported for the current workspace state",
                              action_ids=tuple(a.id for a in actions))
        storage = Path(checkpoint.storage_ref)
        restored = []
        try:
            if checkpoint.kind == "git":
                safety = self.create_git(root, task_id=task_id)
            else:
                safety_paths = [root / self._safe_relative(item["path"])
                                for item in checkpoint.metadata["files"]
                                if (root / self._safe_relative(item["path"])).exists()]
                safety = self.create_filesystem(root, safety_paths, task_id=task_id)
            if checkpoint.kind == "git":
                restore = self.runner(["git", "restore", "--staged", "--worktree", "."],
                                      cwd=root, capture_output=True, text=True, check=False)
                if restore.returncode:
                    raise RuntimeError(restore.stderr or "Git restore failed")
                staged_patch = storage / "staged.patch"
                if staged_patch.stat().st_size:
                    applied = self.runner(["git", "apply", "--index", "--binary",
                                           str(staged_patch)],
                                          cwd=root, capture_output=True, text=True, check=False)
                    if applied.returncode:
                        raise RuntimeError(applied.stderr or "checkpoint staged patch failed")
                unstaged_patch = storage / "unstaged.patch"
                if unstaged_patch.stat().st_size:
                    applied = self.runner(["git", "apply", "--binary", str(unstaged_patch)],
                                          cwd=root, capture_output=True, text=True, check=False)
                    if applied.returncode:
                        raise RuntimeError(applied.stderr or "checkpoint worktree patch failed")
                baseline = {item["path"] for item in checkpoint.metadata["untracked"]}
                current = self.runner(["git", "ls-files", "--others", "--exclude-standard", "-z"],
                                      cwd=root, capture_output=True, text=True, check=False)
                if current.returncode:
                    raise RuntimeError(current.stderr or "cannot inspect untracked files")
                quarantine = storage / "rollback-quarantine"
                for name in [item for item in current.stdout.split("\0") if item and item not in baseline]:
                    relative = self._safe_relative(name)
                    source = (root / relative).resolve(strict=True)
                    destination = quarantine / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), destination)
                source_root = storage / "untracked"
                files = checkpoint.metadata["untracked"]
            else:
                source_root = storage / "files"
                files = checkpoint.metadata["files"]
            for item in files:
                relative = self._safe_relative(item["path"])
                source = source_root / relative
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".tars-restore")
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
                if _hash(destination) != item["sha256"]:
                    raise RuntimeError(f"restored file failed verification: {destination}")
                restored.append(str(destination))
            stamp = now_utc()
            with transaction(immediate=True) as conn:
                conn.execute("UPDATE workspace_checkpoints SET state='restored',restored_at=? "
                             "WHERE id=?", (stamp, checkpoint.id))
            data = {"checkpoint_id": checkpoint.id, "restored": True,
                    "files": restored, "external_effects_reversible": False,
                    "safety_checkpoint_id": safety.data["checkpoint_id"],
                    "quarantine_ref": str(storage / "rollback-quarantine")
                    if checkpoint.kind == "git" else ""}
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=data)
        evidence = self.runtime.evidence("workspace_rollback", root, repr(data),
                                         task_id=task_id, event_uuid=actions[0].event_uuid,
                                         result_ref=checkpoint.id)
        return ToolResult("workspace.rollback", "succeeded", data,
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))
