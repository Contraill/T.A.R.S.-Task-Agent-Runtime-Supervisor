from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import uuid

from .config import STATE_ROOT
from .policy import ScopeRequest, canonical_path
from .secure_paths import AnchoredRoot, select_anchor
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


class WorkspaceRecovery:
    def __init__(self, roots, *, runtime=None, storage_root=None, runner=subprocess.run,
                 max_files=10_000, max_bytes=1_000_000_000):
        self.roots = tuple(canonical_path(root) for root in roots)
        if not self.roots:
            raise ValueError("workspace recovery requires at least one scope root")
        self.runtime = runtime or ToolRuntime()
        self.storage_root = Path(storage_root or (STATE_ROOT / "workspace-checkpoints"))
        self.storage_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.storage_root.chmod(0o700)
        self.storage_root = self.storage_root.resolve(strict=True)
        self._anchors = tuple(AnchoredRoot(root) for root in self.roots)
        self._storage_anchor = AnchoredRoot(self.storage_root)
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
        self._storage_anchor.mkdir((checkpoint_id,))
        return path

    def _root(self, target):
        return select_anchor(self._anchors, target)

    def _remove_storage(self, checkpoint_id):
        try:
            self._storage_anchor.delete((checkpoint_id,), recursive=True)
        except FileNotFoundError:
            pass

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
        root = Path(os.path.abspath(os.fspath(repository)))
        actions = self._authorize("workspace.checkpoint", "read", root, task_id=task_id)
        checkpoint_id = "wcp-" + uuid.uuid4().hex
        storage = self._storage(checkpoint_id)
        root_fd = -1
        try:
            root_anchor, root_parts, root = self._root(root)
            root_fd = root_anchor.open_directory(root_parts)
            def git(arguments):
                return self.runner(
                    ["git", *arguments], cwd=f"/proc/self/fd/{root_fd}",
                    capture_output=True, text=True, check=False,
                    pass_fds=(root_fd,),
                )
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
            self._storage_anchor.atomic_write(
                (checkpoint_id, "staged.patch"), staged.stdout.encode()
            )
            self._storage_anchor.atomic_write(
                (checkpoint_id, "unstaged.patch"), unstaged.stdout.encode()
            )
            untracked_output = git(["ls-files", "--others", "--exclude-standard", "-z"])
            if untracked_output.returncode:
                raise RuntimeError(untracked_output.stderr or "cannot list untracked files")
            untracked = [item for item in untracked_output.stdout.split("\0") if item]
            files = []
            total = 0
            for name in untracked:
                relative = self._safe_relative(name)
                parts = root_parts + relative.parts
                try:
                    source_fd = root_anchor.open(parts, os.O_RDONLY)
                except OSError as exc:
                    raise ValueError(f"unsupported untracked entry: {name}") from exc
                value = os.fstat(source_fd)
                if not stat.S_ISREG(value.st_mode):
                    os.close(source_fd)
                    raise ValueError(f"unsupported untracked entry: {name}")
                size = value.st_size
                total += size
                if len(files) >= self.max_files or total > self.max_bytes:
                    os.close(source_fd)
                    raise ValueError("workspace checkpoint bounds exceeded")
                try:
                    digest, verified_size = self._storage_anchor.copy_fd_to(
                        source_fd, (checkpoint_id, "untracked", *relative.parts)
                    )
                finally:
                    os.close(source_fd)
                if verified_size != size:
                    raise RuntimeError(f"untracked file changed while captured: {name}")
                files.append({"path": relative.as_posix(), "bytes": size,
                              "sha256": digest})
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
            self._remove_storage(checkpoint_id)
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        finally:
            if root_fd >= 0:
                os.close(root_fd)
        self.runtime.finish(actions, state="succeeded", result=data)
        evidence = self.runtime.evidence("workspace_checkpoint", root, repr(data),
                                         task_id=task_id, event_uuid=actions[0].event_uuid,
                                         result_ref=checkpoint.id)
        return ToolResult("workspace.checkpoint", "succeeded", data,
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def create_filesystem(self, root, paths, *, task_id=None):
        workspace = Path(os.path.abspath(os.fspath(root)))
        actions = self._authorize("workspace.checkpoint", "read", workspace, task_id=task_id)
        checkpoint_id = "wcp-" + uuid.uuid4().hex
        storage = self._storage(checkpoint_id)
        try:
            workspace_anchor, workspace_parts, workspace = self._root(workspace)
            files = []
            total = 0
            for value in paths:
                entry = Path(os.path.abspath(os.fspath(value)))
                try:
                    item_relative = entry.relative_to(workspace)
                except ValueError as exc:
                    raise PermissionError(
                        f"checkpoint path is outside the workspace: {entry}"
                    ) from exc
                entry_parts = workspace_parts + item_relative.parts
                try:
                    entry_mode = workspace_anchor.lstat(entry_parts).st_mode
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"checkpoint path does not exist: {entry}"
                    ) from exc
                if stat.S_ISREG(entry_mode):
                    source_fd = workspace_anchor.open(entry_parts, os.O_RDONLY)
                    candidates = ((item_relative.parts, source_fd),)
                elif stat.S_ISDIR(entry_mode):
                    candidates = ((relative[len(workspace_parts):], fd)
                                  for relative, fd in workspace_anchor.walk_files(
                                      entry_parts, reject_symlinks=True))
                else:
                    raise ValueError(f"filesystem snapshots do not follow symlinks: {entry}")
                for relative_parts, source_fd in candidates:
                    size = os.fstat(source_fd).st_size
                    total += size
                    if len(files) >= self.max_files or total > self.max_bytes:
                        if stat.S_ISREG(entry_mode):
                            os.close(source_fd)
                        raise ValueError("workspace checkpoint bounds exceeded")
                    digest, verified_size = self._storage_anchor.copy_fd_to(
                        source_fd, (checkpoint_id, "files", *relative_parts)
                    )
                    if stat.S_ISREG(entry_mode):
                        os.close(source_fd)
                    if verified_size != size:
                        raise RuntimeError("workspace file changed while captured")
                    relative_path = Path(*relative_parts)
                    files.append({"path": relative_path.as_posix(), "bytes": size,
                                  "sha256": digest})
            metadata = {"files": files, "reversible_scope": "captured-files-only",
                        "external_effects_reversible": False}
            checkpoint = self._insert(checkpoint_id, task_id, "filesystem", workspace,
                                      storage, metadata)
            data = {"checkpoint_id": checkpoint.id, "kind": "filesystem",
                    "root": str(workspace), "files": len(files), "bytes": total,
                    "external_effects_reversible": False}
        except Exception as exc:
            self._remove_storage(checkpoint_id)
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
        root_anchor, root_parts, root = self._root(root)
        changes = []
        if checkpoint.kind == "git":
            root_fd = root_anchor.open_directory(root_parts)
            try:
                current_head = self.runner(
                    ["git", "rev-parse", "HEAD"], cwd=f"/proc/self/fd/{root_fd}",
                    capture_output=True, text=True, check=False,
                    pass_fds=(root_fd,),
                ).stdout.strip()
            finally:
                os.close(root_fd)
            changes.append({"operation": "restore_tracked_patch",
                            "supported": current_head == checkpoint.metadata["head"],
                            "reason": "same HEAD required"})
            baseline = {item["path"]: item for item in checkpoint.metadata["untracked"]}
            root_fd = root_anchor.open_directory(root_parts)
            try:
                current_untracked = self.runner(
                    ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                    cwd=f"/proc/self/fd/{root_fd}", capture_output=True, text=True,
                    check=False, pass_fds=(root_fd,),
                )
            finally:
                os.close(root_fd)
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
            try:
                target_parts = root_parts + self._safe_relative(name).parts
                current, _ = root_anchor.hash(target_parts)
            except (FileNotFoundError, OSError, ValueError):
                current = None
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
        root_fd = -1
        try:
            root_anchor, root_parts, root = self._root(root)
            if checkpoint.kind == "git":
                safety = self.create_git(root, task_id=task_id)
            else:
                safety_paths = []
                for item in checkpoint.metadata["files"]:
                    relative = self._safe_relative(item["path"])
                    try:
                        root_anchor.lstat(root_parts + relative.parts)
                    except FileNotFoundError:
                        continue
                    safety_paths.append(root / relative)
                safety = self.create_filesystem(root, safety_paths, task_id=task_id)
            if checkpoint.kind == "git":
                root_fd = root_anchor.open_directory(root_parts)

                def git(arguments, *, extra_fds=()):
                    return self.runner(
                        ["git", *arguments], cwd=f"/proc/self/fd/{root_fd}",
                        capture_output=True, text=True, check=False,
                        pass_fds=(root_fd, *extra_fds),
                    )

                restore = git(["restore", "--staged", "--worktree", "."])
                if restore.returncode:
                    raise RuntimeError(restore.stderr or "Git restore failed")
                for patch_name, indexed in (("staged.patch", True),
                                            ("unstaged.patch", False)):
                    patch_parts = (checkpoint.id, patch_name)
                    if not self._storage_anchor.stat(patch_parts).st_size:
                        continue
                    patch_fd = self._storage_anchor.open(patch_parts, os.O_RDONLY)
                    try:
                        arguments = ["apply"]
                        if indexed:
                            arguments.append("--index")
                        arguments.extend(["--binary", f"/proc/self/fd/{patch_fd}"])
                        applied = git(arguments, extra_fds=(patch_fd,))
                    finally:
                        os.close(patch_fd)
                    if applied.returncode:
                        label = "staged" if indexed else "worktree"
                        raise RuntimeError(
                            applied.stderr or f"checkpoint {label} patch failed"
                        )
                baseline = {item["path"] for item in checkpoint.metadata["untracked"]}
                current = git(["ls-files", "--others", "--exclude-standard", "-z"])
                if current.returncode:
                    raise RuntimeError(current.stderr or "cannot inspect untracked files")
                quarantine = storage / "rollback-quarantine"
                for name in [item for item in current.stdout.split("\0") if item and item not in baseline]:
                    relative = self._safe_relative(name)
                    root_anchor.rename(
                        root_parts + relative.parts, self._storage_anchor,
                        (checkpoint.id, "rollback-quarantine", *relative.parts),
                    )
                files = checkpoint.metadata["untracked"]
            else:
                files = checkpoint.metadata["files"]
            for item in files:
                relative = self._safe_relative(item["path"])
                destination = root / relative
                source_prefix = "untracked" if checkpoint.kind == "git" else "files"
                self._storage_anchor.copy_to(
                    (checkpoint.id, source_prefix, *relative.parts),
                    root_anchor, root_parts + relative.parts,
                )
                digest, _ = root_anchor.hash(root_parts + relative.parts)
                if digest != item["sha256"]:
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
        finally:
            if root_fd >= 0:
                os.close(root_fd)
        self.runtime.finish(actions, state="succeeded", result=data)
        evidence = self.runtime.evidence("workspace_rollback", root, repr(data),
                                         task_id=task_id, event_uuid=actions[0].event_uuid,
                                         result_ref=checkpoint.id)
        return ToolResult("workspace.rollback", "succeeded", data,
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))
