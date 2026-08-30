from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

from .policy import ScopeRequest, canonical_path
from .tool_core import ToolResult, ToolRuntime


class FilesystemTools:
    def __init__(self, roots, *, runtime=None):
        self.roots = tuple(canonical_path(root) for root in roots)
        if not self.roots:
            raise ValueError("filesystem tools require at least one scope root")
        self.runtime = runtime or ToolRuntime()

    def _request(self, tool, effect, target, *, arguments=None, task_id=None, session_id=None,
                 destructive=False):
        return ScopeRequest(
            tool, effect, target, arguments or {}, task_id=task_id, session_id=session_id,
            allowed_paths=self.roots, destructive=destructive,
        )

    def _read_path(self, tool, path, *, task_id=None, session_id=None):
        request = self._request(tool, "read", path, task_id=task_id, session_id=session_id)
        actions = self.runtime.authorize((("read", request),))
        return Path(canonical_path(path)), actions

    def list(self, path, *, task_id=None, session_id=None):
        target, actions = self._read_path("fs.list", path, task_id=task_id, session_id=session_id)
        try:
            rows = [{"name": item.name, "path": str(item), "type": (
                "symlink" if item.is_symlink() else "directory" if item.is_dir() else "file"
            )} for item in sorted(target.iterdir(), key=lambda item: item.name)]
            result = {"path": str(target), "entries": rows}
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=result)
        evidence = self.runtime.evidence("filesystem", target, repr(rows), task_id=task_id,
                                         event_uuid=actions[0].event_uuid)
        return ToolResult("fs.list", "succeeded", result,
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def stat(self, path, *, task_id=None, session_id=None):
        target, actions = self._read_path("fs.stat", path, task_id=task_id, session_id=session_id)
        try:
            value = target.lstat()
            result = {"path": str(target), "size": value.st_size, "mode": value.st_mode,
                      "mtime_ns": value.st_mtime_ns, "is_file": target.is_file(),
                      "is_dir": target.is_dir(), "is_symlink": target.is_symlink()}
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=result)
        evidence = self.runtime.evidence("filesystem", target, repr(result), task_id=task_id,
                                         event_uuid=actions[0].event_uuid)
        return ToolResult("fs.stat", "succeeded", result,
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def read(self, path, *, offset=0, limit=1_000_000, task_id=None, session_id=None):
        if offset < 0 or limit <= 0:
            raise ValueError("invalid read bounds")
        target, actions = self._read_path("fs.read", path, task_id=task_id, session_id=session_id)
        try:
            with target.open("rb") as handle:
                handle.seek(offset)
                payload = handle.read(limit + 1)
            truncated = len(payload) > limit
            payload = payload[:limit]
            text = payload.decode("utf-8", errors="replace")
            result = {"path": str(target), "offset": offset, "content": text,
                      "bytes": len(payload), "truncated": truncated,
                      "sha256": hashlib.sha256(payload).hexdigest()}
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=result)
        evidence = self.runtime.evidence("filesystem", target, payload, task_id=task_id,
                                         event_uuid=actions[0].event_uuid,
                                         metadata={"offset": offset, "truncated": truncated})
        return ToolResult("fs.read", "succeeded", result,
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def search(self, root, query, *, limit=100, task_id=None, session_id=None):
        target, actions = self._read_path("fs.search", root, task_id=task_id, session_id=session_id)
        hits = []
        try:
            for path in sorted(target.rglob("*")):
                if len(hits) >= limit or not path.is_file() or path.is_symlink():
                    continue
                try:
                    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                        if query.casefold() in line.casefold():
                            hits.append({"path": str(path), "line": number, "text": line})
                            if len(hits) >= limit:
                                break
                except OSError:
                    continue
            result = {"root": str(target), "query": query, "hits": hits,
                      "truncated": len(hits) >= limit}
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=result)
        evidence = self.runtime.evidence("filesystem", target, repr(hits), task_id=task_id,
                                         event_uuid=actions[0].event_uuid)
        return ToolResult("fs.search", "succeeded", result,
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def _mutate(self, tool, target, operation, *, arguments=None, approval_id=None,
                task_id=None, session_id=None, destructive=False):
        request = self._request(
            tool, "destructive" if destructive else "write", target,
            arguments=arguments, task_id=task_id, session_id=session_id,
            destructive=destructive,
        )
        actions = self.runtime.authorize((("write", request),), {"write": approval_id})
        try:
            result = operation(Path(canonical_path(target)))
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=result)
        evidence = self.runtime.evidence("filesystem", target, repr(result), task_id=task_id,
                                         event_uuid=actions[0].event_uuid)
        return ToolResult(tool, "succeeded", result,
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def mkdir(self, path, *, parents=False, approval_id=None, task_id=None, session_id=None):
        def operation(target):
            target.mkdir(parents=parents, exist_ok=False)
            return {"path": str(target), "created": True}
        return self._mutate("fs.mkdir", path, operation, arguments={"parents": parents},
                            approval_id=approval_id, task_id=task_id, session_id=session_id)

    def write(self, path, content, *, create=True, approval_id=None, task_id=None, session_id=None):
        payload = content.encode("utf-8")
        def operation(target):
            if not create and not target.exists():
                raise FileNotFoundError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            return {"path": str(target), "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest()}
        return self._mutate("fs.write", path, operation, arguments={"bytes": len(payload)},
                            approval_id=approval_id, task_id=task_id, session_id=session_id)

    def patch(self, path, replacements, *, approval_id=None, task_id=None, session_id=None):
        def operation(target):
            original = target.read_text()
            updated = original
            applied = []
            for old, new in replacements:
                count = updated.count(old)
                if count != 1:
                    raise ValueError(f"patch context must match exactly once; matched {count}")
                updated = updated.replace(old, new, 1)
                applied.append(hashlib.sha256(old.encode()).hexdigest())
            target.write_text(updated)
            return {"path": str(target), "replacements": len(applied),
                    "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
                    "after_sha256": hashlib.sha256(updated.encode()).hexdigest()}
        return self._mutate("fs.patch", path, operation,
                            arguments={"replacements": len(replacements)},
                            approval_id=approval_id, task_id=task_id, session_id=session_id)

    def copy(self, source, destination, *, approval_ids=None, task_id=None, session_id=None):
        requests = (
            ("read", self._request("fs.copy", "read", source, task_id=task_id, session_id=session_id)),
            ("write", self._request("fs.copy", "write", destination, task_id=task_id, session_id=session_id)),
        )
        actions = self.runtime.authorize(requests, approval_ids)
        try:
            src, dst = Path(canonical_path(source)), Path(canonical_path(destination))
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            result = {"source": str(src), "destination": str(dst)}
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=result)
        return ToolResult("fs.copy", "succeeded", result,
                          action_ids=tuple(a.id for a in actions))

    def move(self, source, destination, *, approval_ids=None, task_id=None, session_id=None):
        requests = (
            ("source", self._request("fs.move", "write", source, task_id=task_id, session_id=session_id)),
            ("destination", self._request("fs.move", "write", destination, task_id=task_id, session_id=session_id)),
        )
        actions = self.runtime.authorize(requests, approval_ids)
        try:
            src, dst = Path(canonical_path(source)), Path(canonical_path(destination))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            result = {"source": str(src), "destination": str(dst)}
        except Exception as exc:
            self.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        self.runtime.finish(actions, state="succeeded", result=result)
        return ToolResult("fs.move", "succeeded", result,
                          action_ids=tuple(a.id for a in actions))

    def delete(self, path, *, recursive=False, approval_id=None, task_id=None, session_id=None):
        def operation(target):
            if target.is_dir() and not target.is_symlink():
                if not recursive:
                    target.rmdir()
                else:
                    shutil.rmtree(target)
            else:
                target.unlink()
            return {"path": str(target), "deleted": True, "recursive": recursive}
        return self._mutate("fs.delete", path, operation,
                            arguments={"recursive": recursive}, destructive=True,
                            approval_id=approval_id, task_id=task_id, session_id=session_id)
