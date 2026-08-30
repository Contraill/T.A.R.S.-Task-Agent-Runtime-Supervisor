from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

from .policy import ScopeRequest, canonical_path, normalize_network_target
from .tool_core import ToolResult, ToolRuntime


class GitTools:
    def __init__(self, repository, *, runtime=None, runner=subprocess.run):
        self.repository = canonical_path(repository)
        if not (Path(self.repository) / ".git").exists():
            raise ValueError(f"not a Git repository: {self.repository}")
        self.runtime = runtime or ToolRuntime()
        self.runner = runner

    def _git(self, arguments):
        return self.runner(
            ["git", *arguments], cwd=self.repository, capture_output=True,
            text=True, check=False,
        )

    def _state(self):
        head = self._git(["rev-parse", "HEAD"])
        branch = self._git(["branch", "--show-current"])
        status = self._git(["status", "--porcelain"])
        return {"head": head.stdout.strip() if head.returncode == 0 else None,
                "branch": branch.stdout.strip(), "dirty": bool(status.stdout),
                "status": status.stdout}

    def _run(self, tool, arguments, *, effect="read", approval_id=None,
             task_id=None, session_id=None, destructive=False, evidence_type="git"):
        request = ScopeRequest(
            tool, effect, self.repository, {"argv": ["git", *arguments]},
            task_id=task_id, session_id=session_id, allowed_paths=(self.repository,),
            destructive=destructive,
        )
        actions = self.runtime.authorize((("action", request),), {"action": approval_id})
        before = self._state()
        proc = self._git(arguments)
        after = self._state()
        data = {"exit_code": proc.returncode, "stdout": proc.stdout,
                "stderr": proc.stderr, "before": before, "after": after}
        state = "succeeded" if proc.returncode == 0 else "failed"
        self.runtime.finish(actions, state=state, result=data)
        evidence = self.runtime.evidence(
            evidence_type, self.repository, repr(data), task_id=task_id,
            event_uuid=actions[0].event_uuid, metadata={"tool": tool},
        )
        return ToolResult(tool, state, data, error=proc.stderr if proc.returncode else "",
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))

    def status(self, **kwargs): return self._run("git.status", ["status", "--short", "--branch"], **kwargs)
    def diff(self, *, staged=False, **kwargs): return self._run("git.diff", ["diff", *( ["--cached"] if staged else [])], **kwargs)
    def log(self, *, limit=20, **kwargs): return self._run("git.log", ["log", f"-{int(limit)}", "--format=%H%x09%an%x09%ae%x09%s"], **kwargs)
    def show(self, revision="HEAD", **kwargs): return self._run("git.show", ["show", "--stat", "--oneline", revision], **kwargs)
    def branch(self, name, *, approval_id=None, **kwargs): return self._run("git.branch", ["branch", name], effect="write", approval_id=approval_id, **kwargs)
    def switch(self, name, *, create=False, approval_id=None, **kwargs): return self._run("git.switch", ["switch", *( ["-c"] if create else []), name], effect="write", approval_id=approval_id, **kwargs)

    def commit(self, message, *, approval_id=None, **kwargs):
        result = self._run("git.commit", ["commit", "-m", message], effect="write",
                           approval_id=approval_id, **kwargs)
        if result.succeeded:
            metadata = self._git(["show", "-s", "--format=%H%n%an%n%ae%n%cn%n%ce", "HEAD"])
            values = metadata.stdout.splitlines()
            result.data["commit"] = dict(zip(
                ("sha", "author_name", "author_email", "committer_name", "committer_email"), values
            ))
        return result

    def checkpoint(self, **kwargs):
        request = ScopeRequest(
            "git.checkpoint", "read", self.repository,
            task_id=kwargs.get("task_id"), session_id=kwargs.get("session_id"),
            allowed_paths=(self.repository,),
        )
        actions = self.runtime.authorize((("read", request),))
        state = self._state()
        proc = self._git(["diff", "--binary", "HEAD"])
        diff = proc.stdout
        data = state | {"diff_sha256": hashlib.sha256(diff.encode()).hexdigest(), "diff": diff,
                        "exit_code": proc.returncode, "stderr": proc.stderr}
        result_state = "succeeded" if proc.returncode == 0 else "failed"
        self.runtime.finish(actions, state=result_state, result=data)
        evidence = self.runtime.evidence(
            "git", self.repository, diff, task_id=kwargs.get("task_id"),
            event_uuid=actions[0].event_uuid, metadata={"checkpoint_head": state["head"]},
        )
        return ToolResult("git.checkpoint", result_state, data,
                          error=proc.stderr if proc.returncode else "",
                          action_ids=tuple(action.id for action in actions),
                          evidence_ids=(evidence.id,))

    def rollback(self, revision, *, approval_id=None, **kwargs):
        return self._run("git.rollback", ["reset", "--hard", revision], effect="destructive",
                         destructive=True, approval_id=approval_id, **kwargs)

    def push(self, remote="origin", branch="HEAD", *, approval_ids=None, **kwargs):
        remote_url = self._git(["remote", "get-url", remote])
        if remote_url.returncode != 0:
            return ToolResult("git.push", "failed", {"exit_code": remote_url.returncode},
                              error=remote_url.stderr)
        url = remote_url.stdout.strip()
        if url.startswith("git@"):
            host = url.split("@", 1)[1].split(":", 1)[0]
            url = "https://" + host
        url = normalize_network_target(url, resolve_dns=True)[0]
        requests = (
            ("write", ScopeRequest("git.push", "destructive", self.repository,
                                   {"remote": remote, "branch": branch},
                                   allowed_paths=(self.repository,), destructive=True,
                                   task_id=kwargs.get("task_id"), session_id=kwargs.get("session_id"))),
            ("network", ScopeRequest("git.push", "network", url,
                                     {"remote": remote, "branch": branch},
                                     task_id=kwargs.get("task_id"), session_id=kwargs.get("session_id"))),
        )
        actions = self.runtime.authorize(requests, approval_ids)
        before = self._state()
        proc = self._git(["push", remote, branch])
        after = self._state()
        data = {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                "remote": remote, "branch": branch, "before": before, "after": after}
        state = "succeeded" if proc.returncode == 0 else "failed"
        self.runtime.finish(actions, state=state, result=data)
        evidence = self.runtime.evidence(
            "git", self.repository, repr(data), task_id=kwargs.get("task_id"),
            event_uuid=actions[0].event_uuid, metadata={"tool": "git.push"},
        )
        return ToolResult("git.push", state, data, error=proc.stderr if proc.returncode else "",
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))
