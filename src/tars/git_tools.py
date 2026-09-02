from __future__ import annotations

import hashlib
import os
import re
import subprocess

from .network import network_destination
from .policy import ScopeRequest, canonical_path
from .secure_paths import AnchoredRoot
from .tool_core import ToolResult, ToolRuntime


_REMOTE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}\Z")


def _git_ref(value, *, label="Git ref"):
    value = str(value)
    forbidden = ("..", "@{", "\\", "~", "^", ":", "?", "*", "[", "]")
    components = value.split("/")
    if (
        not value or value.startswith("-") or value.startswith("/") or value.endswith(("/", "."))
        or any(ord(char) <= 32 or ord(char) == 127 for char in value)
        or any(token in value for token in forbidden)
        or any(not part or part.startswith(".") or part.endswith(".lock") for part in components)
    ):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _remote_name(value):
    value = str(value)
    if not _REMOTE_NAME.fullmatch(value) or value.startswith("-") or ".." in value:
        raise ValueError(f"invalid Git remote name: {value!r}")
    return value


class GitTools:
    def __init__(self, repository, *, runtime=None, runner=subprocess.run):
        self.repository = canonical_path(repository)
        self._anchor = AnchoredRoot(self.repository)
        try:
            self._anchor.lstat((".git",))
        except FileNotFoundError:
            self._anchor.close()
            raise ValueError(f"not a Git repository: {self.repository}")
        self._repository_fd = self._anchor.open_directory()
        self.runtime = runtime or ToolRuntime()
        self.runner = runner

    def _git(self, arguments):
        return self.runner(
            ["git", *arguments], cwd=f"/proc/self/fd/{self._repository_fd}",
            capture_output=True, text=True, check=False,
            pass_fds=(self._repository_fd,),
        )

    def close(self):
        if self._repository_fd >= 0:
            os.close(self._repository_fd)
            self._repository_fd = -1
        self._anchor.close()

    def __del__(self):
        self.close()

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
    def show(self, revision="HEAD", **kwargs):
        return self._run("git.show", ["show", "--stat", "--oneline", _git_ref(revision)], **kwargs)

    def branch(self, name, *, approval_id=None, **kwargs):
        return self._run("git.branch", ["branch", _git_ref(name, label="branch name")],
                         effect="write", approval_id=approval_id, **kwargs)

    def switch(self, name, *, create=False, approval_id=None, **kwargs):
        return self._run("git.switch", ["switch", *(["-c"] if create else []),
                                        _git_ref(name, label="branch name")],
                         effect="write", approval_id=approval_id, **kwargs)

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
        return self._run("git.rollback", ["reset", "--hard", _git_ref(revision)], effect="destructive",
                         destructive=True, approval_id=approval_id, **kwargs)

    def push(self, remote="origin", branch="HEAD", *, approval_ids=None, **kwargs):
        remote = _remote_name(remote)
        branch = _git_ref(branch, label="branch name")
        remote_url = self._git(["remote", "get-url", "--push", "--all", remote])
        if remote_url.returncode != 0:
            return ToolResult("git.push", "failed", {"exit_code": remote_url.returncode},
                              error=remote_url.stderr)
        urls = tuple(value for value in remote_url.stdout.splitlines() if value.strip())
        if len(urls) != 1:
            raise PermissionError("Git push requires exactly one explicit remote destination")
        destination = network_destination(urls[0], resolve_dns=True)
        authority = {
            "remote": remote, "branch": branch, "origin": destination.origin,
            "url_sha256": destination.url_sha256,
        }
        requests = (
            ("write", ScopeRequest("git.push", "destructive", self.repository,
                                   authority,
                                   allowed_paths=(self.repository,), destructive=True,
                                   task_id=kwargs.get("task_id"), session_id=kwargs.get("session_id"))),
            ("network", ScopeRequest("git.push", "network", destination.policy_url,
                                     authority,
                                     task_id=kwargs.get("task_id"), session_id=kwargs.get("session_id"))),
        )
        actions = self.runtime.authorize(requests, approval_ids)
        before = self._state()
        resolver_options = ["-c", "http.curloptResolve="]
        for address in destination.addresses:
            pinned = f"[{address}]" if ":" in address else address
            value = f"+{destination.host}:{destination.port}:{pinned}"
            resolver_options.extend(("-c", f"http.curloptResolve={value}"))
        proc = self._git([
            "-c", "http.proxy=", "-c", "http.followRedirects=false",
            *resolver_options,
            "push", "--", destination.request_url, branch,
        ])
        after = self._state()
        data = {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                "remote": remote, "branch": branch, "origin": destination.origin,
                "before": before, "after": after}
        state = "succeeded" if proc.returncode == 0 else "failed"
        self.runtime.finish(actions, state=state, result=data)
        evidence = self.runtime.evidence(
            "git", self.repository, repr(data), task_id=kwargs.get("task_id"),
            event_uuid=actions[0].event_uuid, metadata={"tool": "git.push"},
        )
        return ToolResult("git.push", state, data, error=proc.stderr if proc.returncode else "",
                          action_ids=tuple(a.id for a in actions), evidence_ids=(evidence.id,))
