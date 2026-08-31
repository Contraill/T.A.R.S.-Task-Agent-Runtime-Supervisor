from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess

from .artifact_tools import ArtifactRuntime
from .policy import ScopeRequest
from .tool_core import ToolResult, ToolRuntime


class NotificationTools:
    def __init__(self, *, runtime=None, runner=subprocess.run):
        self.runtime = runtime or ToolRuntime()
        self.runner = runner
        self.binary = shutil.which("notify-send")

    def status(self):
        return {"available": bool(self.binary), "backend": "notify-send" if self.binary else None}

    def send(self, summary, body="", *, urgency="normal", approval_id=None,
             task_id=None, session_id=None):
        if urgency not in {"low", "normal", "critical"}:
            raise ValueError("invalid notification urgency")
        request = ScopeRequest("notify.send", "write", "desktop-notification",
                               {"summary": summary, "body": body,
                                "urgency": urgency}, task_id=task_id, session_id=session_id)
        actions = self.runtime.authorize((("write", request),), {"write": approval_id})
        if not self.binary:
            data = {"available": False, "reason": "notify-send unavailable"}
            self.runtime.finish(actions, state="failed", result=data)
            return ToolResult("notify.send", "unavailable", data, data["reason"],
                              action_ids=tuple(action.id for action in actions))
        proc = self.runner([self.binary, "--urgency", urgency, summary, body],
                           capture_output=True, text=True, check=False)
        data = {"backend": "notify-send", "exit_code": proc.returncode,
                "delivered": proc.returncode == 0, "stderr": proc.stderr}
        state = "succeeded" if proc.returncode == 0 else "failed"
        self.runtime.finish(actions, state=state, result=data)
        return ToolResult("notify.send", state, data, proc.stderr if proc.returncode else "",
                          action_ids=tuple(action.id for action in actions))


class ScreenCaptureTools:
    def __init__(self, roots, *, runtime=None, runner=subprocess.run):
        self.artifacts = ArtifactRuntime(roots, runtime=runtime)
        self.runner = runner

    @staticmethod
    def status():
        if shutil.which("spectacle"):
            return {"available": True, "backend": "spectacle", "portal_aware": True}
        if shutil.which("grim"):
            return {"available": True, "backend": "grim", "portal_aware": False}
        if shutil.which("gnome-screenshot"):
            return {"available": True, "backend": "gnome-screenshot", "portal_aware": True}
        return {"available": False, "backend": None, "portal_aware": False}

    def capture(self, output, *, approval_id=None, task_id=None, session_id=None):
        actions, _, writes = self.artifacts.authorize(
            "screen.capture", (), (output,), approval_ids={"write-0": approval_id},
            task_id=task_id, session_id=session_id,
        )
        destination = writes[0]
        status = self.status()
        if not status["available"]:
            data = status | {"reason": "screen capture backend unavailable"}
            return self.artifacts.result("screen.capture", actions, data,
                                         state="unavailable", error=data["reason"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        commands = {
            "spectacle": ["spectacle", "--background", "--nonotify", "--fullscreen",
                          "--output", str(destination)],
            "grim": ["grim", str(destination)],
            "gnome-screenshot": ["gnome-screenshot", "--file", str(destination)],
        }
        proc = self.runner(commands[status["backend"]], capture_output=True, text=True, check=False)
        verified = proc.returncode == 0 and destination.is_file() and destination.stat().st_size > 0
        data = status | {"path": str(destination), "exit_code": proc.returncode,
                         "verified": verified, "stderr": proc.stderr}
        if verified:
            data.update({"bytes": destination.stat().st_size,
                         "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
        return self.artifacts.result("screen.capture", actions, data, task_id=task_id,
                                     evidence_source=destination if verified else None,
                                     state="succeeded" if verified else "failed",
                                     error="" if verified else proc.stderr or "capture was not created")
