from __future__ import annotations

from .state_events import read_state_events

REASONING_VISIBILITY = {"hidden", "summary", "raw"}


def reasoning_view(mode: str, *, emitted_raw="", summary="") -> str:
    mode = mode.lower()
    if mode not in REASONING_VISIBILITY:
        raise ValueError(f"invalid reasoning visibility: {mode}")
    if mode == "hidden":
        return ""
    if mode == "summary":
        return str(summary or "")
    return str(emitted_raw or "")


def activity_trace(*, session_id=None, task_id=None, after_id=0, limit=200):
    return [
        {
            "id": event.id, "type": event.type, "timestamp": event.timestamp,
            "role": event.source_role, "message": event.message,
            "payload": event.payload,
        }
        for event in read_state_events(
            session_id=session_id, task_id=task_id, after_id=after_id, limit=limit
        )
    ]
