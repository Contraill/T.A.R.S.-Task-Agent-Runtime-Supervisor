from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import getpass
from typing import Mapping


class EntryKind(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    ERROR = "error"
    REASONING = "reasoning"


@dataclass
class TranscriptEntry:
    kind: EntryKind
    text: str
    label: str = ""
    detail: str = ""
    streaming: bool = False
    id: int = 0


@dataclass
class TranscriptModel:
    """Append-only TUI projection; durable conversation state remains canonical."""

    display_name: str
    entries: list[TranscriptEntry] = field(default_factory=list)
    _next_id: int = 1

    def append(self, kind: EntryKind, text: str, *, label: str = "",
               detail: str = "", streaming: bool = False) -> TranscriptEntry:
        entry = TranscriptEntry(kind, str(text), label, detail, streaming, self._next_id)
        self._next_id += 1
        self.entries.append(entry)
        return entry

    def start_assistant(self, label: str, *, detail: str = "") -> TranscriptEntry:
        return self.append(EntryKind.ASSISTANT, "", label=label, detail=detail,
                           streaming=True)

    def stream(self, entry: TranscriptEntry, chunk: str) -> None:
        if entry not in self.entries or not entry.streaming:
            raise ValueError("stream target is not an active transcript entry")
        entry.text += str(chunk)

    def finish(self, entry: TranscriptEntry, fallback="[no final content]") -> None:
        if entry not in self.entries:
            raise ValueError("unknown transcript entry")
        if not entry.text:
            entry.text = fallback
        entry.streaming = False

    def discard(self, entry: TranscriptEntry) -> None:
        if entry in self.entries:
            self.entries.remove(entry)

    def render(self) -> str:
        blocks = [render_entry(entry, self.display_name) for entry in self.entries]
        return "\n\n".join(block for block in blocks if block)


def configured_display_name(cfg: Mapping | None) -> str:
    cfg = cfg or {}
    ui = cfg.get("ui", {}) if isinstance(cfg, Mapping) else {}
    chat = cfg.get("chat", {}) if isinstance(cfg, Mapping) else {}
    override = (ui.get("display_name") if isinstance(ui, Mapping) else None) or (
        chat.get("display_name") if isinstance(chat, Mapping) else None)
    return str(override).strip() if override and str(override).strip() else getpass.getuser()


def _railed(text: str, rail: str) -> str:
    lines = str(text).rstrip().splitlines() or [""]
    return "\n".join(f"{rail} {line}".rstrip() for line in lines)


def render_entry(entry: TranscriptEntry, display_name: str) -> str:
    text = entry.text.rstrip()
    if entry.kind == EntryKind.USER:
        return f"{entry.label or display_name}\n{_railed(text, '┃')}"
    if entry.kind == EntryKind.ASSISTANT:
        label = entry.label or "T.A.R.S."
        suffix = f" · {entry.detail}" if entry.detail else ""
        marker = " …" if entry.streaming else ""
        return f"T.A.R.S. · {label}{suffix}{marker}\n{_railed(text, '│')}"
    if entry.kind == EntryKind.TOOL:
        body = entry.detail or text
        detail = f"\n{_railed(body, '│')}" if body else ""
        return f"├─ {entry.label or 'tool'}{detail}"
    if entry.kind == EntryKind.ERROR:
        return _railed(text, "!")
    if entry.kind == EntryKind.REASONING:
        return f"{entry.label or 'Raw reasoning'}\n{_railed(text, '┊')}"
    return _railed(text, "·")
