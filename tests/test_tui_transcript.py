import base64
import inspect
from types import SimpleNamespace

from prompt_toolkit.selection import SelectionState

from tars import chat_tui
from tars.transcript import (
    EntryKind,
    TranscriptModel,
    configured_display_name,
)


def test_rail_renderer_distinguishes_entries_without_color():
    transcript = TranscriptModel("contrail")
    transcript.append(EntryKind.USER, "hello\nthere")
    assistant = transcript.start_assistant("General")
    transcript.stream(assistant, "working")
    transcript.append(EntryKind.SYSTEM, "model loaded")
    transcript.append(EntryKind.TOOL, "", label="git.diff", detail="completed · 180 ms")
    transcript.append(EntryKind.ERROR, "useful error")

    rendered = transcript.render()
    assert "contrail\n┃ hello\n┃ there" in rendered
    assert "T.A.R.S. · General …\n│ working" in rendered
    assert "· model loaded" in rendered
    assert "├─ git.diff\n│ completed · 180 ms" in rendered
    assert "! useful error" in rendered
    transcript.finish(assistant)
    assert "General …" not in transcript.render()


def test_display_name_prefers_ui_override_then_os_user(monkeypatch):
    monkeypatch.setattr("tars.transcript.getpass.getuser", lambda: "local-user")
    assert configured_display_name({}) == "local-user"
    assert configured_display_name({"chat": {"display_name": "Chat Name"}}) == "Chat Name"
    assert configured_display_name({"ui": {"display_name": "İzzet"},
                                    "chat": {"display_name": "Other"}}) == "İzzet"


def test_stream_updates_same_entry_instead_of_duplicating():
    transcript = TranscriptModel("user")
    entry = transcript.start_assistant("General")
    transcript.stream(entry, "one")
    transcript.stream(entry, " two")
    transcript.finish(entry)
    assert len(transcript.entries) == 1
    assert transcript.entries[0].text == "one two"
    assert transcript.render().count("T.A.R.S. · General") == 1


def test_render_preserves_selection_and_stops_following(monkeypatch):
    ui = object.__new__(chat_tui.ChatTUI)
    ui.transcript = TranscriptModel("user")
    ui.transcript.append(EntryKind.USER, "selected words")
    ui.output = chat_tui.TextArea(text="", read_only=True, focusable=True)
    ui.app = SimpleNamespace(invalidate=lambda: None)
    ui._follow_output = True
    ui._new_output_below = False
    ui._render_output()
    ui.output.buffer.cursor_position = 8
    ui.output.buffer.selection_state = SelectionState(original_cursor_position=2)
    ui._follow_output = False
    ui.transcript.append(EntryKind.SYSTEM, "new output")
    ui._render_output(streamed=True)
    assert ui.output.buffer.cursor_position == 8
    assert ui.output.buffer.selection_state.original_cursor_position == 2
    assert ui._new_output_below


def test_stream_flush_is_chunk_buffered():
    source = inspect.getsource(chat_tui.ChatTUI._consume_stream_event)
    assert "call_later" in source and "0.04" in source


def test_terminal_clipboard_is_bounded_write_only_osc52():
    class Output:
        def __init__(self):
            self.raw = ""
            self.flushed = False

        def write_raw(self, value):
            self.raw += value

        def flush(self):
            self.flushed = True

    output = Output()
    chat_tui.write_terminal_clipboard(output, "selected text")
    assert output.raw.startswith("\x1b]52;c;") and output.raw.endswith("\x07")
    encoded = output.raw.removeprefix("\x1b]52;c;").removesuffix("\x07")
    assert base64.b64decode(encoded).decode() == "selected text"
    assert output.flushed and not hasattr(output, "read")
    with __import__("pytest").raises(ValueError, match="too large"):
        chat_tui.write_terminal_clipboard(output, "12345", max_bytes=4)


def test_tui_temporary_path_uses_buffered_transcript_stream():
    source = inspect.getsource(chat_tui.ChatTUI._stream_temporary)
    assert "self.temporary.stream" in source
    assert "self._consume_stream_event(payload)" in source
