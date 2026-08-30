from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import textwrap
import time

from prompt_toolkit import Application
from prompt_toolkit.application import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, Dimension, DynamicContainer, FormattedTextControl, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from .calibration import get_profile
from .config import CHAT_STATE_ROOT
from .registry import get_model
from .roles import default_role_id, ensure_role_registry, get_role, list_roles, resolve_role_id
from .runtime import chat_completion_stream, runtime_models
from .context import ContextManager, latest_projection, current_context_message_seq
from .conversation import create_conversation, add_message
from .tasks import active_task, append_event, list_tasks, load_task, set_active_task
from .events import read_events, read_events_since
from .runner import create_run, run_task_epoch, request_control, list_runs
from .temporary import TEMPORARY_NOTICE, TemporarySession
from .themes import (
    VALID_LOGOS,
    current_logo,
    current_theme,
    list_themes,
    prompt_toolkit_style,
    set_logo,
    set_theme,
)


STATIC_COMMANDS = [
    ("/role <name>", "Switch role; task state stays untouched"),
    ("/<role>", "Shortcut for any registered role"),
    ("/ask <question>", "Sideband question; main context unchanged"),
    ("/run [id]", "Run one durable reasoning epoch for a task"),
    ("/pause [id]", "Request pause at next safe boundary"),
    ("/resume [id]", "Resume with a new reasoning epoch"),
    ("/cancel [id]", "Request cancellation at next safe boundary"),
    ("/task [id]", "Show active or selected task state"),
    ("/tasks", "List tasks"),
    ("/scheduled", "List scheduled/future tasks"),
    ("/status", "Show runtime, context and task status"),
    ("/context", "Show current ContextManager projection/budget"),
    ("/models", "Show role/model bindings"),
    ("/progress quiet|normal|verbose", "Set progress event visibility"),
    ("/reasoning hidden|summary|raw", "Set reasoning visibility"),
    ("/theme [name]", "Show or change UI color theme"),
    ("/logo [mode]", "Show or change HUD logo mode"),
    ("/new", "Clear conversation only; task state stays"),
    ("/temporary", "Enter or exit ephemeral conversation mode"),
    ("/help", "Show all commands"),
    ("/quit", "Exit chat"),
]


@dataclass
class QueueItem:
    kind: str
    role_id: str
    text: str = ""
    task_id: str | None = None
    run_id: str | None = None


class CommandCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        roles = list_roles()
        role_names = sorted({r.id for r in roles} | {a for r in roles for a in r.aliases})

        if text.startswith("/role "):
            prefix = text[len("/role "):].lower()
            for role in roles:
                if role.id.startswith(prefix):
                    yield Completion(
                        role.id,
                        start_position=-len(prefix),
                        display=role.display_name,
                        display_meta=("enabled" if role.enabled else "disabled"),
                    )
            return

        if text.startswith("/progress "):
            prefix = text[len("/progress "):].lower()
            for value in ("quiet", "normal", "verbose"):
                if value.startswith(prefix):
                    yield Completion(value, start_position=-len(prefix))
            return

        if text.startswith("/reasoning "):
            prefix = text[len("/reasoning "):].lower()
            for value in ("hidden", "summary", "raw"):
                if value.startswith(prefix):
                    yield Completion(value, start_position=-len(prefix))
            return

        if text.startswith("/theme "):
            prefix = text[len("/theme "):].lower()
            for theme in list_themes():
                if theme.id.startswith(prefix):
                    yield Completion(
                        theme.id,
                        start_position=-len(prefix),
                        display=theme.name,
                        display_meta=theme.source,
                    )
            return

        if text.startswith("/logo "):
            prefix = text[len("/logo "):].lower()
            for value in VALID_LOGOS:
                if value.startswith(prefix):
                    yield Completion(value, start_position=-len(prefix))
            return

        if " " in text:
            return

        candidates = [
            ("/role", "switch role"),
            ("/ask", "sideband question"),
            ("/run", "run one task reasoning epoch"),
            ("/pause", "pause at safe boundary"),
            ("/resume", "resume task reasoning"),
            ("/cancel", "cancel at safe boundary"),
            ("/task", "active/selected task"),
            ("/tasks", "list tasks"),
            ("/scheduled", "scheduled/future tasks"),
            ("/status", "runtime + context status"),
            ("/context", "ContextManager projection/budget"),
            ("/models", "role/model bindings"),
            ("/progress", "progress visibility"),
            ("/reasoning", "reasoning visibility"),
            ("/theme", "UI color theme"),
            ("/logo", "HUD logo mode"),
            ("/new", "new conversation"),
            ("/temporary", "ephemeral conversation mode"),
            ("/help", "command help"),
            ("/quit", "exit"),
        ]
        candidates.extend((f"/{name}", "role shortcut") for name in role_names)

        prefix = text.lower()
        for command, meta in sorted(candidates):
            if command.startswith(prefix):
                yield Completion(
                    command,
                    start_position=-len(text),
                    display=command,
                    display_meta=meta,
                )


class ChatTUI:
    def __init__(self, cfg, initial_role=None):
        ensure_role_registry()
        requested = initial_role or default_role_id()
        self.role_id = resolve_role_id(requested)
        if not get_role(self.role_id).enabled:
            raise RuntimeError(f"{get_role(self.role_id).display_name} is disabled")

        self.cfg = cfg
        self.conversation = create_conversation(
            title="T.A.R.S. chat",
            source="chat",
            metadata={"initial_role": self.role_id},
            make_active=True,
        )
        self.reasoning = cfg.get("policy", {}).get("reasoning", {}).get("default_visibility", "hidden")
        if self.reasoning not in {"hidden", "summary", "raw"}:
            self.reasoning = "hidden"
        self.progress_mode = cfg.get("chat", {}).get("progress", "normal")
        if self.progress_mode not in {"quiet", "normal", "verbose"}:
            self.progress_mode = "normal"

        self.theme = current_theme()
        self.logo_mode = current_logo()

        self.context_manager = ContextManager(cfg)
        self.log_lines: list[str] = []
        self.queue: asyncio.Queue | None = None
        self.busy = False
        self.activity = "Ready"
        self.queued = 0
        self.runtime_status = "unknown"
        self.gpu_status = self._gpu_status()
        self._closed = False
        self.live_reasoning = ""
        self.live_content = ""
        self.live_stream_kind = ""
        self.live_stream_role = ""
        self.live_stream_started = None
        self.live_reasoning_chars = 0
        self._event_cursor: dict[str, int] = {}
        self.temporary: TemporarySession | None = None

        CHAT_STATE_ROOT.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(CHAT_STATE_ROOT / "prompt-history.txt"))
        self._persistent_history = history

        self.output = TextArea(
            text="",
            read_only=True,
            focusable=False,
            scrollbar=True,
            wrap_lines=True,
            height=Dimension(weight=1),
        )

        self.input = TextArea(
            height=1,
            multiline=False,
            prompt=FormattedText([("class:prompt", "You › ")]),
            history=history,
            completer=CommandCompleter(),
            complete_while_typing=True,
            accept_handler=self._accept,
            wrap_lines=False,
        )
        self.input.buffer.on_text_changed += self._input_changed

        self.header_control = FormattedTextControl(self._header_fragments)
        self.command_control = FormattedTextControl(self._command_fragments)
        self.footer_control = FormattedTextControl(self._footer_fragments)
        self.live_control = FormattedTextControl(self._live_fragments)

        # The HUD is deliberately outside the conversation buffer. A DynamicContainer
        # selects the logo layout from the current terminal width, while the
        # conversation itself remains independently scrollable.
        self.logo_control = FormattedTextControl(self._logo_fragments)
        self.minimal_logo_control = FormattedTextControl(self._minimal_logo_fragments)
        self._header_compact = self._make_header("compact")
        self._header_minimal = self._make_header("minimal")
        self._header_none = self._make_header("none")
        header = DynamicContainer(self._select_header_container)

        command_palette = ConditionalContainer(
            Frame(
                Window(
                    self.command_control,
                    height=Dimension(preferred=6, max=7),
                    wrap_lines=False,
                ),
                title="Commands",
            ),
            filter=Condition(lambda: self.input.text.startswith("/")),
        )

        live_stream = ConditionalContainer(
            Frame(
                Window(self.live_control, height=Dimension(preferred=5, max=7), wrap_lines=True),
                title="Live Model Stream",
            ),
            filter=Condition(lambda: self.busy),
        )

        footer = Window(self.footer_control, height=1)

        root = HSplit([
            header,
            Frame(self.output, title="Conversation"),
            live_stream,
            command_palette,
            Frame(self.input, title="Input"),
            footer,
        ])

        kb = KeyBindings()

        @kb.add("c-c")
        def _ctrl_c(event):
            if self.input.text:
                self.input.buffer.reset()
                self.activity = "Input cleared"
            else:
                self.activity = "Ready"
            event.app.invalidate()

        @kb.add("c-d")
        def _ctrl_d(event):
            if not self.input.text:
                self._closed = True
                event.app.exit(result=0)

        @kb.add("escape")
        def _escape(event):
            self.input.buffer.complete_state = None
            event.app.invalidate()

        self.style = Style.from_dict(prompt_toolkit_style(self.theme))

        self.app = Application(
            layout=Layout(root, focused_element=self.input),
            key_bindings=kb,
            full_screen=True,
            mouse_support=True,
            style=self.style,
        )

        self._append_system("Enhanced chat UI ready. Type / for commands.")

    # ---------- presentation ----------
    def _profile(self):
        role = get_role(self.role_id)
        if not role.model:
            return None
        try:
            return get_profile(role.model, role.profile)
        except Exception:
            return None

    def _model_name(self):
        role = get_role(self.role_id)
        if not role.model:
            return "unbound"
        try:
            return get_model(role.model).name
        except Exception:
            return role.model

    def _context_state(self):
        profile = self._profile()
        if profile is None:
            return 0, 0, False
        try:
            current_seq = current_context_message_seq(self.conversation.id)
            projection = latest_projection(self.conversation.id, self.role_id)
            if projection is not None and projection.through_message_seq == current_seq:
                return projection.token_count, projection.usable_input, projection.exact
            estimate = self.context_manager.estimate_current(
                self.conversation.id, self.role_id
            )
            return estimate.token_count, estimate.usable_input, False
        except Exception:
            # HUD failure must never make chat unusable.
            return 0, profile.context, False

    def _context_bar(self, used, maximum, width=14):
        if not maximum:
            return "░" * width, 0.0
        ratio = max(0.0, min(1.0, used / maximum))
        filled = min(width, round(ratio * width))
        return "█" * filled + "░" * (width - filled), ratio

    @staticmethod
    def _fmt_tokens(value):
        if value >= 1024:
            kib = value / 1024
            if kib.is_integer():
                return f"{int(kib)}K"
            return f"{kib:.1f}K"
        return str(value)

    @staticmethod
    def _kv_label(value):
        return str(value).upper().replace("_0", "")

    def _terminal_columns(self):
        try:
            return int(get_app().output.get_size().columns)
        except Exception:
            return 100

    def _effective_logo_mode(self):
        columns = self._terminal_columns()
        requested = self.logo_mode
        if requested == "none":
            return "none"
        if requested == "minimal":
            return "minimal" if columns >= 58 else "none"
        if requested == "compact":
            if columns >= 100:
                return "compact"
            return "minimal" if columns >= 58 else "none"
        # auto
        if columns >= 100:
            return "compact"
        if columns >= 70:
            return "minimal"
        return "none"

    def _logo_fragments(self):
        return FormattedText([
            ("class:logo", "  ______   ___      ____   _____"), ("", "\n"),
            ("class:logo", " /_  __/  /   |    / __ \\ / ___/"), ("", "\n"),
            ("class:logo", "  / /    / /| |   / /_/ / \\__ \\"), ("", "\n"),
            ("class:logo", " / /  _ / ___ |_ / _, _/ ___/ /"), ("", "\n"),
            ("class:logo", "/_/  (_)_/  |_(_)_/ |_(_)____(_)")
        ])

    def _minimal_logo_fragments(self):
        return FormattedText([
            ("", "\n"),
            ("class:logo", " T.A.R.S."), ("", "\n"),
            ("class:dim", " runtime"), ("", "\n"),
            ("class:dim", " supervisor"), ("", "\n")
        ])

    def _make_header(self, mode):
        status = Window(self.header_control, height=5, wrap_lines=False)
        if mode == "compact":
            logo = Window(self.logo_control, width=Dimension.exact(36), height=5, wrap_lines=False)
            body = VSplit([logo, Window(width=1, char="│", style="class:dim"), status])
        elif mode == "minimal":
            logo = Window(self.minimal_logo_control, width=Dimension.exact(12), height=5, wrap_lines=False)
            body = VSplit([logo, Window(width=1, char="│", style="class:dim"), status])
        else:
            body = status
        return Frame(body, title="Task & Agent Runtime Supervisor")

    def _select_header_container(self):
        mode = self._effective_logo_mode()
        if mode == "compact":
            return self._header_compact
        if mode == "minimal":
            return self._header_minimal
        return self._header_none

    def _header_fragments(self):
        role = get_role(self.role_id)
        profile = self._profile()
        used, maximum, exact = self._context_state()
        bar, ratio = self._context_bar(used, maximum, width=12)

        if ratio < 0.60:
            ctx_style = "class:context.good"
            pressure = "OK"
        elif ratio < 0.75:
            ctx_style = "class:context.mid"
            pressure = "WATCH"
        elif ratio < 0.85:
            ctx_style = "class:context.high"
            pressure = "HIGH"
        else:
            ctx_style = "class:context.critical"
            pressure = "COMPACT"

        runtime_style = "class:ok" if self.runtime_status in {"unloaded", "loaded"} else "class:warn"
        gpu_style = "class:ok" if self.gpu_status == "suspended" else "class:warn"

        if profile:
            kv = f"{self._kv_label(profile.cache_type_k)}/{self._kv_label(profile.cache_type_v)}"
            profile_text = f"{role.profile} · {profile.context // 1024}K · {kv}"
        else:
            profile_text = f"{role.profile} · uncalibrated"

        task = active_task()
        task_text = "Task —"
        if task:
            pct = "" if task.progress is None else f" {task.progress * 100:.0f}%"
            task_text = f"Task {task.id[-8:]} · {task.state}{pct} · owner {get_role(task.owner_role).display_name}"

        mark = "" if exact else "≈"
        ctx_text = f"{mark}{self._fmt_tokens(used)}/{self._fmt_tokens(maximum)}"
        pct_text = f"{ratio * 100:.0f}%" if maximum else "—"

        return FormattedText([
            ("class:error", " TEMPORARY " if self.temporary else ""),
            ("class:role", f" {role.display_name}"),
            ("class:dim", "  •  "),
            ("class:model", self._model_name()),
            ("class:dim", "  •  "),
            ("class:accent", profile_text),
            ("", "\n "),
            (ctx_style, f"Context [{bar}] {pct_text} · {ctx_text} · {pressure}"),
            ("", "\n "),
            ("class:dim", "Runtime "),
            (runtime_style, self.runtime_status),
            ("class:dim", "  •  GPU "),
            (gpu_style, self.gpu_status),
            ("class:dim", "  •  Activity "),
            ("class:accent", self.activity),
            ("", "\n "),
            ("class:dim", f"Reasoning {self.reasoning.title()}  •  Progress {self.progress_mode.title()}  •  Trace Compact  •  Theme "),
            ("class:accent", self.theme.id),
            ("", "\n "),
            ("class:accent", task_text),
        ])

    def _matching_commands(self):
        text = self.input.text
        roles = list_roles()

        if text.startswith("/role "):
            prefix = text[len("/role "):].strip().lower()
            rows = []
            for role in roles:
                if not prefix or role.id.startswith(prefix) or role.display_name.lower().startswith(prefix):
                    state = "enabled" if role.enabled else "disabled"
                    rows.append((f"/role {role.id}", f"{role.display_name} · {state} · {role.model or 'unbound'}"))
            return rows[:6]

        if text.startswith("/progress "):
            return [(f"/progress {v}", d) for v, d in [
                ("quiet", "only important/final events"),
                ("normal", "meaningful milestones (default)"),
                ("verbose", "smaller steps and tool activity"),
            ]]

        if text.startswith("/reasoning "):
            return [
                ("/reasoning hidden", "hide reasoning text; activity still remains visible"),
                ("/reasoning summary", "show genuine loop/checkpoint outputs, not hidden CoT"),
                ("/reasoning raw", "stream genuine backend reasoning_content"),
            ]

        if text.startswith("/theme "):
            prefix = text[len("/theme "):].strip().lower()
            rows = []
            for theme in list_themes():
                if not prefix or theme.id.startswith(prefix):
                    marker = "current" if theme.id == self.theme.id else theme.source
                    rows.append((f"/theme {theme.id}", f"{theme.name} · {marker}"))
            return rows[:6]

        if text.startswith("/logo "):
            return [(f"/logo {mode}", "current" if mode == self.logo_mode else "HUD logo mode") for mode in VALID_LOGOS]

        if text.startswith("/ask "):
            return [("/ask <question>", "isolated sideband; does not modify main conversation context")]

        query = text.lower()
        rows = list(STATIC_COMMANDS)
        for role in roles:
            rows.append((f"/{role.id}", f"switch to {role.display_name}"))
        if query != "/":
            rows = [row for row in rows if row[0].lower().startswith(query) or query in row[0].lower()]
        return rows[:6]

    def _command_fragments(self):
        rows = self._matching_commands()
        if not rows:
            return FormattedText([("class:warn", " No matching command")])
        fragments = []
        for i, (usage, desc) in enumerate(rows):
            fragments.extend([
                ("class:command", f" {usage:<34}"),
                ("class:meta", desc),
            ])
            if i != len(rows) - 1:
                fragments.append(("", "\n"))
        return FormattedText(fragments)

    def _live_fragments(self):
        if not self.busy:
            return FormattedText([])
        elapsed = 0.0 if self.live_stream_started is None else max(0.0, time.monotonic() - self.live_stream_started)
        role = self.live_stream_role or get_role(self.role_id).display_name
        kind = self.live_stream_kind or "inference"
        fragments = [
            ("class:accent", f" {role} · {kind}"),
            ("class:dim", f" · {elapsed:0.1f}s"),
            ("", "\n"),
        ]
        if self.reasoning == "raw" and self.live_reasoning:
            tail = self.live_reasoning[-2400:]
            fragments.append(("class:reasoning", tail))
        elif self.live_content:
            fragments.append(("", self.live_content[-2400:]))
        elif self.live_reasoning_chars:
            # Hidden/Summary never fabricate or paraphrase chain-of-thought.  They
            # still prove that a real backend reasoning stream is alive.
            fragments.append(("class:reasoning", f"Reasoning stream active · {self.live_reasoning_chars} chars received"))
        else:
            fragments.append(("class:dim", "Preparing context / waiting for first backend token…"))
        return FormattedText(fragments)

    def _footer_fragments(self):
        queue_text = f" · queued {self.queued}" if self.queued else ""
        return FormattedText([
            ("class:footer", f" Enter send · / commands · Tab complete · ↑↓ history · Ctrl-C clear · Ctrl-D exit · {self.activity}{queue_text}"),
        ])

    def _render_output(self):
        text = "\n".join(self.log_lines)
        self.output.buffer.set_document(
            Document(text=text, cursor_position=len(text)),
            bypass_readonly=True,
        )
        self.app.invalidate()

    def _append(self, prefix, text, *, blank=True):
        if blank and self.log_lines:
            self.log_lines.append("")
        clean = str(text).rstrip()
        if "\n" in clean:
            lines = clean.splitlines()
            self.log_lines.append(f"{prefix} {lines[0]}")
            indent = " " * (len(prefix) + 1)
            self.log_lines.extend(indent + line for line in lines[1:])
        else:
            self.log_lines.append(f"{prefix} {clean}")
        self._render_output()

    def _append_system(self, text):
        self._append("T.A.R.S. ·", text, blank=False)

    def _append_user(self, text):
        self._append("You ›", text)

    def _append_assistant(self, role_id, text, sideband=False):
        role = get_role(role_id).display_name
        prefix = f"{role} {'· sideband' if sideband else '›'}"
        self._append(prefix, text or "[no final content]")

    # ---------- status ----------
    @staticmethod
    def _gpu_status():
        devices = sorted(Path("/sys/bus/pci/devices").glob("*"))
        for device in devices:
            try:
                if (device / "vendor").read_text().strip() != "0x10de":
                    continue
                if (device / "class").read_text().strip().startswith("0x030"):
                    return (device / "power/runtime_status").read_text().strip()
            except OSError:
                continue
        return "unknown"

    async def _poll_runtime(self):
        while not self._closed:
            try:
                models = await asyncio.to_thread(runtime_models, self.cfg)
                role = get_role(self.role_id)
                selected = next((m for m in models if m.get("id") == role.runtime_id), None)
                self.runtime_status = selected.get("status", {}).get("value", "unknown") if selected else "unavailable"
            except Exception:
                self.runtime_status = "offline"
            self.gpu_status = self._gpu_status()
            self.app.invalidate()
            await asyncio.sleep(2.0)

    def _status_text(self):
        role = get_role(self.role_id)
        used, maximum, exact = self._context_state()
        exact_text = "exact last turn" if exact else "approx until next inference"
        lines = [
            f"Role: {role.display_name}",
            f"Model: {self._model_name()} · profile {role.profile}",
            f"Runtime: {self.runtime_status} · GPU {self.gpu_status}",
            f"Context: {self._fmt_tokens(used)} / {self._fmt_tokens(maximum)} ({exact_text})",
        ]
        task = active_task()
        if task:
            pct = "—" if task.progress is None else f"{task.progress * 100:.0f}%"
            lines.extend([
                f"Task: {task.id}",
                f"Task state: {task.state} · phase {task.phase or '—'} · progress {pct}",
                f"Goal: {task.goal}",
            ])
        else:
            lines.append("Task: none")
        return "\n".join(lines)

    def _models_text(self):
        rows = []
        for role in list_roles():
            state = "enabled" if role.enabled else "disabled"
            rows.append(f"{role.display_name:<10} {state:<8} {role.model or 'unbound':<24} {role.profile}")
        return "\n".join(rows)

    def _task_text(self, task_id=None):
        try:
            task = load_task(task_id) if task_id else active_task()
        except KeyError as exc:
            return str(exc)
        if task is None:
            return "No active task."
        pct = "—" if task.progress is None else f"{task.progress * 100:.0f}%"
        schedule = ""
        if task.kind == "scheduled" or task.schedule_kind:
            schedule = (
                f"\nSchedule: {task.schedule_kind or 'scheduled'} · "
                f"{task.schedule_expr or '—'} · next {task.next_run_at or '—'} · "
                f"{'enabled' if task.schedule_enabled else 'paused'}"
            )
        return (
            f"{task.id}\n"
            f"Owner: {get_role(task.owner_role).display_name}\n"
            f"State: {task.state} · Epoch: {task.epoch}\n"
            f"Phase: {task.phase or '—'}\n"
            f"Progress: {pct}\n"
            f"Goal: {task.goal}{schedule}"
        )

    def _tasks_text(self, scheduled_only=False):
        rows = list_tasks(limit=20, scheduled_only=scheduled_only)
        if not rows:
            return "No scheduled tasks." if scheduled_only else "No tasks."
        lines = []
        for task in rows:
            marker = "*" if active_task() and active_task().id == task.id else " "
            tail = ""
            if scheduled_only:
                tail = f" · next={task.next_run_at or '—'} · {task.schedule_expr or task.schedule_kind or '—'}"
            lines.append(
                f"{marker} {task.id} · {task.state} · {get_role(task.owner_role).display_name} "
                f"· e{task.epoch}{tail} · {task.title or task.goal}"
            )
        return "\n".join(lines)

    # ---------- input / commands ----------
    def _input_changed(self, _):
        try:
            self.app.invalidate()
        except AttributeError:
            pass

    def _accept(self, buffer: Buffer):
        value = buffer.text.strip()
        if not value:
            return True
        buffer.reset()
        get_app().create_background_task(self._handle_submission(value))
        return True

    async def _handle_submission(self, value):
        if value.startswith("/"):
            handled = await self._handle_command(value)
            if handled:
                return

        self._append_user(value)
        if self.temporary is not None:
            if self.busy or self.queued:
                self._append_system("Wait for the current TEMPORARY response before sending another turn.")
                return
            self._enqueue(QueueItem("temporary", self.role_id, value))
            return
        task = active_task()
        add_message(
            self.conversation.id, "user", value, kind="message", include_in_context=True,
            related_task_id=task.id if task else None, metadata={"role_id": self.role_id},
        )
        self._enqueue(QueueItem("main", self.role_id, value))

    async def _handle_command(self, value):
        parts = value.split(maxsplit=2)
        command = parts[0].lower()

        if command in {"/quit", "/exit"}:
            self._closed = True
            self.app.exit(result=0)
            return True

        if command == "/help":
            text = "\n".join(f"{usage:<36} {desc}" for usage, desc in STATIC_COMMANDS)
            self._append("Help ·", text)
            return True

        if command == "/temporary":
            if self.busy or self.queued:
                self._append_system("Wait for active inference to finish before changing mode.")
                return True
            if self.temporary is None:
                self.temporary = TemporarySession(self.cfg, self.role_id)
                self.input.buffer.history = InMemoryHistory()
                self._append_system(TEMPORARY_NOTICE)
            else:
                self.temporary.close()
                self.temporary = None
                self.input.buffer.history = self._persistent_history
                self._append_system("TEMPORARY ended. Ephemeral state was discarded; normal conversation resumed.")
            self.app.invalidate()
            return True

        if self.temporary is not None and command in {
            "/new", "/run", "/resume", "/pause", "/cancel", "/ask",
        }:
            self._append_system(f"{command} is unavailable in TEMPORARY mode.")
            return True

        if command == "/status":
            self._append("Status ·", self._status_text())
            return True

        if command == "/context":
            self._append("Context ·", self._context_text())
            return True

        if command == "/task":
            task_id = parts[1] if len(parts) >= 2 else None
            self._append("Task ·", self._task_text(task_id))
            return True

        if command == "/tasks":
            self._append("Tasks ·", self._tasks_text(False))
            return True

        if command == "/scheduled":
            self._append("Scheduled ·", self._tasks_text(True))
            return True

        if command == "/models":
            self._append("Models ·", self._models_text())
            return True

        if command == "/new":
            self.conversation = create_conversation(
                title="T.A.R.S. chat",
                source="chat",
                metadata={"initial_role": self.role_id, "created_by": "/new"},
                make_active=True,
            )
            self._append_system(f"New conversation: {self.conversation.id}. Task state unchanged.")
            return True

        if command == "/role":
            if len(parts) < 2:
                self._append_system("Usage: /role <name>")
                return True
            return self._switch_role(parts[1])

        # dynamic /<role> shortcut, including compatibility aliases
        try:
            shortcut = get_role(command[1:])
        except KeyError:
            shortcut = None
        if shortcut is not None:
            return self._switch_role(shortcut.id)

        if command == "/progress":
            if len(parts) != 2 or parts[1] not in {"quiet", "normal", "verbose"}:
                self._append_system("Usage: /progress quiet|normal|verbose")
                return True
            self.progress_mode = parts[1]
            self._append_system(f"Progress visibility: {self.progress_mode}")
            self.app.invalidate()
            return True

        if command == "/reasoning":
            if len(parts) != 2 or parts[1] not in {"hidden", "summary", "raw"}:
                self._append_system("Usage: /reasoning hidden|summary|raw")
                return True
            self.reasoning = parts[1]
            self._append_system(f"Reasoning visibility: {self.reasoning}")
            self.app.invalidate()
            return True

        if command == "/theme":
            if len(parts) == 1:
                available = ", ".join(theme.id for theme in list_themes())
                self._append_system(f"Theme: {self.theme.id}. Available: {available}")
                return True
            try:
                self.theme = set_theme(parts[1])
            except (KeyError, ValueError) as exc:
                self._append_system(str(exc))
                return True
            self.style = Style.from_dict(prompt_toolkit_style(self.theme))
            self.app.style = self.style
            self.activity = f"Theme: {self.theme.id}"
            self._append_system(f"Theme changed to {self.theme.name} ({self.theme.id}).")
            self.app.invalidate()
            return True

        if command == "/logo":
            if len(parts) == 1:
                self._append_system(f"Logo: {self.logo_mode}. Available: {', '.join(VALID_LOGOS)}")
                return True
            try:
                self.logo_mode = set_logo(parts[1])
            except ValueError as exc:
                self._append_system(str(exc))
                return True
            self.activity = f"Logo: {self.logo_mode}"
            self._append_system(f"Logo mode changed to {self.logo_mode}.")
            self.app.invalidate()
            return True

        if command in {"/run", "/resume"}:
            task_id = parts[1] if len(parts) >= 2 else None
            try:
                task = load_task(task_id) if task_id else active_task()
                if task is None:
                    self._append_system("No active task. Use tars task create ... first.")
                    return True
                set_active_task(task.id)
                self._prime_event_cursor(task.id)
                run = create_run(task.id, self.conversation.id)
            except Exception as exc:
                self._append_system(str(exc))
                return True
            self._append_system(f"Task runner queued {run.id}; chat remains interactive.")
            self._enqueue(QueueItem("task", task.owner_role, task_id=task.id, run_id=run.id))
            return True

        if command in {"/pause", "/cancel"}:
            task_id = parts[1] if len(parts) >= 2 else None
            try:
                task = load_task(task_id) if task_id else active_task()
                if task is None:
                    self._append_system("No active task.")
                    return True
                request_control(task.id, command[1:])
            except Exception as exc:
                self._append_system(str(exc))
                return True
            self._append_system(f"{command[1:].title()} requested for {task.id}; applies at safe boundary.")
            return True

        if command == "/ask":
            question = value[len("/ask"):].strip()
            if not question:
                self._append_system("Usage: /ask <question>")
                return True
            self._append("Sideband ?", question)
            task = active_task()
            add_message(
                self.conversation.id, "user", question, kind="sideband",
                include_in_context=False, related_task_id=task.id if task else None,
                metadata={"role_id": self.role_id, "isolated": True},
            )
            self._enqueue(QueueItem("sideband", self.role_id, question))
            return True

        if command.startswith("/"):
            self._append_system(f"Unknown command: {command}. Type /help.")
            return True

        return False

    def _switch_role(self, name):
        try:
            role_id = resolve_role_id(name)
            role = get_role(role_id)
        except KeyError as exc:
            self._append_system(str(exc))
            return True
        if not role.enabled:
            self._append_system(f"{role.display_name} is disabled.")
            return True
        self.role_id = role_id
        if self.temporary is not None:
            self.temporary.role_id = role_id
        self.activity = f"Role: {role.display_name}"
        self._append_system(f"Role switched to {role.display_name}. Task state unchanged.")
        self.app.invalidate()
        return True

    def _enqueue(self, item):
        if self.queue is None:
            self._append_system("Runtime queue is not ready yet.")
            return
        self.queue.put_nowait(item)
        self.queued = self.queue.qsize()
        self.app.invalidate()

    def _context_text(self):
        try:
            projection = self.context_manager.estimate_current(
                self.conversation.id, self.role_id
            )
        except Exception as exc:
            return f"ContextManager unavailable: {exc}"
        mark = "exact" if projection.exact else "approx"
        return (
            f"Role: {get_role(self.role_id).display_name}\n"
            f"Profile: {projection.profile_name} · window {self._fmt_tokens(projection.context_window)}\n"
            f"Usable input: {self._fmt_tokens(projection.usable_input)} "
            f"(reserve {self._fmt_tokens(projection.output_reserve)} + safety {self._fmt_tokens(projection.safety_margin)})\n"
            f"Projection: {self._fmt_tokens(projection.token_count)} · {mark}\n"
            f"Messages: {projection.included_messages} included · {projection.omitted_messages} omitted\n"
            f"Task projection: {projection.task_id or 'none'}"
        )

    # ---------- durable events / streaming ----------
    def _prime_event_cursor(self, task_id):
        try:
            rows = read_events(task_id, limit=1)
            self._event_cursor[task_id] = rows[-1]["id"] if rows else 0
        except Exception:
            self._event_cursor[task_id] = 0

    def _event_visible(self, event):
        if event.get("visibility") == "internal":
            return False
        if self.progress_mode == "verbose":
            return True
        if self.progress_mode == "quiet":
            return event.get("type") in {"result", "error"} or event.get("visibility") == "quiet"
        return event.get("visibility") != "verbose"

    async def _poll_task_events(self):
        current_task = None
        while not self._closed:
            task = active_task()
            if task is None:
                current_task = None
                await asyncio.sleep(0.6)
                continue
            if task.id != current_task:
                current_task = task.id
                if task.id not in self._event_cursor:
                    self._prime_event_cursor(task.id)
            after = self._event_cursor.get(task.id, 0)
            try:
                events = await asyncio.to_thread(read_events_since, task.id, after, 100)
            except Exception:
                events = []
            for event in events:
                self._event_cursor[task.id] = max(self._event_cursor.get(task.id, 0), event["id"])
                if not self._event_visible(event):
                    continue
                etype = event.get("type", "status")
                prefix = {
                    "progress": "Progress ·",
                    "tool": "Tool ·",
                    "reasoning": "Reasoning ·",
                    "result": "Result ·",
                    "error": "Error ·",
                    "checkpoint": "Checkpoint ·",
                    "delegation": "Delegation ·",
                    "handoff": "Handoff ·",
                }.get(etype, "Task ·")
                self._append(prefix, event.get("message") or etype, blank=False)
            await asyncio.sleep(0.6)

    def _reset_live_stream(self, role_id, kind):
        self.live_reasoning = ""
        self.live_content = ""
        self.live_reasoning_chars = 0
        self.live_stream_kind = kind
        self.live_stream_role = get_role(role_id).display_name
        self.live_stream_started = time.monotonic()
        self.app.invalidate()

    def _consume_stream_event(self, event):
        reasoning = event.get("reasoning") or ""
        content = event.get("content") or ""
        if reasoning:
            self.live_reasoning += reasoning
            self.live_reasoning_chars += len(reasoning)
            if len(self.live_reasoning) > 32000:
                self.live_reasoning = self.live_reasoning[-32000:]
        if content:
            self.live_content += content
        self.app.invalidate()

    async def _stream_completion(self, role_id, messages, *, max_tokens=1024, kind="main"):
        self._reset_live_stream(role_id, kind)
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def producer():
            try:
                for event in chat_completion_stream(
                    self.cfg, role_id, messages, max_tokens=max_tokens
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        producer_task = asyncio.create_task(asyncio.to_thread(producer))
        finish = None
        usage = None
        while True:
            typ, payload = await queue.get()
            if typ == "event":
                self._consume_stream_event(payload)
                if payload.get("finish_reason") is not None:
                    finish = payload.get("finish_reason")
                if payload.get("usage"):
                    usage = payload.get("usage")
            elif typ == "error":
                await producer_task
                raise payload
            else:
                break
        await producer_task
        return self.live_content, self.live_reasoning, finish or "unknown", usage

    async def _run_task_stream(self, item):
        self._reset_live_stream(item.role_id, f"task epoch · {item.task_id[-8:]}")
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_stream(event):
            loop.call_soon_threadsafe(queue.put_nowait, ("event", event))

        def producer():
            try:
                result = run_task_epoch(self.cfg, item.run_id, on_stream=on_stream)
                loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        producer_task = asyncio.create_task(asyncio.to_thread(producer))
        result = None
        error = None
        while True:
            typ, payload = await queue.get()
            if typ == "event":
                self._consume_stream_event(payload)
            elif typ == "result":
                result = payload
            elif typ == "error":
                error = payload
            elif typ == "done":
                break
        await producer_task
        if error:
            raise error
        return result

    # ---------- model worker ----------
    @staticmethod
    def _response_parts(response):
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        return (
            message.get("content") or "",
            message.get("reasoning_content") or "",
            choice.get("finish_reason", "unknown"),
        )

    async def _worker(self):
        while not self._closed:
            item = await self.queue.get()
            self.queued = self.queue.qsize()
            self.busy = True
            role_name = get_role(item.role_id).display_name
            self.activity = f"{role_name} · {item.kind}"
            self.app.invalidate()

            try:
                if item.kind == "task":
                    result = await self._run_task_stream(item)
                    if result and self.reasoning == "raw" and result.get("reasoning"):
                        raw = result["reasoning"]
                        shown = raw[-16000:]
                        note = "\n[earlier raw reasoning omitted from display]" if self.live_reasoning_chars > len(shown) else ""
                        self._append(f"{role_name} · raw", shown + note)
                    run_state = result.get("run").state if result and result.get("run") else "unknown"
                    if run_state == "cancelled":
                        self._append_system(
                            f"Task runner cancelled {item.task_id} at a safe boundary; generated output was not promoted."
                        )
                    else:
                        self._append_system(
                            f"Task reasoning epoch finished for {item.task_id}. Durable result/checkpoint written; task paused for the next safe continuation."
                        )

                elif item.kind == "temporary":
                    if self.temporary is None:
                        raise RuntimeError("temporary mode ended before queued inference")
                    response = await asyncio.to_thread(self.temporary.send, item.text)
                    content, raw, finish = self._response_parts(response)
                    if self.reasoning == "raw" and raw:
                        self._append(f"{role_name} · raw", raw[-16000:])
                    self._append_assistant(
                        item.role_id, content or f"[no final content · {finish}]"
                    )

                elif item.kind == "sideband":
                    task = active_task()
                    if task:
                        append_event(task.id, "sideband", item.text, role=item.role_id,
                                     data={"mode": "isolated"})
                    self.activity = f"{role_name} · preparing sideband context"
                    projection = await asyncio.to_thread(
                        self.context_manager.build,
                        self.conversation.id,
                        item.role_id,
                        mode="sideband",
                        sideband_question=item.text,
                        requested_output_tokens=512,
                        exact=True,
                    )
                    content, raw, finish, usage = await self._stream_completion(
                        item.role_id, list(projection.messages), max_tokens=512, kind="sideband"
                    )
                    if self.reasoning == "raw" and raw:
                        shown = raw[-16000:]
                        note = "\n[earlier raw reasoning omitted from display]" if self.live_reasoning_chars > len(shown) else ""
                        self._append(f"{role_name} · raw", shown + note)
                    sideband_final = content or f"[no final content · {finish}]"
                    self._append_assistant(item.role_id, sideband_final, sideband=True)
                    add_message(
                        self.conversation.id, "assistant", sideband_final, kind="sideband",
                        include_in_context=False, related_task_id=task.id if task else None,
                        metadata={"role_id": item.role_id, "finish_reason": finish,
                                  "isolated": True, "usage": usage or {}},
                    )
                    self._append_system("Sideband complete. Main conversation context unchanged.")

                else:
                    self.activity = f"{role_name} · preparing context"
                    projection = await asyncio.to_thread(
                        self.context_manager.build,
                        self.conversation.id,
                        item.role_id,
                        mode="main",
                        requested_output_tokens=1024,
                        exact=True,
                    )
                    if projection.omitted_messages:
                        self._append_system(
                            f"ContextManager omitted {projection.omitted_messages} older messages for this inference; canonical state was preserved."
                        )
                    content, raw, finish, usage = await self._stream_completion(
                        item.role_id, list(projection.messages), max_tokens=1024, kind="chat"
                    )
                    if self.reasoning == "raw" and raw:
                        shown = raw[-16000:]
                        note = "\n[earlier raw reasoning omitted from display]" if self.live_reasoning_chars > len(shown) else ""
                        self._append(f"{role_name} · raw", shown + note)
                    final = content or f"[no final content · {finish}]"
                    self._append_assistant(item.role_id, final)
                    task = active_task()
                    add_message(
                        self.conversation.id, "assistant", final, kind="message",
                        include_in_context=True, related_task_id=task.id if task else None,
                        metadata={
                            "role_id": item.role_id,
                            "finish_reason": finish,
                            "context_projection_id": projection.id,
                            "prompt_tokens": projection.token_count,
                            "prompt_tokens_exact": projection.exact,
                            "usage": usage or {},
                            "reasoning_chars": len(raw),
                        },
                    )

            except Exception as exc:
                self._append("Runtime error ·", str(exc))
            finally:
                self.busy = False
                self.activity = "Ready"
                self.live_reasoning = ""
                self.live_content = ""
                self.live_stream_started = None
                self.queue.task_done()
                self.queued = self.queue.qsize()
                self.app.invalidate()

    def _pre_run(self):
        self.queue = asyncio.Queue()
        self.app.create_background_task(self._worker())
        self.app.create_background_task(self._poll_runtime())
        self.app.create_background_task(self._poll_task_events())

    def run(self):
        try:
            return self.app.run(pre_run=self._pre_run)
        finally:
            if self.temporary is not None:
                self.temporary.close()
                self.temporary = None
            self._closed = True


def run_chat(cfg, *, initial_role=None):
    try:
        ui = ChatTUI(cfg, initial_role=initial_role)
    except Exception as exc:
        # Keep a clear error instead of silently falling back when the registry/profile
        # itself is invalid. ImportError fallback is handled in chat.py.
        print(f"T.A.R.S. chat UI error: {exc}")
        return 2
    result = ui.run()
    return 0 if result is None else int(result)
