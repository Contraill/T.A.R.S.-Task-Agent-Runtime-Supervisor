from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .calibration import get_profile
from .config import CHAT_STATE_ROOT
from .roles import (
    default_role_id,
    ensure_role_registry,
    get_role,
    list_roles,
    resolve_role_id,
)
from .runtime import chat_completion, runtime_models
from .conversation import create_conversation, add_message
from .tasks import active_task, append_event, list_tasks, load_task

console = Console()


def _setup_readline():
    try:
        import readline
    except ImportError:
        return None

    CHAT_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    history = CHAT_STATE_ROOT / "history.txt"

    try:
        readline.read_history_file(history)
    except FileNotFoundError:
        pass

    readline.set_history_length(2000)
    return history


def _save_readline(history):
    if history is None:
        return
    try:
        import readline
        readline.write_history_file(history)
    except Exception:
        pass


def _role_summary(role_id):
    role = get_role(role_id)
    if not role.model:
        return f"{role.display_name} · disabled:no-model"

    try:
        profile = get_profile(role.model, role.profile)
        profile_text = (
            f"{role.profile} · {profile.context // 1024}K · "
            f"{profile.cache_type_k.upper()}/{profile.cache_type_v.upper()}"
        )
    except Exception:
        profile_text = f"{role.profile} · uncalibrated"

    return f"{role.display_name} · {role.model} · {profile_text}"


def _banner(role_id, reasoning, progress_mode):
    role = get_role(role_id)
    body = Text()
    body.append(_role_summary(role_id))
    body.append("\n")
    body.append(
        f"Reasoning: {reasoning.title()} · "
        f"Progress: {progress_mode.title()} · "
        "Trace: Compact · Zero-Idle: On"
    )
    console.print(
        Panel(
            body,
            title="T.A.R.S.",
            subtitle=f"Role: {role.display_name}",
            border_style="cyan",
            expand=False,
        )
    )


def _show_help():
    table = Table(
        title="Chat commands",
        show_header=False,
        box=None,
        pad_edge=False,
    )

    commands = [
        ("/role <name>", "switch role without changing task state"),
        ("/<role>", "shortcut for any registered role"),
        ("/ask <question>", "isolated sideband question; main chat context unchanged"),
        ("/task [id]", "show active or selected task state"),
        ("/tasks", "list tasks"),
        ("/scheduled", "list scheduled/future tasks"),
        ("/status", "runtime + active task status; model-free"),
        ("/models", "show role/model bindings"),
        ("/progress quiet|normal|verbose", "progress event visibility"),
        ("/reasoning hidden|raw", "backend reasoning visibility"),
        ("/new", "clear current conversation only"),
        ("/help", "show this help"),
        ("/quit", "exit chat"),
    ]
    for command, description in commands:
        table.add_row(f"[cyan]{command}[/cyan]", description)

    console.print(table)


def _show_models():
    table = Table(title="Role bindings")
    table.add_column("Role")
    table.add_column("State")
    table.add_column("Model")
    table.add_column("Profile")
    table.add_column("Runtime ID")

    for role in list_roles():
        state = "enabled" if role.enabled else "disabled"
        model = role.model or "unbound"
        table.add_row(
            role.display_name,
            state,
            model,
            role.profile,
            role.runtime_id,
        )

    console.print(table)


def _show_task():
    task = active_task()
    if task is None:
        console.print("[dim]No active task.[/dim]")
        return

    pct = "-" if task.progress is None else f"{task.progress * 100:.0f}%"
    table = Table(title=f"Active task · {task.id}", show_header=False)
    table.add_row("Owner", get_role(task.owner_role).display_name)
    table.add_row("State", task.state)
    table.add_row("Phase", task.phase or "-")
    table.add_row("Progress", pct)
    table.add_row("Goal", task.goal)
    console.print(table)


def _show_runtime_status(cfg):
    task = active_task()
    if task is not None:
        _show_task()
        console.print()

    try:
        models = runtime_models(cfg)
    except Exception as exc:
        console.print(f"[red]Runtime offline:[/red] {exc}")
        return

    table = Table(title="llama-swap")
    table.add_column("Runtime ID")
    table.add_column("Status")
    table.add_column("Name")

    for model in models:
        table.add_row(
            model.get("id", "?"),
            model.get("status", {}).get("value", "unknown"),
            model.get("name", model.get("id", "?")),
        )

    console.print(table)


def _print_response(response, reasoning):
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    raw_reasoning = message.get("reasoning_content") or ""

    if reasoning == "raw" and raw_reasoning:
        console.print(
            Panel(
                raw_reasoning,
                title="Raw reasoning",
                border_style="dim",
            )
        )

    if content:
        console.print()
        console.print("[bold cyan]T.A.R.S. ›[/bold cyan]")
        console.print(Markdown(content))
        console.print()
    else:
        finish = choice.get("finish_reason", "unknown")
        console.print(
            f"[yellow]No final content returned "
            f"(finish_reason={finish}).[/yellow]"
        )

    return content


def _sideband_messages(messages, question):
    # v0.3 foundation: immutable projection of the recent tail.
    # ContextManager will replace this with tokenizer-aware projection.
    tail = messages[-6:]
    return [
        {
            "role": "system",
            "content": (
                "This is a sideband question. Answer it without changing the "
                "main task or treating it as a new instruction for that task."
            ),
        },
        *tail,
        {"role": "user", "content": question},
    ]


def run_chat(cfg, *, initial_role=None):
    ensure_role_registry()

    requested = initial_role or default_role_id()
    try:
        role_id = resolve_role_id(requested)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    role = get_role(role_id)
    if not role.enabled:
        console.print(f"[yellow]{role.display_name} is disabled.[/yellow]")
        return 2

    reasoning = cfg.get("policy", {}).get(
        "reasoning", {}
    ).get("default_visibility", "hidden")
    if reasoning not in {"hidden", "raw"}:
        reasoning = "hidden"

    progress_mode = cfg.get("chat", {}).get("progress", "normal")
    if progress_mode not in {"quiet", "normal", "verbose"}:
        progress_mode = "normal"

    messages = []
    conversation = create_conversation(
        title="T.A.R.S. chat", source="chat-classic",
        metadata={"initial_role": role_id}, make_active=True,
    )
    history = _setup_readline()

    console.print()
    _banner(role_id, reasoning, progress_mode)
    console.print(
        "[dim]Type /help for commands. "
        "Ctrl-D or /quit exits.[/dim]"
    )
    console.print()

    try:
        while True:
            try:
                value = console.input("[bold cyan]You ›[/] ").strip()
            except EOFError:
                console.print()
                break
            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted.[/dim]")
                continue

            if not value:
                continue

            if value.startswith("/"):
                parts = value.split(maxsplit=2)
                command = parts[0].lower()

                if command in {"/quit", "/exit"}:
                    break

                if command == "/help":
                    _show_help()
                    continue

                if command == "/new":
                    messages.clear()
                    conversation = create_conversation(
                        title="T.A.R.S. chat", source="chat-classic",
                        metadata={"initial_role": role_id, "created_by": "/new"}, make_active=True,
                    )
                    console.print(f"[dim]New conversation: {conversation.id}. Task state unchanged.[/dim]")
                    continue

                if command == "/status":
                    _show_runtime_status(cfg)
                    continue

                if command == "/task":
                    _show_task()
                    continue

                if command == "/models":
                    _show_models()
                    continue

                if command == "/role":
                    if len(parts) < 2:
                        console.print("Usage: /role <name>")
                        continue
                    try:
                        requested_role = get_role(parts[1])
                    except KeyError as exc:
                        console.print(f"[yellow]{exc}[/yellow]")
                        continue
                    if not requested_role.enabled:
                        console.print(
                            f"[yellow]{requested_role.display_name} is disabled.[/yellow]"
                        )
                        continue
                    role_id = requested_role.id
                    console.print()
                    _banner(role_id, reasoning, progress_mode)
                    continue

                # Dynamic role shortcuts, including hidden legacy aliases.
                try:
                    shortcut_role = get_role(command[1:])
                except KeyError:
                    shortcut_role = None

                if shortcut_role is not None:
                    if not shortcut_role.enabled:
                        console.print(
                            f"[yellow]{shortcut_role.display_name} is disabled.[/yellow]"
                        )
                        continue
                    role_id = shortcut_role.id
                    console.print()
                    _banner(role_id, reasoning, progress_mode)
                    continue

                if command == "/ask":
                    if len(parts) < 2:
                        console.print("Usage: /ask <question>")
                        continue
                    question = value[len("/ask"):].strip()
                    task = active_task()
                    add_message(
                        conversation.id, "user", question, kind="sideband", include_in_context=False,
                        related_task_id=task.id if task else None, metadata={"role_id": role_id, "isolated": True},
                    )
                    if task is not None:
                        append_event(
                            task.id,
                            "sideband",
                            question,
                            role=role_id,
                            data={"mode": "isolated"},
                        )
                    try:
                        with console.status(
                            f"[cyan]{get_role(role_id).display_name} sideband…[/cyan]",
                            spinner="dots",
                        ):
                            response = chat_completion(
                                cfg,
                                role_id,
                                _sideband_messages(messages, question),
                                max_tokens=512,
                            )
                    except Exception as exc:
                        console.print(
                            Panel(
                                str(exc),
                                title="Sideband runtime error",
                                border_style="red",
                            )
                        )
                        continue

                    console.print("[bold magenta]Sideband ›[/bold magenta]")
                    _print_response(response, reasoning)
                    console.print(
                        "[dim]Main conversation context was not modified.[/dim]"
                    )
                    continue

                if command == "/reasoning":
                    if len(parts) != 2 or parts[1] not in {"hidden", "raw"}:
                        console.print("Usage: /reasoning hidden|raw")
                        continue
                    reasoning = parts[1]
                    console.print(
                        f"[dim]Reasoning visibility: {reasoning}[/dim]"
                    )
                    continue

                if command == "/progress":
                    if len(parts) != 2 or parts[1] not in {"quiet", "normal", "verbose"}:
                        console.print("Usage: /progress quiet|normal|verbose")
                        continue
                    progress_mode = parts[1]
                    console.print(
                        f"[dim]Progress visibility: {progress_mode}[/dim]"
                    )
                    continue

                console.print(
                    f"[yellow]Unknown command:[/yellow] {command}"
                )
                continue

            messages.append({"role": "user", "content": value})

            try:
                with console.status(
                    f"[cyan]{get_role(role_id).display_name} is thinking…[/cyan]",
                    spinner="dots",
                ):
                    response = chat_completion(cfg, role_id, messages)
            except Exception as exc:
                messages.pop()
                console.print(
                    Panel(
                        str(exc),
                        title="Runtime error",
                        border_style="red",
                    )
                )
                continue

            content = _print_response(response, reasoning)
            messages.append({"role": "assistant", "content": content})
            task = active_task()
            add_message(
                conversation.id, "assistant", content or "[no final content]", kind="message",
                include_in_context=True, related_task_id=task.id if task else None,
                metadata={"role_id": role_id},
            )

    finally:
        _save_readline(history)

    console.print("[dim]Session closed.[/dim]")
    return 0
