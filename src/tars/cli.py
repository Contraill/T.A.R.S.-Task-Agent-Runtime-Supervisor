import argparse
import json
import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import __version__
from .calibration import (
    ensure_seed_calibrations,
    list_calibrations,
    load_calibration,
)
from .calibration_engine import CalibrationEngine
from .chat import run_chat
from .config import (
    CACHE_ROOT,
    CALIBRATION_ROOT,
    CONFIG_PATH,
    DATA_ROOT,
    REGISTRY_PATH,
    ROLE_REGISTRY_PATH,
    STATE_ROOT,
    STATE_DB_PATH,
    TASK_ROOT,
    THEME_ROOT,
    UI_PREFS_PATH,
    load_config,
)
from .registry import ensure_registry, get_model, role_for_alias
from .runtime_backends import BACKEND_TYPES, backend_for_name
from .model_lifecycle import import_model, pull_model, remove_model, search_huggingface, verify_model
from .memory import (
    decide_candidate,
    doctor as memory_doctor,
    forget as forget_memory,
    inspect as inspect_memory,
    remember as remember_memory,
    review_candidates,
    search as search_memory,
    status as memory_status,
)
from .roles import (
    bind_model,
    create_role,
    default_role_id,
    ensure_role_registry,
    get_role,
    list_roles,
    remove_role,
    resolve_role_id,
    set_default_role,
    set_role_enabled,
    set_role_profile,
)
from .runtime import runtime_models
from .runtime_config import (
    apply_runtime_config,
    build_runtime_plan,
    render_runtime_config,
    runtime_config_status,
    switch_role_runtime,
    start_runtime_service,
    stop_runtime_service,
    runtime_service_logs,
)
from .state_store import health as state_store_health
from .conversation import create_conversation, list_conversations, load_conversation, list_messages, active_conversation
from .context import ContextManager, latest_projection
from .checkpoints import list_checkpoints, verify_checkpoint
from .themes import (
    VALID_LOGOS,
    current_logo,
    current_theme,
    ensure_ui_store,
    list_themes,
    set_logo,
    set_theme,
)
from .runner import create_run, run_task_epoch, list_runs, request_control
from .orchestration import (
    complete_delegation,
    create_delegation,
    delegation_envelope,
    handoff_task,
    list_delegations,
    list_handoffs,
    route_for_capabilities,
)
from .tasks import (
    active_task,
    append_event,
    clear_active_task,
    create_task,
    ensure_task_store,
    list_tasks,
    load_task,
    read_events,
    set_active_task,
    update_task,
    checkpoint_task,
    canonical_task_state,
)

console = Console()


def command_status(cfg):
    console.print(f"[bold]T.A.R.S.[/bold] {__version__}")
    console.print("Config: ", CONFIG_PATH)
    console.print("Models: ", REGISTRY_PATH)
    console.print("Roles:  ", ROLE_REGISTRY_PATH)
    console.print("Runtime backend:", cfg["runtime"]["backend"])
    console.print("Default role:", get_role(default_role_id()).display_name)

    task = active_task()
    if task is not None:
        console.print(
            f"Active task: {task.id} · {task.state} · "
            f"{get_role(task.owner_role).display_name} · {task.goal}"
        )

    try:
        models = runtime_models(cfg)
    except Exception as exc:
        console.print("[red]Runtime status: OFFLINE[/red]")
        console.print("Runtime error:", exc)
        return 1

    console.print("[green]Runtime status: ONLINE[/green]")
    console.print()

    table = Table(title="Runtime models")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Context")
    table.add_column("Name")

    for model in models:
        table.add_row(
            model.get("id", "?"),
            model.get("status", {}).get("value", "unknown"),
            str(model.get("context_length", "-")),
            model.get("name", model.get("id", "?")),
        )

    console.print(table)

    console.print()
    console.print("[bold]NVIDIA runtime[/bold]")
    found = False

    for device in sorted(Path("/sys/bus/pci/devices").glob("*")):
        vendor = device / "vendor"
        if not vendor.exists():
            continue
        try:
            if vendor.read_text().strip() != "0x10de":
                continue
            found = True
            runtime = (device / "power/runtime_status").read_text().strip()
            cls = (device / "class").read_text().strip()
            console.print(f"  {device.name}  class={cls}  runtime={runtime}")
        except OSError:
            pass

    if not found:
        console.print("  no NVIDIA PCI device found")

    return 0


def command_agents(cfg):
    try:
        models = runtime_models(cfg)
    except Exception as exc:
        console.print("[red]llama-swap unavailable:[/red]", exc)
        return 1

    table = Table(title="Runtime models")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Context")
    table.add_column("Name")

    for model in models:
        table.add_row(
            model.get("id", "?"),
            model.get("status", {}).get("value", "unknown"),
            str(model.get("context_length", "-")),
            model.get("name", model.get("id", "?")),
        )
    console.print(table)
    return 0


def command_paths(_cfg):
    rows = [
        ("config", CONFIG_PATH),
        ("models", REGISTRY_PATH),
        ("roles", ROLE_REGISTRY_PATH),
        ("data", DATA_ROOT),
        ("state", STATE_ROOT),
        ("state db", STATE_DB_PATH),
        ("cache", CACHE_ROOT),
        ("calibration", CALIBRATION_ROOT),
        ("legacy tasks", TASK_ROOT),
        ("ui prefs", UI_PREFS_PATH),
        ("themes", THEME_ROOT),
        ("docs", DATA_ROOT / "docs"),
        ("model files", DATA_ROOT / "models"),
        ("core", DATA_ROOT / "orchestrator"),
        ("tools", DATA_ROOT / "tools"),
    ]
    for label, value in rows:
        console.print(f"{label:<12}: {value}")
    return 0


def command_doctor(cfg):
    failures = 0
    checks = [
        ("central config", CONFIG_PATH.exists()),
        ("model registry", REGISTRY_PATH.exists()),
        ("role registry", ROLE_REGISTRY_PATH.exists()),
        ("state database", STATE_DB_PATH.exists()),
        ("UI preferences", UI_PREFS_PATH.exists()),
        ("theme directory", THEME_ROOT.is_dir()),
        ("data root", DATA_ROOT.is_dir()),
        ("state root", STATE_ROOT.is_dir()),
        ("cache root", CACHE_ROOT.is_dir()),
        ("llama-swap binary", shutil.which("llama-swap") is not None),
    ]

    for label, ok in checks:
        console.print(("[green]OK  [/green] " if ok else "[red]FAIL[/red] ") + label)
        if not ok:
            failures += 1

    try:
        health = state_store_health()
        if health["ok"]:
            counts = health["counts"]
            console.print(
                f"[green]OK  [/green] state DB schema={health['schema_version']} "
                f"integrity={health['integrity']} · "
                f"{counts['conversations']} conversations / {counts['tasks']} tasks / "
                f"{counts['checkpoints']} checkpoints / {counts['task_runs']} runs"
            )
        else:
            console.print(
                f"[red]FAIL[/red] state DB schema={health['schema_version']} "
                f"expected={health['expected_schema_version']} integrity={health['integrity']}"
            )
            failures += 1
    except Exception as exc:
        console.print(f"[red]FAIL[/red] state DB: {exc}")
        failures += 1

    try:
        roles = list_roles()
        enabled = sum(1 for r in roles if r.enabled)
        console.print(f"[green]OK  [/green] roles ({enabled} enabled / {len(roles)} total)")
    except Exception as exc:
        console.print(f"[red]FAIL[/red] roles: {exc}")
        failures += 1

    try:
        runtime_models(cfg)
        console.print("[green]OK  [/green] llama-swap API")
    except Exception as exc:
        console.print(f"[yellow]WARN[/yellow] llama-swap API: {exc}")

    try:
        calibrations = list_calibrations()
        ready = sum(1 for x in calibrations if x["status"] == "ready")
        console.print(f"[green]OK  [/green] calibration store ({ready} ready)")
    except Exception as exc:
        console.print(f"[red]FAIL[/red] calibration store: {exc}")
        failures += 1

    return 1 if failures else 0


def command_model_list():
    registry = ensure_registry()
    table = Table(title="Model Registry")
    table.add_column("Alias")
    table.add_column("Model")
    table.add_column("Role")
    table.add_column("Quant")
    table.add_column("Native ctx", justify="right")
    table.add_column("File")

    for alias, info in registry["models"].items():
        roles = role_for_alias(alias)
        path = Path(info["path"]).expanduser()
        table.add_row(
            alias,
            info["name"],
            ",".join(roles) if roles else "-",
            info.get("quant", "-"),
            str(info.get("native_context", "-")),
            "[green]present[/green]" if path.exists() else "[red]missing[/red]",
        )

    console.print(table)
    return 0


def command_model_info(alias):
    try:
        model = get_model(alias)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    roles = role_for_alias(alias)
    console.print(f"[bold]{model.name}[/bold]")
    console.print(f"alias          : {model.alias}")
    console.print(f"roles          : {', '.join(roles) if roles else '-'}")
    console.print(f"backend        : {model.backend}")
    console.print(f"quant          : {model.quant}")
    console.print(f"native context : {model.native_context}")
    console.print(f"sha256         : {model.sha256}")
    console.print(f"path           : {model.path}")
    console.print(f"file           : {'present' if model.path.exists() else 'missing'}")
    console.print(f"source         : {model.source}")
    console.print(f"license        : {model.license}")
    console.print(f"integrity      : {'verified' if model.integrity_verified else 'pending'}")
    console.print(f"compatibility  : {'compatible' if model.runtime_compatible else 'pending/unsupported'}")

    try:
        cal = load_calibration(alias)
    except Exception as exc:
        console.print(f"calibration    : missing ({exc})")
        return 0

    console.print(f"calibration    : {cal.get('status')} / {cal.get('depth')}")
    for name, profile in cal.get("profiles", {}).items():
        console.print(
            f"  {name:<8} ctx={profile['context']:<7} "
            f"KV={profile.get('cache_type_k')}/{profile.get('cache_type_v')} "
            f"t={profile.get('threads')}"
        )
    return 0


def command_model_search(query, limit):
    try:
        rows = search_huggingface(query, limit=limit)
    except Exception as exc:
        console.print(f"[red]Model search failed:[/red] {exc}")
        return 1
    table = Table(title="Hugging Face GGUF models")
    for column in ("Repository", "Downloads", "Likes", "License"):
        table.add_column(column)
    for row in rows:
        table.add_row(row["id"], str(row["downloads"]), str(row["likes"]), row["license"])
    console.print(table)
    return 0


def command_model_acquire(action, args):
    try:
        if action == "import":
            path = import_model(args.path, args.alias, expected_sha256=args.sha256,
                                name=args.name, license_name=args.license,
                                quant=args.quant, native_context=args.native_context)
        else:
            path = pull_model(args.source, args.alias, expected_sha256=args.sha256,
                              license_name=args.license, revision=args.revision,
                              filename=args.filename, quant=args.quant,
                              native_context=args.native_context)
    except Exception as exc:
        console.print(f"[red]Model {action} failed:[/red] {exc}")
        return 1
    console.print(f"{args.alias}: {path}")
    console.print("Readiness: integrity verified; runtime compatibility checked; calibration pending")
    return 0


def command_model_verify(alias):
    try:
        result = verify_model(alias)
    except Exception as exc:
        console.print(f"[red]Model verification failed:[/red] {exc}")
        return 1
    state = "runtime ready" if result.runtime_ready else "calibration pending"
    console.print(f"{alias}: sha256={result.sha256} · compatible={result.runtime_compatible} · {state}")
    return 0 if result.runtime_compatible else 1


def command_model_remove(alias):
    try:
        deleted = remove_model(alias)
    except Exception as exc:
        console.print(f"[red]Model removal failed:[/red] {exc}")
        return 1
    console.print(f"Removed {alias}; physical artifact {'deleted' if deleted else 'retained'}.")
    return 0


def command_backend_list(cfg):
    table = Table(title="Runtime Backends")
    for column in ("Backend", "Availability", "Health", "Support", "Message"):
        table.add_column(column)
    for name in BACKEND_TYPES:
        status = backend_for_name(name, cfg).status()
        table.add_row(name, "available" if status.available else "unavailable",
                      "healthy" if status.healthy else "not ready", status.support, status.message)
    console.print(table)
    return 0


def command_backend_status(cfg, name):
    try:
        backend = backend_for_name(name, cfg)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print_json(data=backend.diagnostics())
    return 0 if backend.status().healthy else 1


def command_memory(args):
    if args.memory_command == "status":
        result = memory_status()
        console.print_json(data=result)
        return 0 if result["ok"] else 1
    if args.memory_command == "doctor":
        result = memory_doctor()
        console.print_json(data=result)
        return 0 if result["ok"] else 1
    if args.memory_command == "remember":
        entry = remember_memory(
            args.content, kind=args.kind, scope=args.scope, title=args.title,
            source="user", confidence=args.confidence, tags=args.tag,
        )
        console.print_json(data={"id": entry.id, "path": entry.path, "scope": entry.scope})
        return 0
    if args.memory_command == "search":
        hits = search_memory(args.query, scope=args.scope, kind=args.kind, limit=args.limit)
        console.print_json(data=[{
            "id": hit.entry.id, "title": hit.entry.title, "scope": hit.entry.scope,
            "content": hit.entry.content, "score": hit.score,
            "signals": list(hit.signals), "source": hit.entry.source,
        } for hit in hits])
        return 0
    if args.memory_command == "inspect":
        entry = inspect_memory(args.memory_id)
        console.print_json(data={
            "id": entry.id, "kind": entry.kind, "scope": entry.scope,
            "title": entry.title, "content": entry.content, "source": entry.source,
            "confidence": entry.confidence, "tags": list(entry.tags), "path": entry.path,
        })
        return 0
    if args.memory_command == "forget":
        archived = forget_memory(args.memory_id)
        console.print(f"forgot {args.memory_id} · archive={archived}")
        return 0
    if args.memory_command == "review":
        if args.promote or args.reject:
            if not args.candidate_id:
                console.print("[red]candidate id is required for a review decision[/red]")
                return 2
            entry = decide_candidate(args.candidate_id, promote=args.promote, reason=args.reason)
            console.print(f"{args.candidate_id}: {'promoted to ' + entry.id if entry else 'rejected'}")
            return 0
        console.print_json(data=review_candidates())
        return 0
    return 2


def command_model_bindings():
    table = Table(title="Role Bindings")
    table.add_column("Role")
    table.add_column("State")
    table.add_column("Model")
    table.add_column("Profile")
    for role in list_roles():
        table.add_row(
            role.display_name,
            "enabled" if role.enabled else "disabled",
            role.model or "[dim]unbound[/dim]",
            role.profile,
        )
    console.print(table)
    return 0


def command_role_list():
    default = default_role_id()
    table = Table(title="Role Registry")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("State")
    table.add_column("Model")
    table.add_column("Profile")
    table.add_column("Execution")
    table.add_column("Default")

    for role in list_roles():
        table.add_row(
            role.id,
            role.display_name,
            "enabled" if role.enabled else "disabled",
            role.model or "unbound",
            role.profile,
            role.execution,
            "yes" if role.id == default else "",
        )
    console.print(table)
    return 0


def command_role_info(name):
    try:
        role = get_role(name)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(f"[bold]{role.display_name}[/bold] ({role.id})")
    console.print(f"enabled      : {role.enabled}")
    console.print(f"model        : {role.model or 'unbound'}")
    console.print(f"profile      : {role.profile}")
    console.print(f"runtime id   : {role.runtime_id}")
    console.print(f"execution    : {role.execution}")
    console.print(f"capabilities : {', '.join(role.capabilities) or '-'}")
    console.print(f"aliases      : {', '.join(role.aliases) or '-'}")
    console.print(f"description  : {role.description or '-'}")
    return 0


def command_role_create(args):
    try:
        role = create_role(
            args.role_id,
            display_name=args.display_name,
            model="",
            profile="normal",
            execution=args.execution,
            runtime_id=args.runtime_id,
            capabilities=args.capability,
            aliases=args.alias,
            description=args.description or "",
        )
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(f"Created role [bold]{role.display_name}[/bold] ({role.id}).")
    return 0


def command_role_mutation(action, name, value=None):
    try:
        if action == "remove":
            remove_role(name)
        elif action == "enable":
            set_role_enabled(name, True)
        elif action == "disable":
            set_role_enabled(name, False)
        elif action == "default":
            set_default_role(name)
        elif action == "bind":
            bind_model(name, value or "")
        elif action == "profile":
            set_role_profile(name, value)
        else:
            raise ValueError(action)
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print("OK")
    return 0


def command_theme_list():
    current = current_theme().id
    table = Table(title="UI Themes")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Source")
    table.add_column("Current")
    for theme in list_themes():
        table.add_row(theme.id, theme.name, theme.source, "yes" if theme.id == current else "")
    console.print(table)
    return 0


def command_theme_show(name=None):
    try:
        theme = current_theme() if not name else next(t for t in list_themes() if t.id == name.lower())
    except StopIteration:
        console.print(f"[red]unknown theme: {name}[/red]")
        return 2
    console.print(f"[bold]{theme.name}[/bold] ({theme.id})")
    console.print(f"source : {theme.source}")
    for key in ("accent", "muted", "success", "warning", "error", "reasoning", "tool"):
        console.print(f"{key:<10}: {theme.colors.get(key, 'default') or 'default'}")
    return 0


def command_theme_set(name):
    try:
        theme = set_theme(name)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(f"Theme: {theme.id}")
    return 0


def command_logo_show():
    console.print(f"Logo: {current_logo()}")
    console.print("Available: " + ", ".join(VALID_LOGOS))
    return 0


def command_logo_set(mode):
    try:
        value = set_logo(mode)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(f"Logo: {value}")
    return 0


def command_calibration_list():
    table = Table(title="Calibration Store")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Depth")
    table.add_column("Profiles")
    table.add_column("Reasonable max", justify="right")

    for row in list_calibrations():
        table.add_row(
            row["alias"],
            row["status"],
            row["depth"],
            ", ".join(row["profiles"]) or "-",
            str(row.get("reasonable_max_context", "-")),
        )
    console.print(table)
    return 0


def command_calibrate(args):
    aliases = args.models or list(ensure_registry()["models"])
    depth = "max" if args.max else "mid" if args.mid else "min"
    failures = 0
    engine = CalibrationEngine()
    for alias in aliases:
        try:
            result = engine.calibrate(alias, depth=depth, fresh=args.fresh)
        except Exception as exc:
            console.print(f"[red]{alias}: calibration failed:[/red] {exc}")
            failures += 1
            continue
        if result.get("protected_from_shallower"):
            state = f"retained {result.get('depth')} result"
        elif result.get("resumed"):
            state = f"already ready ({result.get('depth')})"
        else:
            state = f"ready ({result.get('depth')})"
        console.print(f"{alias}: {state} · max_ctx={result.get('reasonable_max_context', '-')}")
    return 1 if failures else 0



def command_runtime_plan():
    try:
        plan = build_runtime_plan()
    except (KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        console.print(f"[red]Runtime plan error:[/red] {exc}")
        return 2

    console.print(
        f"[bold]Zero-Idle[/bold] · TTL={plan.global_ttl}s · "
        f"unload={plan.unload_timeout}s · performance-monitor=disabled"
    )
    table = Table(title="Generated Runtime Plan")
    table.add_column("Role")
    table.add_column("Runtime")
    table.add_column("Model")
    table.add_column("Profile")
    table.add_column("Context", justify="right")
    table.add_column("KV")
    table.add_column("CPU")
    table.add_column("Threads")
    table.add_column("NGL")
    table.add_column("Overrides")
    for item in plan.models:
        ngl = "all" if item.ngl is None or str(item.ngl) == "-1" else str(item.ngl)
        table.add_row(
            item.display_name,
            item.runtime_id,
            item.model_alias,
            item.profile_name,
            str(item.context),
            f"{item.cache_type_k}/{item.cache_type_v}",
            item.cpus,
            f"{item.threads}/{item.batch_threads}",
            ngl,
            "yes" if item.tensor_overrides else "-",
        )
    console.print(table)
    return 0


def command_runtime_render():
    try:
        rendered = render_runtime_config(build_runtime_plan())
    except (KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        console.print(f"[red]Runtime render error:[/red] {exc}")
        return 2
    console.print(rendered, markup=False, highlight=False, end="")
    return 0


def command_runtime_status(cfg):
    try:
        status = runtime_config_status(cfg)
    except Exception as exc:
        console.print(f"[red]Runtime config status error:[/red] {exc}")
        return 2
    console.print(f"config       : {status.path}")
    console.print(f"installed    : {status.installed}")
    console.print(f"generated    : {status.generated_sha256[:16]}…")
    console.print(
        "installed sha: "
        + (f"{status.installed_sha256[:16]}…" if status.installed_sha256 else "-")
    )
    console.print(f"matches      : {status.matches_generated}")
    console.print(f"service      : {'active' if status.service_active else 'inactive'}")
    console.print(f"API          : {'healthy' if status.api_healthy else 'unhealthy'}")
    return 0 if status.service_active and status.api_healthy else 1


def command_runtime_apply(cfg, *, yes=False):
    if not yes:
        console.print("[yellow]Refusing to modify runtime without --yes.[/yellow]")
        console.print("Inspect `tars runtime plan` and `tars runtime render` first.")
        return 2
    try:
        result = apply_runtime_config(cfg)
    except Exception as exc:
        console.print(f"[red]Runtime apply failed:[/red] {exc}")
        return 1
    console.print("Runtime config: " + ("updated" if result.changed else "already current"))
    console.print(f"sha256       : {result.sha256}")
    console.print(f"runtime ids  : {', '.join(result.runtime_ids)}")
    console.print(f"backup       : {result.backup_path or '-'}")
    return 0


def command_runtime_switch(cfg, args):
    if not args.model and not args.profile:
        console.print("[red]Specify --model and/or --profile.[/red]")
        return 2
    if not args.yes:
        console.print("[yellow]Refusing runtime switch without --yes.[/yellow]")
        return 2
    try:
        result = switch_role_runtime(
            cfg,
            args.role,
            model_alias=args.model,
            profile_name=args.profile,
        )
    except Exception as exc:
        console.print(f"[red]Runtime switch failed:[/red] {exc}")
        return 1
    console.print(
        f"{result.role_id} -> {result.model_alias} / {result.profile_name} · "
        f"runtime {'updated' if result.apply.changed else 'already current'}"
    )
    return 0


def command_model_binding(cfg, action, role, alias=None):
    try:
        result = switch_role_runtime(
            cfg, role, model_alias=alias,
            unassign=action == "unassign",
        )
    except Exception as exc:
        console.print(f"[red]Model {action} failed:[/red] {exc}")
        return 1
    target = result.model_alias or "unbound"
    console.print(
        f"{result.role_id} -> {target} / {result.profile_name} · "
        f"runtime {'updated' if result.apply.changed else 'already current'}"
    )
    return 0


def command_role_profile(cfg, role, profile):
    try:
        result = switch_role_runtime(cfg, role, profile_name=profile)
    except Exception as exc:
        console.print(f"[red]Profile change failed:[/red] {exc}")
        return 1
    console.print(f"{result.role_id} -> {result.profile_name}")
    return 0


def command_service(action, *, lines=100, follow=False):
    try:
        if action == "start":
            start_runtime_service()
        elif action == "stop":
            stop_runtime_service()
        else:
            return runtime_service_logs(lines=lines, follow=follow)
    except Exception as exc:
        console.print(f"[red]Runtime service {action} failed:[/red] {exc}")
        return 1
    console.print("Runtime service " + ("started." if action == "start" else "stopped."))
    return 0


def _task_table(tasks, *, scheduled_view=False):
    table = Table(title="Scheduled Tasks" if scheduled_view else "Tasks")
    table.add_column("ID")
    table.add_column("State")
    table.add_column("Owner")
    table.add_column("Epoch", justify="right")
    table.add_column("Phase")
    table.add_column("Progress")
    if scheduled_view:
        table.add_column("Next run")
        table.add_column("Trigger")
    table.add_column("Goal")
    active = active_task()
    active_id = active.id if active else None
    for task in tasks:
        pct = "-" if task.progress is None else f"{task.progress * 100:.0f}%"
        marker = "*" if task.id == active_id else ""
        row = [
            marker + task.id,
            task.state,
            get_role(task.owner_role).display_name,
            str(task.epoch),
            task.phase or "-",
            pct,
        ]
        if scheduled_view:
            row.extend([
                task.next_run_at or "-",
                task.schedule_expr or task.schedule_kind or "-",
            ])
        row.append(task.title or task.goal)
        table.add_row(*row)
    console.print(table)


def command_task_list(scheduled_only=False):
    _task_table(list_tasks(scheduled_only=scheduled_only), scheduled_view=scheduled_only)
    return 0


def command_task_create(args):
    try:
        task = create_task(
            args.goal,
            args.role,
            kind=args.kind,
            source="cli",
            make_active=not args.no_activate,
            title=args.title or "",
        )
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(task.id)
    return 0


def command_task_status(task_id=None):
    try:
        task = load_task(task_id) if task_id else active_task()
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    if task is None:
        console.print("[dim]No active task.[/dim]")
        return 0
    _task_table([task])
    return 0


def command_task_show(task_id):
    try:
        task = load_task(task_id)
        state = canonical_task_state(task_id)
        cps = list_checkpoints(task_id, limit=1)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(f"[bold]{task.title or task.id}[/bold]")
    console.print(f"id            : {task.id}")
    console.print(f"conversation  : {task.conversation_id or '-'}")
    console.print(f"owner         : {get_role(task.owner_role).display_name}")
    console.print(f"state         : {task.state}")
    console.print(f"kind          : {task.kind}")
    console.print(f"phase         : {task.phase or '-'}")
    console.print(f"epoch         : {task.epoch}")
    console.print(f"progress      : {'-' if task.progress is None else f'{task.progress*100:.0f}%'}")
    console.print(f"goal          : {task.goal}")
    for key in ("constraints", "decisions", "completed", "open_steps", "failures", "evidence_refs"):
        values = state[key]
        console.print(f"{key:<13}: {', '.join(map(str, values)) if values else '-'}")
    schedule = state["schedule"]
    console.print(
        "schedule      : " + (
            f"{schedule['kind'] or '-'} · {schedule['expr'] or '-'} · next={schedule['next_run_at'] or '-'} · "
            f"{'enabled' if schedule['enabled'] else 'paused'}"
            if task.kind == "scheduled" or schedule["kind"] else "-"
        )
    )
    if cps:
        cp = cps[0]
        console.print(f"checkpoint    : #{cp.seq} {cp.id} · epoch {cp.epoch} · verified={verify_checkpoint(cp.id)}")
    else:
        console.print("checkpoint    : -")
    return 0


def command_task_checkpoint(task_id, reason, advance_epoch):
    try:
        cp = checkpoint_task(task_id, reason=reason, advance_epoch=advance_epoch)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(f"{cp.id} · seq={cp.seq} · epoch={cp.epoch} · sha256={cp.content_sha256[:16]}…")
    return 0


def command_task_checkpoints(task_id, limit):
    try:
        load_task(task_id)
        rows = list_checkpoints(task_id, limit=limit)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    table = Table(title=f"Checkpoints · {task_id}")
    table.add_column("Seq", justify="right")
    table.add_column("Epoch", justify="right")
    table.add_column("ID")
    table.add_column("Created")
    table.add_column("Owner")
    table.add_column("Verified")
    table.add_column("Reason")
    for cp in rows:
        table.add_row(str(cp.seq), str(cp.epoch), cp.id, cp.created_at, cp.owner_role,
                      "yes" if verify_checkpoint(cp.id) else "NO", cp.reason or "-")
    console.print(table)
    return 0


def command_task_events(task_id, limit):
    try:
        load_task(task_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    events = read_events(task_id, limit=limit)
    table = Table(title=f"Events · {task_id}")
    table.add_column("#", justify="right")
    table.add_column("Time")
    table.add_column("Type")
    table.add_column("Role")
    table.add_column("Visibility")
    table.add_column("Message")
    for event in events:
        table.add_row(
            str(event.get("id", "-")),
            event.get("timestamp", "-")[-15:-6],
            event.get("type", "-"),
            event.get("role") or "-",
            event.get("visibility", "normal"),
            event.get("message", ""),
        )
    console.print(table)
    return 0


def command_task_runs(task_id=None, limit=20):
    try:
        if task_id is not None:
            load_task(task_id)
        rows = list_runs(task_id, limit=limit)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    table = Table(title="Task Runs" + (f" · {task_id}" if task_id else ""))
    table.add_column("Run")
    table.add_column("Task")
    table.add_column("State")
    table.add_column("Role")
    table.add_column("Epoch", justify="right")
    table.add_column("Control")
    table.add_column("Finish")
    for run in rows:
        table.add_row(
            run.id, run.task_id, run.state, get_role(run.role_id).display_name,
            str(run.epoch), run.control_request or "-", run.finish_reason or "-",
        )
    console.print(table)
    return 0


def command_task_run(cfg, task_id, reasoning="hidden"):
    try:
        task = load_task(task_id)
        set_active_task(task.id)
        run = create_run(task.id)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    console.print(f"[bold]Runner[/bold] {run.id} · {get_role(task.owner_role).display_name} · epoch {task.epoch}")
    reasoning_started = False

    def on_stream(event):
        nonlocal reasoning_started
        raw = event.get("reasoning") or ""
        content = event.get("content") or ""
        if reasoning == "raw" and raw:
            if not reasoning_started:
                console.print("[magenta]Reasoning ›[/magenta] ", end="")
                reasoning_started = True
            console.print(raw, end="", markup=False, highlight=False)
        if content:
            # Final content is accumulated and printed once below; avoid duplicated output.
            pass

    try:
        result = run_task_epoch(cfg, run.id, on_stream=on_stream)
    except Exception as exc:
        if reasoning_started:
            console.print()
        console.print(f"[red]Runner failed:[/red] {exc}")
        return 1
    if reasoning_started:
        console.print()
    console.print("[bold]Result ›[/bold]")
    console.print(result.get("content") or "[no final content]", markup=False)
    console.print(f"[dim]Run {run.id} committed a durable checkpoint; task is paused for continuation.[/dim]")
    return 0


def command_task_mutation(action, task_id):
    try:
        if action == "activate":
            set_active_task(task_id)
        elif action == "pause":
            request_control(task_id, "pause")
        elif action == "resume":
            request_control(task_id, "resume")
            set_active_task(task_id)
        elif action == "cancel":
            request_control(task_id, "cancel")
        elif action == "complete":
            update_task(task_id, state="completed", phase="completed", progress=1.0)
            clear_active_task(task_id)
        else:
            raise ValueError(action)
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print("OK")
    return 0


def _parse_scope(values):
    result = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"scope entry must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("scope key cannot be empty")
        result[key] = value.strip()
    return result


def command_route(capabilities, task_id=None):
    try:
        decision = route_for_capabilities(capabilities, task_id=task_id)
    except (LookupError, KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    role = get_role(decision.selected_role)
    console.print(f"[bold]{role.display_name}[/bold] ({role.id})")
    console.print("required : " + (", ".join(decision.requested_capabilities) or "-"))
    console.print(f"reason   : {decision.reason}")
    table = Table(title="Routing Candidates")
    table.add_column("Role")
    table.add_column("Extra capabilities", justify="right")
    table.add_column("Capabilities")
    for candidate in decision.candidates:
        table.add_row(
            candidate["role_id"],
            str(candidate["extra_capabilities"]),
            ", ".join(candidate["capabilities"]),
        )
    console.print(table)
    return 0


def command_task_delegate(args):
    try:
        delegation = create_delegation(
            args.task_id,
            args.goal,
            role=args.role,
            required_capabilities=args.capability,
            scope=_parse_scope(args.scope),
            constraints=args.constraint,
            permissions=args.permission,
            evidence_refs=args.evidence,
            expected_result=args.expected_result or "",
        )
    except (LookupError, KeyError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(delegation.id)
    console.print(f"child task : {delegation.child_task_id}")
    console.print(f"role       : {delegation.requested_role}")
    console.print("parent owner unchanged")
    return 0


def command_task_delegations(task_id=None, limit=50):
    try:
        rows = list_delegations(task_id, limit=limit)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    table = Table(title="Delegations" + (f" · {task_id}" if task_id else ""))
    table.add_column("ID")
    table.add_column("Parent")
    table.add_column("Child")
    table.add_column("Role")
    table.add_column("State")
    table.add_column("Result")
    table.add_column("Goal")
    for row in rows:
        table.add_row(
            row.id, row.parent_task_id, row.child_task_id, row.requested_role,
            row.state, row.result_status or "-", row.goal,
        )
    console.print(table)
    return 0


def command_task_delegation_show(delegation_id):
    try:
        envelope = delegation_envelope(delegation_id)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print_json(data=envelope)
    return 0


def command_task_delegation_complete(args):
    try:
        data = json.loads(args.result_json) if args.result_json else {}
        if not isinstance(data, dict):
            raise ValueError("--result-json must decode to a JSON object")
        delegation = complete_delegation(
            args.delegation_id,
            status=args.status,
            summary=args.summary,
            result=data,
        )
    except (json.JSONDecodeError, KeyError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(f"{delegation.id} · {delegation.state} · {delegation.result_status}")
    return 0


def command_task_handoff(task_id, role, reason):
    try:
        record = handoff_task(task_id, role, reason=reason)
    except (KeyError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(
        f"{record.id} · {record.from_role} -> {record.to_role} · checkpoint={record.checkpoint_id}"
    )
    return 0


def command_task_handoffs(task_id=None, limit=50):
    rows = list_handoffs(task_id, limit=limit)
    table = Table(title="Handoffs" + (f" · {task_id}" if task_id else ""))
    table.add_column("ID")
    table.add_column("Task")
    table.add_column("From")
    table.add_column("To")
    table.add_column("Checkpoint")
    table.add_column("Reason")
    for row in rows:
        table.add_row(row.id, row.task_id, row.from_role, row.to_role, row.checkpoint_id, row.reason)
    console.print(table)
    return 0


def command_conversation_list(limit=50):
    rows = list_conversations(limit=limit)
    table = Table(title="Conversations")
    table.add_column("ID")
    table.add_column("State")
    table.add_column("Source")
    table.add_column("Updated")
    table.add_column("Title")
    for conv in rows:
        table.add_row(conv.id, conv.state, conv.source, conv.updated_at, conv.title or "-")
    console.print(table)
    return 0


def command_conversation_create(title):
    conv = create_conversation(title=title or "", source="cli", make_active=True)
    console.print(conv.id)
    return 0


def command_conversation_show(conversation_id, message_limit=20):
    try:
        conv = load_conversation(conversation_id)
        messages = list_messages(conversation_id, limit=message_limit)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print(f"[bold]{conv.title or conv.id}[/bold]")
    console.print(f"id      : {conv.id}")
    console.print(f"state   : {conv.state}")
    console.print(f"source  : {conv.source}")
    console.print(f"created : {conv.created_at}")
    console.print(f"updated : {conv.updated_at}")
    console.print(f"messages: {len(messages)} shown")
    if messages:
        table = Table(title="Recent Messages")
        table.add_column("Seq", justify="right")
        table.add_column("Role")
        table.add_column("Kind")
        table.add_column("Context")
        table.add_column("Content")
        for msg in messages:
            content = msg.content.replace("\n", " ")
            if len(content) > 100:
                content = content[:97] + "..."
            table.add_row(str(msg.seq), msg.role, msg.kind,
                          "yes" if msg.include_in_context else "no", content)
        console.print(table)
    return 0

def command_context_show(cfg, role_name=None, conversation_id=None, exact=False):
    conv = None
    if conversation_id:
        try:
            conv = load_conversation(conversation_id)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            return 2
    else:
        conv = active_conversation()
    if conv is None:
        console.print("[dim]No active conversation.[/dim]")
        return 0

    role_name = role_name or default_role_id()
    manager = ContextManager(cfg)
    try:
        projection = manager.build(
            conv.id, role_name, exact=bool(exact), persist=bool(exact)
        ) if exact else manager.estimate_current(conv.id, role_name)
    except Exception as exc:
        console.print(f"[red]ContextManager error:[/red] {exc}")
        return 2

    role = get_role(projection.role_id)
    console.print(f"[bold]Context Projection[/bold] · {role.display_name}")
    console.print(f"conversation    : {projection.conversation_id}")
    console.print(f"task            : {projection.task_id or '-'}")
    console.print(f"model           : {projection.model_alias}")
    console.print(f"profile         : {projection.profile_name}")
    console.print(f"context window  : {projection.context_window}")
    console.print(f"output reserve  : {projection.output_reserve}")
    console.print(f"safety margin   : {projection.safety_margin}")
    console.print(f"usable input    : {projection.usable_input}")
    console.print(f"projection      : {projection.token_count} ({'exact' if projection.exact else 'approx'})")
    pct = (projection.token_count / projection.usable_input * 100.0) if projection.usable_input else 0.0
    console.print(f"pressure        : {pct:.1f}%")
    console.print(f"messages        : {projection.included_messages} included / {projection.omitted_messages} omitted")
    if exact:
        console.print(f"projection id   : {projection.id}")
    else:
        previous = latest_projection(conv.id, projection.role_id)
        if previous:
            console.print(
                f"last exact/store : {previous.token_count} tokens · "
                f"{'exact' if previous.exact else 'approx'} · {previous.created_at}"
            )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="tars")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat")
    chat.add_argument("--role", default=None)

    sub.add_parser("status")
    sub.add_parser("agents")
    sub.add_parser("paths")
    sub.add_parser("doctor")
    sub.add_parser("help", help="show top-level help")

    model = sub.add_parser("model")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    model_sub.add_parser("list")
    model_sub.add_parser("bindings")
    model_info = model_sub.add_parser("info")
    model_info.add_argument("alias")
    for name in ["assign", "swap"]:
        p = model_sub.add_parser(name)
        p.add_argument("role")
        p.add_argument("alias")
    model_unassign = model_sub.add_parser("unassign")
    model_unassign.add_argument("role")
    model_search = model_sub.add_parser("search")
    model_search.add_argument("query")
    model_search.add_argument("--limit", type=int, default=10)
    model_pull = model_sub.add_parser("pull")
    model_pull.add_argument("source")
    model_pull.add_argument("--alias", required=True)
    model_pull.add_argument("--filename")
    model_pull.add_argument("--revision", default="main")
    model_import = model_sub.add_parser("import")
    model_import.add_argument("path")
    model_import.add_argument("--alias", required=True)
    for p in (model_pull, model_import):
        p.add_argument("--sha256")
        p.add_argument("--license", default="unknown")
        p.add_argument("--quant", default="unknown")
        p.add_argument("--native-context", type=int, default=0)
    model_import.add_argument("--name")
    model_verify = model_sub.add_parser("verify")
    model_verify.add_argument("alias")
    model_remove = model_sub.add_parser("remove")
    model_remove.add_argument("alias")
    backend = sub.add_parser("backend", help="inspect local runtime backends")
    backend_sub = backend.add_subparsers(dest="backend_command", required=True)
    backend_sub.add_parser("list")
    backend_status = backend_sub.add_parser("status")
    backend_status.add_argument("backend", choices=sorted(BACKEND_TYPES))

    memory = sub.add_parser("memory", help="inspect and maintain durable personal memory")
    memory_sub = memory.add_subparsers(dest="memory_command", required=True)
    memory_sub.add_parser("status")
    memory_sub.add_parser("doctor")
    memory_search = memory_sub.add_parser("search")
    memory_search.add_argument("query")
    memory_search.add_argument("--scope")
    memory_search.add_argument("--kind", choices=sorted({"system", "profile", "projects", "episodic", "reference"}))
    memory_search.add_argument("--limit", type=int, default=10)
    memory_inspect = memory_sub.add_parser("inspect")
    memory_inspect.add_argument("memory_id")
    memory_remember = memory_sub.add_parser("remember")
    memory_remember.add_argument("content")
    memory_remember.add_argument("--kind", default="profile", choices=sorted({"system", "profile", "projects", "episodic", "reference"}))
    memory_remember.add_argument("--scope", default="global")
    memory_remember.add_argument("--title", default="")
    memory_remember.add_argument("--confidence", type=float, default=1.0)
    memory_remember.add_argument("--tag", action="append", default=[])
    memory_forget = memory_sub.add_parser("forget")
    memory_forget.add_argument("memory_id")
    memory_review = memory_sub.add_parser("review")
    memory_review.add_argument("candidate_id", nargs="?")
    decision = memory_review.add_mutually_exclusive_group()
    decision.add_argument("--promote", action="store_true")
    decision.add_argument("--reject", action="store_true")
    memory_review.add_argument("--reason", default="")

    role = sub.add_parser("role")
    role_sub = role.add_subparsers(dest="role_command", required=True)
    role_sub.add_parser("list")
    role_info = role_sub.add_parser("info")
    role_info.add_argument("name")
    role_create = role_sub.add_parser("create")
    role_create.add_argument("role_id")
    role_create.add_argument("--display-name")
    role_create.add_argument("--execution", default="chat")
    role_create.add_argument("--runtime-id")
    role_create.add_argument("--capability", action="append", default=[])
    role_create.add_argument("--alias", action="append", default=[])
    role_create.add_argument("--description")
    for name in ["remove", "enable", "disable", "set-default"]:
        p = role_sub.add_parser(name)
        p.add_argument("name")
    role_profile = role_sub.add_parser("profile")
    role_profile.add_argument("name")
    role_profile.add_argument("profile", choices=["compact", "normal", "extended"])

    sub.add_parser("start", help="start the llama-swap user service")
    sub.add_parser("stop", help="stop the llama-swap user service")
    logs = sub.add_parser("logs", help="show llama-swap user-service logs")
    logs.add_argument("-n", "--lines", type=int, default=100)
    logs.add_argument("-f", "--follow", action="store_true")

    theme = sub.add_parser("theme")
    theme_sub = theme.add_subparsers(dest="theme_command", required=True)
    theme_sub.add_parser("list")
    theme_show = theme_sub.add_parser("show")
    theme_show.add_argument("name", nargs="?")
    theme_set = theme_sub.add_parser("set")
    theme_set.add_argument("name")

    logo = sub.add_parser("logo")
    logo_sub = logo.add_subparsers(dest="logo_command", required=True)
    logo_sub.add_parser("show")
    logo_set = logo_sub.add_parser("set")
    logo_set.add_argument("mode", choices=VALID_LOGOS)

    calibration = sub.add_parser("calibration")
    calibration_sub = calibration.add_subparsers(dest="calibration_command", required=True)
    calibration_sub.add_parser("list")

    calibrate = sub.add_parser("calibrate", help="run objective local-model calibration")
    calibrate.add_argument("models", nargs="*")
    depth = calibrate.add_mutually_exclusive_group()
    depth.add_argument("--mid", action="store_true")
    depth.add_argument("--max", action="store_true")
    calibrate.add_argument("--fresh", action="store_true", help="ignore compatible stage cache")

    runtime = sub.add_parser("runtime", help="generated local runtime configuration")
    runtime_sub = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_sub.add_parser("plan", help="show Role/Model/Calibration runtime plan")
    runtime_sub.add_parser("render", help="render candidate llama-swap YAML without writing")
    runtime_sub.add_parser("status", help="compare installed and generated runtime config")
    runtime_apply = runtime_sub.add_parser(
        "apply", help="atomically apply generated runtime config"
    )
    runtime_apply.add_argument(
        "--yes", action="store_true",
        help="confirm config replacement, service restart and health-checked rollback",
    )
    runtime_switch = runtime_sub.add_parser(
        "switch", help="transactionally change a Role model/profile and apply runtime"
    )
    runtime_switch.add_argument("role")
    runtime_switch.add_argument("--model")
    runtime_switch.add_argument("--profile", choices=["compact", "normal", "extended"])
    runtime_switch.add_argument(
        "--yes", action="store_true",
        help="confirm Role Registry and runtime config transaction",
    )

    task = sub.add_parser("task")
    task_sub = task.add_subparsers(dest="task_command", required=True)
    task_list = task_sub.add_parser("list")
    task_list.add_argument("--scheduled", action="store_true", help="show only scheduled/future tasks")
    task_create = task_sub.add_parser("create")
    task_create.add_argument("goal")
    task_create.add_argument("--title")
    task_create.add_argument("--role", default="general")
    task_create.add_argument("--kind", default="primary", choices=["primary", "delegation", "sideband", "scheduled"])
    task_create.add_argument("--no-activate", action="store_true")
    task_status = task_sub.add_parser("status")
    task_status.add_argument("task_id", nargs="?")
    task_show = task_sub.add_parser("show")
    task_show.add_argument("task_id")
    task_events = task_sub.add_parser("events")
    task_events.add_argument("task_id")
    task_events.add_argument("--limit", type=int, default=50)
    task_cp = task_sub.add_parser("checkpoint")
    task_cp.add_argument("task_id")
    task_cp.add_argument("--reason", default="manual checkpoint")
    task_cp.add_argument("--advance-epoch", action="store_true")
    task_cps = task_sub.add_parser("checkpoints")
    task_cps.add_argument("task_id")
    task_cps.add_argument("--limit", type=int, default=20)
    task_run = task_sub.add_parser("run")
    task_run.add_argument("task_id")
    task_run.add_argument("--reasoning", choices=["hidden", "raw"], default="hidden")
    task_runs = task_sub.add_parser("runs")
    task_runs.add_argument("task_id", nargs="?")
    task_runs.add_argument("--limit", type=int, default=20)
    for name in ["activate", "pause", "resume", "cancel", "complete"]:
        p = task_sub.add_parser(name)
        p.add_argument("task_id")

    route = sub.add_parser("route", help="capability-based AUTO role routing")
    route.add_argument("--capability", action="append", default=[])
    route.add_argument("--task", dest="task_id")

    task_delegate = task_sub.add_parser("delegate")
    task_delegate.add_argument("task_id")
    task_delegate.add_argument("goal")
    task_delegate.add_argument("--role")
    task_delegate.add_argument("--capability", action="append", default=[])
    task_delegate.add_argument("--scope", action="append", default=[], metavar="KEY=VALUE")
    task_delegate.add_argument("--constraint", action="append", default=[])
    task_delegate.add_argument("--permission", action="append", default=[])
    task_delegate.add_argument("--evidence", action="append", default=[])
    task_delegate.add_argument("--expected-result", default="")

    task_delegations = task_sub.add_parser("delegations")
    task_delegations.add_argument("task_id", nargs="?")
    task_delegations.add_argument("--limit", type=int, default=50)

    task_delegation_show = task_sub.add_parser("delegation-show")
    task_delegation_show.add_argument("delegation_id")

    task_delegation_complete = task_sub.add_parser("delegation-complete")
    task_delegation_complete.add_argument("delegation_id")
    task_delegation_complete.add_argument("--status", required=True, choices=["success", "partial", "failed", "cancelled"])
    task_delegation_complete.add_argument("--summary", required=True)
    task_delegation_complete.add_argument("--result-json", default="")

    task_handoff = task_sub.add_parser("handoff")
    task_handoff.add_argument("task_id")
    task_handoff.add_argument("role")
    task_handoff.add_argument("--reason", default="manual handoff")

    task_handoffs = task_sub.add_parser("handoffs")
    task_handoffs.add_argument("task_id", nargs="?")
    task_handoffs.add_argument("--limit", type=int, default=50)

    conv = sub.add_parser("conversation")
    conv_sub = conv.add_subparsers(dest="conversation_command", required=True)
    conv_list = conv_sub.add_parser("list")
    conv_list.add_argument("--limit", type=int, default=50)
    conv_create = conv_sub.add_parser("create")
    conv_create.add_argument("--title", default="")
    conv_show = conv_sub.add_parser("show")
    conv_show.add_argument("conversation_id")
    conv_show.add_argument("--messages", type=int, default=20)

    context = sub.add_parser("context")
    context_sub = context.add_subparsers(dest="context_command", required=True)
    context_show = context_sub.add_parser("show")
    context_show.add_argument("--role", default=None)
    context_show.add_argument("--conversation", default=None)
    context_show.add_argument(
        "--exact", action="store_true",
        help="use target llama.cpp tokenizer; may load the selected model",
    )

    return parser


def main():
    ensure_registry()
    ensure_role_registry()
    ensure_seed_calibrations()
    ensure_task_store()
    ensure_ui_store()

    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config()

    if args.command is None:
        return run_chat(cfg)
    if args.command == "chat":
        return run_chat(cfg, initial_role=args.role)
    if args.command == "status":
        return command_status(cfg)
    if args.command == "agents":
        return command_agents(cfg)
    if args.command == "paths":
        return command_paths(cfg)
    if args.command == "doctor":
        return command_doctor(cfg)
    if args.command == "help":
        parser.print_help()
        return 0
    if args.command in {"start", "stop"}:
        return command_service(args.command)
    if args.command == "logs":
        return command_service("logs", lines=args.lines, follow=args.follow)

    if args.command == "model":
        if args.model_command == "list":
            return command_model_list()
        if args.model_command == "bindings":
            return command_model_bindings()
        if args.model_command == "info":
            return command_model_info(args.alias)
        if args.model_command in {"assign", "swap"}:
            return command_model_binding(cfg, args.model_command, args.role, args.alias)
        if args.model_command == "unassign":
            return command_model_binding(cfg, "unassign", args.role)
        if args.model_command == "search":
            return command_model_search(args.query, args.limit)
        if args.model_command in {"pull", "import"}:
            return command_model_acquire(args.model_command, args)
        if args.model_command == "verify":
            return command_model_verify(args.alias)
        if args.model_command == "remove":
            return command_model_remove(args.alias)
    if args.command == "backend":
        if args.backend_command == "list":
            return command_backend_list(cfg)
        return command_backend_status(cfg, args.backend)
    if args.command == "memory":
        return command_memory(args)

    if args.command == "role":
        if args.role_command == "list":
            return command_role_list()
        if args.role_command == "info":
            return command_role_info(args.name)
        if args.role_command == "create":
            return command_role_create(args)
        if args.role_command == "remove":
            return command_role_mutation("remove", args.name)
        if args.role_command == "enable":
            return command_role_mutation("enable", args.name)
        if args.role_command == "disable":
            return command_role_mutation("disable", args.name)
        if args.role_command == "set-default":
            return command_role_mutation("default", args.name)
        if args.role_command == "profile":
            return command_role_profile(cfg, args.name, args.profile)

    if args.command == "theme":
        if args.theme_command == "list":
            return command_theme_list()
        if args.theme_command == "show":
            return command_theme_show(args.name)
        if args.theme_command == "set":
            return command_theme_set(args.name)

    if args.command == "logo":
        if args.logo_command == "show":
            return command_logo_show()
        if args.logo_command == "set":
            return command_logo_set(args.mode)

    if args.command == "calibration":
        if args.calibration_command == "list":
            return command_calibration_list()
    if args.command == "calibrate":
        return command_calibrate(args)

    if args.command == "runtime":
        if args.runtime_command == "plan":
            return command_runtime_plan()
        if args.runtime_command == "render":
            return command_runtime_render()
        if args.runtime_command == "status":
            return command_runtime_status(cfg)
        if args.runtime_command == "apply":
            return command_runtime_apply(cfg, yes=args.yes)
        if args.runtime_command == "switch":
            return command_runtime_switch(cfg, args)

    if args.command == "task":
        if args.task_command == "list":
            return command_task_list(args.scheduled)
        if args.task_command == "create":
            return command_task_create(args)
        if args.task_command == "status":
            return command_task_status(args.task_id)
        if args.task_command == "show":
            return command_task_show(args.task_id)
        if args.task_command == "events":
            return command_task_events(args.task_id, args.limit)
        if args.task_command == "checkpoint":
            return command_task_checkpoint(args.task_id, args.reason, args.advance_epoch)
        if args.task_command == "checkpoints":
            return command_task_checkpoints(args.task_id, args.limit)
        if args.task_command == "run":
            return command_task_run(cfg, args.task_id, args.reasoning)
        if args.task_command == "runs":
            return command_task_runs(args.task_id, args.limit)
        if args.task_command == "delegate":
            return command_task_delegate(args)
        if args.task_command == "delegations":
            return command_task_delegations(args.task_id, args.limit)
        if args.task_command == "delegation-show":
            return command_task_delegation_show(args.delegation_id)
        if args.task_command == "delegation-complete":
            return command_task_delegation_complete(args)
        if args.task_command == "handoff":
            return command_task_handoff(args.task_id, args.role, args.reason)
        if args.task_command == "handoffs":
            return command_task_handoffs(args.task_id, args.limit)
        if args.task_command in {"activate", "pause", "resume", "cancel", "complete"}:
            return command_task_mutation(args.task_command, args.task_id)

    if args.command == "route":
        return command_route(args.capability, args.task_id)

    if args.command == "conversation":
        if args.conversation_command == "list":
            return command_conversation_list(args.limit)
        if args.conversation_command == "create":
            return command_conversation_create(args.title)
        if args.conversation_command == "show":
            return command_conversation_show(args.conversation_id, args.messages)

    if args.command == "context":
        if args.context_command == "show":
            return command_context_show(
                cfg, args.role, args.conversation, args.exact
            )

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
