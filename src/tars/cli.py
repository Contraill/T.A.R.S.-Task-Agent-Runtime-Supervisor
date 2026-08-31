import argparse
import json
import shutil
import ssl
import threading
import time
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
from .memory_maintenance import list_runs as list_maintenance_runs, run_maintenance
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
from .context_epochs import list_epochs, search_transcript
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
from .temporary import run_temporary
from .policy import ScopeGuard, ScopeRequest, add_rule as add_policy_rule, list_rules as list_policy_rules
from .approvals import ApprovalBroker
from .action_journal import list_actions as list_audit_actions, load_action
from .execution_backends import ContainerBackend, HostBackend, SSHBackend
from .tool_registry import ToolRegistry
from .evidence import list_records as list_evidence_records
from .agent_loop import submit_task_control
from .control_queue import list_controls as list_task_controls
from .delegation import (accept as accept_child, cancel as cancel_child,
                         create_child, load_contract as load_child_contract)
from .workspace_recovery import (WorkspaceRecovery, load as load_workspace_checkpoint,
                                 list_checkpoints as list_workspace_checkpoints)
from .skills import SkillRegistry
from .mcp import (MCPClient, list_servers as list_mcp_servers,
                  register as register_mcp_server, set_enabled as set_mcp_enabled)
from .scheduler import (Scheduler, create_schedule, edit_schedule, list_runs as list_schedule_runs,
                        condition_registry, health as scheduler_health, list_schedules,
                        load_schedule, remove_schedule, require_condition_support,
                        set_enabled as set_schedule_enabled)
from .core_auth import (DEFAULT_PERMISSIONS, PERMISSIONS as CLIENT_PERMISSIONS,
                        create_pairing, list_clients as list_core_clients,
                        revoke as revoke_core_client)
from .core_api import CoreAPI, CoreServerConfig, make_server
from .runtime_routing import LocalRuntimeRouter
from .extensions import ExtensionLoader

console = Console()


def command_schedule_list():
    table = Table(title="Schedules")
    for column in ("ID", "Task", "Kind", "Expression", "Next", "State"):
        table.add_column(column)
    for item in list_schedules():
        table.add_row(item.id, item.task_id, item.kind, item.expression,
                      item.next_run_at or "-", "enabled" if item.enabled else "paused")
    console.print(table)


def command_schedule_show(schedule_id):
    item = load_schedule(schedule_id)
    console.print_json(data={
        "id": item.id, "task_id": item.task_id, "kind": item.kind,
        "expression": item.expression, "timezone": item.timezone,
        "next_run_at": item.next_run_at, "missed_policy": item.missed_policy,
        "max_catch_up": item.max_catch_up, "enabled": item.enabled,
        "concurrency_key": item.concurrency_key,
        "max_concurrency": item.max_concurrency,
        "delivery_target": item.delivery_target,
        "revision": item.revision, "removed_at": item.removed_at,
    })


def command_schedule_add(cfg, args):
    if args.kind == "condition":
        name = args.expression.split("@", 1)[0].strip()
        if name not in condition_registry(cfg):
            raise ValueError(f"condition is not configured: {name}")
    item = create_schedule(
        args.task_id, args.kind, args.expression, next_run_at=args.next,
        missed_policy=args.missed, max_catch_up=args.max_catch_up,
        concurrency_key=args.concurrency_key, max_concurrency=args.max_concurrency,
        delivery_target=args.delivery_target)
    console.print(f"[green]Registered[/green] {item.id} · next {item.next_run_at or '-'}")


def command_schedule_runs(schedule_id, limit):
    table = Table(title="Schedule Run Journal")
    for column in ("Run", "Schedule", "Planned", "State", "Attempt", "Checkpoint"):
        table.add_column(column)
    for run in list_schedule_runs(schedule_id, limit=limit):
        table.add_row(run.id, run.schedule_id, run.planned_for, run.state,
                      str(run.attempt), run.checkpoint_id or "-")
    console.print(table)


def command_schedule_run_due(cfg):
    conditions = condition_registry(cfg)
    require_condition_support(conditions)
    engine = Scheduler(max_concurrency=int(cfg.get("scheduler", {}).get("max_concurrency", 1)),
                       conditions=conditions)
    recovered = engine.recover()
    claimed = engine.claim_due()

    completed = engine.execute_claimed(_schedule_executor(cfg))
    console.print(f"Recovered {recovered} · claimed {len(claimed)} · completed {len(completed)}")
    return 1 if any(run.state == "failed" for run in completed) else 0


def command_client_list():
    table = Table(title="Core Clients")
    for column in ("ID", "Name", "Principal", "State", "Permissions", "Last seen"):
        table.add_column(column)
    for client in list_core_clients():
        table.add_row(client.id, client.name, client.principal_id, client.state,
                      ", ".join(client.permissions), client.last_seen_at or "-")
    console.print(table)


def _schedule_executor(cfg):
    def execute(scheduled_run):
        run = create_run(scheduled_run.task_id)
        outcome = run_task_epoch(cfg, run.id)
        finished = outcome["run"]
        if finished.state in {"failed", "cancelled"}:
            raise RuntimeError(
                f"task run {finished.id} {finished.state}: "
                f"{finished.error or finished.finish_reason}")
        return {"task_run_id": finished.id, "state": finished.state,
                "finish_reason": finished.finish_reason,
                "checkpoint_id": outcome.get("checkpoint_id")}
    return execute


def _core_server_options(cfg, args):
    core_cfg = cfg.get("core", {})
    host = args.host if args.host is not None else core_cfg.get("host", "127.0.0.1")
    port = args.port if args.port is not None else int(core_cfg.get("port", 8765))
    allow_remote = (args.allow_remote if args.allow_remote is not None
                    else bool(core_cfg.get("allow_remote", False)))
    return host, port, allow_remote


def command_core_serve(cfg, args):
    host, port, allow_remote = _core_server_options(cfg, args)
    context = None
    if args.cert or args.key:
        if not args.cert or not args.key:
            raise ValueError("both --cert and --key are required for TLS")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert, args.key)
    conditions = condition_registry(cfg)
    require_condition_support(conditions)
    server = make_server(
        CoreServerConfig(host, port, allow_remote=allow_remote,
                         ssl_context=context),
        api=CoreAPI(allow_remote_pairing=allow_remote and context is not None,
                    conditions=conditions))
    scheduler = Scheduler(
        max_concurrency=int(cfg.get("scheduler", {}).get("max_concurrency", 1)),
        conditions=conditions)
    scheduler_stop = threading.Event()
    scheduler_thread = threading.Thread(
        target=scheduler.run_forever, args=(_schedule_executor(cfg),),
        kwargs={"stop": scheduler_stop}, name="tars-scheduler", daemon=True)
    scheduler_thread.start()
    deadline = time.monotonic() + 2
    while scheduler_thread.is_alive() and not scheduler._wake_socket and time.monotonic() < deadline:
        time.sleep(0.01)
    if not scheduler_thread.is_alive() or scheduler._wake_socket is None:
        server.server_close()
        raise RuntimeError("Core scheduler failed to start")
    scheme = "https" if context else "http"
    console.print(f"T.A.R.S. Core listening on {scheme}://{host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        scheduler_stop.set()
        scheduler.wake()
        scheduler_thread.join(timeout=5)


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


def command_runtime_route(cfg, args):
    route = LocalRuntimeRouter(cfg).resolve(
        args.role, task_id=args.task, required_capabilities=args.capability,
        context_tokens=args.context_tokens, require_reasoning=args.reasoning,
        require_tools=args.tools)
    console.print_json(data={
        "id": route.id, "state": route.state,
        "requested_role": route.requested_role, "selected_role": route.selected_role,
        "model": route.model_alias, "backend": route.backend,
        "runtime_id": route.runtime_id, "profile": route.profile,
        "requested": route.requested, "reasons": list(route.reasons),
        "backend_status": route.backend_status,
        "runtime_capabilities": route.runtime_capabilities,
        "model_capabilities": route.model_capabilities,
    })
    return 0 if route.ready else 1


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

    colibri = backend_for_name("colibri", cfg).diagnostics()
    if not colibri["configured"]:
        console.print("[yellow]WARN[/yellow] Oracle / Colibri: not configured (optional)")
    elif colibri["healthy"]:
        console.print(
            f"[green]OK  [/green] Oracle / Colibri ({colibri['message']}; "
            f"TTL {colibri['ttl_seconds']}s)")
    else:
        console.print(f"[red]FAIL[/red] Oracle / Colibri: {colibri['message']}")
        failures += 1

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
    console.print(f"thinking       : {'on/off toggle' if model.thinking_control == 'toggle' else 'unverified'}")

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


def command_extension_list(cfg):
    table = Table(title="Extension Boundaries")
    for column in ("Kind", "Name", "Provenance", "Enabled", "Trusted", "Distribution"):
        table.add_column(column)
    for item in ExtensionLoader(cfg).discover():
        table.add_row(item.kind, item.name, item.provenance,
                      "yes" if item.enabled else "no", "yes" if item.trusted else "no",
                      item.distribution or "-")
    console.print(table)
    return 0


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
    if args.memory_command == "maintain":
        run = run_maintenance(trigger=args.trigger, apply=args.apply)
        console.print_json(data={
            "id": run.id, "trigger": run.trigger, "mode": run.mode,
            "status": run.status, "report": run.report,
            "actions": list(run.actions), "rollback_refs": list(run.rollback_refs),
        })
        return 0 if run.status == "completed" else 1
    if args.memory_command == "maintenance-runs":
        console.print_json(data=[{
            "id": run.id, "trigger": run.trigger, "mode": run.mode,
            "status": run.status, "created_at": run.created_at,
            "model_provenance": run.model_provenance,
        } for run in list_maintenance_runs(args.limit)])
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


def command_task_control(task_id, kind, message):
    try:
        control, feedback = submit_task_control(task_id, kind, message)
    except (KeyError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print_json(data={"id": control.id, "task_id": control.task_id,
                             "seq": control.seq, "kind": control.kind,
                             "state": control.state, "feedback": feedback})
    return 0


def command_task_controls(task_id, limit=50):
    try:
        rows = list_task_controls(task_id, limit=limit)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print_json(data=[{"id": row.id, "seq": row.seq, "kind": row.kind,
                              "state": row.state, "message": row.message,
                              "payload": row.payload, "created_at": row.created_at,
                              "applied_at": row.applied_at} for row in rows])
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


def command_task_child_control(args):
    try:
        if args.task_command == "child-cancel":
            result = cancel_child(args.delegation_id)
        elif args.task_command == "child-accept":
            result = accept_child(args.delegation_id, accept_result=not args.reject,
                                  reason=args.reason)
            result = result.__dict__
        else:
            result = load_child_contract(args.delegation_id).__dict__
    except (KeyError, ValueError, RuntimeError, PermissionError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print_json(data=result)
    return 0


def command_task_child_create(args):
    try:
        contract = json.loads(args.contract_json)
        if not isinstance(contract, dict):
            raise ValueError("--contract-json must decode to an object")
        record = create_child(args.task_id, args.goal, **contract)
    except (json.JSONDecodeError, KeyError, ValueError, RuntimeError, PermissionError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print_json(data=record.__dict__)
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


def command_skill(args):
    registry = SkillRegistry()
    try:
        if args.skill_command == "list":
            data = [item.summary() | {"valid": item.valid, "errors": list(item.errors)}
                    for item in registry.discover(project_path=args.project, role=args.role,
                                                  include_invalid=args.invalid)]
        elif args.skill_command == "show":
            skill = registry.load(args.name, project_path=args.project, role=args.role)
            data = skill.descriptor.summary() | {"instructions": skill.instructions,
                                                 "resources": list(skill.resources)}
        else:
            data = registry.doctor(project_path=args.project, role=args.role)
    except (KeyError, ValueError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print_json(data=data)
    return 0


def command_mcp(args):
    try:
        if args.mcp_command == "list":
            data = [{"name": item.name, "transport": item.transport,
                     "enabled": item.enabled, "tool_filter": item.tool_filter,
                     "effect_policy": item.effect_policy} for item in list_mcp_servers()]
        elif args.mcp_command == "register":
            config = json.loads(args.config_json)
            tool_filter = json.loads(args.filter_json)
            effects = json.loads(args.effects_json)
            item = register_mcp_server(args.name, args.transport, config,
                                       tool_filter=tool_filter, effect_policy=effects)
            data = {"name": item.name, "transport": item.transport,
                    "enabled": item.enabled}
        elif args.mcp_command in {"enable", "disable"}:
            item = set_mcp_enabled(args.name, args.mcp_command == "enable")
            data = {"name": item.name, "enabled": item.enabled}
        else:
            client = MCPClient(args.name, connection_approval_id=args.connect_approval)
            try:
                if args.mcp_command == "tools":
                    data = client.discover_tools()
                else:
                    arguments = json.loads(args.arguments_json)
                    result = client.call_tool(args.tool, arguments, target=args.target,
                                              approval_id=args.approval)
                    data = {"tool": result.tool, "state": result.state,
                            "data": result.data, "error": result.error,
                            "action_ids": list(result.action_ids),
                            "evidence_ids": list(result.evidence_ids)}
            finally:
                client.close()
    except (json.JSONDecodeError, KeyError, ValueError, RuntimeError, PermissionError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print_json(data=data)
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


def command_context_epochs(task_id, limit):
    console.print_json(data=[{
        "id": epoch.id, "task_id": epoch.task_id, "epoch": epoch.epoch,
        "from_message_seq": epoch.from_message_seq,
        "through_message_seq": epoch.through_message_seq,
        "checkpoint_id": epoch.checkpoint_id,
        "archived_messages": len(epoch.archived_messages), "created_at": epoch.created_at,
    } for epoch in list_epochs(task_id, limit)])
    return 0


def command_context_search(conversation_id, query, limit):
    console.print_json(data=search_transcript(conversation_id, query, limit=limit))
    return 0


def command_scope(args):
    if args.scope_command == "rules":
        console.print_json(data=list_policy_rules())
        return 0
    if args.scope_command == "rule-add":
        rule_id = add_policy_rule(
            args.effect, args.action, target=args.target or "",
            target_kind="path" if args.path else None,
        )
        console.print(rule_id)
        return 0
    request = ScopeRequest(
        tool=args.tool, effect=args.effect, target=args.target or "",
        arguments=json.loads(args.arguments), task_id=args.task,
        session_id=args.session, allowed_paths=tuple(args.allow_path),
        allowed_hosts=tuple(args.allow_host), destructive=args.destructive,
        elevated=args.elevated, sandbox_escape=args.sandbox_escape,
    )
    decision = ScopeGuard().evaluate(request)
    console.print_json(data={
        "action": decision.action, "risk_class": decision.risk_class,
        "effect": decision.effect, "target": decision.target,
        "reason": decision.reason, "rule_id": decision.rule_id,
        "normalized_arguments": decision.normalized_arguments,
    })
    return 1 if decision.action == "deny" else 0


def command_approvals(args):
    broker = ApprovalBroker()
    if args.approve or args.deny:
        approval = broker.decide(
            args.approve or args.deny, approve=bool(args.approve), reason=args.reason,
        )
        console.print(f"{approval.id}: {approval.state}")
        return 0
    console.print_json(data=[{
        "id": item.id, "state": item.state, "risk_class": item.risk_class,
        "tool": item.tool, "target": item.target, "scope": item.scope,
        "task_id": item.task_id, "session_id": item.session_id,
        "created_at": item.created_at, "decision_reason": item.decision_reason,
    } for item in broker.list(state=args.state, limit=args.limit)])
    return 0


def command_audit(args):
    if args.action_id:
        rows = [load_action(args.action_id)]
    else:
        rows = list_audit_actions(task_id=args.task, state=args.state, limit=args.limit)
    console.print_json(data=[{
        "id": item.id, "task_id": item.task_id, "session_id": item.session_id,
        "event_uuid": item.event_uuid, "tool": item.tool,
        "arguments": item.normalized_arguments, "target": item.target,
        "effect": item.effect, "risk_class": item.risk_class,
        "policy_action": item.policy_action, "policy_reason": item.policy_reason,
        "approval_id": item.approval_id, "state": item.state,
        "result": item.result, "created_at": item.created_at,
        "started_at": item.started_at, "completed_at": item.completed_at,
    } for item in rows])
    return 0


def command_execution_backend(args):
    backends = {
        "host": HostBackend(), "container": ContainerBackend(), "ssh": SSHBackend(),
    }
    names = [args.backend] if args.backend else list(backends)
    rows = []
    for name in names:
        status = backends[name].status()
        rows.append({
            "backend": status.backend, "available": status.available,
            "support": status.support, "message": status.message,
        })
    console.print_json(data=rows)
    return 0 if all(row["available"] for row in rows) or not args.backend else 1


def command_tool_list():
    console.print_json(data=[{
        "name": item.name, "capability": item.capability, "effect": item.effect,
        "native": item.native, "available": item.available, "support": item.support,
    } for item in ToolRegistry().list()])
    return 0


def command_evidence(args):
    console.print_json(data=[{
        "id": item.id, "task_id": item.task_id, "event_uuid": item.event_uuid,
        "type": item.evidence_type, "source": item.source,
        "sha256": item.content_sha256, "result_ref": item.result_ref,
        "metadata": item.metadata, "created_at": item.created_at,
    } for item in list_evidence_records(
        task_id=args.task, evidence_type=args.type, limit=args.limit,
    )])
    return 0


def command_workspace(args):
    try:
        if args.workspace_command == "list":
            rows = list_workspace_checkpoints(task_id=args.task, limit=args.limit)
            console.print_json(data=[{"id": row.id, "task_id": row.task_id,
                                      "kind": row.kind, "root": row.root,
                                      "state": row.state, "metadata": row.metadata,
                                      "created_at": row.created_at,
                                      "restored_at": row.restored_at} for row in rows])
            return 0
        if args.workspace_command == "checkpoint":
            recovery = WorkspaceRecovery((args.root,))
            result = (recovery.create_filesystem(args.root, args.path, task_id=args.task)
                      if args.path else recovery.create_git(args.root, task_id=args.task))
        else:
            checkpoint = load_workspace_checkpoint(args.checkpoint_id)
            recovery = WorkspaceRecovery((checkpoint.root,))
            if args.workspace_command == "preview":
                result = recovery.preview(args.checkpoint_id, task_id=args.task)
            else:
                result = recovery.rollback(args.checkpoint_id, approval_id=args.approval,
                                           task_id=args.task)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    console.print_json(data={"tool": result.tool, "state": result.state,
                             "data": result.data, "error": result.error,
                             "evidence_ids": result.evidence_ids})
    return 0 if result.succeeded else 1


def build_parser():
    parser = argparse.ArgumentParser(prog="tars")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat")
    chat.add_argument("--role", default=None)
    temporary = sub.add_parser("temporary", help="start an ephemeral conversation")
    temporary.add_argument("--role", default=None)

    sub.add_parser("status")
    sub.add_parser("agents")
    sub.add_parser("paths")
    sub.add_parser("doctor")
    sub.add_parser("help", help="show top-level help")

    scope = sub.add_parser("scope", help="inspect deterministic execution policy")
    scope_sub = scope.add_subparsers(dest="scope_command", required=True)
    scope_explain = scope_sub.add_parser("explain")
    scope_explain.add_argument("tool")
    scope_explain.add_argument("effect", choices=sorted({
        "read", "write", "execute", "network", "service", "remote", "secret",
        "elevated", "destructive", "sandbox_escape",
    }))
    scope_explain.add_argument("target", nargs="?", default="")
    scope_explain.add_argument("--arguments", default="{}")
    scope_explain.add_argument("--task")
    scope_explain.add_argument("--session")
    scope_explain.add_argument("--allow-path", action="append", default=[])
    scope_explain.add_argument("--allow-host", action="append", default=[])
    scope_explain.add_argument("--destructive", action="store_true")
    scope_explain.add_argument("--elevated", action="store_true")
    scope_explain.add_argument("--sandbox-escape", action="store_true")
    scope_sub.add_parser("rules")
    scope_rule = scope_sub.add_parser("rule-add")
    scope_rule.add_argument("effect", choices=sorted({
        "read", "write", "execute", "network", "service", "remote", "secret",
        "elevated", "destructive", "sandbox_escape",
    }))
    scope_rule.add_argument("action", choices=["allow", "deny", "ask"])
    scope_rule.add_argument("target", nargs="?", default="")
    scope_rule.add_argument("--path", action="store_true", help="treat target as a filesystem root")

    approvals = sub.add_parser("approvals", help="inspect or decide approval requests")
    approval_decision = approvals.add_mutually_exclusive_group()
    approval_decision.add_argument("--approve")
    approval_decision.add_argument("--deny")
    approvals.add_argument("--reason", default="")
    approvals.add_argument("--state", choices=["pending", "approved", "denied", "expired", "consumed"])
    approvals.add_argument("--limit", type=int, default=50)

    audit = sub.add_parser("audit", help="inspect guarded action truth")
    audit.add_argument("action_id", nargs="?")
    audit.add_argument("--task")
    audit.add_argument("--state", choices=["proposed", "running", "succeeded", "failed", "denied", "cancelled", "unknown"])
    audit.add_argument("--limit", type=int, default=50)

    execution_backend = sub.add_parser(
        "execution-backend", help="inspect host, container and SSH execution backends",
    )
    execution_backend.add_argument("backend", nargs="?", choices=["host", "container", "ssh"])
    tool = sub.add_parser("tool", help="inspect native semantic tools")
    tool.add_subparsers(dest="tool_command", required=True).add_parser("list")
    evidence = sub.add_parser("evidence", help="inspect task evidence records")
    evidence.add_argument("--task")
    evidence.add_argument("--type")
    evidence.add_argument("--limit", type=int, default=50)
    workspace = sub.add_parser("workspace", help="inspect and recover bounded workspace checkpoints")
    workspace_sub = workspace.add_subparsers(dest="workspace_command", required=True)
    workspace_list = workspace_sub.add_parser("list")
    workspace_list.add_argument("--task")
    workspace_list.add_argument("--limit", type=int, default=50)
    workspace_checkpoint = workspace_sub.add_parser("checkpoint")
    workspace_checkpoint.add_argument("root")
    workspace_checkpoint.add_argument("--path", action="append", default=[])
    workspace_checkpoint.add_argument("--task")
    workspace_preview = workspace_sub.add_parser("preview")
    workspace_preview.add_argument("checkpoint_id")
    workspace_preview.add_argument("--task")
    workspace_rollback = workspace_sub.add_parser("rollback")
    workspace_rollback.add_argument("checkpoint_id")
    workspace_rollback.add_argument("--approval", required=True)
    workspace_rollback.add_argument("--task")

    skill = sub.add_parser("skill", help="discover and validate procedural skills")
    skill_sub = skill.add_subparsers(dest="skill_command", required=True)
    for name in ("list", "doctor"):
        item = skill_sub.add_parser(name)
        item.add_argument("--project")
        item.add_argument("--role")
        if name == "list":
            item.add_argument("--invalid", action="store_true")
    skill_show = skill_sub.add_parser("show")
    skill_show.add_argument("name")
    skill_show.add_argument("--project")
    skill_show.add_argument("--role")

    mcp = sub.add_parser("mcp", help="manage guarded MCP interoperability")
    mcp_sub = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_sub.add_parser("list")
    mcp_register = mcp_sub.add_parser("register")
    mcp_register.add_argument("name")
    mcp_register.add_argument("transport", choices=["stdio", "streamable-http"])
    mcp_register.add_argument("--config-json", required=True)
    mcp_register.add_argument("--filter-json", default="{}")
    mcp_register.add_argument("--effects-json", default="{}")
    for name in ("enable", "disable"):
        item = mcp_sub.add_parser(name)
        item.add_argument("name")
    mcp_tools = mcp_sub.add_parser("tools")
    mcp_tools.add_argument("name")
    mcp_tools.add_argument("--connect-approval")
    mcp_call = mcp_sub.add_parser("call")
    mcp_call.add_argument("name")
    mcp_call.add_argument("tool")
    mcp_call.add_argument("--arguments-json", default="{}")
    mcp_call.add_argument("--target", default="")
    mcp_call.add_argument("--approval")
    mcp_call.add_argument("--connect-approval")

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
    memory_maintain = memory_sub.add_parser("maintain")
    memory_maintain.add_argument(
        "--trigger", default="explicit",
        choices=["explicit", "session_close", "context_rollover", "scheduled"],
    )
    memory_maintain.add_argument("--apply", action="store_true")
    memory_runs = memory_sub.add_parser("maintenance-runs")
    memory_runs.add_argument("--limit", type=int, default=50)

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
    runtime_route = runtime_sub.add_parser("route", help="inspect an exact local Role route")
    runtime_route.add_argument("role")
    runtime_route.add_argument("--task")
    runtime_route.add_argument("--capability", action="append", default=[])
    runtime_route.add_argument("--context-tokens", type=int, default=0)
    runtime_route.add_argument("--reasoning", action="store_true")
    runtime_route.add_argument("--tools", action="store_true")

    schedule = sub.add_parser("schedule", help="durable model-free scheduling")
    schedule_sub = schedule.add_subparsers(dest="schedule_command", required=True)
    schedule_sub.add_parser("list")
    schedule_sub.add_parser("status")
    schedule_show = schedule_sub.add_parser("show")
    schedule_show.add_argument("schedule_id")
    schedule_add = schedule_sub.add_parser("add")
    schedule_add.add_argument("task_id")
    schedule_add.add_argument("kind", choices=["one-shot", "recurring", "condition"])
    schedule_add.add_argument("expression")
    schedule_add.add_argument("--next")
    schedule_add.add_argument("--missed", choices=["skip", "run-once", "catch-up"], default="run-once")
    schedule_add.add_argument("--max-catch-up", type=int, default=1)
    schedule_add.add_argument("--concurrency-key", default="default")
    schedule_add.add_argument("--max-concurrency", type=int, default=1)
    schedule_add.add_argument("--delivery-target", default="")
    for name in ("pause", "resume", "remove"):
        command = schedule_sub.add_parser(name)
        command.add_argument("schedule_id")
    schedule_edit = schedule_sub.add_parser("edit")
    schedule_edit.add_argument("schedule_id")
    schedule_edit.add_argument("--expression")
    schedule_edit.add_argument("--next")
    schedule_edit.add_argument("--missed", choices=["skip", "run-once", "catch-up"])
    schedule_edit.add_argument("--max-catch-up", type=int)
    schedule_edit.add_argument("--concurrency-key")
    schedule_edit.add_argument("--max-concurrency", type=int)
    schedule_edit.add_argument("--delivery-target")
    schedule_runs = schedule_sub.add_parser("runs")
    schedule_runs.add_argument("schedule_id", nargs="?")
    schedule_runs.add_argument("--limit", type=int, default=50)
    schedule_sub.add_parser("run-due")

    core = sub.add_parser("core", help="authoritative authenticated Core API")
    core_sub = core.add_subparsers(dest="core_command", required=True)
    core_serve = core_sub.add_parser("serve")
    core_serve.add_argument("--host")
    core_serve.add_argument("--port", type=int)
    core_serve.add_argument("--allow-remote", action=argparse.BooleanOptionalAction,
                            default=None)
    core_serve.add_argument("--cert")
    core_serve.add_argument("--key")

    client = sub.add_parser("client", help="Core client pairing and revocation")
    client_sub = client.add_subparsers(dest="client_command", required=True)
    client_sub.add_parser("list")
    client_pair = client_sub.add_parser("pair")
    client_pair.add_argument("--permission", action="append",
                             choices=sorted(CLIENT_PERMISSIONS))

    client_pair.add_argument("--principal", default="local-owner")
    client_pair.add_argument("--ttl", type=int, default=300)
    client_revoke = client_sub.add_parser("revoke")
    client_revoke.add_argument("client_id")

    extension = sub.add_parser("extension", help="inspect built-in and third-party boundaries")
    extension_sub = extension.add_subparsers(dest="extension_command", required=True)
    extension_sub.add_parser("list")

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
    task_control = task_sub.add_parser("control")
    task_control.add_argument("task_id")
    task_control.add_argument("kind", choices=["cancel", "interrupt", "approval", "redirect",
                                               "message", "pause", "resume"])
    task_control.add_argument("message", nargs="?", default="")
    task_controls = task_sub.add_parser("controls")
    task_controls.add_argument("task_id")
    task_controls.add_argument("--limit", type=int, default=50)
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

    for name in ("child-show", "child-cancel"):
        child_command = task_sub.add_parser(name)
        child_command.add_argument("delegation_id")
    child_accept = task_sub.add_parser("child-accept")
    child_accept.add_argument("delegation_id")
    child_accept.add_argument("--reject", action="store_true")
    child_accept.add_argument("--reason", default="")
    child_create = task_sub.add_parser("child-create")
    child_create.add_argument("task_id")
    child_create.add_argument("goal")
    child_create.add_argument("--contract-json", required=True)

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
    context_epochs = context_sub.add_parser("epochs")
    context_epochs.add_argument("task_id")
    context_epochs.add_argument("--limit", type=int, default=50)
    context_search = context_sub.add_parser("search")
    context_search.add_argument("conversation_id")
    context_search.add_argument("query")
    context_search.add_argument("--limit", type=int, default=50)

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
    if args.command == "temporary":
        return run_temporary(cfg, role_id=args.role or default_role_id())
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
    if args.command == "scope":
        try:
            return command_scope(args)
        except (json.JSONDecodeError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            return 2
    if args.command == "approvals":
        try:
            return command_approvals(args)
        except (KeyError, RuntimeError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            return 2
    if args.command == "audit":
        try:
            return command_audit(args)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            return 2
    if args.command == "execution-backend":
        return command_execution_backend(args)
    if args.command == "tool":
        return command_tool_list()
    if args.command == "evidence":
        return command_evidence(args)
    if args.command == "workspace":
        return command_workspace(args)
    if args.command == "skill":
        return command_skill(args)
    if args.command == "mcp":
        return command_mcp(args)
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
        if args.runtime_command == "route":
            return command_runtime_route(cfg, args)

    if args.command == "schedule":
        if args.schedule_command == "list":
            return command_schedule_list()
        if args.schedule_command == "status":
            report = scheduler_health(condition_registry(cfg))
            console.print_json(data=report)
            return 0 if report["ok"] else 1
        if args.schedule_command == "show":
            return command_schedule_show(args.schedule_id)
        if args.schedule_command == "add":
            return command_schedule_add(cfg, args)
        if args.schedule_command == "pause":
            set_schedule_enabled(args.schedule_id, False)
            return 0
        if args.schedule_command == "resume":
            set_schedule_enabled(args.schedule_id, True)
            return 0
        if args.schedule_command == "remove":
            remove_schedule(args.schedule_id)
            return 0
        if args.schedule_command == "edit":
            item = edit_schedule(
                args.schedule_id, expression=args.expression, next_run_at=args.next,
                missed_policy=args.missed, max_catch_up=args.max_catch_up,
                concurrency_key=args.concurrency_key,
                max_concurrency=args.max_concurrency,
                delivery_target=args.delivery_target)
            console.print(f"[green]Updated[/green] {item.id} · revision {item.revision}")
            return 0
        if args.schedule_command == "runs":
            return command_schedule_runs(args.schedule_id, args.limit)
        if args.schedule_command == "run-due":
            return command_schedule_run_due(cfg)

    if args.command == "core" and args.core_command == "serve":
        return command_core_serve(cfg, args)

    if args.command == "client":
        if args.client_command == "list":
            return command_client_list()
        if args.client_command == "pair":
            pairing = create_pairing(
                permissions=args.permission or DEFAULT_PERMISSIONS,
                principal_id=args.principal, ttl_seconds=args.ttl)
            console.print_json(data=pairing)
            return 0

        if args.client_command == "revoke":
            client = revoke_core_client(args.client_id)
            console.print(f"[green]Revoked[/green] {client.id} · {client.name}")
            return 0

    if args.command == "extension" and args.extension_command == "list":
        return command_extension_list(cfg)

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
        if args.task_command == "control":
            return command_task_control(args.task_id, args.kind, args.message)
        if args.task_command == "controls":
            return command_task_controls(args.task_id, args.limit)
        if args.task_command == "delegate":
            return command_task_delegate(args)
        if args.task_command == "delegations":
            return command_task_delegations(args.task_id, args.limit)
        if args.task_command == "delegation-show":
            return command_task_delegation_show(args.delegation_id)
        if args.task_command == "delegation-complete":
            return command_task_delegation_complete(args)
        if args.task_command in {"child-show", "child-cancel", "child-accept"}:
            return command_task_child_control(args)
        if args.task_command == "child-create":
            return command_task_child_create(args)
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
        if args.context_command == "epochs":
            return command_context_epochs(args.task_id, args.limit)
        if args.context_command == "search":
            return command_context_search(args.conversation_id, args.query, args.limit)

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
