from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.request

from .calibration import load_calibration
from .config import (
    LLAMA_SERVER_PATH,
    LLAMA_SWAP_CONFIG_PATH,
    LLAMA_SWAP_SERVICE,
    RUNTIME_CONFIG_BACKUP_ROOT,
    runtime_base_url,
)
from .registry import get_model
from .roles import (
    get_role,
    list_roles,
    load_role_registry,
    resolve_role_id,
    save_role_registry,
)

START_PORT = 10001
GLOBAL_TTL = 30
UNLOAD_TIMEOUT = 10
SEND_LOADING_STATE = False
PERFORMANCE_MONITOR_DISABLED = True
CONCURRENCY_LIMIT = 1


@dataclass(frozen=True)
class RuntimeModelPlan:
    role_id: str
    display_name: str
    runtime_id: str
    execution: str
    model_alias: str
    model_name: str
    model_path: Path
    model_sha256: str
    quant: str
    profile_name: str
    context: int
    cache_type_k: str
    cache_type_v: str
    cpus: str
    threads: int
    batch_threads: int
    ngl: str | int | None
    tensor_overrides: str | None

    @property
    def tools(self) -> bool:
        return self.execution == "loop"


@dataclass(frozen=True)
class RuntimePlan:
    models: tuple[RuntimeModelPlan, ...]
    llama_server: Path
    start_port: int = START_PORT
    global_ttl: int = GLOBAL_TTL
    unload_timeout: int = UNLOAD_TIMEOUT
    send_loading_state: bool = SEND_LOADING_STATE
    performance_monitor_disabled: bool = PERFORMANCE_MONITOR_DISABLED


@dataclass(frozen=True)
class RuntimeConfigStatus:
    path: Path
    installed: bool
    matches_generated: bool
    installed_sha256: str | None
    generated_sha256: str
    service_active: bool
    api_healthy: bool


@dataclass(frozen=True)
class RuntimeApplyResult:
    changed: bool
    path: Path
    backup_path: Path | None
    sha256: str
    runtime_ids: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeSwitchResult:
    role_id: str
    model_alias: str
    profile_name: str
    apply: RuntimeApplyResult


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _yaml_quote(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _ngl_text(value) -> str:
    if value is None or str(value).strip() in {"-1", "all", "ALL"}:
        return "all"
    return str(value)


def _runtime_command(model: RuntimeModelPlan, llama_server: Path) -> str:
    args = [
        "taskset", "-c", model.cpus,
        str(llama_server),
        "-m", str(model.model_path),
        "-c", str(model.context),
        "-ngl", _ngl_text(model.ngl),
    ]
    if model.tensor_overrides:
        args.extend(["-ot", model.tensor_overrides])
    args.extend([
        "-fa", "on",
        "--load-mode", "auto",
        "-ctk", model.cache_type_k,
        "-ctv", model.cache_type_v,
        "-t", str(model.threads),
        "-tb", str(model.batch_threads),
        "-np", "1",
        "--no-context-shift",
        "--host", "127.0.0.1",
        "--port",
    ])
    return " ".join(shlex.quote(part) for part in args) + " ${PORT}"


def build_runtime_plan(*, require_files: bool = False) -> RuntimePlan:
    plans = []
    runtime_ids = set()

    for role in list_roles(include_disabled=False):
        if not role.model:
            continue
        if role.runtime_id in runtime_ids:
            raise ValueError(f"duplicate enabled runtime id: {role.runtime_id}")

        model = get_model(role.model)
        if model.backend != "llama.cpp":
            raise ValueError(
                f"role {role.id} uses backend {model.backend!r}; "
                "v0.5.0 RuntimeConfigGenerator only renders llama.cpp models"
            )

        calibration = load_calibration(model.alias)
        if calibration.get("status") != "ready":
            raise RuntimeError(
                f"calibration for {model.alias} is {calibration.get('status', 'unknown')!r}"
            )
        calibrated_sha = calibration.get("model_sha256")
        if calibrated_sha and calibrated_sha != model.sha256:
            raise RuntimeError(
                f"calibration SHA mismatch for {model.alias}; recalibration required"
            )

        profile_data = (calibration.get("profiles") or {}).get(role.profile)
        if not isinstance(profile_data, dict):
            raise KeyError(f"{model.alias} has no calibration profile {role.profile!r}")

        context = int(profile_data["context"])
        if model.native_context and context > model.native_context:
            raise ValueError(
                f"{model.alias}/{role.profile} context {context} exceeds "
                f"native context {model.native_context}"
            )

        plan = RuntimeModelPlan(
            role_id=role.id,
            display_name=role.display_name,
            runtime_id=role.runtime_id,
            execution=role.execution,
            model_alias=model.alias,
            model_name=model.name,
            model_path=model.path,
            model_sha256=model.sha256,
            quant=model.quant,
            profile_name=role.profile,
            context=context,
            cache_type_k=str(profile_data.get("cache_type_k", "f16")),
            cache_type_v=str(profile_data.get("cache_type_v", "f16")),
            cpus=str(profile_data.get("cpus", "0-11")),
            threads=int(profile_data.get("threads", 1)),
            batch_threads=int(profile_data.get("batch_threads", profile_data.get("threads", 1))),
            ngl=profile_data.get("ngl"),
            tensor_overrides=profile_data.get("tensor_overrides"),
        )
        if not plan.cpus:
            raise ValueError(f"empty CPU affinity for {model.alias}/{role.profile}")
        if plan.threads < 1 or plan.batch_threads < 1:
            raise ValueError(f"invalid thread count for {model.alias}/{role.profile}")
        if require_files and not plan.model_path.is_file():
            raise FileNotFoundError(f"model file missing: {plan.model_path}")

        plans.append(plan)
        runtime_ids.add(role.runtime_id)

    if require_files and not LLAMA_SERVER_PATH.is_file():
        raise FileNotFoundError(f"llama-server missing: {LLAMA_SERVER_PATH}")

    return RuntimePlan(models=tuple(plans), llama_server=LLAMA_SERVER_PATH)


def render_runtime_config(plan: RuntimePlan | None = None) -> str:
    plan = plan or build_runtime_plan()
    if not plan.performance_monitor_disabled:
        raise ValueError("Zero-Idle policy forbids llama-swap performance monitoring")
    if plan.global_ttl <= 0 or plan.unload_timeout <= 0:
        raise ValueError("Zero-Idle policy requires finite positive unload timers")

    lines = [
        f"startPort: {plan.start_port}",
        "",
        f"globalTTL: {plan.global_ttl}",
        f"unloadTimeout: {plan.unload_timeout}",
        "sendLoadingState: " + ("true" if plan.send_loading_state else "false"),
        "",
        "performance:",
        "  disabled: " + ("true" if plan.performance_monitor_disabled else "false"),
        "",
        "models:" if plan.models else "models: {}",
    ]

    for model in plan.models:
        lines.extend([
            f"  {model.runtime_id}:",
            f"    name: {_yaml_quote('T.A.R.S. ' + model.display_name)}",
            "    description: " + _yaml_quote(
                f"{model.model_name} {model.quant} — calibrated "
                f"{model.profile_name}: {model.context} ctx"
            ),
            "",
            f"    ttl: {plan.global_ttl}",
            f"    concurrencyLimit: {CONCURRENCY_LIMIT}",
        ])
        if model.tools:
            lines.extend([
                "",
                "    capabilities:",
                "      in:",
                "        - text",
                "      out:",
                "        - text",
                "      tools: true",
                f"      context: {model.context}",
            ])
        lines.extend([
            "",
            "    cmd: |",
            "      " + _runtime_command(model, plan.llama_server),
            "",
            "    checkEndpoint: /health",
            "",
        ])

    rendered = "\n".join(lines).rstrip() + "\n"
    required = (
        "globalTTL: 30",
        "unloadTimeout: 10",
        "sendLoadingState: false",
        "performance:\n  disabled: true",
    )
    missing = [item for item in required if item not in rendered]
    if missing:
        raise RuntimeError("generated config violates Zero-Idle invariants: " + ", ".join(missing))
    return rendered


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        try:
            dir_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _service_active() -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", LLAMA_SWAP_SERVICE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _restart_service() -> None:
    subprocess.run(
        ["systemctl", "--user", "restart", LLAMA_SWAP_SERVICE],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )


def start_runtime_service() -> None:
    subprocess.run(
        ["systemctl", "--user", "start", LLAMA_SWAP_SERVICE], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
    )


def stop_runtime_service() -> None:
    subprocess.run(
        ["systemctl", "--user", "stop", LLAMA_SWAP_SERVICE], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
    )


def runtime_service_logs(*, lines: int = 100, follow: bool = False) -> int:
    args = ["journalctl", "--user", "-u", LLAMA_SWAP_SERVICE, "-n", str(lines)]
    if follow:
        args.append("-f")
    return subprocess.run(args, check=False).returncode


def _api_runtime_ids(cfg, *, timeout=2.0) -> set[str]:
    with urllib.request.urlopen(
        runtime_base_url(cfg) + "/v1/models", timeout=timeout
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }


def _wait_healthy(cfg, expected_ids: set[str], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = "runtime did not become healthy"
    while time.monotonic() < deadline:
        try:
            if not _service_active():
                last_error = f"{LLAMA_SWAP_SERVICE} is not active"
            else:
                actual = _api_runtime_ids(cfg)
                missing = expected_ids - actual
                if not missing:
                    return
                last_error = "missing runtime ids: " + ", ".join(sorted(missing))
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(last_error)


def runtime_config_status(cfg) -> RuntimeConfigStatus:
    generated = render_runtime_config(build_runtime_plan())
    generated_sha = _sha256_text(generated)
    installed = LLAMA_SWAP_CONFIG_PATH.exists()
    installed_text = (
        LLAMA_SWAP_CONFIG_PATH.read_text(encoding="utf-8") if installed else None
    )
    installed_sha = _sha256_text(installed_text) if installed_text is not None else None
    try:
        api_healthy = bool(_api_runtime_ids(cfg))
    except Exception:
        api_healthy = False
    return RuntimeConfigStatus(
        path=LLAMA_SWAP_CONFIG_PATH,
        installed=installed,
        matches_generated=installed_text == generated,
        installed_sha256=installed_sha,
        generated_sha256=generated_sha,
        service_active=_service_active(),
        api_healthy=api_healthy,
    )


def apply_runtime_config(cfg) -> RuntimeApplyResult:
    plan = build_runtime_plan(require_files=True)
    candidate = render_runtime_config(plan)
    digest = _sha256_text(candidate)
    expected_ids = {model.runtime_id for model in plan.models}

    was_active = _service_active()
    previous = None
    backup_path = None
    if LLAMA_SWAP_CONFIG_PATH.exists():
        previous = LLAMA_SWAP_CONFIG_PATH.read_text(encoding="utf-8")
        if previous == candidate:
            if was_active:
                _wait_healthy(cfg, expected_ids)
            else:
                try:
                    start_runtime_service()
                    _wait_healthy(cfg, expected_ids)
                finally:
                    stop_runtime_service()
            return RuntimeApplyResult(
                changed=False,
                path=LLAMA_SWAP_CONFIG_PATH,
                backup_path=None,
                sha256=digest,
                runtime_ids=tuple(sorted(expected_ids)),
            )
        RUNTIME_CONFIG_BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        backup_path = RUNTIME_CONFIG_BACKUP_ROOT / f"llama-swap-{stamp}.yaml"
        shutil.copy2(LLAMA_SWAP_CONFIG_PATH, backup_path)

    _atomic_write(LLAMA_SWAP_CONFIG_PATH, candidate)
    try:
        if was_active:
            _restart_service()
        else:
            start_runtime_service()
        _wait_healthy(cfg, expected_ids)
        if not was_active:
            stop_runtime_service()
    except Exception as exc:
        if previous is None:
            try:
                LLAMA_SWAP_CONFIG_PATH.unlink()
            except FileNotFoundError:
                pass
        else:
            _atomic_write(LLAMA_SWAP_CONFIG_PATH, previous)

        rollback_error = None
        try:
            if was_active:
                _restart_service()
                if previous is not None:
                    _wait_healthy(cfg, expected_ids=set(), timeout=10.0)
            else:
                stop_runtime_service()
        except Exception as rollback_exc:
            rollback_error = rollback_exc

        message = f"runtime apply failed; previous config restored: {exc}"
        if rollback_error is not None:
            message += f"; rollback health warning: {rollback_error}"
        raise RuntimeError(message) from exc

    return RuntimeApplyResult(
        changed=True,
        path=LLAMA_SWAP_CONFIG_PATH,
        backup_path=backup_path,
        sha256=digest,
        runtime_ids=tuple(sorted(expected_ids)),
    )


def switch_role_runtime(
    cfg,
    role_name: str,
    *,
    model_alias: str | None = None,
    profile_name: str | None = None,
    unassign: bool = False,
) -> RuntimeSwitchResult:
    role_id = resolve_role_id(role_name)
    current = get_role(role_id)
    target_model = "" if unassign else (model_alias or current.model)
    target_profile = profile_name or current.profile
    if not target_model and not unassign:
        raise ValueError(f"role {role_id} has no model binding")
    if target_profile not in {"compact", "normal", "extended"}:
        raise ValueError("profile must be compact, normal or extended")

    if target_model:
        model = get_model(target_model)
        calibration = load_calibration(model.alias)
        if calibration.get("status") != "ready":
            raise RuntimeError(f"calibration for {target_model} is not ready")
        if target_profile not in (calibration.get("profiles") or {}):
            raise KeyError(f"{target_model} has no calibration profile {target_profile!r}")

    before = load_role_registry()
    candidate = copy.deepcopy(before)
    info = candidate["roles"][role_id]
    info["model"] = target_model
    info["profile"] = target_profile
    if not unassign and not info.get("enabled", True):
        raise ValueError(f"role {role_id} is disabled")

    save_role_registry(candidate)
    try:
        applied = apply_runtime_config(cfg)
    except Exception:
        save_role_registry(before)
        raise

    return RuntimeSwitchResult(
        role_id=role_id,
        model_alias=target_model,
        profile_name=target_profile,
        apply=applied,
    )
