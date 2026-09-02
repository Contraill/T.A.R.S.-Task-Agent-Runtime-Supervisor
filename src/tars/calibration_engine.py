from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Protocol

from .calibration import calibration_path, load_calibration
from .config import CALIBRATION_ROOT, LLAMA_BENCH_PATH, LLAMA_SERVER_PATH
from .model_integrity import (
    current_artifact_handle,
    current_model_artifact_handle,
    inspect_model_artifact,
    require_current_model_artifact,
)
from .ownership import (
    MODEL_EXECUTION_RESOURCE,
    add_external_process_fence,
    held_by,
    model_execution_owner,
    model_execution_scope,
    remove_external_process_fence,
)
from .process_supervision import spawn_supervised
from .registry import get_model
from .runtime_config import GLOBAL_TTL, UNLOAD_TIMEOUT

DEPTH_RANK = {"min": 1, "mid": 2, "max": 3}
CALIBRATION_SLOT_WAIT_SECONDS = 30.0
CALIBRATION_PROCESS_TIMEOUT_SECONDS = 1800.0
CALIBRATION_POLL_SECONDS = 0.25
STAGES = {
    "min": ("fit", "throughput", "finalize", "zero_idle"),
    "mid": ("fit", "throughput", "context_kv", "cpu", "resources", "finalize", "zero_idle"),
    "max": ("fit", "throughput", "context_kv", "cpu", "resources", "placement", "pressure", "finalize", "zero_idle"),
}


@dataclass(frozen=True)
class Candidate:
    context: int
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    cpus: str = "0"
    threads: int = 1
    batch_threads: int = 1
    ngl: str | int | None = "all"
    tensor_overrides: str | None = None


@dataclass(frozen=True)
class Measurement:
    success: bool
    pp_tps: float = 0.0
    tg_tps: float = 0.0
    ram_peak_bytes: int = 0
    vram_peak_bytes: int = 0
    error: str = ""
    placement: dict | None = None


class CalibrationProbe(Protocol):
    def fingerprint(self) -> dict: ...
    def measure(self, model_path: Path, candidate: Candidate, *, pressure: float = 0.0,
                fit: bool = False) -> Measurement: ...
    def zero_idle(self) -> dict: ...


def _require_model_execution_owner():
    owner = model_execution_owner()
    if owner is None or not held_by(*MODEL_EXECUTION_RESOURCE, owner):
        raise RuntimeError("calibration probe requires model execution ownership")
    return owner


def _spawn_fenced(args, **kwargs):
    owner = _require_model_execution_owner()
    supervised = spawn_supervised(args, start_gated=True, **kwargs)
    identities = (
        {"pid": supervised.process.pid, "start": supervised.supervisor_start},
        {"pid": supervised.child_pid, "start": supervised.child_start},
    )
    try:
        if not add_external_process_fence(
                *MODEL_EXECUTION_RESOURCE, owner, identities):
            raise RuntimeError("model execution ownership was lost before process start")
        supervised.release_start_gate()
        return supervised, owner, identities
    except Exception:
        try:
            supervised.stop(timeout=5)
        finally:
            supervised.close_control()
        raise


def _finish_fenced(supervised, owner, identities):
    try:
        if supervised.process.poll() is None:
            supervised.stop(timeout=5)
        if not remove_external_process_fence(
                *MODEL_EXECUTION_RESOURCE, owner, identities):
            raise RuntimeError("model execution ownership was lost during process cleanup")
    finally:
        supervised.close_control()


def _command_text(args, timeout=15) -> str:
    supervised = None
    owner = None
    identities = ()
    try:
        supervised, owner, identities = _spawn_fenced(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = supervised.process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            supervised.stop(timeout=5)
            stdout, stderr = supervised.process.communicate()
        return (stdout or stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""
    finally:
        if supervised is not None:
            _finish_fenced(supervised, owner, identities)


def _mem_available() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _gpu_snapshot() -> list[dict]:
    text = _command_text([
        "nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append({"uuid": parts[0], "name": parts[1], "memory_total_mib": int(parts[2]),
                         "memory_used_mib": int(parts[3]), "driver": parts[4]})
    return rows


def _process_rss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _gpu_runtime_statuses() -> list[str]:
    statuses = []
    for device in sorted(Path("/sys/bus/pci/devices").glob("*")):
        try:
            if (device / "vendor").read_text().strip() == "0x10de":
                statuses.append((device / "power/runtime_status").read_text().strip())
        except OSError:
            continue
    return statuses


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def hardware_fingerprint() -> dict:
    _require_model_execution_owner()
    bench = inspect_model_artifact(LLAMA_BENCH_PATH)
    server = inspect_model_artifact(LLAMA_SERVER_PATH)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    stable = {
        "schema": 1,
        "machine": platform.machine(),
        "kernel": platform.release(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count() or 1,
        "cpu_affinity": affinity,
        "memory_total_bytes": _mem_total(),
        "gpus": _gpu_snapshot(),
        "llama_bench": f"sha256:{bench.sha256}",
        "llama_server": f"sha256:{server.sha256}",
        "llama_bench_sha256": bench.sha256,
        "llama_server_sha256": server.sha256,
    }
    stable["digest"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    stable["captured_at"] = datetime.now(timezone.utc).isoformat()
    return stable


def _mem_total() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class LlamaBenchProbe:
    """Objective llama.cpp calibration probe used by the local reference backend."""

    def __init__(self):
        self._bench_sha256 = ""
        self._model = None

    def bind_model(self, model):
        self._model = model

    def fingerprint(self) -> dict:
        fingerprint = hardware_fingerprint()
        self._bench_sha256 = fingerprint["llama_bench_sha256"]
        return fingerprint

    def measure(self, model_path: Path, candidate: Candidate, *, pressure: float = 0.0,
                fit: bool = False) -> Measurement:
        _require_model_execution_owner()
        expected_bench = self._bench_sha256
        if not expected_bench:
            expected_bench = inspect_model_artifact(LLAMA_BENCH_PATH).sha256
        bench_handle = current_artifact_handle(
            LLAMA_BENCH_PATH, expected_bench, label="llama-bench")
        if self._model is not None:
            if Path(model_path).expanduser().absolute() != Path(
                    self._model.path).expanduser().absolute():
                raise RuntimeError("calibration probe model path differs from its binding")
            model_handle = current_model_artifact_handle(self._model)
        else:
            model_inspection = inspect_model_artifact(model_path)
            model_handle = current_artifact_handle(
                model_path, model_inspection.sha256, label="calibration model")
        prompt = max(512, int(candidate.context * pressure)) if pressure else 512
        bench_ref = f"/proc/self/fd/{bench_handle.fileno()}"
        model_ref = f"/proc/self/fd/{model_handle.fileno()}"
        args = [
            "taskset", "-c", candidate.cpus, bench_ref,
            "-m", model_ref, "-p", str(prompt), "-n", "64",
            "-r", "1", "-o", "json", "-ctk", candidate.cache_type_k,
            "-ctv", candidate.cache_type_v, "-t", str(candidate.threads), "-fa", "on",
            "-ngl", "999" if str(candidate.ngl) in {"all", "-1", "None"} else str(candidate.ngl),
        ]
        if fit:
            args.extend(["-fitt", "256", "-fitc", str(candidate.context)])
        if candidate.tensor_overrides:
            args.extend(["-ot", candidate.tensor_overrides])
        before_gpu = _gpu_snapshot()
        supervised = None
        owner = None
        identities = ()
        try:
            supervised, owner, identities = _spawn_fenced(
                args, inherited_fds=(bench_handle.fileno(), model_handle.fileno()),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            process = supervised.process
        except OSError as exc:
            return Measurement(False, error=str(exc))
        try:
            deadline = time.monotonic() + CALIBRATION_PROCESS_TIMEOUT_SECONDS
            ram_peak = 0
            vram_peak_mib = 0
            baseline_vram = sum(row.get("memory_used_mib", 0) for row in before_gpu)
            while process.poll() is None:
                ram_peak = max(ram_peak, _process_rss(supervised.child_pid))
                current_vram = sum(row.get("memory_used_mib", 0) for row in _gpu_snapshot())
                vram_peak_mib = max(vram_peak_mib, max(0, current_vram - baseline_vram))
                if time.monotonic() >= deadline:
                    supervised.stop(timeout=5)
                    stdout, stderr = process.communicate()
                    return Measurement(False, ram_peak_bytes=ram_peak,
                                       vram_peak_bytes=vram_peak_mib * 1024 * 1024,
                                       error="llama-bench timed out")
                time.sleep(CALIBRATION_POLL_SECONDS)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                return Measurement(False, ram_peak_bytes=ram_peak,
                                   vram_peak_bytes=vram_peak_mib * 1024 * 1024,
                                   error=(stderr or stdout)[-1000:])
            try:
                rows = json.loads(stdout)
                if isinstance(rows, dict):
                    rows = [rows]
            except json.JSONDecodeError as exc:
                return Measurement(False, error=f"invalid llama-bench JSON: {exc}")
            pp = [float(row.get("avg_ts", 0)) for row in rows
                  if int(row.get("n_prompt", 0)) > 0]
            tg = [float(row.get("avg_ts", 0)) for row in rows
                  if int(row.get("n_gen", 0)) > 0]
            placement = next(({
                "n_gpu_layers": row.get("n_gpu_layers"),
                "n_cpu_moe": row.get("n_cpu_moe"),
                "tensor_split": row.get("tensor_split"),
            } for row in rows if isinstance(row, dict)), {})
            return Measurement(
                True, max(pp, default=0.0), max(tg, default=0.0), ram_peak,
                vram_peak_mib * 1024 * 1024, placement=placement)
        finally:
            _finish_fenced(supervised, owner, identities)

    def zero_idle(self) -> dict:
        deadline = time.monotonic() + 30
        processes = _command_text(["pgrep", "-af", "[l]lama-server"])
        statuses = _gpu_runtime_statuses()
        while not processes.strip() and statuses and any(x != "suspended" for x in statuses):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.5)
            statuses = _gpu_runtime_statuses()
        return {
            "passed": not bool(processes.strip()) and all(x == "suspended" for x in statuses),
            "llama_server_processes": processes.splitlines(),
            "nvidia_runtime_status": statuses,
            "global_ttl": GLOBAL_TTL,
            "unload_timeout": UNLOAD_TIMEOUT,
        }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _input_digest(stage: str, model_sha: str, fingerprint: str, depth: str, prior: dict) -> str:
    value = {"schema": 1, "stage": stage, "model_sha256": model_sha,
             "fingerprint": fingerprint, "depth": depth, "prior": prior}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _candidate(context: int, threads: int, cpus: str, **changes) -> Candidate:
    values = asdict(Candidate(context=context, threads=threads, batch_threads=threads, cpus=cpus))
    values.update(changes)
    return Candidate(**values)


def _score(measurement: Measurement) -> float:
    return measurement.pp_tps + measurement.tg_tps * 8.0


class CalibrationEngine:
    def __init__(self, probe: CalibrationProbe | None = None, *, root: Path = CALIBRATION_ROOT):
        self.probe = probe or LlamaBenchProbe()
        self.root = root

    def calibrate(self, alias: str, *, depth: str = "min", fresh: bool = False) -> dict:
        if depth not in DEPTH_RANK:
            raise ValueError("calibration depth must be min, mid or max")
        model = get_model(alias)
        if model.backend != "llama.cpp":
            raise ValueError(f"local calibration does not apply to backend {model.backend!r}")
        if not model.path.is_file():
            raise FileNotFoundError(model.path)
        with model_execution_scope(
            operation="calibration",
            timeout=CALIBRATION_SLOT_WAIT_SECONDS,
            metadata={"model_alias": str(alias), "depth": depth},
        ):
            require_current_model_artifact(model)
            bind_model = getattr(self.probe, "bind_model", None)
            if bind_model is not None:
                bind_model(model)
            return self._calibrate_owned(model, alias, depth=depth, fresh=fresh)

    def _calibrate_owned(self, model, alias: str, *, depth: str, fresh: bool) -> dict:
        fingerprint = self.probe.fingerprint()
        fingerprint_digest = fingerprint["digest"]
        try:
            existing = load_calibration(alias)
        except FileNotFoundError:
            existing = None
        if existing and existing.get("status") == "ready":
            old_depth = existing.get("depth", "min")
            same_artifact = existing.get("model_sha256") == model.sha256
            same_runtime = existing.get("fingerprint_digest") == fingerprint_digest
            if (same_artifact and same_runtime
                    and DEPTH_RANK.get(old_depth, 0) > DEPTH_RANK[depth]):
                return {**existing, "protected_from_shallower": True}
            if same_artifact and same_runtime and old_depth == depth and not fresh:
                return {**existing, "resumed": True}

        run_root = self.root / "engine" / alias / fingerprint_digest
        state = {"model": model, "fingerprint": fingerprint, "results": {}, "cache_events": []}
        for stage in STAGES[depth]:
            prior = state["results"]
            digest = _input_digest(stage, model.sha256, fingerprint_digest, depth, prior)
            stage_path = run_root / "stages" / f"{stage}.json"
            cached = None
            if stage_path.exists() and not fresh:
                cached = json.loads(stage_path.read_text(encoding="utf-8"))
                if cached.get("input_digest") != digest:
                    state["cache_events"].append({"stage": stage, "status": "stale"})
                    cached = None
                else:
                    state["cache_events"].append({"stage": stage, "status": "resumed"})
            elif fresh and stage_path.exists():
                state["cache_events"].append({"stage": stage, "status": "fresh"})
            if cached is None:
                output = self._run_stage(stage, state, depth)
                cached = {"schema": 1, "stage": stage, "input_digest": digest,
                          "completed_at": datetime.now(timezone.utc).isoformat(), "output": output}
                _atomic_json(stage_path, cached)
            state["results"][stage] = cached["output"]

        payload = state["results"]["finalize"]
        zero_idle = state["results"]["zero_idle"]
        if not zero_idle.get("passed"):
            raise RuntimeError("Zero-Idle validation failed after calibration")
        payload["zero_idle"] = zero_idle
        payload["fingerprint"] = fingerprint
        payload["fingerprint_digest"] = fingerprint_digest
        payload["stage_cache"] = str(run_root / "stages")
        payload["stage_cache_events"] = state["cache_events"]
        _atomic_json(calibration_path(model.sha256), payload)
        return payload

    def _run_stage(self, stage: str, state: dict, depth: str) -> dict:
        model = state["model"]
        affinity = state["fingerprint"].get("cpu_affinity") or [0]
        cpus = f"{min(affinity)}-{max(affinity)}" if len(affinity) > 1 else str(affinity[0])
        threads = max(1, min(len(affinity), 12))
        native = model.native_context or 32768
        if stage == "fit":
            contexts = sorted(set([min(native, 32768), min(native, 65536)]))
            trials = []
            for context in contexts:
                candidate = _candidate(context, threads, cpus)
                measurement = self.probe.measure(model.path, candidate, fit=True)
                candidate_data = asdict(candidate)
                fitted_ngl = (measurement.placement or {}).get("n_gpu_layers")
                if fitted_ngl is not None:
                    candidate_data["ngl"] = fitted_ngl
                trials.append({"candidate": candidate_data, "measurement": asdict(measurement)})
            successful = [trial for trial in trials if trial["measurement"]["success"]]
            if not successful:
                raise RuntimeError(f"no FIT candidate succeeded for {model.alias}")
            return {"trials": trials, "selected": max(successful, key=lambda x: x["candidate"]["context"])}
        base = state["results"]["fit"]["selected"]["candidate"]
        if stage == "throughput":
            measurement = self.probe.measure(model.path, Candidate(**base))
            if not measurement.success:
                raise RuntimeError(f"throughput measurement failed: {measurement.error}")
            return {"candidate": base, "measurement": asdict(measurement)}
        if stage == "context_kv":
            trials = []
            contexts = sorted(set([min(native, base["context"]), min(native, base["context"] * 2), native]))
            for context in contexts:
                for kv in ("q8_0", "f16"):
                    candidate = Candidate(**{**base, "context": context, "cache_type_k": kv, "cache_type_v": kv})
                    m = self.probe.measure(model.path, candidate, fit=True)
                    trials.append({"candidate": asdict(candidate), "measurement": asdict(m)})
            successful = [trial for trial in trials if trial["measurement"]["success"]]
            if not successful:
                raise RuntimeError("context/KV search found no usable candidate")
            selected = max(successful, key=lambda x: (x["candidate"]["context"], _score(Measurement(**x["measurement"]))))
            return {"trials": trials, "selected": selected}
        if stage == "cpu":
            source = state["results"].get("context_kv", {}).get("selected", {"candidate": base})["candidate"]
            affinity = state["fingerprint"].get("cpu_affinity") or [0]
            midpoint = max(1, len(affinity) // 2)
            cpu_sets = [affinity, affinity[:midpoint], affinity[midpoint:]]
            trials = []
            for cpu_set in cpu_sets:
                if not cpu_set:
                    continue
                cpu_text = f"{min(cpu_set)}-{max(cpu_set)}" if len(cpu_set) > 1 else str(cpu_set[0])
                count = min(len(cpu_set), threads)
                candidate = Candidate(**{**source, "cpus": cpu_text,
                                          "threads": count, "batch_threads": count})
                m = self.probe.measure(model.path, candidate)
                trials.append({"candidate": asdict(candidate), "measurement": asdict(m)})
            selected = max((x for x in trials if x["measurement"]["success"]),
                           key=lambda x: _score(Measurement(**x["measurement"])))
            return {"trials": trials, "selected": selected}
        if stage == "resources":
            source = state["results"].get("cpu", state["results"]["throughput"])
            return {"ram_available_bytes": _mem_available(), "gpus": _gpu_snapshot(),
                    "selected_measurement": source.get("selected", source).get("measurement", {})}
        if stage == "placement":
            source = state["results"]["cpu"]["selected"]["candidate"]
            trials = []
            for ngl in ("all", 41, 20):
                candidate = Candidate(**{**source, "ngl": ngl})
                m = self.probe.measure(model.path, candidate)
                trials.append({"candidate": asdict(candidate), "measurement": asdict(m)})
            selected = max((x for x in trials if x["measurement"]["success"]),
                           key=lambda x: _score(Measurement(**x["measurement"])))
            return {"trials": trials, "selected": selected}
        if stage == "pressure":
            source = state["results"]["placement"]["selected"]["candidate"]
            trials = []
            for pressure in (0.25, 0.75):
                m = self.probe.measure(model.path, Candidate(**source), pressure=pressure)
                trials.append({"pressure": pressure, "measurement": asdict(m)})
            return {"candidate": source, "trials": trials}
        if stage == "finalize":
            return self._finalize(state, depth)
        if stage == "zero_idle":
            return self.probe.zero_idle()
        raise ValueError(stage)

    def _finalize(self, state: dict, depth: str) -> dict:
        model = state["model"]
        results = state["results"]
        selected = results.get("placement", {}).get("selected") or results.get("cpu", {}).get("selected") or results.get("context_kv", {}).get("selected") or results["throughput"]
        candidate = selected["candidate"]
        metric = selected.get("measurement", results["throughput"]["measurement"])
        reasonable_max = min(model.native_context or candidate["context"], candidate["context"])
        compact_context = min(reasonable_max, 32768)
        normal_context = min(reasonable_max, max(compact_context, reasonable_max // 2))
        pressure = results.get("pressure", {}).get("trials", [])

        def profile(context):
            value = {key: candidate[key] for key in (
                "cache_type_k", "cache_type_v", "cpus", "threads", "batch_threads", "ngl", "tensor_overrides")}
            value["context"] = context
            value["metrics"] = {"pp_tps": metric.get("pp_tps", 0), "tg_tps": metric.get("tg_tps", 0),
                                "ram_peak_bytes": metric.get("ram_peak_bytes", 0),
                                "vram_peak_bytes": metric.get("vram_peak_bytes", 0)}
            if pressure:
                value["metrics"]["pressure"] = pressure
            return value

        return {
            "schema": 2, "model_alias": model.alias, "model_sha256": model.sha256,
            "status": "ready", "depth": depth, "source": "calibration-engine",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "profiles": {"compact": profile(compact_context), "normal": profile(normal_context),
                         "extended": profile(reasonable_max), "reasonable_max": profile(reasonable_max)},
            "reasonable_max_context": reasonable_max,
        }
