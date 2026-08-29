import json
import platform
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import CALIBRATION_ROOT, MODEL_CALIBRATION_ROOT
from .models import RuntimeProfile
from .registry import ensure_registry


def _cmd_text(args):
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = (result.stdout or result.stderr).strip()
        return text
    except Exception:
        return ""


def capture_fingerprint():
    llama_server = (
        Path.home()
        / ".local/src/llama.cpp/build-cuda/bin/llama-server"
    )
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "power_profile": _cmd_text(["powerprofilesctl", "get"]),
        "llama_cpp": _cmd_text([str(llama_server), "--version"]),
    }


def _latest_fit_file(filename):
    runs = CALIBRATION_ROOT / "runs"
    if not runs.exists():
        return None

    for run in sorted(runs.iterdir(), reverse=True):
        candidate = run / "fit-kv" / filename
        if candidate.exists():
            return candidate
    return None


def _fit_details(alias, context):
    filename = f"{alias}-{context}-q8-q8.args"
    path = _latest_fit_file(filename)
    if path is None:
        return {}

    tokens = shlex.split(path.read_text(encoding="utf-8").strip())
    result = {"source_fit": str(path)}

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "-ngl" and i + 1 < len(tokens):
            result["ngl"] = tokens[i + 1]
            i += 2
            continue
        if token == "-ot" and i + 1 < len(tokens):
            result["tensor_overrides"] = tokens[i + 1]
            i += 2
            continue
        i += 1

    return result


def _profile(context, threads, *, alias, metrics=None):
    data = {
        "context": context,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "cpus": "0-11",
        "threads": threads,
        "batch_threads": threads,
    }
    data.update(_fit_details(alias, context))
    if metrics:
        data["metrics"] = metrics
    return data


def seed_payloads(registry=None):
    registry = registry or ensure_registry()
    by_alias = registry["models"]

    return {
        "qwen3.5-9b": {
            "schema": 1,
            "model_alias": "qwen3.5-9b",
            "model_sha256": by_alias["qwen3.5-9b"]["sha256"],
            "status": "ready",
            "depth": "max",
            "source": "manual-max-run-20260829",
            "profiles": {
                "compact": _profile(
                    32768, 2, alias="daily",
                    metrics={"pp_tps": 2498.21, "tg_tps": 41.36},
                ),
                "normal": _profile(
                    65536, 2, alias="daily",
                    metrics={
                        "pp_tps": 2331.99,
                        "tg_tps": 39.78,
                        "pressure_75_tg_tps": 36.13,
                    },
                ),
                "extended": _profile(
                    69632, 2, alias="daily",
                    metrics={
                        "pressure_25_tg_tps": 39.96,
                        "pressure_75_tg_tps": 35.87,
                    },
                ),
            },
            "reasonable_max_context": 69632,
        },
        "kat-coder-v2.5-dev": {
            "schema": 1,
            "model_alias": "kat-coder-v2.5-dev",
            "model_sha256": by_alias["kat-coder-v2.5-dev"]["sha256"],
            "status": "ready",
            "depth": "max",
            "source": "manual-max-run-20260829",
            "profiles": {
                "compact": _profile(
                    32768, 12, alias="coder",
                    metrics={"pp_tps": 650.20, "tg_tps": 37.29},
                ),
                "normal": _profile(
                    196608, 12, alias="coder",
                    metrics={
                        "pp_tps": 529.38,
                        "tg_tps": 33.74,
                        "pressure_75_tg_tps": 23.59,
                    },
                ),
                "extended": _profile(
                    262144, 12, alias="coder",
                    metrics={
                        "pp_tps": 515.21,
                        "tg_tps": 32.64,
                        "pressure_75_tg_tps": 20.30,
                    },
                ),
            },
            "reasonable_max_context": 262144,
        },
        "agents-a1": {
            "schema": 1,
            "model_alias": "agents-a1",
            "model_sha256": by_alias["agents-a1"]["sha256"],
            "status": "ready",
            "depth": "max",
            "source": "manual-max-run-20260829",
            "profiles": {
                "compact": _profile(
                    32768, 12, alias="operator",
                    metrics={"pp_tps": 657.42, "tg_tps": 36.75},
                ),
                "normal": _profile(
                    196608, 12, alias="operator",
                    metrics={
                        "pp_tps": 506.38,
                        "tg_tps": 33.67,
                        "pressure_75_tg_tps": 23.60,
                    },
                ),
                "extended": _profile(
                    262144, 12, alias="operator",
                    metrics={
                        "pp_tps": 491.77,
                        "tg_tps": 32.68,
                        "pressure_75_tg_tps": 20.24,
                    },
                ),
            },
            "reasonable_max_context": 262144,
        },
    }


def calibration_path(sha256):
    return MODEL_CALIBRATION_ROOT / f"{sha256}.json"


def ensure_seed_calibrations():
    registry = ensure_registry()
    MODEL_CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)

    for alias, payload in seed_payloads(registry).items():
        path = calibration_path(payload["model_sha256"])
        if path.exists():
            continue
        payload["fingerprint"] = capture_fingerprint()
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def load_calibration(alias):
    registry = ensure_registry()
    info = registry["models"].get(alias)
    if info is None:
        raise KeyError(f"unknown model alias: {alias}")

    path = calibration_path(info["sha256"])
    if not path.exists():
        raise FileNotFoundError(
            f"no calibration for {alias}: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_profile(alias, profile_name="normal"):
    data = load_calibration(alias)
    profiles = data.get("profiles", {})
    if profile_name not in profiles:
        raise KeyError(
            f"{alias} has no calibration profile {profile_name!r}"
        )
    return RuntimeProfile.from_dict(
        profile_name,
        profiles[profile_name],
    )


def list_calibrations():
    registry = ensure_registry()
    rows = []

    for alias, model in registry["models"].items():
        path = calibration_path(model["sha256"])
        if not path.exists():
            rows.append({
                "alias": alias,
                "status": "missing",
                "depth": "-",
                "profiles": [],
            })
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "alias": alias,
            "status": data.get("status", "unknown"),
            "depth": data.get("depth", "unknown"),
            "profiles": list(data.get("profiles", {}).keys()),
            "reasonable_max_context": data.get("reasonable_max_context"),
        })

    return rows
