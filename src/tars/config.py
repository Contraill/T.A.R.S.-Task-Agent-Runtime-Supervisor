from pathlib import Path
import tomllib

CONFIG_PATH = Path.home() / ".config/tars/config.toml"
DATA_ROOT = Path.home() / ".local/share/tars"
STATE_ROOT = Path.home() / ".local/state/tars"
CACHE_ROOT = Path.home() / ".cache/tars"

REGISTRY_PATH = DATA_ROOT / "model-registry.toml"
ROLE_REGISTRY_PATH = DATA_ROOT / "role-registry.toml"
CALIBRATION_ROOT = STATE_ROOT / "calibration"
MODEL_CALIBRATION_ROOT = CALIBRATION_ROOT / "models"
CHAT_STATE_ROOT = STATE_ROOT / "chat"
STATE_DB_PATH = STATE_ROOT / "tars-state.sqlite3"
EVIDENCE_ROOT = STATE_ROOT / "evidence"
ARTIFACT_ROOT = STATE_ROOT / "artifacts"
# Legacy v0.3 task store paths retained for one-way migration/rollback.
TASK_ROOT = STATE_ROOT / "tasks"
TASK_INDEX_PATH = TASK_ROOT / "index.json"
TASK_EVENTS_ROOT = TASK_ROOT / "events"
THEME_ROOT = Path.home() / ".config/tars/themes"
UI_PREFS_PATH = Path.home() / ".config/tars/ui.toml"
PERSONA_ROOT = Path.home() / ".config/tars/persona"
IDENTITY_PATH = PERSONA_ROOT / "IDENTITY.md"
SOUL_PATH = PERSONA_ROOT / "SOUL.md"
ROLE_PERSONA_ROOT = PERSONA_ROOT / "roles"

LLAMA_SWAP_CONFIG_PATH = Path.home() / ".config/tars/llama-swap/config.yaml"
LLAMA_SERVER_PATH = Path.home() / ".local/src/llama.cpp/build-cuda/bin/llama-server"
LLAMA_BENCH_PATH = LLAMA_SERVER_PATH.with_name("llama-bench")
LLAMA_SWAP_SERVICE = "tars-llama-swap.service"
RUNTIME_CONFIG_BACKUP_ROOT = STATE_ROOT / "runtime-config-backups"
MODEL_ARTIFACT_ROOT = DATA_ROOT / "model-artifacts/sha256"
MODEL_DOWNLOAD_ROOT = CACHE_ROOT / "downloads"


def load_config():
    with CONFIG_PATH.open("rb") as handle:
        data = tomllib.load(handle)
    runtime = data.setdefault("runtime", {})
    if "backend" not in runtime:
        legacy = runtime.get("provider", "llama-swap")
        runtime["backend"] = "llama.cpp" if legacy == "llama-swap" else legacy
    return data


def expand_path(value):
    return Path(value).expanduser().resolve()


def runtime_base_url(cfg):
    return cfg["runtime"]["base_url"].rstrip("/")
