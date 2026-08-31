from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import tempfile
import tomllib
import uuid
import zipfile

from . import __version__
from . import config, state_store
from .secret_store import parse_reference
from .policy import redact
from . import memory


BUNDLE_FORMAT = "tars-state"
BUNDLE_VERSION = 1
MAX_BUNDLE_BYTES = 16 * 1024 * 1024 * 1024
MANIFEST = "manifest.json"
DB_MEMBER = "state/tars-state.sqlite3"


@dataclass(frozen=True)
class BackupPaths:
    config_path: Path = config.CONFIG_PATH
    data_root: Path = config.DATA_ROOT
    state_root: Path = config.STATE_ROOT
    theme_root: Path = config.THEME_ROOT
    ui_prefs_path: Path = config.UI_PREFS_PATH
    persona_root: Path = config.PERSONA_ROOT
    state_db_path: Path = config.STATE_DB_PATH


@dataclass(frozen=True)
class BundleReport:
    bundle: str
    files: int
    bytes: int
    bundle_version: int
    schema_version: int
    source_version: str
    excluded: tuple[str, ...]
    reconciliation: dict


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_files(paths):
    singles = (
        (paths.config_path, "config/config.toml"),
        (paths.ui_prefs_path, "config/ui.toml"),
        (paths.data_root / "model-registry.toml", "data/model-registry.toml"),
        (paths.data_root / "role-registry.toml", "data/role-registry.toml"),
    )
    for source, member in singles:
        if source.is_file():
            if source.is_symlink():
                raise ValueError(f"backup refuses symlinked portable source: {source}")
            yield source, member
    roots = (
        (paths.persona_root, "config/persona"),
        (paths.theme_root, "config/themes"),
        (paths.config_path.parent / "skills", "config/skills"),
        (paths.data_root / "memory", "data/memory"),
        (paths.state_root / "calibration", "state/calibration"),
        (paths.state_root / "evidence", "state/evidence"),
    )
    for root, prefix in roots:
        if root.is_dir():
            for source in sorted(item for item in root.rglob("*") if item.is_file()):
                if source.is_symlink():
                    continue
                yield source, str(PurePosixPath(prefix) / source.relative_to(root).as_posix())


def _validate_config_secrets(path):
    if not path.is_file():
        return
    with path.open("rb") as handle:
        value = tomllib.load(handle)

    def walk(item, key=""):
        if isinstance(item, dict):
            for child_key, child in item.items():
                walk(child, str(child_key).casefold())
        elif isinstance(item, list):
            for child in item:
                walk(child, key)
        elif isinstance(item, str) and any(
                marker in key for marker in ("password", "secret", "token", "credential",
                                              "authorization", "api_key", "apikey")):
            try:
                parse_reference(item)
            except ValueError as exc:
                raise ValueError(
                    f"backup refused plaintext-like configured secret field: {key}") from exc
    walk(value)


def _snapshot_database(source, destination):
    if not source.is_file():
        raise FileNotFoundError(f"state database is unavailable: {source}")
    incoming = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    outgoing = sqlite3.connect(destination)
    try:
        incoming.backup(outgoing)
        outgoing.execute("DELETE FROM core_pairings")
        outgoing.execute(
            "UPDATE core_clients SET token_salt='',token_hash='excluded',state='revoked',"
            "revoked_at=COALESCE(revoked_at,created_at)")
        for row in outgoing.execute("SELECT id,metadata_json FROM core_clients").fetchall():
            try:
                metadata = json.loads(row[1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid Core client metadata: {row[0]}") from exc
            outgoing.execute("UPDATE core_clients SET metadata_json=? WHERE id=?", (
                json.dumps(redact(metadata), sort_keys=True, separators=(",", ":")), row[0]))
        for row in outgoing.execute("SELECT name,config_json FROM mcp_servers").fetchall():
            try:
                mcp_config = json.loads(row[1])
                for reference in mcp_config.get("env", {}).values():
                    parse_reference(reference)
                if mcp_config.get("authorization_ref"):
                    parse_reference(mcp_config["authorization_ref"])
                sensitive = re.compile(
                    r"(?i)^--?(?:api[-_]?key|authorization|password|secret|token)(?:=|$)")
                if any(sensitive.search(str(value))
                       for value in mcp_config.get("argv", ())):
                    raise ValueError("credential in MCP argv")
            except (AttributeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"backup refused invalid or plaintext-like MCP credentials: {row[0]}") from exc
        result = outgoing.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError(f"database snapshot integrity failed: {result}")
        row = outgoing.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        schema = int(row[0]) if row else 0
        outgoing.commit()
        return schema
    finally:
        outgoing.close()
        incoming.close()


def create_bundle(destination, *, paths=None):
    paths = paths or BackupPaths()
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"backup destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _validate_config_secrets(paths.config_path)
    with tempfile.TemporaryDirectory(prefix="tars-backup-") as temporary:
        temporary = Path(temporary)
        db = temporary / "state.sqlite3"
        schema = _snapshot_database(paths.state_db_path, db)
        sources = [(db, DB_MEMBER), *_portable_files(paths)]
        names = [name for _, name in sources]
        if len(names) != len(set(names)):
            raise RuntimeError("backup member collision")
        files = {name: {"sha256": _sha256(source), "size": source.stat().st_size}
                 for source, name in sources}
        manifest = {
            "format": BUNDLE_FORMAT, "bundle_version": BUNDLE_VERSION,
            "source_version": __version__, "schema_version": schema,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "excluded": ["model weights/artifacts", "secret values", "browser profiles/cookies",
                         "cache/downloads", "process logs", "workspace checkpoint payloads",
                         "pairing credentials",
                         "client bearer verifiers"],
        }
        partial = destination.with_name(destination.name + ".partial")
        try:
            with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for source, name in sources:
                    archive.write(source, name)
                archive.writestr(MANIFEST, json.dumps(
                    manifest, sort_keys=True, separators=(",", ":")))
            partial.replace(destination)
            destination.chmod(0o600)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    return _report(destination, manifest, reconciliation={})


def _read_and_validate(bundle, staging):
    bundle = Path(bundle).expanduser().resolve()
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if names.count(MANIFEST) != 1 or len(names) != len(set(names)):
            raise ValueError("bundle has a missing or duplicate manifest/member")
        manifest = json.loads(archive.read(MANIFEST))
        if manifest.get("format") != BUNDLE_FORMAT:
            raise ValueError("unsupported backup format")
        version = int(manifest.get("bundle_version", 0))
        if version > BUNDLE_VERSION:
            raise ValueError(f"backup version {version} is newer than supported {BUNDLE_VERSION}")
        if version < 1:
            raise ValueError("unsupported legacy backup version")
        schema = int(manifest.get("schema_version", 0))
        if schema < 1 or schema > state_store.SCHEMA_VERSION:
            raise ValueError(
                f"backup schema {schema} is incompatible with {state_store.SCHEMA_VERSION}")
        declared = manifest.get("files")
        if not isinstance(declared, dict) or set(names) != {MANIFEST, *declared}:
            raise ValueError("bundle members do not match the manifest")
        declared_bytes = sum(int(item.get("size", -1)) for item in declared.values())
        if declared_bytes < 0 or declared_bytes > MAX_BUNDLE_BYTES:
            raise ValueError("backup declared size is invalid or exceeds the safety limit")
        info = {item.filename: item for item in archive.infolist()}
        for name, expected in declared.items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"unsafe backup member: {name}")
            if info[name].file_size != int(expected["size"]):
                raise ValueError(f"backup member size does not match manifest: {name}")
            target = staging.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o600)
            if target.stat().st_size != int(expected["size"]) or _sha256(target) != expected["sha256"]:
                raise ValueError(f"backup checksum mismatch: {name}")
    db = staging / DB_MEMBER
    conn = sqlite3.connect(db)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"restored database integrity failed: {integrity}")
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        database_schema = int(row[0]) if row else 0
        if database_schema != int(manifest["schema_version"]):
            raise ValueError(
                f"database schema {database_schema} does not match manifest "
                f"{manifest['schema_version']}")
        state_store.migrate_connection(conn)
    finally:
        conn.close()
    for member in ("config/config.toml", "config/ui.toml",
                   "data/model-registry.toml", "data/role-registry.toml"):
        path = staging / member
        if path.is_file():
            with path.open("rb") as handle:
                tomllib.load(handle)
    return manifest


def inspect_bundle(bundle):
    with tempfile.TemporaryDirectory(prefix="tars-inspect-") as temporary:
        manifest = _read_and_validate(bundle, Path(temporary))
    return _report(bundle, manifest, reconciliation={})


def _destinations(paths):
    return {
        "config/config.toml": paths.config_path,
        "config/ui.toml": paths.ui_prefs_path,
        "data/model-registry.toml": paths.data_root / "model-registry.toml",
        "data/role-registry.toml": paths.data_root / "role-registry.toml",
        DB_MEMBER: paths.state_db_path,
        "config/persona": paths.persona_root,
        "config/themes": paths.theme_root,
        "config/skills": paths.config_path.parent / "skills",
        "data/memory": paths.data_root / "memory",
        "state/calibration": paths.state_root / "calibration",
        "state/evidence": paths.state_root / "evidence",
    }


def _reconciliation(paths, manifest):
    missing_models = []
    registry = paths.data_root / "model-registry.toml"
    if registry.is_file():
        with registry.open("rb") as handle:
            for alias, model in tomllib.load(handle).get("models", {}).items():
                model_path = Path(str(model.get("path", ""))).expanduser()
                if model_path and not model_path.is_file():
                    missing_models.append(str(alias))
    missing_workspaces = []
    missing_mcp_commands = []
    unresolved_secret_refs = set()
    checkpoint_payloads = []
    if paths.config_path.is_file():
        with paths.config_path.open("rb") as handle:
            configured = tomllib.load(handle)
        def collect(item, key=""):
            if isinstance(item, dict):
                for child_key, child in item.items():
                    if str(child_key).casefold() == "scopes":
                        continue
                    collect(child, str(child_key).casefold())
            elif isinstance(item, list):
                for child in item:
                    collect(child, key)
            elif isinstance(item, str) and any(marker in key for marker in (
                    "password", "secret", "token", "credential", "authorization",
                    "api_key", "apikey")):
                try:
                    parsed = parse_reference(item)
                except ValueError:
                    return
                if parsed.provider != "env" or parsed.key not in os.environ:
                    unresolved_secret_refs.add(parsed.value)
        collect(configured)
    conn = sqlite3.connect(paths.state_db_path)
    try:
        for (value,) in conn.execute("SELECT canonical_path FROM project_refs"):
            if not Path(value).expanduser().exists():
                missing_workspaces.append(value)
        for checkpoint_id, root in conn.execute(
                "SELECT id,root FROM workspace_checkpoints"):
            checkpoint_payloads.append(checkpoint_id)
            if not Path(root).expanduser().exists():
                missing_workspaces.append(root)
        for _, transport, raw in conn.execute(
                "SELECT name,transport,config_json FROM mcp_servers"):
            value = json.loads(raw)
            for reference in value.get("env", {}).values():
                parsed = parse_reference(reference)
                if parsed.provider != "env" or parsed.key not in os.environ:
                    unresolved_secret_refs.add(parsed.value)
            if value.get("authorization_ref"):
                parsed = parse_reference(value["authorization_ref"])
                if parsed.provider != "env" or parsed.key not in os.environ:
                    unresolved_secret_refs.add(parsed.value)
            if transport == "stdio" and value.get("argv"):
                command = str(value["argv"][0])
                available = (Path(command).is_file() if "/" in command else shutil.which(command))
                if not available:
                    missing_mcp_commands.append(command)
    finally:
        conn.close()
    return {"model_assets_required": sorted(missing_models),
            "unresolved_secret_references": sorted(unresolved_secret_refs),
            "missing_workspace_paths": sorted(set(missing_workspaces)),
            "missing_mcp_commands": sorted(set(missing_mcp_commands)),
            "workspace_checkpoint_payloads_require_recreation": sorted(checkpoint_payloads),
            "runtime_and_calibration_revalidation_required": True,
            "external_effects_rolled_back": False,
            "source_version": manifest["source_version"]}


def restore_bundle(bundle, *, paths=None, replace=False):
    if not replace:
        raise PermissionError("restore requires explicit replace=True")
    paths = paths or BackupPaths()
    paths.state_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=".tars-restore-", dir=paths.state_root.parent) as temporary:
        staging = Path(temporary) / "staging"
        staging.mkdir()
        manifest = _read_and_validate(bundle, staging)
        recovery = Path(temporary) / "recovery"
        recovery.mkdir()
        applied = []
        rebuilt_memory_entries = 0
        try:
            for member, destination in _destinations(paths).items():
                source = staging / member
                matching = source.is_file() or source.is_dir()
                if not matching:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                saved = recovery / str(len(applied))
                existed = destination.exists()
                incoming = destination.with_name(
                    f".{destination.name}.restore-{uuid.uuid4().hex}")
                try:
                    if source.is_dir():
                        shutil.copytree(source, incoming)
                        for directory in (incoming, *(
                                item for item in incoming.rglob("*") if item.is_dir())):
                            directory.chmod(0o700)
                        for file in (item for item in incoming.rglob("*") if item.is_file()):
                            file.chmod(0o600)
                    else:
                        shutil.copy2(source, incoming)
                        incoming.chmod(0o600)
                    if existed:
                        destination.replace(saved)
                    applied.append((destination, saved, existed))
                    incoming.replace(destination)
                except Exception:
                    if incoming.is_dir():
                        shutil.rmtree(incoming)
                    else:
                        incoming.unlink(missing_ok=True)
                    raise
            rebuilt_memory_entries = memory.rebuild_index(
                memory_root=paths.data_root / "memory",
                state_db_path=paths.state_db_path)
        except Exception:
            for destination, saved, existed in reversed(applied):
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink(missing_ok=True)
                if existed:
                    saved.replace(destination)
            raise
    reconciliation = _reconciliation(paths, manifest)
    reconciliation["memory_index_entries_rebuilt"] = rebuilt_memory_entries
    return _report(bundle, manifest, reconciliation=reconciliation)


def _report(bundle, manifest, *, reconciliation):
    files = manifest["files"]
    return BundleReport(
        str(Path(bundle).expanduser().resolve()), len(files),
        sum(int(item["size"]) for item in files.values()),
        int(manifest["bundle_version"]), int(manifest["schema_version"]),
        str(manifest["source_version"]), tuple(manifest["excluded"]), reconciliation)
