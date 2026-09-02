from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import tempfile
import tomllib
import uuid
import zipfile

from . import __version__
from . import config, state_store
from .secret_store import parse_reference
from .policy import redact
from .secure_paths import AnchoredRoot, select_anchor
from . import memory


BUNDLE_FORMAT = "tars-state"
BUNDLE_VERSION = 1
MAX_BUNDLE_BYTES = 16 * 1024 * 1024 * 1024
MANIFEST = "manifest.json"
DB_MEMBER = "state/tars-state.sqlite3"
RESTORE_JOURNAL_VERSION = 1
RESTORE_PREFIX = ".tars-restore-"
RESTORE_JOURNAL = "journal.json"
RESTORE_LOCK = ".tars-restore.lock"
RESTORE_MARKER = config.RESTORE_RECOVERY_MARKER


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


def _portable_roots(paths):
    return tuple(dict.fromkeys((
        paths.config_path.parent,
        paths.data_root,
        paths.state_root,
    )))


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


def _create_bundle(destination, *, paths):
    requested = Path(destination).expanduser().absolute()
    requested.parent.mkdir(parents=True, exist_ok=True)
    destination_root = AnchoredRoot(requested.parent.resolve(strict=True))
    destination = destination_root.path / requested.name
    try:
        destination_root.lstat((requested.name,))
    except FileNotFoundError:
        pass
    else:
        destination_root.close()
        raise FileExistsError(f"backup destination already exists: {destination}")
    try:
        with tempfile.TemporaryDirectory(prefix="tars-backup-") as temporary:
            temporary = Path(temporary)
            db = temporary / "state.sqlite3"
            schema = _snapshot_database(paths.state_db_path, db)
            portable = []
            with ExitStack() as stack:
                anchors = [stack.enter_context(AnchoredRoot(root.resolve(strict=True)))
                           for root in _portable_roots(paths) if root.is_dir()]
                for source, member in _portable_files(paths):
                    anchor, parts, _ = select_anchor(anchors, source)
                    staged_source = temporary / "portable" / member
                    staged_source.parent.mkdir(parents=True, exist_ok=True)
                    with anchor.reader(parts) as input_handle, staged_source.open("wb") as output:
                        shutil.copyfileobj(input_handle, output)
                    staged_source.chmod(0o600)
                    portable.append((staged_source, member))
            staged_config = temporary / "portable" / "config/config.toml"
            _validate_config_secrets(staged_config)
            sources = [(db, DB_MEMBER), *portable]
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
                "excluded": ["model weights/artifacts", "secret values",
                             "browser profiles/cookies", "cache/downloads", "process logs",
                             "workspace checkpoint payloads", "pairing credentials",
                             "client bearer verifiers"],
            }
            staged = temporary / "bundle.tars-backup"
            with zipfile.ZipFile(staged, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for source, name in sources:
                    archive.write(source, name)
                archive.writestr(MANIFEST, json.dumps(
                    manifest, sort_keys=True, separators=(",", ":")))
            source_fd = os.open(staged, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                destination_root.copy_fd_to(source_fd, (requested.name,))
            finally:
                os.close(source_fd)
    finally:
        destination_root.close()
    return _report(destination, manifest, reconciliation={})


def _read_and_validate(bundle, staging):
    requested = Path(bundle).expanduser().absolute()
    bundle_root = AnchoredRoot(requested.parent.resolve(strict=True))
    try:
        bundle_handle = bundle_root.reader((requested.name,))
        source = bundle_handle.__enter__()
    except Exception:
        bundle_root.close()
        raise
    archive = None
    try:
        archive = zipfile.ZipFile(source)
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
    finally:
        if archive is not None:
            archive.close()
        bundle_handle.__exit__(None, None, None)
        bundle_root.close()
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


def _path_identity(value):
    mode = value.st_mode
    if stat.S_ISREG(mode):
        kind = "file"
    elif stat.S_ISDIR(mode):
        kind = "directory"
    else:
        raise ValueError("restore paths must be regular files or directories")
    return {"device": int(value.st_dev), "inode": int(value.st_ino), "kind": kind}


def _same_identity(value, expected):
    if value is None or expected is None:
        return value is None and expected is None
    try:
        actual = _path_identity(value)
    except ValueError:
        return False
    return actual == expected


def _valid_serialized_identity(value, *, kind=None):
    if not isinstance(value, dict) or set(value) != {"device", "inode", "kind"}:
        return False
    if (not isinstance(value["device"], int) or isinstance(value["device"], bool)
            or not isinstance(value["inode"], int) or isinstance(value["inode"], bool)
            or value["device"] < 0 or value["inode"] < 0
            or value["kind"] not in {"file", "directory"}):
        return False
    return kind is None or value["kind"] == kind


def _optional_lstat(root, name):
    try:
        return root.lstat((name,))
    except FileNotFoundError:
        return None


def _fsync_tree_fd(fd):
    value = os.fstat(fd)
    if stat.S_ISREG(value.st_mode):
        os.fsync(fd)
        return
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError("restore staging contains a special filesystem object")
    for name in os.listdir(fd):
        child = os.open(
            name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=fd)
        try:
            _fsync_tree_fd(child)
        finally:
            os.close(child)
    os.fsync(fd)


def _fsync_tree(root, name):
    fd = root.open((name,), os.O_RDONLY)
    try:
        _fsync_tree_fd(fd)
    finally:
        os.close(fd)


def _fsync_restore_path(root, name):
    fd = root.open((name,), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_restore_path(root, name, *, expected=None):
    current = _optional_lstat(root, name)
    if current is None:
        return False
    if expected is not None and not _same_identity(current, expected):
        raise RuntimeError(f"restore recovery identity changed: {name}")
    identity = _path_identity(current)
    root.delete((name,), recursive=identity["kind"] == "directory")
    os.fsync(root.fd)
    return True


def _rename_restore_path(root, source, destination):
    root.rename((source,), root, (destination,))
    os.fsync(root.fd)


def _write_restore_journal(root, transaction_name, journal):
    payload = json.dumps(journal, sort_keys=True, separators=(",", ":")).encode()
    root.atomic_write((transaction_name, RESTORE_JOURNAL), payload)


def _restore_state_root(root, paths, *, create=False):
    state_root = Path(paths.state_root).absolute()
    expected_parent = Path(os.path.abspath(state_root.parent))
    if expected_parent != root.path:
        raise RuntimeError("restore state root is outside its locked parent")
    if create:
        root.makedirs((state_root.name,))
        os.fsync(root.fd)
    fd = root.open_directory((state_root.name,))
    try:
        return AnchoredRoot.from_fd(fd, display=root.path / state_root.name)
    finally:
        os.close(fd)


def _write_restore_marker(root, paths, transaction_id):
    with _restore_state_root(root, paths, create=True) as state_root:
        state_root.atomic_write((RESTORE_MARKER,), (transaction_id + "\n").encode())


def _read_restore_marker(root, paths):
    try:
        state_root = _restore_state_root(root, paths)
    except FileNotFoundError:
        return None
    with state_root:
        try:
            payload = state_root.read_bytes((RESTORE_MARKER,), limit=128)
        except FileNotFoundError:
            return None
    if len(payload) > 128:
        raise RuntimeError("restore recovery marker exceeds its safety limit")
    try:
        transaction_id = payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("restore recovery marker is invalid") from exc
    if not re.fullmatch(r"[0-9a-f]{32}", transaction_id):
        raise RuntimeError("restore recovery marker is invalid")
    return transaction_id


def _clear_restore_marker(root, paths, transaction_id):
    current = _read_restore_marker(root, paths)
    if current is None:
        return
    if current != transaction_id:
        raise RuntimeError("restore recovery marker belongs to another transaction")
    with _restore_state_root(root, paths) as state_root:
        _remove_restore_path(state_root, RESTORE_MARKER)


def _read_restore_journal(root, transaction_name):
    try:
        payload = root.read_bytes((transaction_name, RESTORE_JOURNAL), limit=1024 * 1024)
    except FileNotFoundError:
        transaction_fd = root.open_directory((transaction_name,))
        try:
            names = os.listdir(transaction_fd)
        finally:
            os.close(transaction_fd)
        if names and any(not name.startswith(f".{RESTORE_JOURNAL}.tars-") for name in names):
            raise RuntimeError(
                f"interrupted legacy restore lacks a durable journal: {transaction_name}")
        root.delete((transaction_name,), recursive=True)
        os.fsync(root.fd)
        return None
    if len(payload) > 1024 * 1024:
        raise RuntimeError(f"restore journal exceeds its safety limit: {transaction_name}")
    try:
        journal = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid restore journal: {transaction_name}") from exc
    if not isinstance(journal, dict):
        raise RuntimeError(f"invalid restore journal: {transaction_name}")
    return journal


def _expected_restore_destination(paths, member):
    try:
        destination = Path(_destinations(paths)[member]).absolute()
    except KeyError as exc:
        raise RuntimeError(f"restore journal contains an unknown member: {member}") from exc
    parent = Path(os.path.abspath(destination.parent))
    return parent, destination.name, str(parent / destination.name)


def _validate_restore_journal(root, transaction_name, journal, paths):
    transaction_id = transaction_name.removeprefix(RESTORE_PREFIX)
    if (not re.fullmatch(r"[0-9a-f]{32}", transaction_id)
            or journal.get("version") != RESTORE_JOURNAL_VERSION
            or journal.get("transaction_id") != transaction_id):
        raise RuntimeError(f"unsupported or mismatched restore journal: {transaction_name}")
    if journal.get("status") not in {
            "validating", "preparing", "prepared", "applying", "rebuilding",
            "rolling_back", "rolled_back", "committed"}:
        raise RuntimeError(f"invalid restore journal state: {transaction_name}")
    items = journal.get("items")
    if not isinstance(items, list):
        raise RuntimeError(f"invalid restore journal items: {transaction_name}")
    members = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("member"), str):
            raise RuntimeError(f"invalid restore journal item: {transaction_name}")
        member = item["member"]
        if member in members:
            raise RuntimeError(f"duplicate restore journal member: {member}")
        members.add(member)
        parent, name, destination = _expected_restore_destination(paths, member)
        incoming = f".{name}.restore-{transaction_id}-new"
        saved = f".{name}.restore-{transaction_id}-old"
        if (item.get("parent") != str(parent)
                or item.get("destination") != destination
                or item.get("name") != name
                or item.get("incoming") != incoming
                or item.get("saved") != saved):
            raise RuntimeError(f"restore journal path authority mismatch: {member}")
        expected_parent = item.get("parent_identity")
        if not _valid_serialized_identity(expected_parent, kind="directory"):
            raise RuntimeError(f"restore journal lacks parent identity: {member}")
        with AnchoredRoot(parent) as parent_root:
            if _path_identity(os.fstat(parent_root.fd)) != expected_parent:
                raise RuntimeError(f"restore parent identity changed: {parent}")
        if not isinstance(item.get("existed"), bool):
            raise RuntimeError(f"restore journal existence state is invalid: {member}")
        original = item.get("original_identity")
        incoming_identity = item.get("incoming_identity")
        if ((item["existed"] and not _valid_serialized_identity(original))
                or (not item["existed"] and original is not None)
                or (incoming_identity is not None
                    and not _valid_serialized_identity(incoming_identity))):
            raise RuntimeError(f"restore journal object identity is invalid: {member}")
    return items


@contextmanager
def _restore_lock(paths):
    lock_parent = Path(paths.state_root).parent
    root = AnchoredRoot.open_or_create(lock_parent)
    lock_fd = -1
    try:
        lock_fd = root.open((RESTORE_LOCK,), os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield root
    finally:
        if lock_fd >= 0:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        root.close()


def _open_restore_parent(item):
    root = AnchoredRoot(item["parent"])
    if _path_identity(os.fstat(root.fd)) != item["parent_identity"]:
        root.close()
        raise RuntimeError(f"restore parent identity changed: {item['parent']}")
    return root


def _rollback_restore(root, transaction_name, journal, paths):
    items = _validate_restore_journal(root, transaction_name, journal, paths)
    journal["status"] = "rolling_back"
    _write_restore_journal(root, transaction_name, journal)
    for item in reversed(items):
        with _open_restore_parent(item) as parent:
            destination = _optional_lstat(parent, item["name"])
            incoming = _optional_lstat(parent, item["incoming"])
            saved = _optional_lstat(parent, item["saved"])
            if saved is not None:
                if not _same_identity(saved, item.get("original_identity")):
                    raise RuntimeError(
                        f"restore recovery backup identity changed: {item['member']}")
                if destination is not None:
                    if not _same_identity(destination, item.get("incoming_identity")):
                        raise RuntimeError(
                            f"restore recovery destination changed: {item['member']}")
                    _remove_restore_path(
                        parent, item["name"], expected=item.get("incoming_identity"))
                _rename_restore_path(parent, item["saved"], item["name"])
            elif incoming is not None:
                expected = item.get("incoming_identity")
                if expected is not None and not _same_identity(incoming, expected):
                    raise RuntimeError(
                        f"restore recovery staging identity changed: {item['member']}")
                _remove_restore_path(parent, item["incoming"], expected=expected)
            elif _same_identity(destination, item.get("incoming_identity")):
                if item["existed"]:
                    raise RuntimeError(
                        f"restore recovery lost its prior destination: {item['member']}")
                _remove_restore_path(
                    parent, item["name"], expected=item.get("incoming_identity"))
            elif item["existed"] and not _same_identity(
                    destination, item.get("original_identity")):
                raise RuntimeError(
                    f"restore recovery cannot prove the prior destination: {item['member']}")
            elif not item["existed"] and destination is not None:
                raise RuntimeError(
                    f"restore recovery found an unrelated destination: {item['member']}")
    journal["status"] = "rolled_back"
    _write_restore_journal(root, transaction_name, journal)
    _clear_restore_marker(root, paths, journal["transaction_id"])
    root.delete((transaction_name,), recursive=True)
    os.fsync(root.fd)


def _finalize_committed_restore(root, transaction_name, journal, paths):
    items = _validate_restore_journal(root, transaction_name, journal, paths)
    for item in items:
        with _open_restore_parent(item) as parent:
            destination = _optional_lstat(parent, item["name"])
            if not _same_identity(destination, item.get("incoming_identity")):
                raise RuntimeError(
                    f"committed restore destination identity changed: {item['member']}")
            _remove_restore_path(
                parent, item["saved"], expected=item.get("original_identity"))
            _remove_restore_path(
                parent, item["incoming"], expected=item.get("incoming_identity"))
    _clear_restore_marker(root, paths, journal["transaction_id"])
    root.delete((transaction_name,), recursive=True)
    os.fsync(root.fd)


def _recover_interrupted_restores(root, paths):
    recovered = 0
    transaction_ids = set()
    for name, value in root.list():
        if not name.startswith(RESTORE_PREFIX):
            continue
        if not stat.S_ISDIR(value.st_mode):
            raise RuntimeError(f"restore transaction path is not a directory: {name}")
        journal = _read_restore_journal(root, name)
        if journal is None:
            continue
        transaction_ids.add(journal.get("transaction_id"))
        if journal.get("status") == "committed":
            _finalize_committed_restore(root, name, journal, paths)
        elif journal.get("status") == "validating":
            _validate_restore_journal(root, name, journal, paths)
            _clear_restore_marker(root, paths, journal["transaction_id"])
            root.delete((name,), recursive=True)
            os.fsync(root.fd)
        else:
            _rollback_restore(root, name, journal, paths)
        recovered += 1
    marker = _read_restore_marker(root, paths)
    if marker is not None and marker not in transaction_ids:
        raise RuntimeError(
            "restore recovery marker has no matching durable journal")
    return recovered


def recover_interrupted_restore(*, paths=None):
    """Resolve every durable restore journal to its old or committed state."""
    paths = paths or BackupPaths()
    with _restore_lock(paths) as root:
        return _recover_interrupted_restores(root, paths)


def restore_recovery_required(*, paths=None):
    paths = paths or BackupPaths()
    try:
        (Path(paths.state_root) / RESTORE_MARKER).lstat()
    except FileNotFoundError:
        return False
    return True


def create_bundle(destination, *, paths=None):
    """Create a bundle only after resolving any interrupted local restore."""
    paths = paths or BackupPaths()
    with _restore_lock(paths) as root:
        _recover_interrupted_restores(root, paths)
        return _create_bundle(destination, paths=paths)


def _build_restore_items(staging, paths, transaction_id):
    items = []
    anchors = {}
    try:
        for member, requested_destination in _destinations(paths).items():
            source = staging / member
            if not (source.is_file() or source.is_dir()):
                continue
            destination = Path(requested_destination).absolute()
            parent_path = Path(os.path.abspath(destination.parent))
            key = str(parent_path)
            parent = anchors.get(key)
            if parent is None:
                parent = AnchoredRoot.open_or_create(parent_path)
                anchors[key] = parent
            name = destination.name
            incoming = f".{name}.restore-{transaction_id}-new"
            saved = f".{name}.restore-{transaction_id}-old"
            if (_optional_lstat(parent, incoming) is not None
                    or _optional_lstat(parent, saved) is not None):
                raise RuntimeError(f"restore transaction path collision: {member}")
            current = _optional_lstat(parent, name)
            original = _path_identity(current) if current is not None else None
            items.append({
                "member": member,
                "parent": key,
                "parent_identity": _path_identity(os.fstat(parent.fd)),
                "destination": str(parent_path / name),
                "name": name,
                "incoming": incoming,
                "saved": saved,
                "existed": current is not None,
                "original_identity": original,
                "incoming_identity": None,
            })
    except Exception:
        for anchor in anchors.values():
            anchor.close()
        raise
    return items, anchors


def _prepare_restore_items(source_root, items, anchors, journal, root, transaction_name):
    for item in items:
        parent = anchors[item["parent"]]
        source_parts = PurePosixPath(item["member"]).parts
        source_root.copy_to(source_parts, parent, (item["incoming"],))
        _fsync_tree(parent, item["incoming"])
        item["incoming_identity"] = _path_identity(
            parent.lstat((item["incoming"],)))
        _write_restore_journal(root, transaction_name, journal)


def _apply_restore_items(items, anchors, journal, root, transaction_name):
    journal["status"] = "applying"
    journal["applied_count"] = 0
    journal["applying_index"] = None
    _write_restore_journal(root, transaction_name, journal)
    for index, item in enumerate(items):
        parent = anchors[item["parent"]]
        journal["applying_index"] = index
        _write_restore_journal(root, transaction_name, journal)
        current = _optional_lstat(parent, item["name"])
        if item["existed"]:
            if not _same_identity(current, item["original_identity"]):
                raise RuntimeError(
                    f"restore destination changed before commit: {item['member']}")
            _fsync_restore_path(parent, item["name"])
            _rename_restore_path(parent, item["name"], item["saved"])
        elif current is not None:
            raise RuntimeError(
                f"restore destination appeared before commit: {item['member']}")
        _rename_restore_path(parent, item["incoming"], item["name"])
        journal["applied_count"] = index + 1
        _write_restore_journal(root, transaction_name, journal)
    journal["applying_index"] = None


def restore_bundle(bundle, *, paths=None, replace=False):
    if not replace:
        raise PermissionError("restore requires explicit replace=True")
    paths = paths or BackupPaths()
    with _restore_lock(paths) as restore_root:
        _recover_interrupted_restores(restore_root, paths)
        transaction_id = uuid.uuid4().hex
        transaction_name = RESTORE_PREFIX + transaction_id
        restore_root.mkdir((transaction_name,))
        os.fsync(restore_root.fd)
        journal = {
            "version": RESTORE_JOURNAL_VERSION,
            "transaction_id": transaction_id,
            "status": "validating",
            "items": [],
        }
        _write_restore_journal(restore_root, transaction_name, journal)
        _write_restore_marker(restore_root, paths, transaction_id)
        transaction_path = Path(f"/proc/self/fd/{restore_root.fd}") / transaction_name
        staging = transaction_path / "staging"
        restore_root.mkdir((transaction_name, "staging"))
        staging_fd = restore_root.open_directory((transaction_name, "staging"))
        source_root = AnchoredRoot.from_fd(staging_fd, display=staging)
        os.close(staging_fd)
        manifest = None
        anchors = {}
        committed = False
        try:
            manifest = _read_and_validate(bundle, staging)
            items, anchors = _build_restore_items(staging, paths, transaction_id)
            journal["items"] = items
            journal["status"] = "preparing"
            _write_restore_journal(restore_root, transaction_name, journal)
            _prepare_restore_items(
                source_root, items, anchors, journal, restore_root, transaction_name)
            journal["status"] = "prepared"
            _write_restore_journal(restore_root, transaction_name, journal)
            _apply_restore_items(
                items, anchors, journal, restore_root, transaction_name)
            journal["status"] = "rebuilding"
            _write_restore_journal(restore_root, transaction_name, journal)
            rebuilt_memory_entries = memory.rebuild_index(
                memory_root=paths.data_root / "memory",
                state_db_path=paths.state_db_path)
            reconciliation = _reconciliation(paths, manifest)
            reconciliation["memory_index_entries_rebuilt"] = rebuilt_memory_entries
            journal["status"] = "committed"
            _write_restore_journal(restore_root, transaction_name, journal)
            committed = True
            _finalize_committed_restore(
                restore_root, transaction_name, journal, paths)
        except Exception:
            if not committed:
                durable_journal = _read_restore_journal(
                    restore_root, transaction_name)
                if durable_journal is not None and durable_journal.get(
                        "status") == "committed":
                    committed = True
                else:
                    _rollback_restore(
                        restore_root, transaction_name,
                        durable_journal or journal, paths)
            raise
        finally:
            source_root.close()
            for anchor in anchors.values():
                anchor.close()
    return _report(bundle, manifest, reconciliation=reconciliation)


def _report(bundle, manifest, *, reconciliation):
    files = manifest["files"]
    return BundleReport(
        str(Path(bundle).expanduser().absolute()), len(files),
        sum(int(item["size"]) for item in files.values()),
        int(manifest["bundle_version"]), int(manifest["schema_version"]),
        str(manifest["source_version"]), tuple(manifest["excluded"]), reconciliation)
