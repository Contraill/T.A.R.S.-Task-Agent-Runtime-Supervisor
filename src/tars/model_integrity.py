from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
import re

from .secure_paths import AnchoredRoot

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_OWNER_CACHE = ContextVar("tars_verified_model_artifacts", default=None)


@dataclass
class _CachedArtifact:
    inspection: "ModelArtifactInspection"
    handle: object
    context: object


def _close_cached_artifact(cached):
    try:
        cached.context.__exit__(None, None, None)
    except (OSError, ValueError):
        pass


@contextmanager
def model_artifact_cache_scope():
    cache = {}
    token = _OWNER_CACHE.set(cache)
    try:
        yield
    finally:
        for cached in tuple(cache.values()):
            _close_cached_artifact(cached)
        cache.clear()
        _OWNER_CACHE.reset(token)


@dataclass(frozen=True)
class ModelArtifactInspection:
    path: Path
    sha256: str
    size: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int
    header: bytes


@contextmanager
def open_model_artifact(path):
    """Open and hash one no-follow artifact, retaining the exact verified file."""
    requested = Path(path).expanduser().absolute()
    anchor = AnchoredRoot(requested.parent.resolve(strict=True))
    digest = hashlib.sha256()
    size = 0
    header = b""
    try:
        with anchor.reader((requested.name,)) as handle:
            identity = os.fstat(handle.fileno())
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                if len(header) < 24:
                    header = (header + chunk)[:24]
                digest.update(chunk)
                size += len(chunk)
            handle.seek(0)
            inspection = ModelArtifactInspection(
                requested, digest.hexdigest(), size, identity.st_dev,
                identity.st_ino, identity.st_mtime_ns, identity.st_ctime_ns,
                header,
            )
            yield inspection, handle
    finally:
        anchor.close()


def inspect_model_artifact(path) -> ModelArtifactInspection:
    with open_model_artifact(path) as (inspection, _handle):
        return inspection


def _artifact_stat(path):
    requested = Path(path).expanduser().absolute()
    anchor = AnchoredRoot(requested.parent.resolve(strict=True))
    try:
        with anchor.reader((requested.name,)) as handle:
            value = os.fstat(handle.fileno())
    finally:
        anchor.close()
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _validated_digest(expected, *, label):
    expected = str(expected)
    if not SHA256_RE.fullmatch(expected):
        raise RuntimeError(f"{label} has no valid SHA-256 identity")
    return expected.casefold()


def _cached_artifact(path, expected, *, label):
    from .ownership import held_by, model_execution_owner, MODEL_EXECUTION_RESOURCE
    owner = model_execution_owner()
    cache = _OWNER_CACHE.get()
    if owner is None or cache is None or not held_by(
            *MODEL_EXECUTION_RESOURCE, owner):
        raise RuntimeError(f"{label} requires model execution ownership")
    requested = str(Path(path).expanduser().absolute())
    key = (owner.token, requested, expected)
    cached = cache.get(key)
    if (cached is not None
            and _artifact_stat(path) == (
                cached.inspection.device, cached.inspection.inode,
                cached.inspection.size, cached.inspection.mtime_ns,
                cached.inspection.ctime_ns)):
        return cached

    if cached is not None:
        _close_cached_artifact(cached)
        cache.pop(key, None)
    context = open_model_artifact(path)
    inspection, handle = context.__enter__()
    if inspection.sha256 != expected:
        context.__exit__(None, None, None)
        raise RuntimeError(
            f"{label} bytes no longer match the verified SHA-256 identity")
    cached = _CachedArtifact(inspection, handle, context)
    cache[key] = cached
    return cached


def require_current_model_artifact(model) -> ModelArtifactInspection:
    label = f"model {getattr(model, 'alias', '')!r} artifact"
    expected = _validated_digest(getattr(model, "sha256", ""), label=label)
    owner_cache = _OWNER_CACHE.get()
    if owner_cache is not None:
        return _cached_artifact(
            model.path, expected, label=label).inspection
    inspection = inspect_model_artifact(model.path)
    if inspection.sha256 != expected:
        raise RuntimeError(
            f"{label} bytes no longer match the verified SHA-256 identity")
    return inspection


def current_model_artifact_handle(model):
    label = f"model {getattr(model, 'alias', '')!r} artifact"
    expected = _validated_digest(getattr(model, "sha256", ""), label=label)
    return _cached_artifact(model.path, expected, label=label).handle


def current_artifact_handle(path, expected_sha256, *, label="artifact"):
    expected = _validated_digest(expected_sha256, label=label)
    return _cached_artifact(path, expected, label=label).handle


def model_artifact_matches(model) -> bool:
    try:
        require_current_model_artifact(model)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def calibration_runtime_artifacts_match(calibration, server_path, bench_path) -> bool:
    fingerprint = calibration.get("fingerprint") if isinstance(calibration, dict) else None
    if not isinstance(fingerprint, dict):
        return False
    expected = (
        (server_path, fingerprint.get("llama_server_sha256")),
        (bench_path, fingerprint.get("llama_bench_sha256")),
    )
    for path, digest in expected:
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            return False
        try:
            actual = inspect_model_artifact(path).sha256
        except OSError:
            return False
        if actual != digest.casefold():
            return False
    return True
