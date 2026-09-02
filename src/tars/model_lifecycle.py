from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import urllib.parse

from .calibration import calibration_path, load_calibration
from .config import MODEL_ARTIFACT_ROOT, MODEL_DOWNLOAD_ROOT
from .network import network_destination, open_bound
from .model_integrity import inspect_model_artifact
from .registry import (
    ensure_registry,
    get_model,
    role_for_alias,
    save_registry,
    validate_model_alias,
)
from .secure_paths import AnchoredRoot

COMPATIBILITY_MANIFEST = json.loads(
    Path(__file__).with_name("compatibility_manifest.json").read_text(encoding="utf-8")
)


@dataclass(frozen=True)
class VerificationResult:
    alias: str
    sha256: str
    integrity_verified: bool
    runtime_compatible: bool
    calibration_ready: bool

    @property
    def runtime_ready(self) -> bool:
        return self.integrity_verified and self.runtime_compatible and self.calibration_ready


def _snapshot_bound_file(source: Path, destination: Path):
    requested = source.expanduser().absolute()
    anchor = AnchoredRoot(requested.parent.resolve(strict=True))
    digest = hashlib.sha256()
    size = 0
    try:
        with anchor.reader((requested.name,)) as input_handle, destination.open("wb") as output:
            while True:
                chunk = input_handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                output.write(chunk)
    finally:
        anchor.close()
    return digest.hexdigest(), size


def _preflight(target: Path, required: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free
    if free < required:
        raise OSError(f"insufficient disk space: need {required} bytes, have {free}")


def _artifact_path(digest: str) -> Path:
    return MODEL_ARTIFACT_ROOT / digest[:2] / f"{digest}.gguf"


def _commit_artifact(source: Path, digest: str) -> Path:
    MODEL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    anchor = AnchoredRoot(MODEL_ARTIFACT_ROOT.resolve(strict=True))
    parts = (digest[:2], f"{digest}.gguf")
    target = anchor.path.joinpath(*parts)
    try:
        try:
            value, _ = anchor.hash(parts)
        except FileNotFoundError:
            pass
        else:
            if value != digest:
                raise RuntimeError(f"content-addressed artifact is corrupt: {target}")
            return target
        source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            copied_digest, _ = anchor.copy_fd_to(source_fd, parts)
        finally:
            os.close(source_fd)
        if copied_digest != digest:
            anchor.delete(parts)
            raise RuntimeError("model source changed during artifact commit")
        return target
    finally:
        anchor.close()


def _register(alias: str, path: Path, digest: str, *, name: str, source: str,
              source_revision: str = "", license_name: str = "unknown",
              quant: str = "unknown", native_context: int = 0, size: int,
              runtime_compatible: bool) -> None:
    validate_model_alias(alias)
    registry = ensure_registry()
    if alias in registry["models"]:
        raise ValueError(f"model alias already exists: {alias}")
    candidate = json.loads(json.dumps(registry))
    candidate["version"] = max(3, int(candidate.get("version", 2)))
    candidate["models"][alias] = {
        "name": name, "path": str(path), "sha256": digest,
        "artifact_sha256": digest, "backend": "llama.cpp", "quant": quant,
        "native_context": native_context, "source": source,
        "source_revision": source_revision, "license": license_name,
        "size": int(size), "integrity_verified": True,
        "runtime_compatible": bool(runtime_compatible),
    }
    save_registry(candidate)


def _gguf_header_compatible(header: bytes) -> bool:
    if len(header) < 24 or header[:4] != b"GGUF":
        return False
    version, tensor_count, metadata_count = struct.unpack("<IQQ", header[4:24])
    return version in {2, 3} and tensor_count >= 0 and metadata_count >= 0


def _gguf_compatible(path: Path) -> bool:
    with path.open("rb") as handle:
        return _gguf_header_compatible(handle.read(24))


def import_model(path: str | Path, alias: str, *, expected_sha256: str | None = None,
                 name: str | None = None, license_name: str = "unknown",
                 quant: str = "unknown", native_context: int = 0) -> Path:
    validate_model_alias(alias)
    source = Path(path).expanduser().absolute()
    with tempfile.TemporaryDirectory(prefix="tars-model-import-") as temporary:
        staged = Path(temporary) / "source.gguf"
        digest, size = _snapshot_bound_file(source, staged)
        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {digest}")
        compatible = _gguf_compatible(staged)
        if not compatible:
            raise ValueError("unsupported local model: expected a GGUF artifact")
        if not _artifact_path(digest).exists():
            _preflight(MODEL_ARTIFACT_ROOT, size)
        artifact = _commit_artifact(staged, digest)
    _register(alias, artifact, digest, name=name or source.stem, source=f"file:{source}",
              license_name=license_name, quant=quant, native_context=native_context,
              size=size, runtime_compatible=compatible)
    return artifact


def _open_url(url, *, method="GET", headers=None, timeout=30, max_redirects=5):
    current = str(url)
    for _ in range(max_redirects + 1):
        destination = network_destination(current, resolve_dns=True)
        response = open_bound(
            destination, method=method, headers=headers, timeout=timeout,
        )
        if response.status not in {301, 302, 303, 307, 308}:
            return response
        location = response.headers.get("Location")
        response.close()
        if not location:
            raise RuntimeError("model download redirect omitted its destination")
        current = urllib.parse.urljoin(destination.request_url, location)
    raise RuntimeError("model download redirect limit exceeded")


def _download(url: str, partial: Path, *, expected_size: int | None = None) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    if expected_size is None:
        try:
            with _open_url(url, method="HEAD", timeout=15) as response:
                length = response.headers.get("Content-Length")
            expected_size = int(length) if length is not None else None
        except (OSError, ValueError):
            expected_size = None
    required = max(0, (expected_size or 0) - offset)
    _preflight(partial, required)
    headers = {}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with _open_url(url, headers=headers, timeout=30) as response:
        status = getattr(response, "status", 200)
        mode = "ab" if offset and status == 206 else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)


def pull_model(source: str, alias: str, *, expected_sha256: str | None = None,
               license_name: str = "unknown", revision: str = "main",
               filename: str | None = None, quant: str = "unknown",
               native_context: int = 0) -> Path:
    validate_model_alias(alias)
    if source.startswith(("http://", "https://")):
        url = source
        destination = network_destination(source, resolve_dns=False)
        source_name = destination.policy_url
        if destination.policy_url != destination.request_url:
            source_name += f"#url-sha256={destination.url_sha256}"
    else:
        if not filename:
            raise ValueError("Hugging Face pulls require --filename")
        quoted = "/".join(urllib.parse.quote(x, safe="") for x in source.split("/"))
        url = f"https://huggingface.co/{quoted}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(filename)}"
        source_name = f"huggingface:{source}"
    MODEL_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    partial = MODEL_DOWNLOAD_ROOT / f"{alias}.partial"
    _download(url, partial)
    with tempfile.TemporaryDirectory(prefix="tars-model-download-") as temporary:
        staged = Path(temporary) / "download.gguf"
        digest, size = _snapshot_bound_file(partial, staged)
        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {digest}")
        compatible = _gguf_compatible(staged)
        if not compatible:
            raise ValueError("downloaded artifact is not GGUF-compatible")
        artifact = _commit_artifact(staged, digest)
    _register(alias, artifact, digest, name=filename or Path(urllib.parse.urlparse(url).path).name,
              source=source_name, source_revision=revision, license_name=license_name,
              quant=quant, native_context=native_context, size=size,
              runtime_compatible=compatible)
    partial.unlink(missing_ok=True)
    return artifact


def search_huggingface(query: str, *, limit: int = 10) -> list[dict]:
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(
        {"search": query, "filter": "gguf", "limit": int(limit), "sort": "downloads", "direction": -1}
    )
    with _open_url(url, timeout=15) as response:
        rows = json.loads(response.read().decode("utf-8"))
    return [{"id": row.get("id", ""), "downloads": row.get("downloads", 0),
             "likes": row.get("likes", 0), "license": (row.get("cardData") or {}).get("license", "unknown")}
            for row in rows]


def verify_model(alias: str) -> VerificationResult:
    validate_model_alias(alias)
    model = get_model(alias)
    if not model.path.is_file():
        raise FileNotFoundError(model.path)
    inspection = inspect_model_artifact(model.path)
    digest = inspection.sha256
    if digest != model.sha256:
        registry = ensure_registry()
        registry["models"][alias]["integrity_verified"] = False
        registry["models"][alias]["runtime_compatible"] = False
        save_registry(registry)
        raise ValueError(f"SHA-256 mismatch for {alias}: expected {model.sha256}, got {digest}")
    compatible = (model.backend == "llama.cpp" and
                  _gguf_header_compatible(inspection.header))
    try:
        calibration = load_calibration(alias)
        calibrated = (calibration.get("status") == "ready" and
                      calibration.get("model_sha256", digest) == digest)
    except (FileNotFoundError, KeyError):
        calibrated = False
    registry = ensure_registry()
    registry["models"][alias]["integrity_verified"] = True
    registry["models"][alias]["runtime_compatible"] = compatible
    save_registry(registry)
    return VerificationResult(alias, digest, True, compatible, calibrated)


def remove_model(alias: str) -> bool:
    validate_model_alias(alias)
    assigned = role_for_alias(alias)
    if assigned:
        raise ValueError(f"model {alias} is assigned to Roles: {', '.join(assigned)}")
    registry = ensure_registry()
    if alias not in registry["models"]:
        raise KeyError(f"unknown model alias: {alias}")
    removed = registry["models"][alias]
    path = Path(removed["path"]).expanduser()
    digest = removed.get("artifact_sha256", removed.get("sha256", ""))
    candidate = json.loads(json.dumps(registry))
    del candidate["models"][alias]
    save_registry(candidate)
    digest_shared = any(
        info.get("artifact_sha256", info.get("sha256", "")) == digest
        for info in candidate["models"].values()
    )
    valid_digest = bool(re.fullmatch(r"[0-9a-f]{64}", digest))
    if valid_digest and not digest_shared:
        calibration_path(digest).unlink(missing_ok=True)
    shared = any(Path(info["path"]).expanduser() == path for info in candidate["models"].values())
    expected = _artifact_path(digest) if valid_digest else None
    if not shared and expected is not None and path.absolute() == expected.absolute():
        MODEL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        anchor = AnchoredRoot(MODEL_ARTIFACT_ROOT.resolve(strict=True))
        try:
            anchor.delete((digest[:2], f"{digest}.gguf"))
            try:
                anchor.delete((digest[:2],))
            except OSError:
                pass
        finally:
            anchor.close()
        return True
    return False
