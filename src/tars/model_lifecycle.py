from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import urllib.parse
import urllib.request

from .calibration import calibration_path, load_calibration
from .config import MODEL_ARTIFACT_ROOT, MODEL_DOWNLOAD_ROOT
from .registry import ensure_registry, get_model, role_for_alias, save_registry

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight(target: Path, required: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free
    if free < required:
        raise OSError(f"insufficient disk space: need {required} bytes, have {free}")


def _artifact_path(digest: str) -> Path:
    return MODEL_ARTIFACT_ROOT / digest[:2] / f"{digest}.gguf"


def _commit_artifact(source: Path, digest: str) -> Path:
    target = _artifact_path(digest)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256(target) != digest:
            raise RuntimeError(f"content-addressed artifact is corrupt: {target}")
        return target
    fd, temp_name = tempfile.mkstemp(prefix=".artifact-", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def _register(alias: str, path: Path, digest: str, *, name: str, source: str,
              source_revision: str = "", license_name: str = "unknown",
              quant: str = "unknown", native_context: int = 0) -> None:
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
        "size": path.stat().st_size, "integrity_verified": True,
        "runtime_compatible": _gguf_compatible(path),
    }
    save_registry(candidate)


def _gguf_compatible(path: Path) -> bool:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:4] != b"GGUF":
        return False
    version, tensor_count, metadata_count = struct.unpack("<IQQ", header[4:24])
    return version in {2, 3} and tensor_count >= 0 and metadata_count >= 0


def import_model(path: str | Path, alias: str, *, expected_sha256: str | None = None,
                 name: str | None = None, license_name: str = "unknown",
                 quant: str = "unknown", native_context: int = 0) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = _sha256(source)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    if not _gguf_compatible(source):
        raise ValueError("unsupported local model: expected a GGUF artifact")
    if not _artifact_path(digest).exists():
        _preflight(MODEL_ARTIFACT_ROOT, source.stat().st_size)
    artifact = _commit_artifact(source, digest)
    _register(alias, artifact, digest, name=name or source.stem, source=f"file:{source}",
              license_name=license_name, quant=quant, native_context=native_context)
    return artifact


def _download(url: str, partial: Path, *, expected_size: int | None = None) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    if expected_size is None:
        try:
            head = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(head, timeout=15) as response:
                length = response.headers.get("Content-Length")
            expected_size = int(length) if length is not None else None
        except (OSError, ValueError):
            expected_size = None
    required = max(0, (expected_size or 0) - offset)
    _preflight(partial, required)
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=30) as response:
        status = getattr(response, "status", 200)
        mode = "ab" if offset and status == 206 else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)


def pull_model(source: str, alias: str, *, expected_sha256: str | None = None,
               license_name: str = "unknown", revision: str = "main",
               filename: str | None = None, quant: str = "unknown",
               native_context: int = 0) -> Path:
    if source.startswith(("http://", "https://")):
        url = source
        source_name = source
    else:
        if not filename:
            raise ValueError("Hugging Face pulls require --filename")
        quoted = "/".join(urllib.parse.quote(x, safe="") for x in source.split("/"))
        url = f"https://huggingface.co/{quoted}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(filename)}"
        source_name = f"huggingface:{source}"
    MODEL_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    partial = MODEL_DOWNLOAD_ROOT / f"{alias}.partial"
    _download(url, partial)
    digest = _sha256(partial)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    if not _gguf_compatible(partial):
        raise ValueError("downloaded artifact is not GGUF-compatible")
    artifact = _commit_artifact(partial, digest)
    _register(alias, artifact, digest, name=filename or Path(urllib.parse.urlparse(url).path).name,
              source=source_name, source_revision=revision, license_name=license_name,
              quant=quant, native_context=native_context)
    partial.unlink(missing_ok=True)
    return artifact


def search_huggingface(query: str, *, limit: int = 10) -> list[dict]:
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(
        {"search": query, "filter": "gguf", "limit": int(limit), "sort": "downloads", "direction": -1}
    )
    with urllib.request.urlopen(url, timeout=15) as response:
        rows = json.loads(response.read().decode("utf-8"))
    return [{"id": row.get("id", ""), "downloads": row.get("downloads", 0),
             "likes": row.get("likes", 0), "license": (row.get("cardData") or {}).get("license", "unknown")}
            for row in rows]


def verify_model(alias: str) -> VerificationResult:
    model = get_model(alias)
    if not model.path.is_file():
        raise FileNotFoundError(model.path)
    digest = _sha256(model.path)
    if digest != model.sha256:
        raise ValueError(f"SHA-256 mismatch for {alias}: expected {model.sha256}, got {digest}")
    compatible = model.backend == "llama.cpp" and _gguf_compatible(model.path)
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
    if digest and not digest_shared:
        calibration_path(digest).unlink(missing_ok=True)
    shared = any(Path(info["path"]).expanduser() == path for info in candidate["models"].values())
    if not shared and path.is_file() and MODEL_ARTIFACT_ROOT in path.parents:
        path.unlink()
        try:
            path.parent.rmdir()
        except OSError:
            pass
        return True
    return False
