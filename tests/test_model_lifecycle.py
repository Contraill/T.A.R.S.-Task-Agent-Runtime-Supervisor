from contextlib import contextmanager
import hashlib
import io
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from tars import model_lifecycle as lifecycle
from tars import registry as model_registry


def _registry():
    return {"version": 3, "models": {}}


@pytest.fixture(autouse=True)
def isolated_registry_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(
        model_registry, "REGISTRY_PATH", tmp_path / "model-registry.toml"
    )
    monkeypatch.setattr(model_registry, "STATE_ROOT", tmp_path / "state")


def test_compatibility_manifest_is_versioned_and_architecture_neutral():
    manifest = lifecycle.COMPATIBILITY_MANIFEST
    assert manifest["schema_version"] == 1
    assert "no required model architecture" in manifest["tested"]["llama.cpp"]["architecture_policy"]


def _gguf_payload(data=b"model-data"):
    return b"GGUF" + struct.pack("<IQQ", 3, 0, 0) + data


def test_import_verify_and_deduplicate(monkeypatch, tmp_path):
    source = tmp_path / "model.gguf"
    source.write_bytes(_gguf_payload())
    registry = _registry()
    monkeypatch.setattr(lifecycle, "MODEL_ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(lifecycle, "ensure_registry", lambda: registry)
    monkeypatch.setattr(lifecycle, "save_registry", lambda data: (registry.clear(), registry.update(data)))

    first = lifecycle.import_model(source, "one")
    second_source = tmp_path / "copy.gguf"
    second_source.write_bytes(source.read_bytes())
    second = lifecycle.import_model(second_source, "two")

    assert first == second
    assert len(list((tmp_path / "artifacts").rglob("*.gguf"))) == 1
    monkeypatch.setattr(lifecycle, "get_model", lambda alias: SimpleNamespace(
        alias=alias, path=first, sha256=registry["models"][alias]["sha256"], backend="llama.cpp"))
    monkeypatch.setattr(lifecycle, "load_calibration", lambda alias: {"status": "missing"})
    result = lifecycle.verify_model("one")
    assert result.runtime_compatible
    assert not result.runtime_ready


def test_import_model_binds_source_before_parent_replacement(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    original_payload = _gguf_payload(b"authorized-model")
    (source_root / "model.gguf").write_bytes(original_payload)
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    (replacement_root / "model.gguf").write_bytes(_gguf_payload(b"outside-model"))
    displaced_root = tmp_path / "displaced"
    artifact_root = tmp_path / "artifacts"
    registry = _registry()
    monkeypatch.setattr(lifecycle, "MODEL_ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(lifecycle, "ensure_registry", lambda: registry)
    monkeypatch.setattr(
        lifecycle,
        "save_registry",
        lambda data: (registry.clear(), registry.update(data)),
    )

    original_reader = lifecycle.AnchoredRoot.reader
    source_identity = source_root.resolve()
    replaced = False

    @contextmanager
    def replacing_reader(self, parts, **kwargs):
        nonlocal replaced
        if not replaced and self.path == source_identity:
            source_root.rename(displaced_root)
            source_root.symlink_to(replacement_root, target_is_directory=True)
            replaced = True
        with original_reader(self, parts, **kwargs) as handle:
            yield handle

    monkeypatch.setattr(lifecycle.AnchoredRoot, "reader", replacing_reader)

    artifact = lifecycle.import_model(source_root / "model.gguf", "bound")

    assert replaced
    assert artifact.read_bytes() == original_payload
    assert registry["models"]["bound"]["sha256"] == hashlib.sha256(
        original_payload
    ).hexdigest()


def test_import_rejects_bad_hash(monkeypatch, tmp_path):
    source = tmp_path / "model.gguf"
    source.write_bytes(_gguf_payload())
    monkeypatch.setattr(lifecycle, "MODEL_ARTIFACT_ROOT", tmp_path / "artifacts")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        lifecycle.import_model(source, "bad", expected_sha256="0" * 64)


def test_failed_reverification_revokes_stale_integrity_flags(monkeypatch, tmp_path):
    path = tmp_path / "model.gguf"
    original = _gguf_payload(b"original")
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    registry = {"version": 3, "models": {"model": {
        "name": "Model", "path": str(path), "sha256": digest,
        "backend": "llama.cpp", "quant": "Q4", "native_context": 4096,
        "integrity_verified": True, "runtime_compatible": True,
    }}}
    monkeypatch.setattr(lifecycle, "ensure_registry", lambda: registry)
    monkeypatch.setattr(
        lifecycle, "save_registry",
        lambda data: None,
    )
    # Keep the lookup bound to the in-memory fixture without involving user state.
    monkeypatch.setattr(
        lifecycle, "get_model",
        lambda alias: SimpleNamespace(
            alias=alias, path=path, sha256=digest, backend="llama.cpp"),
    )
    path.write_bytes(_gguf_payload(b"mutated"))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        lifecycle.verify_model("model")
    assert registry["models"]["model"]["integrity_verified"] is False
    assert registry["models"]["model"]["runtime_compatible"] is False


@pytest.mark.parametrize("alias", [
    "../outside",
    "nested/model",
    ".hidden",
    "model:\n[injected]",
    "model\x00suffix",
])
def test_model_alias_is_rejected_before_import_or_download_paths(
        monkeypatch, tmp_path, alias):
    source = tmp_path / "model.gguf"
    source.write_bytes(_gguf_payload())
    monkeypatch.setattr(
        lifecycle, "_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("download path was reached")),
    )
    with pytest.raises(ValueError, match="invalid model alias"):
        lifecycle.import_model(source, alias)
    with pytest.raises(ValueError, match="invalid model alias"):
        lifecycle.pull_model("https://example.invalid/model.gguf", alias)


def test_disk_preflight_rejects_insufficient_space(monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="insufficient disk space"):
        lifecycle._preflight(tmp_path / "artifact", 2)


def test_download_resumes_partial_file(monkeypatch, tmp_path):
    partial = tmp_path / "download.partial"
    partial.write_bytes(b"GGUF")
    observed = {}

    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Length": "8"}
        def __enter__(self): return self
        def __exit__(self, *args): return False

    def open_url(url, *, method="GET", headers=None, timeout):
        if method == "HEAD":
            return Response(b"")
        observed["range"] = headers.get("Range")
        return Response(b"rest")

    monkeypatch.setattr(lifecycle, "_open_url", open_url)
    lifecycle._download("https://example.invalid/model.gguf", partial)
    assert observed["range"] == "bytes=4-"
    assert partial.read_bytes() == b"GGUFrest"


def test_remove_refuses_assigned_model(monkeypatch):
    monkeypatch.setattr(lifecycle, "role_for_alias", lambda alias: ["builder"])
    with pytest.raises(ValueError, match="assigned"):
        lifecycle.remove_model("model")


def test_remove_retains_shared_artifact(monkeypatch, tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact = artifact_root / "aa" / "shared.gguf"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(_gguf_payload())
    registry = {"version": 3, "models": {
        "one": {"path": str(artifact), "sha256": "same"},
        "two": {"path": str(artifact), "sha256": "same"}}}
    monkeypatch.setattr(lifecycle, "MODEL_ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(lifecycle, "role_for_alias", lambda alias: [])
    monkeypatch.setattr(lifecycle, "ensure_registry", lambda: registry)
    monkeypatch.setattr(lifecycle, "save_registry", lambda data: (registry.clear(), registry.update(data)))
    monkeypatch.setattr(lifecycle, "calibration_path", lambda digest: tmp_path / "calibration.json")
    assert lifecycle.remove_model("one") is False
    assert artifact.exists()


def test_legacy_registry_defaults_to_llama_cpp_backend(monkeypatch, tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(
        'version = 2\n[models."legacy"]\nname = "Legacy"\npath = "/tmp/m.gguf"\n'
        'sha256 = "abc"\nquant = "Q4"\nnative_context = 4096\n'
    )
    monkeypatch.setattr(model_registry, "REGISTRY_PATH", path)
    loaded = model_registry.load_registry()
    assert loaded["models"]["legacy"]["backend"] == "llama.cpp"
