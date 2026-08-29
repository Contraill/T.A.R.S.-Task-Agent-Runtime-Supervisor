import io
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from tars import model_lifecycle as lifecycle


def _registry():
    return {"version": 3, "models": {}}


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


def test_import_rejects_bad_hash(monkeypatch, tmp_path):
    source = tmp_path / "model.gguf"
    source.write_bytes(_gguf_payload())
    monkeypatch.setattr(lifecycle, "MODEL_ARTIFACT_ROOT", tmp_path / "artifacts")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        lifecycle.import_model(source, "bad", expected_sha256="0" * 64)


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

    def urlopen(request, timeout):
        if request.get_method() == "HEAD":
            return Response(b"")
        observed["range"] = request.get_header("Range")
        return Response(b"rest")

    monkeypatch.setattr(lifecycle.urllib.request, "urlopen", urlopen)
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
