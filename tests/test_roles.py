import pytest

from tars import roles
from tars import registry


@pytest.fixture(autouse=True)
def isolated_installation_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(roles, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(registry, "STATE_ROOT", tmp_path / "state")


def _registry(runtime_id):
    return {
        "version": 1,
        "default_role": "general",
        "roles": {
            "general": {
                "enabled": True,
                "runtime_id": runtime_id,
                "model": "",
            },
        },
    }


@pytest.fixture
def isolated_role_registry(monkeypatch, tmp_path):
    path = tmp_path / "role-registry.toml"
    monkeypatch.setattr(roles, "ROLE_REGISTRY_PATH", path)
    monkeypatch.setattr(roles, "ensure_registry", lambda: {"models": {}})
    return path


def test_role_registry_rejects_runtime_id_structure_injection(
        isolated_role_registry):
    data = _registry("daily:\n  injected: true")
    with pytest.raises(ValueError, match="invalid runtime id"):
        roles.save_role_registry(data)
    assert not isolated_role_registry.exists()


def test_role_registry_load_rejects_manually_injected_runtime_id(
        isolated_role_registry):
    data = _registry("daily:\n  injected: true")
    isolated_role_registry.write_text(
        roles.serialize_role_registry(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid runtime id"):
        roles.load_role_registry()


def test_model_registry_load_rejects_manually_injected_alias(monkeypatch, tmp_path):
    path = tmp_path / "model-registry.toml"
    path.write_text(
        'version = 3\n[models."../outside"]\nname = "bad"\npath = "/tmp/bad"\n'
        'sha256 = "abc"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(registry, "REGISTRY_PATH", path)
    with pytest.raises(ValueError, match="invalid model alias"):
        registry.load_registry()
