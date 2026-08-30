from tars import __version__
from tars.roles import LEGACY_ROLE_ALIASES
from tars import config


def test_version():
    assert __version__ == "0.6.2"


def test_legacy_role_aliases():
    assert LEGACY_ROLE_ALIASES["daily"] == "general"
    assert LEGACY_ROLE_ALIASES["coder"] == "builder"


def test_legacy_runtime_provider_config_migrates_in_memory(monkeypatch, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[runtime]\nprovider = "llama-swap"\nbase_url = "http://127.0.0.1:8080"\n')
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    loaded = config.load_config()
    assert loaded["runtime"]["backend"] == "llama.cpp"
    assert loaded["runtime"]["provider"] == "llama-swap"
