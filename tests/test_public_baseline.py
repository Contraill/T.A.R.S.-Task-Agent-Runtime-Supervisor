from tars import __version__
from tars.roles import LEGACY_ROLE_ALIASES


def test_version():
    assert __version__ == "0.4.2"


def test_legacy_role_aliases():
    assert LEGACY_ROLE_ALIASES["daily"] == "general"
    assert LEGACY_ROLE_ALIASES["coder"] == "builder"
