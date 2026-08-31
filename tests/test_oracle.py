from types import SimpleNamespace

import pytest

from tars import oracle
from tars.runtime_routing import RuntimeRouteUnavailable


class Router:
    route = SimpleNamespace(
        ready=True, model_alias="heavy", backend="colibri", reasons=(), id="rrt-one")

    def __init__(self, cfg):
        self.cfg = cfg

    def resolve(self, role, **kwargs):
        assert role == "oracle"
        assert kwargs["required_capabilities"] == ("deep-reasoning",)
        return self.route


def test_oracle_availability_is_truthful_for_unbound_and_ready_states():
    ready = oracle.availability({}, router_factory=Router)
    assert ready.configured and ready.ready and ready.state == "ready"

    class Missing(Router):
        route = SimpleNamespace(
            ready=False, model_alias="", backend="", reasons=("Role Oracle has no model binding",),
            id="rrt-missing")

    missing = oracle.availability({}, router_factory=Missing)
    assert not missing.configured and missing.state == "not-configured"


def test_oracle_delegation_requires_explicit_evidence_contract(monkeypatch):
    with pytest.raises(ValueError, match="input evidence"):
        oracle.create_oracle_delegation(
            {}, "task-one", "review", evidence_refs=(),
            required_evidence_types=("analysis",), parent_authority={}, parent_tools=(),
            router_factory=Router)
    monkeypatch.setattr(oracle, "load_evidence", lambda evidence_id: SimpleNamespace(id=evidence_id))
    captured = {}
    monkeypatch.setattr(oracle, "create_child", lambda *args, **kwargs: captured | kwargs)
    result = oracle.create_oracle_delegation(
        {}, "task-one", "review", evidence_refs=("ev-one",),
        required_evidence_types=("analysis", "citations"),
        parent_authority={"paths": []}, parent_tools=(), router_factory=Router)
    assert result["role"] == "oracle"
    assert result["evidence_refs"] == ("ev-one",)
    assert result["completion"]["required_evidence_types"] == ("analysis", "citations")


def test_oracle_delegation_fails_before_creation_when_route_unavailable(monkeypatch):
    class Unavailable(Router):
        route = SimpleNamespace(
            ready=False, model_alias="heavy", backend="colibri",
            reasons=("Colibri is offline",), id="rrt-offline")

    monkeypatch.setattr(
        oracle, "create_child", lambda *args, **kwargs: pytest.fail("child was created"))
    monkeypatch.setattr(oracle, "load_evidence", lambda evidence_id: SimpleNamespace(id=evidence_id))
    with pytest.raises(RuntimeRouteUnavailable, match="Colibri is offline"):
        oracle.create_oracle_delegation(
            {}, "task-one", "review", evidence_refs=("ev-one",),
            required_evidence_types=("analysis",), parent_authority={}, parent_tools=(),
            router_factory=Unavailable)
