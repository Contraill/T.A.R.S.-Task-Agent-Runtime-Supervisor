from __future__ import annotations

from dataclasses import dataclass

from .delegation import create_child
from .evidence import load as load_evidence
from .runtime_routing import LocalRuntimeRouter, RuntimeRouteUnavailable


@dataclass(frozen=True)
class OracleAvailability:
    configured: bool
    ready: bool
    state: str
    reasons: tuple[str, ...]
    route_id: str | None = None


def availability(cfg, *, router_factory=LocalRuntimeRouter) -> OracleAvailability:
    """Probe the optional Oracle route without loading Heavy model state."""
    try:
        route = router_factory(cfg).resolve(
            "oracle", required_capabilities=("deep-reasoning",), persist=False)
    except Exception as exc:
        return OracleAvailability(
            False, False, "error", (f"{type(exc).__name__}: {exc}",))
    configured = bool(route.model_alias and route.backend == "colibri")
    state = "ready" if route.ready else ("unavailable" if configured else "not-configured")
    return OracleAvailability(configured, route.ready, state, route.reasons, route.id)


def create_oracle_delegation(
        cfg, parent_task_id, goal, *, evidence_refs, required_evidence_types,
        parent_authority, parent_tools, authority=None, tools=(), budget=None,
        workspace=None, constraints=(), router_factory=LocalRuntimeRouter):
    """Create an Oracle child only with explicit input and output evidence contracts."""
    evidence_refs = tuple(dict.fromkeys(map(str, evidence_refs)))
    required_evidence_types = tuple(dict.fromkeys(map(str, required_evidence_types)))
    if not evidence_refs:
        raise ValueError("Oracle delegation requires explicit input evidence references")
    if not required_evidence_types:
        raise ValueError("Oracle delegation requires explicit output evidence types")
    for evidence_id in evidence_refs:
        load_evidence(evidence_id)
    route = router_factory(cfg).resolve(
        "oracle", required_capabilities=("deep-reasoning",), persist=True)
    if not route.ready:
        raise RuntimeRouteUnavailable(
            "Oracle is unavailable: " + ("; ".join(route.reasons) or "not configured"))
    return create_child(
        parent_task_id, goal, role="oracle", required_capabilities=("deep-reasoning",),
        tools=tools, authority=authority or {}, parent_authority=parent_authority,
        parent_tools=parent_tools, budget=budget, workspace=workspace,
        completion={"required_evidence_types": required_evidence_types,
                    "summary_required": True},
        constraints=constraints, evidence_refs=evidence_refs)
