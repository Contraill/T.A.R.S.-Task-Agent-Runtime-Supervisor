from __future__ import annotations

from dataclasses import dataclass

from .roles import get_role, resolve_role_id


POLICIES = {"auto", "off", "on", "low", "medium", "high"}


@dataclass(frozen=True)
class ThinkingCapability:
    supported: bool
    modes: tuple[str, ...]
    mechanism: str


@dataclass(frozen=True)
class ThinkingDecision:
    requested: str
    effective: str
    mechanism: str
    reason: str


TOGGLE_CAPABILITY = ThinkingCapability(
    True, ("off", "on"), "llama.cpp chat_template_kwargs.enable_thinking")
UNSUPPORTED_CAPABILITY = ThinkingCapability(False, (), "unavailable")


def capability_for_model(model):
    if getattr(model, "backend", "") == "llama.cpp" and getattr(
            model, "thinking_control", "unknown") == "toggle":
        return TOGGLE_CAPABILITY
    return UNSUPPORTED_CAPABILITY


def configured_policy(cfg, role_name):
    role_id = resolve_role_id(role_name)
    thinking = cfg.get("thinking", {}) if isinstance(cfg, dict) else {}
    roles = thinking.get("roles", {}) if isinstance(thinking, dict) else {}
    return str(roles.get(role_id, thinking.get("default", "auto"))).casefold()


def decide(cfg, role_name, capability, *, requested=None, operation="chat",
           task_active=False, requires_tools=False, complex_task=False):
    policy = str(requested or configured_policy(cfg, role_name)).casefold()
    if policy not in POLICIES:
        raise ValueError(f"unknown thinking policy: {policy}")
    if not capability.supported:
        if policy == "auto":
            return ThinkingDecision("auto", "off", capability.mechanism,
                                    "active model/backend has no thinking control")
        raise ValueError(f"{policy.title()} thinking is not supported by the active model/backend.")
    if policy != "auto":
        if policy not in capability.modes:
            raise ValueError(f"{policy.title()} thinking is not supported by the active model/backend.")
        return ThinkingDecision(policy, policy, capability.mechanism,
                                "explicit user or Role policy")
    role = get_role(resolve_role_id(role_name))
    expensive = (requires_tools or complex_task or
                 operation in {"task", "agent", "delegation"} or
                 role.execution in {"loop", "delegate"})
    effective = "on" if expensive and "on" in capability.modes else "off"
    return ThinkingDecision("auto", effective, capability.mechanism,
                            "task/tool complexity" if expensive else "ordinary conversation")
