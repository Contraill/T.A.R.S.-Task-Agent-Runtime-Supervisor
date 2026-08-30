from __future__ import annotations

from dataclasses import dataclass

from .calibration import get_profile
from .roles import get_role, resolve_role_id


DEFAULT_GENERATION_TOKENS = 8192
DEFAULT_SAFETY_MARGIN = 1024
SIDEBAND_GENERATION_TOKENS = 512


class GenerationBudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationBudget:
    role_id: str
    context_window: int
    output_reserve: int
    safety_margin: int
    usable_input: int
    explicit_ceiling: int | None


def generation_budget(cfg, role_name, *, requested_tokens=None):
    role_id = resolve_role_id(role_name)
    role = get_role(role_id)
    if not role.enabled or not role.model:
        raise RuntimeError(f"role {role.display_name!r} has no enabled model binding")
    profile = get_profile(role.model, role.profile)
    context_cfg = cfg.get("context", {}) if isinstance(cfg, dict) else {}
    generation_cfg = cfg.get("generation", {}) if isinstance(cfg, dict) else {}
    configured = generation_cfg.get(
        "output_tokens", context_cfg.get("output_reserve_tokens", DEFAULT_GENERATION_TOKENS))
    reserve = int(configured if requested_tokens is None else requested_tokens)
    safety = max(0, int(context_cfg.get("safety_margin_tokens", DEFAULT_SAFETY_MARGIN)))
    window = int(profile.context)
    if reserve <= 0:
        raise GenerationBudgetError("output reserve must be positive")
    if reserve + safety >= window:
        raise GenerationBudgetError(
            f"generation budget exceeds active profile: window={window}, "
            f"reserve={reserve}, safety={safety}")
    return GenerationBudget(role_id, window, reserve, safety,
                            window - reserve - safety,
                            int(requested_tokens) if requested_tokens is not None else None)


def generation_ceiling(budget, actual_input_tokens):
    remaining = budget.context_window - int(actual_input_tokens) - budget.safety_margin
    if remaining <= 0:
        raise GenerationBudgetError(
            f"input leaves no generation capacity: window={budget.context_window}, "
            f"input={actual_input_tokens}, safety={budget.safety_margin}")
    return min(remaining, budget.explicit_ceiling) if budget.explicit_ceiling else remaining


def generation_outcome(content, finish_reason):
    content = str(content or "")
    finish = str(finish_reason or "unknown")
    exhausted = finish == "length"
    return {"state": "exhausted" if exhausted else "succeeded",
            "normal_success": bool(content) and not exhausted,
            "empty": not bool(content), "finish_reason": finish}
