from __future__ import annotations

from dataclasses import dataclass

from .context import ContextManager, ContextProjection
from .context_epochs import rollover


@dataclass(frozen=True)
class ContextPressure:
    ratio: float
    level: str
    soft: float
    hard: float
    emergency: float


def pressure_for(projection: ContextProjection, *, soft=0.70, hard=0.85, emergency=0.95):
    if not 0 < soft < hard < emergency <= 1:
        raise ValueError("context watermarks must satisfy 0 < soft < hard < emergency <= 1")
    ratio = projection.pressure
    level = "normal"
    if ratio >= emergency:
        level = "emergency"
    elif ratio >= hard:
        level = "hard"
    elif ratio >= soft:
        level = "soft"
    return ContextPressure(ratio, level, soft, hard, emergency)


class ContextEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.manager = ContextManager(cfg)

    def prepare(self, conversation_id, role_name, *, task_id=None, exact=True,
                requested_output_tokens=1024, auto_rollover=True):
        projection = self.manager.build(
            conversation_id, role_name, task_id_override=task_id,
            mode="task" if task_id else "main", exact=exact, persist=True,
            requested_output_tokens=requested_output_tokens,
        )
        config = self.cfg.get("context", {})
        pressure = pressure_for(
            projection, soft=float(config.get("soft_watermark", 0.70)),
            hard=float(config.get("hard_watermark", 0.85)),
            emergency=float(config.get("emergency_watermark", 0.95)),
        )
        epoch = None
        if auto_rollover and task_id and pressure.level in {"hard", "emergency"}:
            epoch = rollover(task_id, reason=f"context {pressure.level} watermark")
            projection = self.manager.build(
                conversation_id, role_name, task_id_override=task_id, mode="task",
                exact=exact, persist=True, requested_output_tokens=requested_output_tokens,
            )
            pressure = pressure_for(projection, soft=pressure.soft, hard=pressure.hard,
                                    emergency=pressure.emergency)
        return projection, pressure, epoch
