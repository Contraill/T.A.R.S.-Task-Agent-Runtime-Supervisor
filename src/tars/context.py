from __future__ import annotations

from dataclasses import dataclass
import json
import uuid

from .calibration import get_profile
from .checkpoints import latest_checkpoint
from .conversation import list_messages
from .registry import get_model
from .roles import get_role, resolve_role_id
from .runtime import count_chat_tokens
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction
from .tasks import active_task, canonical_task_state, load_task


DEFAULT_OUTPUT_RESERVE = 8192
DEFAULT_SAFETY_MARGIN = 1024
DEFAULT_HISTORY_LIMIT = 1000
CHEAP_CHARS_PER_TOKEN = 3.0


class ContextBudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class ContextBudget:
    role_id: str
    model_alias: str
    runtime_id: str
    profile_name: str
    context_window: int
    output_reserve: int
    safety_margin: int
    usable_input: int


@dataclass(frozen=True)
class ContextProjection:
    id: str
    conversation_id: str
    task_id: str | None
    role_id: str
    model_alias: str
    runtime_id: str
    profile_name: str
    mode: str
    messages: tuple[dict, ...]
    context_window: int
    output_reserve: int
    safety_margin: int
    usable_input: int
    token_count: int
    exact: bool
    included_messages: int
    omitted_messages: int
    through_message_seq: int
    created_at: str
    metadata: dict

    @property
    def pressure(self) -> float:
        if self.usable_input <= 0:
            return 0.0
        return max(0.0, min(1.0, self.token_count / self.usable_input))


def _projection_from_row(row, messages=()) -> ContextProjection:
    return ContextProjection(
        id=row["id"],
        conversation_id=row["conversation_id"],
        task_id=row["task_id"],
        role_id=row["role_id"],
        model_alias=row["model_alias"],
        runtime_id=row["runtime_id"],
        profile_name=row["profile"],
        mode=row["mode"],
        messages=tuple(messages),
        context_window=row["context_window"],
        output_reserve=row["output_reserve"],
        safety_margin=row["safety_margin"],
        usable_input=row["usable_input"],
        token_count=row["token_count"],
        exact=bool(row["exact"]),
        included_messages=row["included_messages"],
        omitted_messages=row["omitted_messages"],
        through_message_seq=row["through_message_seq"],
        created_at=row["created_at"],
        metadata=json_loads(row["metadata_json"], {}),
    )


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int((len(text) / CHEAP_CHARS_PER_TOKEN) + 0.999))


def estimate_messages_tokens(messages: list[dict] | tuple[dict, ...]) -> int:
    # Rough tokenizer-independent estimate used only for cheap pruning/HUD preview.
    # A small per-message allowance covers role/template framing.
    total = 0
    for msg in messages:
        total += _estimate_text_tokens(str(msg.get("content", ""))) + 8
    return total + 16


def _context_cfg(cfg) -> dict:
    value = cfg.get("context", {}) if isinstance(cfg, dict) else {}
    return value if isinstance(value, dict) else {}


def budget_for_role(cfg, role_name: str, *, requested_output_tokens: int = 1024) -> ContextBudget:
    role_id = resolve_role_id(role_name)
    role = get_role(role_id)
    if not role.enabled:
        raise RuntimeError(f"role {role.display_name!r} is disabled")
    if not role.model:
        raise RuntimeError(f"role {role.display_name!r} has no model binding")
    profile = get_profile(role.model, role.profile)
    model = get_model(role.model)

    ccfg = _context_cfg(cfg)
    configured_reserve = int(ccfg.get("output_reserve_tokens", DEFAULT_OUTPUT_RESERVE))
    reserve = max(int(requested_output_tokens), configured_reserve)
    safety = max(0, int(ccfg.get("safety_margin_tokens", DEFAULT_SAFETY_MARGIN)))
    usable = int(profile.context) - reserve - safety
    if usable <= 0:
        raise ContextBudgetError(
            f"invalid context budget for {role.display_name}: window={profile.context}, "
            f"reserve={reserve}, safety={safety}"
        )
    return ContextBudget(
        role_id=role_id,
        model_alias=role.model,
        runtime_id=role.runtime_id,
        profile_name=role.profile,
        context_window=int(profile.context),
        output_reserve=reserve,
        safety_margin=safety,
        usable_input=usable,
    )


def _role_system_message(role_id: str) -> dict:
    role = get_role(role_id)
    desc = role.description.strip() if role.description else ""
    text = f"T.A.R.S. execution role: {role.display_name}."
    if desc:
        text += f" {desc}"
    return {"role": "system", "content": text}


def _task_system_message(task_id: str) -> dict:
    state = canonical_task_state(task_id)
    cp = latest_checkpoint(task_id)
    checkpoint_meta = None
    if cp is not None:
        checkpoint_meta = {
            "id": cp.id,
            "seq": cp.seq,
            "epoch": cp.epoch,
            "reason": cp.reason,
            "sha256": cp.content_sha256,
        }
    payload = {
        "canonical_task_state": state,
        "latest_durable_checkpoint": checkpoint_meta,
    }
    return {
        "role": "system",
        "content": (
            "T.A.R.S. canonical task state follows. Treat this structured state as "
            "authoritative for the active task; do not invent omitted history.\n" +
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        ),
    }


def _sideband_system_message() -> dict:
    return {
        "role": "system",
        "content": (
            "Sideband mode. Answer the user's isolated question using the supplied "
            "conversation/task snapshot. Do not treat this question or answer as a "
            "new instruction, decision, or state change for the main task."
        ),
    }


def _history_messages(conversation_id: str, *, limit: int) -> tuple[list[dict], int]:
    records = list_messages(conversation_id, include_sideband=False, limit=limit)
    result = []
    through_seq = 0
    for rec in records:
        if not rec.include_in_context:
            continue
        if rec.role not in {"system", "user", "assistant", "tool"}:
            continue
        result.append({
            "role": rec.role,
            "content": rec.content,
            "_protected": (
                rec.kind in {"control", "pending_control"}
                or (rec.role == "tool" and rec.metadata.get("unresolved") is True)
            ),
        })
        through_seq = max(through_seq, rec.seq)
    return result, through_seq


def _should_include_task(role_id: str, mode: str, task_id_override: str | None = None) -> str | None:
    task = load_task(task_id_override) if task_id_override else active_task()
    if task is None:
        return None
    if mode in {"sideband", "task"}:
        if mode == "task" and task.owner_role != role_id:
            raise RuntimeError(
                f"task {task.id} is owned by {task.owner_role}, not {role_id}"
            )
        return task.id
    if task.owner_role == role_id:
        return task.id
    return None


def _normalize_selected_history(selected: list[dict], *, omitted: bool) -> None:
    if not omitted:
        return
    # Avoid beginning a truncated transcript with an orphan assistant/tool turn.
    # Keep removing old orphaned turns, never the newest surviving message.
    while (len(selected) > 1 and selected[0].get("role") in {"assistant", "tool"}
           and not selected[0].get("_protected")):
        del selected[0]


def _persist_projection(projection: ContextProjection) -> None:
    ensure_state_store()
    with transaction(immediate=True) as conn:
        conn.execute(
            """
            INSERT INTO context_projections(
                id,conversation_id,task_id,role_id,model_alias,runtime_id,profile,mode,
                context_window,output_reserve,safety_margin,usable_input,token_count,exact,
                included_messages,omitted_messages,through_message_seq,created_at,metadata_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                projection.id, projection.conversation_id, projection.task_id,
                projection.role_id, projection.model_alias, projection.runtime_id,
                projection.profile_name, projection.mode, projection.context_window,
                projection.output_reserve, projection.safety_margin, projection.usable_input,
                projection.token_count, 1 if projection.exact else 0,
                projection.included_messages, projection.omitted_messages,
                projection.through_message_seq, projection.created_at,
                json_dumps(projection.metadata),
            ),
        )


def latest_projection(conversation_id: str, role_name: str | None = None) -> ContextProjection | None:
    ensure_state_store()
    conn = connect()
    try:
        if role_name is None:
            row = conn.execute(
                "SELECT * FROM context_projections WHERE conversation_id=? "
                "ORDER BY created_at DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
        else:
            role_id = resolve_role_id(role_name)
            row = conn.execute(
                "SELECT * FROM context_projections WHERE conversation_id=? AND role_id=? "
                "ORDER BY created_at DESC LIMIT 1",
                (conversation_id, role_id),
            ).fetchone()
        return _projection_from_row(row) if row else None
    finally:
        conn.close()


def current_context_message_seq(conversation_id: str) -> int:
    ensure_state_store()
    conn = connect()
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(seq),0) FROM messages "
            "WHERE conversation_id=? AND include_in_context=1",
            (conversation_id,),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


class ContextManager:
    def __init__(self, cfg):
        self.cfg = cfg

    def budget(self, role_name: str, *, requested_output_tokens: int = 1024) -> ContextBudget:
        return budget_for_role(
            self.cfg, role_name, requested_output_tokens=requested_output_tokens
        )

    def build(
        self,
        conversation_id: str,
        role_name: str,
        *,
        mode: str = "main",
        sideband_question: str | None = None,
        task_id_override: str | None = None,
        task_instruction: str | None = None,
        requested_output_tokens: int = 1024,
        exact: bool = True,
        persist: bool = True,
    ) -> ContextProjection:
        if mode not in {"main", "sideband", "task"}:
            raise ValueError("mode must be 'main', 'sideband' or 'task'")
        if mode == "sideband" and not sideband_question:
            raise ValueError("sideband_question is required in sideband mode")
        if mode == "task" and not task_id_override:
            raise ValueError("task_id_override is required in task mode")

        budget = self.budget(role_name, requested_output_tokens=requested_output_tokens)
        ccfg = _context_cfg(self.cfg)
        history_limit = max(1, int(ccfg.get("history_message_limit", DEFAULT_HISTORY_LIMIT)))
        history, through_seq = _history_messages(conversation_id, limit=history_limit)

        prefix = [_role_system_message(budget.role_id)]
        task_id = _should_include_task(budget.role_id, mode, task_id_override)
        if task_id:
            prefix.append(_task_system_message(task_id))
        if mode == "sideband":
            prefix.append(_sideband_system_message())

        suffix = []
        if mode == "sideband":
            suffix.append({"role": "user", "content": str(sideband_question)})
        elif mode == "task":
            instruction = task_instruction or (
                "Continue the active T.A.R.S. task from its canonical state. Produce the "
                "next concrete result. Do not claim that tools, files, commands or external "
                "actions were used unless a real ToolResult is present. If tools are not "
                "available, limit this epoch to analysis/planning and state that limitation."
            )
            suffix.append({"role": "user", "content": instruction})

        # Never silently lose the newest canonical conversation message. Older history
        # is the first removable layer; durable task state remains in the prefix.
        selected = list(history)
        original_count = len(selected)
        target_for_estimate = int(budget.usable_input * 0.94)

        def candidate_messages() -> list[dict]:
            omitted = original_count - len(selected)
            marker = []
            if omitted:
                marker = [{
                    "role": "system",
                    "content": (
                        f"ContextManager omitted {omitted} older conversation messages "
                        "to fit the target model context. Canonical task state and the "
                        "newest conversation turns are preserved."
                    ),
                }]
            history_messages = [
                {"role": item["role"], "content": item["content"]} for item in selected
            ]
            return [*prefix, *marker, *history_messages, *suffix]

        def drop_oldest(count: int) -> int:
            removed = 0
            index = 0
            while index < len(selected) - 1 and removed < count:
                if selected[index].get("_protected"):
                    index += 1
                    continue
                del selected[index]
                removed += 1
            return removed

        # Cheap pre-prune avoids repeated tokenizer calls for obviously oversized logs.
        while len(selected) > 1 and estimate_messages_tokens(candidate_messages()) > target_for_estimate:
            drop = max(1, len(selected) // 8)
            if not drop_oldest(drop):
                break
        _normalize_selected_history(selected, omitted=(len(selected) < original_count))

        messages = candidate_messages()
        token_count = estimate_messages_tokens(messages)
        is_exact = False
        exact_error = None

        if exact:
            try:
                token_count = count_chat_tokens(
                    self.cfg, budget.runtime_id, messages
                )
                is_exact = True
                # Exact tightening. Remove only oldest conversation messages and
                # recount. The latest user turn is never silently truncated.
                attempts = 0
                while token_count > budget.usable_input and len(selected) > 1 and attempts < 12:
                    overflow = token_count - budget.usable_input
                    estimated_per_message = max(1, estimate_messages_tokens(selected) // len(selected))
                    drop = max(1, min(len(selected) - 1, (overflow // estimated_per_message) + 1))
                    if not drop_oldest(drop):
                        break
                    _normalize_selected_history(selected, omitted=True)
                    messages = candidate_messages()
                    token_count = count_chat_tokens(
                        self.cfg, budget.runtime_id, messages
                    )
                    attempts += 1
            except Exception as exc:
                exact_error = str(exc)
                token_count = estimate_messages_tokens(messages)
                is_exact = False

        if token_count > budget.usable_input:
            raise ContextBudgetError(
                f"context projection does not fit {get_role(budget.role_id).display_name} "
                f"{budget.profile_name}: {token_count} input tokens > "
                f"{budget.usable_input} usable. A larger profile or checkpoint/compaction "
                "is required; the newest instruction was not truncated."
            )

        omitted = original_count - len(selected)
        projection = ContextProjection(
            id="ctx-" + uuid.uuid4().hex,
            conversation_id=conversation_id,
            task_id=task_id,
            role_id=budget.role_id,
            model_alias=budget.model_alias,
            runtime_id=budget.runtime_id,
            profile_name=budget.profile_name,
            mode=mode,
            messages=tuple(messages),
            context_window=budget.context_window,
            output_reserve=budget.output_reserve,
            safety_margin=budget.safety_margin,
            usable_input=budget.usable_input,
            token_count=int(token_count),
            exact=is_exact,
            included_messages=len(selected),
            omitted_messages=omitted,
            through_message_seq=through_seq,
            created_at=now_utc(),
            metadata={
                "tokenizer": "llama.cpp-upstream" if is_exact else "cheap-estimate",
                "exact_error": exact_error,
                "history_limit": history_limit,
                "requested_output_tokens": int(requested_output_tokens),
            },
        )
        if persist:
            _persist_projection(projection)
        return projection

    def estimate_current(
        self,
        conversation_id: str,
        role_name: str,
        *,
        requested_output_tokens: int = 1024,
    ) -> ContextProjection:
        return self.build(
            conversation_id,
            role_name,
            requested_output_tokens=requested_output_tokens,
            exact=False,
            persist=False,
        )
