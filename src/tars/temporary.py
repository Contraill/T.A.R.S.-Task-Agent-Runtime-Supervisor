from __future__ import annotations

from dataclasses import dataclass, field

from .context import budget_for_role, estimate_messages_tokens
from .memory import search as search_memory
from .prompt_compiler import PromptCompiler
from .roles import resolve_role_id
from .runtime import chat_completion, chat_completion_stream


TEMPORARY_NOTICE = (
    "TEMPORARY SESSION. New T.A.R.S. conversation, task, checkpoint, context and memory "
    "state is ephemeral. External tool side effects, if any, remain real."
)


@dataclass
class TemporarySession:
    cfg: dict
    role_id: str
    compiler: PromptCompiler = field(default_factory=PromptCompiler)
    turns: list[dict] = field(default_factory=list)
    closed: bool = False

    def __post_init__(self):
        self.role_id = resolve_role_id(self.role_id)

    def _messages(self, latest_text, *, requested_output_tokens=None):
        hits = search_memory(latest_text, limit=5, initialize=False)
        memories = [
            f"[{hit.entry.id} · {hit.entry.source}] {hit.entry.content}" for hit in hits
        ]
        compiled = self.compiler.compile(
            role_name=self.role_id, personal_memory=memories, create_identity=False,
        )
        compiled_messages = list(compiled.messages)
        if compiled_messages and compiled_messages[0].get("role") == "system":
            compiled_messages[0] = {
                "role": "system",
                "content": TEMPORARY_NOTICE + "\n\n" + compiled_messages[0]["content"],
            }
        else:
            compiled_messages.insert(0, {"role": "system", "content": TEMPORARY_NOTICE})
        prefix = compiled_messages
        selected = list(self.turns)
        budget = budget_for_role(
            self.cfg, self.role_id, requested_output_tokens=requested_output_tokens
        )
        while len(selected) > 1 and estimate_messages_tokens([*prefix, *selected]) > budget.usable_input:
            # Drop complete oldest exchanges where possible; never drop the latest turn.
            drop = 2 if len(selected) > 2 else 1
            del selected[:drop]
        messages = [*prefix, *selected]
        if estimate_messages_tokens(messages) > budget.usable_input:
            raise RuntimeError("temporary conversation does not fit the active context budget")
        return messages

    def send(self, text, *, requested_output_tokens=None, complete=chat_completion,
             thinking="auto"):
        if self.closed:
            raise RuntimeError("temporary session is closed")
        text = str(text).strip()
        if not text:
            raise ValueError("temporary message cannot be empty")
        self.turns.append({"role": "user", "content": text})
        try:
            response = complete(
                self.cfg, self.role_id,
                self._messages(text, requested_output_tokens=requested_output_tokens),
                max_tokens=requested_output_tokens,
                thinking=thinking,
            )
        except Exception:
            self.turns.pop()
            raise
        message = (response.get("choices") or [{}])[0].get("message") or {}
        content = str(message.get("content") or "")
        finish = (response.get("choices") or [{}])[0].get("finish_reason")
        if finish == "length" and not content:
            return response
        self.turns.append({"role": "assistant", "content": content})
        return response

    def stream(self, text, *, requested_output_tokens=None,
               stream=chat_completion_stream, thinking="auto"):
        """Yield backend events while retaining all new state only in memory."""
        if self.closed:
            raise RuntimeError("temporary session is closed")
        text = str(text).strip()
        if not text:
            raise ValueError("temporary message cannot be empty")
        self.turns.append({"role": "user", "content": text})
        content_parts = []
        finish = None
        committed = False
        try:
            messages = self._messages(
                text, requested_output_tokens=requested_output_tokens)
            for event in stream(
                    self.cfg, self.role_id, messages,
                    max_tokens=requested_output_tokens, thinking=thinking,
                    operation="temporary"):
                content_parts.append(str(event.get("content") or ""))
                if event.get("finish_reason") is not None:
                    finish = event["finish_reason"]
                yield event
            content = "".join(content_parts)
            if not (finish == "length" and not content):
                self.turns.append({"role": "assistant", "content": content})
            committed = True
        finally:
            if not committed and self.turns and self.turns[-1].get("role") == "user":
                self.turns.pop()

    def close(self):
        self.turns.clear()
        self.closed = True


def run_temporary(cfg, *, role_id, input_fn=input, output_fn=print):
    session = TemporarySession(cfg, role_id)
    output_fn("TEMPORARY · new T.A.R.S. state will not be persisted")
    try:
        while True:
            try:
                value = input_fn("TEMPORARY You › ").strip()
            except EOFError:
                break
            if value.lower() in {"/exit", "/quit", "/temporary", "/temporary exit"}:
                break
            if not value:
                continue
            response = session.send(value)
            content = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            output_fn(content)
    finally:
        session.close()
    return 0
