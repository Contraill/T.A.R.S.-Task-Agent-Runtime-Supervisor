from __future__ import annotations

from dataclasses import dataclass
import json

from .checkpoints import latest_checkpoint
from .context import estimate_messages_tokens
from .conversation import list_messages
from .identity import load_identity
from .memory import search as search_memory
from .projects import discover_project_context
from .roles import get_role, resolve_role_id
from .tasks import canonical_task_state


@dataclass(frozen=True)
class PromptSource:
    name: str
    content: str
    tokens: int
    protected: bool
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledPrompt:
    messages: tuple[dict, ...]
    sources: tuple[PromptSource, ...]
    total_tokens: int

    def explain(self):
        return {
            "total_tokens": self.total_tokens,
            "sources": [
                {"name": source.name, "tokens": source.tokens,
                 "protected": source.protected, "provenance": list(source.provenance)}
                for source in self.sources
            ],
        }


def _source(name, content, *, protected=False, provenance=()):
    content = str(content or "").strip()
    return PromptSource(name, content, estimate_messages_tokens([{"role": "system", "content": content}]),
                        protected, tuple(provenance))


class PromptCompiler:
    def compile(self, *, role_name, conversation_id=None, task_id=None, project_path=None,
                personal_memory=(), memory_query=None, memory_scope=None,
                evidence=(), skills=(), tool_schemas=(), pending_controls=(),
                recent_limit=100):
        role_id = resolve_role_id(role_name)
        role = get_role(role_id)
        identity = load_identity(role_id)
        sources = [
            _source("base_identity", f"{identity.identity}\n\n{identity.soul}", protected=True,
                    provenance=identity.sources[:2]),
            _source("role_overlay", f"Role: {role.display_name}\n{role.description}\n{identity.role_overlay}",
                    protected=True, provenance=identity.sources[2:]),
            _source("capabilities", json.dumps(list(role.capabilities), ensure_ascii=False)),
        ]
        recalled = list(map(str, personal_memory))
        memory_provenance = []
        if memory_query:
            hits = search_memory(memory_query, scope=memory_scope)
            recalled.extend(hit.entry.content for hit in hits)
            memory_provenance.extend(hit.entry.id for hit in hits)
        if recalled:
            sources.append(_source("personal_memory", "\n".join(recalled),
                                   provenance=memory_provenance))
        if project_path:
            project = discover_project_context(project_path)
            sources.append(_source("project_context", project.content,
                                   provenance=tuple(map(str, project.files))))
        if task_id:
            task = canonical_task_state(task_id)
            checkpoint = latest_checkpoint(task_id)
            payload = {"task": task, "checkpoint": checkpoint.state if checkpoint else None}
            sources.append(_source("task_state", json.dumps(payload, ensure_ascii=False), protected=True))
        if evidence:
            sources.append(_source("evidence", "\n".join(map(str, evidence))))
        if skills:
            sources.append(_source("skills", "\n".join(map(str, skills))))
        if tool_schemas:
            sources.append(_source("tool_schemas", json.dumps(list(tool_schemas), ensure_ascii=False)))
        if pending_controls:
            sources.append(_source("pending_controls", "\n".join(map(str, pending_controls)), protected=True))
        messages = [{"role": "system", "content": f"[{s.name}]\n{s.content}"}
                    for s in sources if s.content]
        if conversation_id:
            records = list_messages(conversation_id, include_sideband=False, limit=recent_limit)
            history = [{"role": r.role, "content": r.content} for r in records
                       if r.role in {"system", "user", "assistant", "tool"}]
            history_tokens = estimate_messages_tokens(history)
            sources.append(PromptSource("recent_conversation", "", history_tokens, True))
            messages.extend(history)
        return CompiledPrompt(tuple(messages), tuple(sources), estimate_messages_tokens(messages))
