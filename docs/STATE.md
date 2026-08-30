# State and prompt composition

SQLite is the canonical transactional store for conversations, sessions, tasks, task events, task controls, task checkpoints, schedules and run/delivery journals, workspace-checkpoint metadata, child delegation contracts, MCP server configuration, Role state and project references. Schema upgrades are additive and preserve existing records. Task checkpoints remain immutable, while ordered state events provide stable identifiers and timestamps for session, model, control and execution activity.

Identity is loaded from:

```text
~/.config/tars/persona/IDENTITY.md
~/.config/tars/persona/SOUL.md
~/.config/tars/persona/roles/<role>.md
```

Role overlays extend shared identity; they do not replace it.

PromptCompiler composes named sources for identity, Role capabilities, personal memory, project context, canonical task/checkpoint state, evidence, skills, filtered tool schemas, recent conversation and pending controls. Its explanation surface reports token allocation and provenance by source without exposing hidden reasoning.

Project discovery prefers `TARS.md` and `.tars.md`, then reads compatible context such as `AGENTS.md`, `CLAUDE.md`, README files and project manifests. These files supply context only and never grant permission.

Reasoning visibility supports Hidden, Summary and Raw. Raw displays only reasoning emitted by the runtime backend. Activity Trace is an independent view of durable state and execution events.
