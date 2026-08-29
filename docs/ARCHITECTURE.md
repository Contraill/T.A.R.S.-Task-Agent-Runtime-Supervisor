# Architecture

T.A.R.S. separates persistent state from model context.

```text
Client / terminal UI
        |
T.A.R.S. Core
        |
+-------+-----------+-------------+
|                   |             |
StateStore      Orchestrator   Scheduler
|                   |
ContextManager   TaskRunner
|                   |
Memory           ScopeGuard
                    |
                ToolRegistry
                    |
             RuntimeProvider
                    |
          local or external model
```

## Core rules

**Task state is not model context.** Conversations, tasks, decisions, checkpoints and evidence are canonical state. A model receives a projection built for its own context window.

**Roles are not models.** A Role owns purpose, capabilities, execution style and policy. A model/provider is bound to a Role and may be replaced.

**One task has one owner.** Delegation does not change ownership. Handoff does, and must be transactional.

**Security is deterministic.** ScopeGuard and the tool executor decide what can run. The model does not grant itself permission.

**Zero-Idle is a runtime requirement.** No inference model should remain resident merely because the supervisor, scheduler or UI is alive.

## Current state

v0.4.2 contains the state/context/streaming foundation. Tool execution and full orchestration are intentionally not faked before their policy layers exist.
