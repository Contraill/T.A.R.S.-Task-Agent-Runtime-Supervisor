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
             RuntimeBackend
                    |
              local model
```

## Core rules

**Task state is not model context.** Conversations, tasks, decisions, checkpoints and evidence are canonical state. A model receives a projection built for its own context window.

**Roles are not models.** A Role owns purpose, capabilities, execution style and policy. Its model and local RuntimeBackend binding may be replaced.

**Routing preserves semantics.** Local routing validates the exact Role binding, backend health, model capabilities and calibrated context. Unavailable bindings fail explicitly; they are not silently replaced.

**One task has one owner.** Delegation does not change ownership. Handoff does, and must be transactional.

**Security is deterministic.** ScopeGuard and the tool executor decide what can run. The model does not grant itself permission.

**Zero-Idle is a runtime requirement.** No inference model should remain resident merely because the supervisor, scheduler or UI is alive.

## Current state

The implemented platform includes local runtime lifecycle, canonical continuity state, guarded tool execution, durable delegation and a model-free scheduler. Schedules reference canonical tasks and checkpoints; they do not keep inference models resident while waiting. Core API and portability layers remain separate later boundaries.
