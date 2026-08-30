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

**One task has one owner.** Delegation does not change ownership. Handoff does, and must be transactional.

**Security is deterministic.** ScopeGuard and the tool executor decide what can run. The model does not grant itself permission.

**Zero-Idle is a runtime requirement.** No inference model should remain resident merely because the supervisor, scheduler or UI is alive.

## Current state

The v0.5 line contains runtime configuration, model lifecycle, objective calibration and the local RuntimeBackend boundary. v0.6.0 adds the canonical session/event store, shared identity and PromptCompiler. Tool execution remains gated on the later ScopeGuard and ToolRegistry layers.
