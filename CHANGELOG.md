# Changelog

## 0.8.0 - Durable Scheduler and TUI Transcript

- replaced the TUI's joined-line transcript with structured user, assistant, system, tool and error entries
- stream ordinary assistant content directly into the selectable Conversation viewer using buffered updates
- added rail-based message hierarchy, OS/configured display names, scroll follow control and new-output indication
- added mouse and keyboard transcript navigation plus selection-only clipboard copy
- added durable one-shot, recurring interval and model-free condition/watch schedules
- added idempotent run journals, missed-run policies, checkpoint-aware restart recovery and bounded concurrency
- added durable delivery state plus schedule pause, resume, edit, archive, health and run inspection controls
- kept scheduler waiting event-driven and independent of inference model residency
- streamed Temporary-mode TUI responses through the same buffered Conversation entries
- added bounded write-only OSC 52 export for selected transcript text
- wired typed model-free condition predicates into CLI execution with truthful rejection when unavailable
- added cross-process scheduler wake notifications and immediate claimed-backlog handling

## 0.7.6 - Skills, MCP and Generation Policy

- added deterministic global, project and Role skill discovery with progressive disclosure and validation
- added guarded MCP stdio and streamable HTTP clients, registry, filtering, namespaced schemas and evidence
- added a minimal policy-controlled MCP server surface for state health inspection
- unified context reservation with dynamic context-bounded generation ceilings across chat, Temporary, task and agent paths
- added truthful generation-exhaustion metadata and explicit text continuation
- added model-capability-based thinking `auto`, `off` and `on` controls independent of reasoning visibility

## 0.7.5 - Delegation and Subagents

- added durable child contracts with bounded tools, authority, budgets, workspaces and completion evidence
- enforced parent-to-child permission and resource ceilings for nested delegation
- added cooperative cancel, join and timeout lifecycle truth with serialized local inference
- added explicit parent acceptance and staged child-memory promotion
- added child contract inspection and control commands

## 0.7.4 - Workspace Checkpoints and Recovery

- added durable Git-aware and bounded filesystem workspace checkpoints
- added rollback previews with explicit supported scope and external-effect limitations
- added same-HEAD Git worktree/index restoration with untracked-file quarantine
- added pre-rollback safety checkpoints and post-restore hash verification
- added agent-loop pre-mutation checkpoint hooks and workspace inspection commands

## 0.7.3 - Agent Loop and Live Steering

- added a canonical model/action/policy/tool/evidence/checkpoint loop with real ToolResult enforcement
- added deterministic repetition, no-progress, time, tool-failure, context-pressure and unsafe-retry guards
- added durable ordered task controls for queued messages, interrupts, redirects, approvals, pause, resume and cancel
- added truthful cancellable and pending non-cancellable interrupt handling at safe tool boundaries
- added evidence-backed completion contracts and durable control inspection surfaces
- added TUI queued-message feedback and Esc interrupt submission

## 0.7.2 - Native Tool Foundation

- added typed filesystem, terminal/process, Git, user-service, Pacman and structured system tools
- added bounded HTTP requests with DNS/private-network validation, redirect policy, response limits and conditional caching
- added optional Tavily search, extract and crawl capabilities using an environment secret reference
- added isolated Playwright browser profiles, downloads, stable element references and destination enforcement
- added task-linked EvidenceRecords for filesystem, execution, Git, HTTP, browser, service and system observations
- added bounded archive, PDF, document, spreadsheet, image and checksum utilities with capability reporting
- added typed desktop notification and screen-capture boundaries
- added native-first tool selection and inspection surfaces

## 0.7.1 - Execution Backends

- added a guarded execution contract shared by host, container and SSH targets
- added direct-argv host execution with an explicit Bash boundary when shell syntax is required
- added rootless container command construction with resource limits, network-off defaults, no implicit image pulls and explicit workspace/mount policy
- added experimental SSH execution through registered targets, bounded commands and paths, strict host verification and credential references
- connected backend execution to approvals and truthful pre/post action journal records

## 0.7.0 - Policy Core, Approvals and Audit

- added deterministic ScopeGuard decisions for filesystem, process, network, service, remote, secret, elevated, destructive and sandbox-escape effects
- added canonical path and symlink scope enforcement plus private-network and SSRF destination rejection
- added scoped one-call, task, session, target and explicit persistent approvals
- added an action journal that records policy truth before execution and real redacted results afterward
- added policy, approval and audit inspection commands

## 0.6.4 - Memory Maintenance and Reflection

- added inspectable maintenance runs for explicit, session-close, context-rollover and scheduled triggers
- added model-free duplicate, supersession, expiry, index-drift and prompt-pressure audits
- added recoverable expired-memory cleanup and derived-index repair with rollback references
- added model-assisted reflection staging with required model/backend provenance
- kept reflection proposals in candidate review rather than promoting model output directly
- added memory maintenance and maintenance-run inspection commands

## 0.6.3 - Temporary Sessions

- added `tars temporary` and interactive `/temporary` entry points
- added coherent in-memory multi-turn sessions using existing identity, memory and context budgets read-only
- prevented temporary turns from entering conversation, task, checkpoint, context-projection, memory and scheduler stores
- used in-memory command history while the TUI is in Temporary mode
- blocked durable task controls and sideband persistence from Temporary mode
- restored the pre-temporary normal conversation after exit and discarded ephemeral state on exit or crash

## 0.6.2 - Context Engine and Context Epochs

- added explicit soft, hard and emergency context-pressure watermarks
- added atomic task checkpoint, transcript archival and Context Epoch advancement
- protected latest instructions, pending controls and unresolved ToolResults during rollover
- kept compacted transcript slices derived while canonical messages and task state remain durable
- added exact lexical transcript search across active and archived context
- added epoch inspection and transcript-search commands

## 0.6.1 - Memory Core

- added a human-readable canonical memory corpus with structured provenance, scope, confidence, expiry, tags and supersession metadata
- added immutable local revision archives for destructive memory changes
- added staged candidate review and promotion policy owned by MemoryManager APIs
- added rebuildable SQLite/FTS5 lexical retrieval with scope, expiry, recency and deduplication signals
- added explainable memory recall integration for PromptCompiler
- added memory status, search, inspect, review, remember, forget and doctor commands

## 0.6.0 - Session Core, Identity and PromptCompiler

- extended the canonical SQLite state store with durable sessions, general state events, Role state and project references
- added append-safe session and activity event APIs with stable identifiers and timestamps
- added shared identity and per-Role overlay discovery from the persona directory
- added native and compatibility project-context discovery without treating context files as permission sources
- added a PromptCompiler with explicit source composition and token-allocation explanations
- separated backend-emitted reasoning visibility from the durable Activity Trace

## 0.5.3 - Runtime Backend Boundary

- added a local RuntimeBackend contract for status, capabilities, lifecycle, inference, streaming and diagnostics
- moved the working llama.cpp/llama-swap path behind the reference LlamaCppBackend
- normalized backend-emitted content, reasoning, tool calls, finish state and usage
- added an explicit unavailable ColibriBackend boundary for later Oracle integration
- added backend inspection commands and legacy runtime-config migration
- kept external inference providers outside the v1.0 runtime architecture

## 0.5.2 - Calibration Engine

- added resumable `calibrate` commands with minimum, mid and maximum depth
- added stable hardware/runtime fingerprints and stale stage-cache detection
- added adaptive FIT, context/KV, CPU and placement searches using objective llama.cpp benchmarks
- recorded prompt-processing, token-generation, RAM and VRAM measurements
- generated compact, normal, extended and reasonable-maximum runtime profiles
- protected higher-depth results from shallower replacement
- required a finite Zero-Idle check before calibration promotion

## 0.5.1 - Model Lifecycle

- added Hugging Face GGUF search and resumable download support
- added local GGUF import with disk-space and SHA-256 preflight checks
- added content-addressed artifact storage with deduplication
- added integrity, llama.cpp compatibility and calibration-aware readiness checks
- recorded source, revision, license and artifact metadata in the Model Registry
- added safe removal with assigned-Role and shared-artifact protection
- added a versioned backend compatibility manifest

## 0.5.0 - Runtime Configuration

- added RuntimeConfigGenerator from Role, Model Registry and Calibration state
- added read-only runtime plan, render and status commands
- added guarded runtime apply with atomic config replacement, health checks and rollback
- added transactional Role/model/profile switching with Role Registry rollback on failure
- added `model assign`, `model unassign`, `model swap` and `role profile` management commands
- added user-service `start`, `stop` and `logs` commands
- preserved the pre-apply service state during validation and rollback
- supported valid generated configuration when no local Role is bound
- enforced Zero-Idle llama-swap policy invariants in generated configs
- added `tars help` as a top-level help alias
- documented the future Temporary session persistence contract

## 0.4.3 - Orchestration Core

- added capability-based AUTO Role routing without role-name special cases
- added structured DelegationRequest/DelegationResult persistence
- added child delegation tasks that preserve parent ownership
- added checkpoint-gated transactional task handoff
- added durable routing, delegation and handoff history
- added CLI surfaces for routing, delegation and handoff inspection

This repository starts its public Git history at the v0.4.2 development baseline. Earlier work is summarized here rather than reconstructed as artificial commits.

## 0.4.2 — 2026-08-29

Public development baseline.

- background task-run foundation
- SSE final/reasoning streaming
- durable task runs and safe-boundary controls
- SQLite state schema v3
- role-aware ContextManager and tokenizer-aware budgeting
- pinned terminal HUD and theme system
- dynamic Role Registry
- Model Registry and calibration-store foundations
- Zero-Idle llama-swap reference runtime

## Pre-public development

v0.1 through v0.4.1 established the local runtime, model calibration, Role Registry, terminal UI, canonical task state, immutable checkpoints and ContextManager. Detailed benchmark history is intentionally kept out of the public repository's normal source tree.
