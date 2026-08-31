# Roadmap to 1.0

The public repository starts at v0.4.2. Version numbers below describe capability gates, not calendar promises.

## v0.4.3 — orchestration core

Status: implemented in the v0.4.3 development milestone.

- single task owner
- durable delegation request/result with bounded child tasks
- verified immutable checkpoint before handoff ownership changes
- capability-based AUTO routing with no Role-name special cases
- durable routing, delegation and handoff history

Exit gates: delegation preserves parent ownership; handoff cannot switch ownership without a verified checkpoint; AUTO routing considers only enabled, model-bound Roles that cover the required capabilities.

## v0.5.0 — runtime configuration

Status: implemented.

- generated llama-swap config from Role + Model + Calibration state
- read-only runtime plan/render/status surfaces
- transactional Role/model/profile switching
- runtime health checks and automatic config rollback
- Zero-Idle policy invariants enforced by the generator

Exit gates: generated normal profiles match the calibrated reference runtime; apply never leaves a broken config active; failed switch restores both runtime config and Role Registry; no runtime management path enables persistent performance monitoring.

## v0.5.1 — model manager

Status: implemented.

- pull/import/verify/remove
- resumable downloads
- source/license/hash metadata
- no hard-coded model recommendation requirement

Local llama.cpp models become runtime-ready only after integrity verification,
compatibility validation and calibration for the current artifact. Physical
artifacts are addressed by SHA-256 and retained while any registry entry refers
to them.

## v0.5.2 — calibration engine

Status: implemented.

- default minimum depth, plus `--mid` and `--max`
- hardware/runtime fingerprinting
- cached/resumable objective tuning
- compact/normal/extended profiles
- Zero-Idle validation

Stage results are keyed by the model artifact and a stable hardware/runtime
fingerprint. Higher-depth results are not replaced by shallower runs.

## v0.5.3 — runtime backend boundary

Status: implemented.

- local RuntimeBackend contract independent of Roles
- reference LlamaCppBackend over llama-swap
- status, capabilities, lifecycle, inference, streaming and diagnostics surfaces
- backend-emitted reasoning and tool-call normalization
- explicit unavailable ColibriBackend boundary for later Oracle integration

External cloud inference providers are outside the v1.0 product scope.

## v0.6.0 — session core, identity and PromptCompiler

Status: implemented.

- schema-versioned sessions, state events, Role state and project references
- shared identity with explicit per-Role overlays
- native `TARS.md` / `.tars.md` project context and compatibility-source discovery
- explicit prompt sources with inspectable token allocation
- backend-emitted reasoning visibility kept separate from Activity Trace

## v0.6.1 — memory core

Status: implemented.

- human-readable canonical memory corpus and immutable revision history
- structured scope, provenance, confidence, expiry, tags and supersession
- staged candidate review and deterministic promotion policy
- rebuildable SQLite/FTS5 lexical retrieval with explainable ranking signals
- memory inspection, search, remember, forget, review and repair commands

## v0.6.2 — context engine and context epochs

Status: implemented.

- actual runtime-profile context budgets with exact tokenizer support
- soft, hard and emergency pressure watermarks
- atomic task checkpoint, transcript archival and epoch advancement
- protected latest instructions, controls and unresolved ToolResults
- lexical search over active and archived transcript messages

## v0.6.x — identity, memory and context epochs

- PromptCompiler and capability-aware system prompt
- Persona/Identity primitives
- MemoryManager and user/project memory
- automatic context epochs, retrieval and compaction

The mechanism lands here; the final First Awakening remains late in the release train.

## v0.6.3 — temporary sessions

Status: implemented.

- `tars temporary` and `/temporary` isolated ephemeral sessions
- temporary conversation/task/context/tool trace lives in memory only
- existing persistent identity/preferences/memory may be read, never mutated by the temporary session
- exiting temporary mode returns to the pre-temporary normal state without promoting temporary messages or task state
- temporary sessions cannot create work that must survive the session boundary, including scheduled/future tasks
- external tool side effects remain real even though T.A.R.S. does not persist its own temporary state

See [TEMPORARY.md](TEMPORARY.md) for the persistence contract.

## v0.6.4 — memory maintenance and reflection

Status: implemented.

- optional explicit, session-close, context-rollover and scheduled triggers
- model-free duplicate, supersession, expiry, index-drift and prompt-pressure audits
- recoverable expiry cleanup and derived-index repair
- reflection proposals staged for review with model/backend provenance
- inspectable maintenance actions and rollback references

## v0.7.x — tools and real agent loop

### v0.7.0 — policy core, approvals and audit

Status: implemented.

- deterministic effect and risk classification outside inference
- canonical filesystem scopes with traversal and symlink-escape protection
- destination restrictions for network actions and private-network SSRF rejection
- ephemeral and explicit persistent approval scopes
- redacted action records linked to canonical state events

See [POLICY.md](POLICY.md) for the policy and audit contracts.

### v0.7.1 — execution backends

Status: implemented.

- reference-tested guarded host execution
- tested rootless Podman/Docker command boundary with bounded resources, mounts and network
- ephemeral workspaces by default and separately authorized writable host workspaces
- experimental, mock-tested SSH execution through registered bounded targets
- backend results attached to the v0.7.0 action journal

See [EXECUTION.md](EXECUTION.md) for backend support and isolation behavior.

### v0.7.2 — native tools, research, evidence and browser

Status: implemented.

- native filesystem, terminal/process, Git, user-service, Pacman and system inspection tools
- direct argv and explicit Bash/PTY boundaries with bounded output and durable background logs
- HTTP destination validation, SSRF protection, redirects, limits and cache validators
- optional Tavily search/extract/crawl with `env:TAVILY_API_KEY`; mock-tested without a live credential
- isolated Playwright browser profile/download policy and stable element references; live-tested on the reference environment and mock-tested in dependency-minimal CI
- ZIP/tar archive handling, artifact checksums and capability-reported PDF, document, spreadsheet and image utilities
- typed desktop notifications and capability-reported screen capture
- lightweight task/event-linked EvidenceRecords

See [TOOLS.md](TOOLS.md) for tool contracts and verified support boundaries.

### v0.7.3 — verified agent loop and live steering

Status: implemented.

- canonical context → model decision → policy/tool → real ToolResult → evidence → checkpoint loop
- evidence-backed completion contracts; model text alone cannot complete a task
- repetition, no-progress, time, tool-failure, context-pressure and unsafe-retry guards
- durable priority-ordered message, interrupt, approval, redirect, pause, resume and cancel controls
- next-safe-boundary queued delivery with truthful cancellable/non-cancellable interrupt state
- TUI queued-message feedback and Esc interrupt submission

See [AGENT_LOOP.md](AGENT_LOOP.md) for loop, control and completion contracts.

### v0.7.4 — workspace checkpoints and rollback

Status: implemented.

- durable Git-aware checkpoints preserving tracked index/worktree and bounded untracked files
- bounded non-Git captured-file snapshots with symlink and resource limits
- explicit rollback preview and destructive approval
- same-HEAD Git recovery without branch/history movement
- new untracked files moved to checkpoint quarantine rather than deleted
- automatic pre-rollback safety checkpoint and post-restore hash verification
- agent-loop pre-mutation checkpoint hook
- external effects explicitly marked non-reversible

See [WORKSPACE_RECOVERY.md](WORKSPACE_RECOVERY.md) for support and recovery truth.

### v0.7.5 — bounded delegation and subagents

Status: implemented.

- parent/child task contracts with separate canonical task/event state and bounded context
- immutable tool, filesystem, effect, destination, remote-target and secret-reference ceilings
- nested permission and resource budgets that can only narrow parent authority
- cancel, join and cooperative timeout truth with local GPU inference concurrency of one
- isolated/shared read-only/read-write workspace policies with exclusive mutation scheduling
- explicit completion evidence, parent acceptance and staged child-memory promotion

See [DELEGATION.md](DELEGATION.md) for contracts, scheduling and acceptance semantics.

### v0.7.6 — skills and guarded MCP interoperability

Status: implemented.

- global, project and Role skill scopes with deterministic override order
- metadata-only discovery followed by explicit bounded instruction loading
- validation, version metadata and doctor reporting; skill text never grants authority
- stdio and streamable HTTP MCP transports with registry, enable/disable and filtering
- stable server-qualified tool names, progressive schema disclosure and conservative effects
- ScopeGuard, ApprovalBroker, ActionJournal and EvidenceRecord integration for MCP calls
- minimal controlled MCP server surface with real ToolResult enforcement
- dynamic profile/context-bounded generation ceilings and truthful on/off thinking capability

See [SKILLS_MCP.md](SKILLS_MCP.md) and [GENERATION.md](GENERATION.md) for support and policy details.

## v0.8.x — durable automation, remote use and portability

### v0.8.0 — durable scheduler

Status: implemented.

- one-shot, recurring interval and condition/watch schedules over canonical tasks
- durable idempotent run journal, checkpoint recovery and missed-run policy
- bounded concurrency, delivery state and pause/edit/archive controls
- event-driven waiting with no inference model residency

See [SCHEDULER.md](SCHEDULER.md) for scheduling and recovery semantics.

### v0.8.1 — Core API and authenticated clients

Status: implemented.

- one authoritative Core over canonical conversations, tasks, controls and events
- one-time pairing, hashed credentials, revocation and per-client permissions
- resumable task-event streaming and native client access
- authenticated schedule controls with immediate Core scheduler wakeups
- loopback defaults with explicit TLS requirements for direct remote binding
- principal-ready client identity without per-client state models

See [REMOTE.md](REMOTE.md) for API, pairing and transport boundaries.

### v0.8.2 — local runtime routing

Status: implemented.

- exact Role to local backend/model/runtime/profile routing
- artifact, integrity, calibration, health, context and capability validation
- mandatory completion, stream and task-inference preflight against backend inventory
- effective context bounded by both calibrated profile and backend-reported model context
- durable ready/unavailable route evidence with no silent substitution
- explicit task-owner handoff requirement and local-only inference policy
- on-demand load and finite-unload lifecycle contracts

See [RUNTIME_BACKENDS.md](RUNTIME_BACKENDS.md) for route and lifecycle truth.

### v0.8.3 — Oracle / Colibri

Status: implemented.

- optional loopback-local Colibri adapter with health, version, capability and model probes
- on-demand Heavy lifecycle with bounded short TTL and no model load during inspection
- normalized content/reasoning streams and backend-native Context Engine token counts
- truthful unbound, not-configured, unavailable and healthy states in routing and doctor
- Oracle child delegation with explicit input and required result-evidence contracts

Oracle remains optional. Model installation and hardware acceptance are separate operator actions.

### v0.8.4 — extension boundaries

Status: implemented.

- versioned minimal providers for runtime and tool extension classes
- metadata-only discovery with explicit enable and in-process trust gates
- authoritative local-only runtime routing and guarded tool dispatch integration
- non-shadowing extension namespaces and visible built-in/third-party provenance
- MCP retained as the preferred low-coupling external tool boundary

See [EXTENSIONS.md](EXTENSIONS.md) for contracts and trust behavior.

### v0.8.5 — SecretStore

Status: implemented.

- provider-based secret references with optional per-consumer scopes
- environment and injected external-manager adapters without plaintext persistence
- authoritative resolution in execution/SSH, MCP, web-service and Core client paths
- response and process-output redaction at credential-bearing boundaries

See [SECRETS.md](SECRETS.md) for reference and consumer behavior.

### v0.8.6 — Backup, restore and migration

Status: implemented.

- checksummed and versioned `.tarsbundle` format with live SQLite snapshots
- model weights, secrets, browser state, caches and process logs excluded by default
- portable client metadata retained without transferable bearer verification material
- complete pre-mutation validation, schema-aware migration and staged local rollback
- truthful destination reconciliation for missing model assets and unresolved secrets

The v0.8 family is complete. Installer and interactive Welcome Back presentation remain v0.9 work.

## v0.9.0 — installer/setup

The setup wizard will:

- detect hardware and runtime dependencies
- choose/create the default workspace (`~/TARS-Workspace` proposed default)
- offer the starter Role templates General, Builder, Operator and Oracle
- let the user rename, edit, remove or add Roles before finishing
- explain why duplicating near-identical Roles usually adds complexity rather than capability
- bind installed/imported/downloaded models without presenting a fixed tested-model list as a requirement
- choose calibration depth for local models
- choose starter tools/skills and permission defaults
- configure memory/privacy
- optionally configure private remote access
- run doctor and Zero-Idle acceptance

At least one enabled, usable default Role is required before setup becomes READY.

## v0.9.1 — final prompt and First Awakening

Only after the installed capabilities are real and healthy:

- build the final system prompt
- learn the user from setup data and explicit conversation
- ask only useful missing questions
- persist confirmed preferences/memory

A restored installation runs environment recognition and Welcome Back instead.

## v0.9.2 — updater and recovery

- periodic, configurable GitHub Release check
- no silent update installation
- `tars update check`
- `tars update` staged download/install
- release digest/checksum verification
- schema/compat checks
- `doctor` before promotion
- automatic rollback on failed update

The installed release is not maintained with an in-place `git pull`.

## v0.9.5 — release candidate

Clean-install, upgrade, restore, remote-client, scheduled-task, tool-loop, security and Zero-Idle acceptance tests.

## v1.0.0

Stable Linux-first release after all gates above pass.
