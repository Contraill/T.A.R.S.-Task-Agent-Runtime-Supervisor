# T.A.R.S.

**Task & Agent Runtime Supervisor**

T.A.R.S. is a local-first agent runtime and control plane for Linux. The project is built around a simple idea: model context is temporary; tasks, memory, permissions and runtime state belong to the supervisor.

The current codebase is a pre-1.0 development snapshot. It already runs on the reference installation, but the generic installer, model manager and real tool loop are still under construction. If you clone it today, treat it as development software rather than a finished end-user package.

## Current state

Implemented in the current reference build:

- llama.cpp + llama-swap runtime integration
- on-demand model loading and Zero-Idle behavior
- dynamic Role Registry
- Model Registry and calibration store foundations
- pinned terminal UI with themes, command palette and context HUD
- SQLite-backed conversations, tasks, events and immutable checkpoints
- role-aware ContextManager with native tokenizer budgeting
- streaming model output and backend-emitted reasoning visibility
- durable task-run state with safe-boundary pause/cancel controls
- generated, transactional llama-swap runtime configuration
- transactional Role assignment, unassignment, model swap and profile changes
- user-service start, stop and log controls
- capability-based delegation, handoff and AUTO routing
- resumable Hugging Face downloads and local GGUF import
- integrity, compatibility and calibration-aware model readiness
- content-addressed model artifacts and safe removal

Calibration automation is the next local-model lifecycle gate.

## Roles

A Role is a working mode: a bundle of purpose, capabilities, execution policy and model binding. It is not the model itself.

The setup wizard will offer four starter Role templates:

- **General** — conversation, planning and everyday work
- **Builder** — creating, editing and repairing projects, code, documents and automations
- **Operator** — tool-first work on systems, files, services, networks and APIs
- **Oracle** — deep analysis, review and second-opinion reasoning

They are starting points, not fixed architecture. Users can rename, edit, remove or add Roles. More Roles are not automatically better; a separate Role is useful when the model, permission boundary, execution style or behavior materially differs.

`compact`, `normal` and `extended` are **runtime profiles**, not Roles.

## Workspace

First-time setup will ask for a default workspace. The proposed default is:

```text
~/TARS-Workspace
```

The workspace is the default place for project work and the initial scope used by tool policy. It is not meant to be an artificial prison: work outside it can be allowed explicitly through ScopeGuard policy.

## Models and providers

Local llama.cpp is the reference backend. T.A.R.S. is being designed so a Role can also bind to an external provider without changing the task or context architecture.

The current v0.4.2 source still carries the reference machine's initial model-registry seed. That is development bootstrap state, not a model recommendation. v0.5.x replaces that assumption with the generic model/provider lifecycle.

Planned provider classes include:

- local llama.cpp / llama-swap
- Colibri for optional heavy local workloads
- OpenAI API
- OpenAI-compatible HTTP endpoints
- provider-specific OAuth-backed services where appropriate

External adapters that have not been exercised against a live service will be marked experimental/best-effort rather than presented as tested.

## Remote use

The default remains local-only. The planned remote model is one authoritative T.A.R.S. Core with one or more clients.

For private remote access, the preferred design follows the same useful pattern seen in OpenClaw: keep the Core bound to loopback and optionally publish only the T.A.R.S. API through **Tailscale Serve**. Raw llama.cpp/llama-swap ports should not be exposed to the network.

Tailscale is optional. T.A.R.S. will still keep its own client pairing and permission scopes above the transport.

## Portability

A portable `.tarsbundle` will contain the state that makes an installation *your* T.A.R.S. — identity, memory, preferences, Roles, tasks, schedules, skills, tool configuration and calibration history.

Model weight files are deliberately excluded. Restore happens into an already-installed T.A.R.S.; the destination uses models already present there or offers to resolve missing model manifests. Calibration history travels with the bundle and is reused only when the destination hardware/runtime fingerprint is compatible.

A restore is a **Welcome Back / environment recognition** flow, not a second first-time awakening.

## Updates

Installed releases will periodically check the repository's latest stable GitHub Release, with a configurable interval and opt-out. The check only reports availability; it does not silently update the program.

The intended flow is:

```text
A newer T.A.R.S. release is available: vX.Y.Z
Run `tars update` to install it.
```

`tars update` will use a staged install, integrity verification, compatibility checks, `doctor`, and rollback on failure. Release installs will not be updated by blindly running `git pull` inside the installed application tree.

## Development

The installed application state belongs under XDG-style user paths such as `~/.config/tars`, `~/.local/share/tars` and `~/.local/state/tars`. The source repository is separate.

For a development checkout:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

Running the current CLI requires a compatible development configuration/runtime. The generic setup wizard is planned for v0.9.0; there is intentionally no claim that this snapshot is a one-command fresh install.

See [docs/ROADMAP.md](docs/ROADMAP.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the current plan.

## Name and inspiration

In this project, **T.A.R.S. means Task & Agent Runtime Supervisor**. The name is also a deliberate nod to the fictional TARS character from *Interstellar*, which helped inspire the idea of a practical, direct assistant.

This is an independent open-source project. It is not affiliated with, sponsored by, or endorsed by the film's studios, production companies, creators or rights holders. The repository does not include film artwork, character models, dialogue, audio, logos or other production assets.

The source code in this repository is licensed under the Apache License 2.0.
Third-party names and trademarks remain the property of their respective owners.
See [LEGAL.md](LEGAL.md) for additional information.
