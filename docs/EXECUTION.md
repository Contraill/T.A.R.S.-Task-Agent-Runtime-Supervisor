# Execution Backends

Execution requests describe an operation and target without embedding container or SSH orchestration in model-generated shell text. GuardedExecutor applies ScopeGuard and ApprovalBroker decisions, creates the pre-action journal record, invokes the selected backend, and records the real result.

HostBackend is reference-tested for explicitly authorized trusted local work. Direct argv execution is the default. Shell syntax uses an explicit `/bin/bash -lc` boundary. Environment values use `env:NAME` references and are resolved outside model-visible arguments.

ContainerBackend is tested with mocked runtime integration and becomes available when a rootless Podman or Docker runtime is verified. It applies CPU, RAM, process and time limits; disables networking by default; forbids implicit image pulls; and creates an ephemeral workspace by default. Host-backed read-only or read-write workspaces are explicit. A writable host workspace requires write authorization. Mounts outside the workspace require separate sandbox-escape approval. No host credentials are inherited implicitly.

SSHBackend is experimental and mock-tested because no live remote target is part of the reference environment. It accepts only registered targets, bounded command names and bounded remote working directories. Strict host-key checking and batch mode are mandatory. Identity files are resolved from credential references rather than model text.

```text
tars execution-backend
tars execution-backend host
tars execution-backend container
tars execution-backend ssh
```
