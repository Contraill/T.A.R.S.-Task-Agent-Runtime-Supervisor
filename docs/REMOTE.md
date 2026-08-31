# Remote Core and clients

The remote model is deliberately small: one authoritative Core and multiple authenticated clients.

```text
computer A: Core + canonical state + models/tools
computer B: T.A.R.S. client
```

Clients read and write the same canonical conversations, tasks, controls and events used locally. Per-client state is limited to authentication, permission and presentation metadata; it is not a second task or conversation store.

## Native API

The versioned `/v1` API provides Core health, conversation/message access, canonical task state, schedule controls, task controls and resumable task-event streams. Event streams use durable event IDs, so a reconnect can continue after the last observed ID without duplicating canonical task state. Schedule mutations notify the Core-owned scheduler loop immediately, including while it is waiting for a later timestamp. Raw llama.cpp and llama-swap endpoints are not proxied.

The long-lived Core owns the lightweight scheduler control loop. Waiting schedules and connected clients do not load an inference model. Only a claimed scheduled task enters the existing task runner and runtime lifecycle.

Client pairing begins with `tars client pair`. The displayed high-entropy one-time code expires and can be exchanged once. Core persists only hashes of pairing codes and bearer credentials; the client token is returned once. `tars client list` and `tars client revoke` inspect and revoke access.

Permissions are explicit scopes for status, conversations, tasks, controls and client administration. A `principal_id` is recorded separately from client identity so later multi-principal policy can extend the model without changing canonical stores.

## Network defaults

1. Core binds to loopback by default.
2. Raw llama.cpp/llama-swap ports remain private to the Core host.
3. Remote access is explicit opt-in.
4. The T.A.R.S. API requires application-level client identity and scopes.
5. A direct non-loopback bind requires both explicit `--allow-remote` and a TLS certificate/key.

## Tailscale

Tailscale is an optional recommended transport. The useful reference pattern is OpenClaw's Tailscale integration: keep the gateway on `127.0.0.1` and use Tailscale Serve for tailnet-only HTTPS access. T.A.R.S. should follow the pattern, not depend on OpenClaw.

Supported transport patterns:

```text
local             default, loopback only
tailscale-serve   private tailnet access, recommended remote mode
tailnet-direct    advanced direct tailnet bind
```

Public Funnel-style exposure is not a default T.A.R.S. mode.

Tailscale identity may be used as one input to authentication, but it does not replace T.A.R.S. pairing/permission scopes. If trusted identity headers are accepted, their source must be verified rather than trusted merely because a header exists.

Loopback operation:

```text
tars client pair
tars core serve
```

Direct remote operation requires deliberate TLS configuration:

```text
tars core serve --host PRIVATE_ADDRESS --allow-remote --cert CERT.pem --key KEY.pem
```
