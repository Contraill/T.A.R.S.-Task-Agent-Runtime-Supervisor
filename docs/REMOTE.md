# Remote Core and clients

The v1 remote model is deliberately small: one authoritative Core and multiple clients.

```text
computer A: Core + canonical state + models/tools
computer B: T.A.R.S. client
```

The same chat/task protocol is used locally and remotely.

## Network defaults

1. Core binds to loopback by default.
2. Raw llama.cpp/llama-swap ports remain private to the Core host.
3. Remote access is explicit opt-in.
4. The T.A.R.S. API requires application-level client identity and scopes.

## Tailscale

Tailscale is an optional recommended transport. The useful reference pattern is OpenClaw's Tailscale integration: keep the gateway on `127.0.0.1` and use Tailscale Serve for tailnet-only HTTPS access. T.A.R.S. should follow the pattern, not depend on OpenClaw.

Planned modes:

```text
local             default, loopback only
tailscale-serve   private tailnet access, recommended remote mode
tailnet-direct    advanced direct tailnet bind
```

Public Funnel-style exposure is not a default T.A.R.S. mode.

Tailscale identity may be used as one input to authentication, but it does not replace T.A.R.S. pairing/permission scopes. If trusted identity headers are accepted, their source must be verified rather than trusted merely because a header exists.
