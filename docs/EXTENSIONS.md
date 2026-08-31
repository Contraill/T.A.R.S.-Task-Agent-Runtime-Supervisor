# Extension boundaries

T.A.R.S. exposes a small versioned provider boundary for runtime backends and tools. Discovery reads packaging metadata only. It does not import third-party code.

Third-party providers use Python entry-point groups `tars.runtime_backends` and `tars.tools`. A provider declares `api_version = 1`, its exact `kind` and `name`, and a callable `create` factory. An identifier must appear in both `extensions.enabled` and `extensions.trusted` before T.A.R.S. imports it. `tars extension list` shows provenance and activation state without loading providers.

Runtime providers are accepted only when they implement the complete backend contract and explicitly guarantee local-only inference plus probe-only Zero-Idle behavior. The normal Role router, inventory/capability checks, context limits and prepare/release lifecycle remain authoritative.

In-process tool providers use `ext.PROVIDER.*` names and cannot shadow existing tools. Registration requires a non-empty deterministic set of keyed `ScopeRequest` values; execution passes through the canonical policy/action-journal runtime and must return a real `ToolResult`. Extension-defined pre-execution and cancellation hooks are not installed into Core control flow. MCP remains the preferred boundary for ordinary external tools because it keeps implementation code out of the Core process.

```toml
[extensions]
enabled = ["runtime_backend:example"]
trusted = ["runtime_backend:example"]
```

Trust permits code execution inside the Core process. It is distinct from model/tool permissions and does not bypass policy.
