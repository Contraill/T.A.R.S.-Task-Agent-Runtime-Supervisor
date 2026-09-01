# Skills and MCP

Skills are Markdown procedures under global, project or Role roots. Discovery reads validated name, description and version metadata first. Full instructions and bounded resources enter PromptCompiler only when explicitly selected. Project and Role scopes override matching global names deterministically. Symlink escape, malformed metadata, oversized instructions and out-of-root resources fail validation. Skill content is context, never ScopeGuard authorization.

The MCP registry supports stdio and streamable HTTP servers, enable/disable state, include/exclude filters and explicit per-tool authority contracts. Tool summaries omit input schemas until selected. Names use the stable `mcp.SERVER.TOOL` namespace and cannot shadow native tools. Missing classification defaults to elevated denial.

An authority contract derives every policy scope from the validated argument object sent to the server. A scope declares its effect and may extract a scalar or array target with an RFC 6901 JSON pointer. Path scopes also declare fixed `allowed_paths`; network scopes declare fixed HTTPS/HTTP origins in `allowed_hosts`. Compound tools declare multiple scopes. For example:

```json
{
  "write_file": {
    "scopes": [{
      "name": "destination",
      "effect": "write",
      "target": "/path",
      "target_kind": "path",
      "allowed_paths": ["/workspace"]
    }]
  }
}
```

The value is supplied through `tars mcp register --effects-json`. A legacy string effect remains an opaque, exact-argument contract; it does not gain a caller-selected target or path/network scope. `tars mcp call` accepts only the actual `--arguments-json`. Compound calls accept a keyed `--approvals-json` mapping.

MCP process connections, HTTP destinations and tool calls use ScopeGuard and ApprovalBroker. Connection authorization binds the immutable server configuration. Call authorization binds the server revision, trusted authority contract, discovered input schema and the exact argument snapshot later serialized to `tools/call`. Registry or schema changes therefore invalidate earlier call authority. Real calls produce ActionJournal and EvidenceRecord entries; errors, disconnects and MCP `isError` results remain failures. HTTP redirects are bounded, same-host and SSRF-validated. Credentials use `env:` references and resolved values are redacted from results.

Every stdio server is a non-persistent child supervised by the same parent-death topology used for managed background processes. Abrupt owner death and explicit close kill the helper and its descendant tree, including detached process groups. Close interrupts an in-flight request, waits for pipe and stderr-drain quiescence, and then releases resolved secret values. Externally managed persistent MCP services use the streamable HTTP transport instead of the stdio child lifecycle. The bundled minimal stdio server exposes only state-store health and applies the same argument-derived scope binder and real ToolResult contract.
