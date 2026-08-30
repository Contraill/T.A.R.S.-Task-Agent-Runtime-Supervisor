# Skills and MCP

Skills are Markdown procedures under global, project or Role roots. Discovery reads validated name, description and version metadata first. Full instructions and bounded resources enter PromptCompiler only when explicitly selected. Project and Role scopes override matching global names deterministically. Symlink escape, malformed metadata, oversized instructions and out-of-root resources fail validation. Skill content is context, never ScopeGuard authorization.

The MCP registry supports stdio and streamable HTTP servers, enable/disable state, include/exclude filters and explicit per-tool effect policy. Tool summaries omit input schemas until selected. Names use the stable `mcp.SERVER.TOOL` namespace and cannot shadow native tools. Missing effect classification defaults to elevated denial.

MCP process connections, HTTP destinations and tool calls use ScopeGuard and ApprovalBroker. Real calls produce ActionJournal and EvidenceRecord entries; errors, disconnects and MCP `isError` results remain failures. HTTP redirects are bounded, same-host and SSRF-validated. Credentials use `env:` references and resolved values are redacted from results. The bundled minimal stdio server exposes only state-store health and still applies the normal policy and real ToolResult contract.
