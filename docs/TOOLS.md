# Native Tools

Native tools carry typed targets, effects and structured results through ScopeGuard, ApprovalBroker, ActionJournal and EvidenceRecord. Selection prefers a matching native tool, then a trusted MCP tool, then the terminal escape hatch.

Filesystem tools provide bounded list, stat, read, search, mkdir, copy, move, patch, write and delete operations. Canonical path scopes apply to every source and destination. Patch replacements require an exact context match. Delete is destructive.

Terminal execution defaults to direct argv. Bash is an explicit shell boundary and PTY allocation is explicit. Background host processes expose list, poll, wait, logs, input, signal and kill state. Full logs remain in restricted local process-log files while model-visible output is bounded. Resolved environment-reference values are redacted.

Git tools return observed branch, HEAD, dirty state and commit identity. Push requires separate high-impact and network authorization. User-service mutations verify post-state. Pacman is the reference package backend; installation and upgrade use full `-Syu` semantics so semantic tools cannot introduce unsupported partial upgrades.

HTTP supports bounded GET, HEAD and explicitly authorized state-changing methods. Every destination and redirect is validated against private-network and DNS policy. Cross-host redirects require a separate request.

Archive tools list, create and safely extract ZIP and tar-class archives. Extraction rejects traversal, links and special archive members before extracting. Artifact hashing provides streamed checksums and optional expected-digest verification. HTTP downloads return a verified byte count and SHA-256 digest.

PDF tools use installed deterministic backends. Poppler provides metadata, text/search and rendering on the reference system. The optional `pypdf` backend provides regenerated merge, split, reorder, rotate, page deletion, form filling and export operations with structural page-count verification. Reliable annotation and content-safe redaction are reported unavailable when no suitable backend exists; arbitrary in-place text editing is not claimed.

Document tools inspect and extract bounded text formats and apply exact patch-style edits. Optional `python-docx` adds DOCX extraction and LibreOffice, when installed, provides headless conversion. Spreadsheet tools support CSV ranges without extra dependencies; optional `openpyxl` adds XLSX ranges, sheets, formulas and CSV export. Optional Pillow provides deterministic image information, resize, crop, rotate, conversion and compression. Every format reports its actual runtime capability.

Desktop utilities expose typed notification and screen-capture operations through available Linux backends. Screen output must be created and non-empty before success is reported. Clipboard access is intentionally deferred.

Tavily is optional and mock-tested in the current reference environment because no credential was available. Missing `env:TAVILY_API_KEY` disables only Tavily operations. HTTP and browser tools remain available independently.

Browser automation uses a dedicated T.A.R.S. profile and download directory. Personal profiles require explicit opt-in. Navigation and subresources remain inside authorized public destinations. Snapshots expose stable element references. Raw evaluation is elevated and unavailable under the default policy. The reference environment completed a live Chromium navigation, snapshot and screenshot check; installations without Playwright/Chromium report the capability unavailable.

MCP tools remain externally supplied capabilities. Server-qualified names cannot shadow native tools. Connections and calls pass through the same policy, approval, audit and evidence path as native tools; unknown effects default to elevated denial. Trusted registry contracts derive call effects and targets from the validated argument snapshot actually sent to the server. A parallel caller-supplied target is not accepted. Streamable HTTP destinations and redirects receive SSRF validation. Registry configuration stores secret references rather than resolved values.

```text
tars tool list
tars evidence [--task TASK_ID] [--type TYPE]
tars skill list|show|doctor
tars mcp list|register|enable|disable|tools|call
```
