# Secret references

T.A.R.S. stores portable secret references, not plaintext credential values. A reference has the form `provider:key`; the built-in Linux environment adapter uses `env:VARIABLE`. External secret managers can supply a provider implementing `get(key)` without changing consumers.

SecretStore requires a named consumer for every resolution. Optional `[secrets.scopes]` entries restrict an exact reference to an allowlist such as `web:tavily`, `mcp:server-name`, `ssh:target-name`, `execution:host`, `execution:container`, `execution:background`, or `core:client`.

Production credential paths resolve only at their use boundary:

- execution environment values immediately before process creation;
- SSH identity references while constructing an authorized invocation;
- MCP environment or authorization values while connecting/requesting;
- web-service tokens while constructing the approved request;
- remote Core tokens while constructing the authorization header.

Resolved values are not inserted into prompts, action arguments, durable configuration, or normal result objects. Process and integration output paths redact values they can receive back. Pairing still returns a one-time Core credential to the caller; store it in a secret provider and configure clients with its reference when persistence is needed.
