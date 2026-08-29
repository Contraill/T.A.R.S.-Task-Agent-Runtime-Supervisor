# Security policy

T.A.R.S. is an agent runtime that is expected to gain filesystem, process, network and service-management tools. Security boundaries therefore live in code, not in model obedience.

Current and planned rules:

- local-only networking by default
- raw model-runtime ports are not remote control surfaces
- ScopeGuard validates execution-time permissions
- the default workspace is an initial tool scope, not permission to the whole home directory
- secrets are referenced from protected storage/environment rather than committed to config
- remote clients require application-level identity/permissions even when transported over a private network
- destructive actions require explicit policy and, where appropriate, confirmation
- portable backups exclude secret material by default

For remote access, Tailscale Serve is the preferred optional path because it lets the Core remain loopback-bound. LAN/public binds increase attack surface and are not the default.

If you find a vulnerability, use GitHub's private security-advisory reporting for the repository if it is enabled. Do not publish credentials, private model endpoints, personal memory or exploit details in a public issue.

Pre-1.0 note: the security model is still being implemented. Do not expose the development snapshot directly to the public internet.
