# Update design

Release installs should know when a newer stable version exists without turning update checks into an always-running service.

Planned default behavior:

- check no more than once every 24 hours while the CLI is already being used
- configurable interval and a complete opt-out
- use the public GitHub Releases metadata for the project
- cache the last check/result locally
- never wake a model or GPU for an update check
- show one unobtrusive notice for a newer version
- never install an update without an explicit user action

Commands:

```text
tars update check
tars update --dry-run
tars update
tars rollback
```

Install path:

```text
latest release metadata
-> select platform artifact
-> download to staging
-> verify release digest/checksum
-> inspect schema/compatibility
-> backup current install/state metadata
-> staged install
-> tars doctor --compat
-> promote atomically
-> rollback automatically if validation fails
```

Development checkouts may use Git normally. Installed releases should not mutate themselves with an unverified `git pull`.
