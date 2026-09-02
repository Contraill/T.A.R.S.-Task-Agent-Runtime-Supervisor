# Portability

A `.tarsbundle` transfers personal T.A.R.S. state into another compatible installation. It does not install the application itself.

## Included

- identity/persona and onboarding state
- memory and project memory
- preferences and UI settings
- Role definitions and bindings
- model manifests, sources and hashes
- full calibration history and fingerprints
- skills and portable tool/MCP configuration
- conversations, tasks, events, checkpoints and schedules
- portable workspace configuration

## Excluded

- GGUF/model weight files
- arbitrary external OS binaries/packages
- secret values, Core bearer verifiers and pairing codes
- browser profiles, cookies and session databases
- caches, downloads and process logs
- workspace recovery payloads (their metadata is retained and marked for recreation)

On the destination machine, model manifests are matched against installed paths. The restore report identifies missing model assets; it does not download them.

Calibration always travels with the bundle. Compatible fingerprints can remain ready; incompatible results remain as history and become stale.

`tars backup create`, `inspect`, and `restore --replace` use the production bundle path. Restore validates archive membership, declared sizes, checksums, database integrity and version compatibility before replacing local state. It migrates the staged database to the current schema, records a durable restore journal before replacement, fsyncs same-parent old/new objects, rebuilds the memory index from canonical files, and records one durable commit point. Process death before that point is rolled back; process death after it finalizes the new state. Normal startup fails closed while recovery is required. The next backup create/restore recovers automatically, or `tars backup recover` can be run explicitly.

The restore report identifies missing model assets, secret references, workspace paths, MCP commands and recovery payloads. Runtime/calibration revalidation is explicitly required. Restore does not claim rollback of external effects. Interactive Welcome Back presentation remains later productization work.
