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
- plaintext secrets by default

On the destination machine, model manifests are matched against already-installed models. Missing models can be offered for download/import.

Calibration always travels with the bundle. Compatible fingerprints can remain ready; incompatible results remain as history and become stale.

Restore finishes with environment recognition: hardware, runtime-backend availability, tool dependencies, workspace paths and network mode are compared against the saved installation. The user is asked only about meaningful differences, then T.A.R.S. enters a Welcome Back flow.
