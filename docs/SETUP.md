# Setup design

The final setup wizard is planned for v0.9.0. This document records the intended behavior so earlier subsystems do not paint the installer into a corner.

## Workspace

Setup asks for one default workspace and proposes:

```text
~/TARS-Workspace
```

The directory can be changed or mapped to an existing project root. It becomes the default working directory and initial tool scope. On restore to another machine, a missing workspace path is remapped during environment recognition rather than silently recreated in an unexpected location.

## Starter Roles

Setup offers four editable templates:

| Role | Intended use |
| --- | --- |
| General | conversation, planning, everyday assistance |
| Builder | creating/editing projects, code, documents and automations |
| Operator | tool-first system/API work |
| Oracle | deep review and second-opinion reasoning |

Users may rename, edit, delete or add Roles. A Role may stay unbound/disabled until a local model/backend is assigned.

The wizard should say explicitly that more Roles are not inherently better. Create another Role when its model, execution style, permission boundary or behavior is meaningfully different.

## Models

Setup does not present one hard-coded model list as "the correct models". It can discover/import/download models and show compatibility metadata. Reference-tested models may be documented separately, but they are not architecture.

Local llama.cpp models require hardware calibration. Later Colibri integration defines its own local readiness checks.
