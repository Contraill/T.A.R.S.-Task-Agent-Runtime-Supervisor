# Model lifecycle

T.A.R.S. stores local model artifacts by SHA-256 and keeps human-facing aliases in the Model Registry. Multiple aliases may refer to one physical artifact.

## Commands

```text
tars model search <query>
tars model pull <repository-or-url> --filename <file.gguf> --alias <alias>
tars model import <file.gguf> --alias <alias>
tars model verify <alias>
tars model info <alias>
tars model remove <alias>
```

Downloads resume from a partial cache file when the source supports HTTP range requests. `--sha256` supplies an expected digest. Pull and import perform disk-space preflight, verify the artifact digest, require GGUF compatibility, then update the registry atomically.

Model and Role registry mutations share a stable cross-process transaction lock. Each
committed registry generation is persisted, so a caller attempting to save a stale
snapshot is rejected instead of erasing a concurrent update. Model removal and Role
binding use the same transaction boundary, preventing dangling bindings. Portable
backup and restore share an installation-wide outer lock with these transactions.

Readiness progresses through acquisition, integrity verification, runtime compatibility and calibration. A local llama.cpp model is not eligible for generated runtime configuration until calibration for its exact SHA-256 artifact is ready.

Run `tars calibrate <alias>` for the minimum objective pass, or add `--mid` or `--max` for broader context, CPU, placement and pressure searches. `--fresh` ignores compatible stage cache entries without allowing a shallower result to replace a deeper one.

Removal is refused while any Role is assigned to the alias. Removing an alias does not delete a physical artifact that another registry entry still references.

The compatibility manifest is versioned in the implementation. It describes supported backends and readiness requirements; it is not a model recommendation list.
