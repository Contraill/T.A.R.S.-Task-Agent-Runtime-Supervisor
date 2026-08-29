# Calibration engine

Local llama.cpp calibration searches for a runtime configuration that fits the current hardware and measures objective execution behavior. It does not evaluate model personality, writing, reasoning or coding quality.

```text
tars calibrate
tars calibrate <alias...>
tars calibrate <alias...> --mid
tars calibrate <alias...> --max
tars calibrate <alias...> --max --fresh
```

The default minimum pass establishes FIT and baseline prompt-processing/token-generation throughput. Mid depth adds context/KV, CPU affinity/thread and resource searches. Maximum depth adds placement refinement and long-context pressure measurements.

Each stage is cached beneath the calibration state directory. Cache inputs include the model SHA-256, requested depth, preceding stage results and a stable hardware/runtime fingerprint. Incompatible entries are marked stale and recomputed. `--fresh` recomputes stages for the requested depth. A ready higher-depth result is never replaced by a shallower run.

Successful output contains `compact`, `normal`, `extended` and `reasonable_max` profiles with context, KV types, CPU placement, threads, GPU placement and measured PP/TG, RAM and VRAM data. Promotion requires the benchmark process to exit, no idle llama-server process, finite unload policy, and return of runtime-managed NVIDIA devices to suspended state.
