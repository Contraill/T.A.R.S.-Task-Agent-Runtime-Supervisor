# Runtime backends

A Role binds to a model alias. The model record identifies a local RuntimeBackend, keeping runtime mechanics separate from Role policy, task ownership and durable state.

```text
Role
  -> model binding
  -> RuntimeBackend
       -> LlamaCppBackend
       -> ColibriBackend
```

The backend contract covers identity, availability, health, runtime and model capabilities, reasoning/tool metadata, load/unload lifecycle, inference, normalized streams and diagnostics.

## LlamaCppBackend

LlamaCppBackend is the reference-tested implementation. It uses the existing llama-swap endpoint and preserves generated configuration, calibrated Role bindings, atomic apply/rollback and Zero-Idle behavior. Model loading remains on demand; finite llama-swap TTL policy manages unload.

## ColibriBackend

ColibriBackend reserves the local Heavy-runtime boundary needed by later Oracle integration. In v0.5.3 it reports unavailable and raises an explicit backend-unavailable error for lifecycle or inference calls. It does not claim live Colibri support.

```text
tars backend list
tars backend status llama.cpp
tars backend status colibri
```

External cloud inference providers, compatible remote inference endpoints, cloud fallback and provider-specific inference authentication are outside the v1.0 runtime architecture.
