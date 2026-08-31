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

## Local routing

`LocalRuntimeRouter` resolves an exact requested Role to its configured model alias, local backend, runtime ID and calibrated profile. A route is ready only when the Role is enabled and semantically capable, its local artifact/integrity/compatibility/calibration state is ready, the backend is available and healthy, requested model capabilities and modalities are verified, and the requested context fits the profile.

Unavailable routes are durable inspection records with concrete reasons. The router does not search for another Role, model or backend when an exact binding fails. A task cannot route through a different owner Role until an explicit handoff changes canonical ownership.

Lifecycle preparation calls the selected backend's on-demand load contract. Release calls its finite unload/TTL contract. Route inspection itself does not run inference or keep a model loaded.

```text
tars runtime route general --capability conversation
tars runtime route builder --task TASK --capability code --tools
```

## LlamaCppBackend

LlamaCppBackend is the reference-tested implementation. It uses the existing llama-swap endpoint and preserves generated configuration, calibrated Role bindings, atomic apply/rollback and Zero-Idle behavior. Model loading remains on demand; finite llama-swap TTL policy manages unload.

## ColibriBackend

ColibriBackend reserves the local Heavy-runtime boundary needed by later Oracle integration. In v0.5.3 it reports unavailable and raises an explicit backend-unavailable error for lifecycle or inference calls. It does not claim live Colibri support.

```text
tars backend list
tars backend status llama.cpp
tars backend status colibri
```

External cloud inference providers, compatible remote inference endpoints, cloud fallback and provider-specific inference authentication are outside the v1.0 runtime architecture and are rejected by local routing.
