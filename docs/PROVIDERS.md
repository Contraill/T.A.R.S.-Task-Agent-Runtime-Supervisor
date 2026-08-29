# Model providers

A Role binds to a model record; the model record points at a RuntimeProvider. This keeps provider choice out of task ownership and memory.

Planned provider classes:

| Provider | Status | Authentication | Local calibration |
| --- | --- | --- | --- |
| llama.cpp / llama-swap | reference | local process | yes |
| Colibri | planned | local process/API | yes, provider-specific |
| OpenAI API | planned experimental | API key / Bearer | no |
| OpenAI-compatible HTTP | planned experimental | endpoint-specific | no |
| OAuth-backed provider | planned adapter mechanism | provider-specific OAuth | no |

OpenAI's public API uses API keys. OAuth should therefore be treated as a generic provider authentication mechanism rather than mislabeled as the normal OpenAI API auth path.

External credentials must not be stored in the portable bundle as plaintext. Provider config stores a secret reference; the Core resolves the secret from an environment variable, OS key store, secret service or another supported secure source.

## Support labels

A provider should report one of:

- `reference-tested`
- `tested`
- `experimental`
- `community-reported`

Protocol unit tests can run with mocks. A live service is not called "tested" unless it has actually been exercised against that service/version.
