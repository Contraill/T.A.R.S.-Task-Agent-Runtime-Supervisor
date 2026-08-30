# Memory

Durable memory is stored as human-readable Markdown under `~/.local/share/tars/memory/`. Entries carry structured identifiers, scope, provenance, timestamps, confidence, supersession, expiry and tags. Profile, project, episodic, reference and system memory remain separate corpus classes.

SQLite/FTS5 is a rebuildable query index, not the canonical representation. Baseline retrieval is fully local and combines lexical matching with scope, expiry, recency and deduplication. Each result reports its memory identifier, source and ranking signals. Embeddings are optional and are not required for normal recall.

Models may stage candidates but cannot write durable memory directly. Promotion validates content, applies scope, rejects exact duplicates and records provenance. Explicit user `remember` commands enter the corpus directly. Superseded entries remain inspectable but are excluded from normal recall.

Before deletion, the current document is copied to an immutable local revision archive. `memory doctor` validates the corpus and rebuilds the derived index.

```text
tars memory status
tars memory search "query" [--scope SCOPE]
tars memory inspect MEMORY_ID
tars memory review
tars memory remember "fact" [--kind profile] [--scope global]
tars memory forget MEMORY_ID
tars memory doctor
```
