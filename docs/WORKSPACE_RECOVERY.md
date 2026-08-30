# Workspace recovery

Workspace checkpoints preserve bounded local filesystem state. They do not claim to reverse remote APIs, sent messages, network mutations, service effects or arbitrary remote-system changes.

Git checkpoints require a repository with a resolvable HEAD. They retain separate binary patches for the index and worktree plus bounded untracked regular files. Rollback is supported only while HEAD is unchanged, so recovery never moves a branch or rewrites history. New untracked files are moved into checkpoint quarantine. A safety checkpoint of the current state is created before restore.

Non-Git checkpoints copy only explicitly selected regular files under an authorized root. Symlinks are rejected. File-count and byte limits bound snapshot size. Rollback restores only captured files and verifies their hashes; unrelated files are untouched.

Rollback always requires destructive authorization. Preview reports the supported operations and states that external effects are not reversible.

```text
tars workspace checkpoint REPOSITORY
tars workspace checkpoint ROOT --path FILE [--path FILE]
tars workspace list
tars workspace preview CHECKPOINT_ID
tars workspace rollback CHECKPOINT_ID --approval APPROVAL_ID
```
