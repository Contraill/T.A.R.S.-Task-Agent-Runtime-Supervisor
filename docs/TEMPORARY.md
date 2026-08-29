# Temporary sessions

Temporary mode is an ephemeral T.A.R.S. session. It is intended for work that may use normal reasoning and tools but must not become part of T.A.R.S.'s durable personal state.

Planned entry points:

```text
tars temporary
/temporary
```

## Persistence contract

While a temporary session is active, T.A.R.S. may keep the current conversation, task state, intermediate results and tool trace in memory so the session remains coherent. That state is never promoted to the normal persistent stores.

Temporary mode must not write temporary-session data to:

- conversation or message history
- durable tasks, task events or task runs
- checkpoints
- user/project memory or the user model
- retrieval indexes
- scheduled/future tasks
- persistent reasoning, progress or tool trace history

Closing temporary mode discards the entire ephemeral session. A crash also discards it; temporary sessions are intentionally not recoverable.

## Existing context

`tars temporary` starts a new ephemeral session and may read already-persisted identity, preferences and memory that the user has allowed T.A.R.S. to use. Those persistent sources are read-only for the lifetime of the temporary session.

Entering `/temporary` from a normal conversation creates an isolated ephemeral branch from a read-only snapshot of the current context. Leaving temporary mode returns to the normal conversation at the pre-temporary point. Temporary messages do not appear in normal history afterwards.

## Tools

Tools remain available subject to the same ScopeGuard and permission policy as normal operation. T.A.R.S. does not persist its own temporary tool/activity trace after the session ends.

Temporary mode is not an undo mechanism or operating-system privacy sandbox. External side effects remain real. A tool may create or edit files, make Git commits, change a service, contact an API, or cause logs to be written by another program or remote system. T.A.R.S. cannot erase those effects merely because its own session is temporary.

## Scheduling and delegation

Immediate ephemeral delegation is allowed once the agent loop supports it. Anything that must survive the session boundary is not allowed. In particular, a temporary session cannot create a future/cron task because there would be no durable owner state after the temporary session is destroyed.

## UI

Temporary state should always be conspicuous in the CLI/TUI, for example with a pinned `TEMPORARY` indicator. Exiting temporary mode should explicitly confirm that the ephemeral state was discarded.
