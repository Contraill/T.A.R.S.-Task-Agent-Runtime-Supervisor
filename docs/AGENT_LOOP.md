# Agent loop and live control

The agent loop builds on canonical task, conversation, context, policy, tool, evidence and checkpoint state. A model emits a structured decision; it does not execute tools or authorize itself. Registered tools must return a real `ToolResult`. Exceptions and invalid return values remain failures and are never promoted into successful execution records.

```text
context → model decision → ScopeGuard/approval → tool → ToolResult
        → evidence → pending controls → task state → checkpoint → continue
```

Completion uses an explicit contract. Required tool results and EvidenceRecord types must exist and belong to the task. A model completion statement without the required evidence is rejected. Repetition, no-progress, elapsed-time, iteration, tool-failure, context-pressure and unsafe-retry guards pause or fail bounded work.

Task controls are stored durably and ordered per task. Boundary priority is cancel, immediate interrupt/pause, approval response, redirect/resume, queued message, then ordinary continuation. Messages of equal priority retain submission order. Redirect becomes the canonical current task instruction. A reconnect does not remove pending controls.

During a tool call, ordinary messages receive:

```text
Message queued for submission after the next tool call.
```

Esc in the TUI submits an immediate interrupt. A registered cancellation callback records its actual response. Without a safe cancellation callback, the interrupt remains pending, the current tool reaches a truthful boundary, its real result is recorded, and no old-plan action follows before the interrupt is applied.

```text
tars task control TASK_ID message "new instruction"
tars task control TASK_ID redirect "do not modify the database"
tars task control TASK_ID interrupt "stop and inspect"
tars task controls TASK_ID
```
