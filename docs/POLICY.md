# Execution Policy

ScopeGuard evaluates tool intent before execution. Model output cannot grant permission or override a decision. Effects cover filesystem access, process execution, network access, services, remote targets, secrets, elevated operations, destructive operations and sandbox escape.

Filesystem targets are canonicalized before evaluation. A filesystem tool requires an authorized path scope or an explicit persistent path rule; traversal and symlink escape are denied. Network targets permit only HTTP(S), reject embedded credentials and non-public address classes, and may be restricted to an explicit destination set. Execution tools must repeat destination validation when connecting so DNS changes cannot bypass policy.

Default policy permits scoped reads; asks for writes, execution, public network access, service changes, remote access, secret use, destructive operations and sandbox escape; and denies elevation. Sandbox escape therefore cannot execute without an approval or explicit user rule. Explicit rules may narrow or override these defaults.

Approval scopes are one call, current task, current session, a specific target, or an explicit persistent user rule. One-call approvals are atomically consumed. Ordinary approvals do not silently create permanent configuration.

The action journal records normalized redacted arguments, target, effect, risk, policy decision and approval reference before execution. Completion records contain the real terminal state and redacted result. Denied and failed actions remain denied or failed; model text is not execution evidence.

```text
tars scope explain TOOL EFFECT [TARGET] [--allow-path PATH] [--allow-host HOST]
tars scope rules
tars scope rule-add EFFECT ACTION [TARGET] [--path]
tars approvals [--state STATE]
tars approvals --approve APPROVAL_ID [--reason TEXT]
tars approvals --deny APPROVAL_ID [--reason TEXT]
tars audit [ACTION_ID] [--task TASK_ID] [--state STATE]
```
