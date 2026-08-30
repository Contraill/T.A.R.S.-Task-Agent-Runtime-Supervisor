# Delegation and subagents

A delegation creates a normal canonical child task plus a durable child contract. The contract bounds model/tool budget, tool names, filesystem and effect scope, network and remote targets, secret references, workspace sharing and completion evidence. Nested contracts may only narrow their parent contract.

Local inference children share one GPU slot by default. Tool-only children may run concurrently. Shared writable workspaces use an exclusive scheduler lock; isolated and read-only work may proceed independently. Cancellation and timeout are cooperative: the child receives a cancellation event and remains running until its current operation reaches a truthful boundary.

Completed child work is not automatically accepted. The parent verifies the configured evidence contract and explicitly accepts or rejects the result. Memory proposed by a child stays in the existing MemoryManager candidate flow and cannot be promoted before parent acceptance.

Use `tars task child-create`, `child-show`, `child-cancel` and `child-accept` to inspect and control durable contracts. Root child creation requires explicit parent authority and tool ceilings; skills and model text cannot supply authorization.
