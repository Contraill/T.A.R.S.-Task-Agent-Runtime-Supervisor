# Context engine

Context is a disposable projection of canonical identity, memory, project, task, evidence, tool and conversation state. The active Role's calibrated runtime profile supplies the context limit; exact builds use the target llama.cpp tokenizer.

Pressure is classified at soft, hard and emergency watermarks. Cheap history pruning occurs before exact tokenization. Durable task truth, the latest user instruction, pending controls and unresolved ToolResults are protected from ordinary pruning.

At hard pressure, Context Epoch rollover atomically:

```text
checkpoint canonical task state
archive the eligible transcript slice
remove that slice from active context
preserve protected instructions and unresolved work
advance the task epoch
```

Archived transcript is derived context data. Canonical messages, immutable checkpoints and task events remain available after rollover. Exact lexical transcript search covers both active and archived turns.

```text
tars context show [--exact]
tars context epochs TASK_ID
tars context search CONVERSATION_ID "query"
```
