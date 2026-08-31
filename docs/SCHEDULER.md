# Durable scheduler

The scheduler is a lightweight control-plane component over canonical tasks. A schedule references one task; it does not create a second task or permission model. Supported kinds are one-shot timestamps, recurring intervals such as `every 10m`, and named model-free condition watches such as `network-online@every 1m`.

Each due occurrence receives a stable idempotency key and an append-preserved run-journal record. Missed occurrences can be skipped, collapsed to one run, or caught up to a configured bound. Global and keyed concurrency ceilings prevent unbounded replay. Paused and archived schedules retain their completed run journal.

Claimed work captures the latest immutable task checkpoint. A scheduler restart reclaims interrupted runs under the same idempotency key and increments their attempt count. Executors receive that checkpoint reference and the canonical task remains the source of continuation truth.

Successful runs create durable delivery state. Delivery adapters can retry failed delivery independently without rerunning inference or changing the run result.

Condition callbacks are explicit lightweight predicates. CLI schedules use named predicates from `scheduler.conditions`; supported built-ins inspect path existence or canonical task state without inference. An unavailable predicate is rejected before execution and is never replaced with model polling. Watches fire on a false-to-true edge and sleep until their next check.

`tars schedule run-due` performs recovery, validates condition availability, claims due work and invokes the existing task runner. When nothing is due, schedule listing, health checks and waiting do not contact a runtime or load a model. The event-driven service loop waits until the next timestamp or an explicit wake event. Schedule add, edit and resume operations notify a local Unix datagram endpoint, including from another process, so earlier work interrupts the current wait. Already-claimed recovery backlog always produces an immediate wake.

Useful commands:

```text
tars schedule list
tars schedule status
tars schedule add TASK one-shot 2026-09-01T09:00:00Z
tars schedule add TASK recurring "every 30m" --missed run-once
tars schedule pause SCHEDULE
tars schedule resume SCHEDULE
tars schedule edit SCHEDULE --expression "every 2h"
tars schedule runs [SCHEDULE]
tars schedule run-due
tars schedule remove SCHEDULE
```
