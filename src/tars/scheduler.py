from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import select
import socket
import threading
import uuid
from typing import Callable, Mapping

from .checkpoints import latest_checkpoint
from . import state_store as _state_store
from .state_store import connect, ensure_state_store, json_dumps, json_loads, now_utc, transaction
from .tasks import append_event, load_task


SCHEDULE_KINDS = {"one-shot", "recurring", "condition"}
MISSED_POLICIES = {"skip", "run-once", "catch-up"}
RUNNING_STATES = {"claimed", "running"}
_INTERVAL = re.compile(r"^(?:every\s+|interval:)(\d+)([smhd]?)$", re.I)


def _wake_path() -> Path:
    return _state_store.STATE_DB_PATH.with_name("scheduler-wake.sock")


def notify_scheduler() -> bool:
    """Wake a live scheduler through a write-only local datagram."""
    path = _wake_path()
    if not path.exists():
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.sendto(b"schedule-changed", str(path))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def condition_registry(cfg: Mapping | None) -> dict[str, Callable[[], bool]]:
    """Build explicitly configured, model-free condition predicates."""
    scheduler_cfg = (cfg or {}).get("scheduler", {})
    definitions = scheduler_cfg.get("conditions", {}) if isinstance(scheduler_cfg, Mapping) else {}
    if not isinstance(definitions, Mapping):
        raise ValueError("scheduler.conditions must be a table")
    result = {}
    for name, definition in definitions.items():
        if not isinstance(definition, Mapping):
            raise ValueError(f"condition {name} must be a table")
        kind = str(definition.get("type", ""))
        if kind == "path-exists":
            path = Path(str(definition.get("path", ""))).expanduser()
            if not str(definition.get("path", "")).strip():
                raise ValueError(f"condition {name} requires path")
            result[str(name)] = lambda path=path: path.exists()
        elif kind == "task-state":
            task_id = str(definition.get("task_id", "")).strip()
            expected = str(definition.get("state", "completed")).strip()
            if not task_id:
                raise ValueError(f"condition {name} requires task_id")
            result[str(name)] = lambda task_id=task_id, expected=expected: (
                load_task(task_id).state == expected)
        else:
            raise ValueError(f"condition {name} has unsupported type: {kind or '(missing)'}")
    return result


def require_condition_support(conditions: Mapping[str, Callable[[], bool]]) -> None:
    missing = sorted({
        item.expression.split("@", 1)[0].strip()
        for item in list_schedules(enabled=True) if item.kind == "condition"
        and item.expression.split("@", 1)[0].strip() not in conditions
    })
    if missing:
        raise RuntimeError("unavailable schedule conditions: " + ", ".join(missing))


def _dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: str | datetime) -> str:
    return _dt(value).isoformat().replace("+00:00", "Z")


def interval_seconds(expression: str) -> int:
    match = _INTERVAL.fullmatch(str(expression).strip())
    if not match:
        raise ValueError("recurring expression must be 'every N[s|m|h|d]' or 'interval:N[s|m|h|d]'")
    amount = int(match.group(1))
    if amount < 1:
        raise ValueError("schedule interval must be positive")
    return amount * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2).lower()]


@dataclass(frozen=True)
class Schedule:
    id: str
    task_id: str
    kind: str
    expression: str
    timezone: str
    next_run_at: str | None
    missed_policy: str
    max_catch_up: int
    enabled: bool
    concurrency_key: str
    max_concurrency: int
    delivery_target: str
    condition_state: bool
    revision: int
    created_at: str
    updated_at: str
    removed_at: str | None


@dataclass(frozen=True)
class ScheduleRun:
    id: str
    schedule_id: str
    task_id: str
    planned_for: str
    idempotency_key: str
    state: str
    attempt: int
    checkpoint_id: str | None
    result: dict
    error: str
    claimed_at: str
    started_at: str | None
    finished_at: str | None


def _schedule(row) -> Schedule:
    return Schedule(
        row["id"], row["task_id"], row["kind"], row["expression"], row["timezone"],
        row["next_run_at"], row["missed_policy"], row["max_catch_up"], bool(row["enabled"]),
        row["concurrency_key"], row["max_concurrency"], row["delivery_target"],
        bool(row["condition_state"]), row["revision"], row["created_at"], row["updated_at"],
        row["removed_at"])


def _run(row) -> ScheduleRun:
    return ScheduleRun(
        row["id"], row["schedule_id"], row["task_id"], row["planned_for"],
        row["idempotency_key"], row["state"], row["attempt"], row["checkpoint_id"],
        json_loads(row["result_json"], {}), row["error"], row["claimed_at"],
        row["started_at"], row["finished_at"])


def _initial_next(kind: str, expression: str, next_run_at: str | None, now: datetime) -> str:
    if next_run_at:
        return _iso(next_run_at)
    if kind == "one-shot":
        return _iso(expression)
    if kind == "recurring":
        return _iso(now + timedelta(seconds=interval_seconds(expression)))
    interval_seconds(expression.split("@", 1)[1] if "@" in expression else "every 60s")
    poll = expression.split("@", 1)[1] if "@" in expression else "every 60s"
    return _iso(now + timedelta(seconds=interval_seconds(poll)))


def create_schedule(task_id: str, kind: str, expression: str, *, next_run_at=None,
                    missed_policy="run-once", max_catch_up=1, timezone_name="UTC",
                    concurrency_key="default", max_concurrency=1,
                    delivery_target="", now=None) -> Schedule:
    ensure_state_store()
    task = load_task(task_id)
    if task.state in {"completed", "cancelled"}:
        raise RuntimeError(f"cannot schedule {task.state} task {task.id}")
    if kind not in SCHEDULE_KINDS:
        raise ValueError(f"invalid schedule kind: {kind}")
    if missed_policy not in MISSED_POLICIES:
        raise ValueError(f"invalid missed-run policy: {missed_policy}")
    if timezone_name != "UTC":
        raise ValueError("v0.8.0 schedule storage is UTC; convert local times before registration")
    current = _dt(now or now_utc())
    due = _initial_next(kind, expression, next_run_at, current)
    schedule_id = "sch-" + uuid.uuid4().hex
    stamp = _iso(current)
    with transaction(immediate=True) as conn:
        conn.execute(
            """INSERT INTO schedules(
               id,task_id,kind,expression,timezone,next_run_at,missed_policy,max_catch_up,
               enabled,concurrency_key,max_concurrency,delivery_target,condition_state,
               revision,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,1,?,?,?,?,1,?,?)""",
            (schedule_id, task.id, kind, expression, timezone_name, due, missed_policy,
             int(max_catch_up), concurrency_key, int(max_concurrency), delivery_target,
             0, stamp, stamp),
        )
        conn.execute(
            "UPDATE tasks SET schedule_kind=?, schedule_expr=?, "
            "next_run_at=?, schedule_enabled=1, updated_at=? WHERE id=?",
            (kind, expression, due, stamp, task.id),
        )
    append_event(task.id, "status", f"Schedule {schedule_id} registered", role=task.owner_role,
                 data={"schedule_id": schedule_id, "kind": kind, "next_run_at": due})
    notify_scheduler()
    return load_schedule(schedule_id)


def load_schedule(schedule_id: str) -> Schedule:
    ensure_state_store()
    with connect() as conn:
        row = conn.execute("SELECT * FROM schedules WHERE id=?", (schedule_id,)).fetchone()
    if not row:
        raise KeyError(f"unknown schedule: {schedule_id}")
    return _schedule(row)


def schedule_for_task(task_id: str) -> Schedule | None:
    ensure_state_store()
    with connect() as conn:
        row = conn.execute("SELECT * FROM schedules WHERE task_id=? AND removed_at IS NULL", (task_id,)).fetchone()
    return _schedule(row) if row else None


def list_schedules(*, enabled=None, limit=100) -> list[Schedule]:
    ensure_state_store()
    sql = "SELECT * FROM schedules WHERE removed_at IS NULL"
    params = []
    if enabled is not None:
        sql += " AND enabled=?"
        params.append(1 if enabled else 0)
    sql += " ORDER BY COALESCE(next_run_at,'9999') LIMIT ?"
    params.append(int(limit))
    with connect() as conn:
        return [_schedule(row) for row in conn.execute(sql, params).fetchall()]


def edit_schedule(schedule_id: str, *, expression=None, next_run_at=None,
                  missed_policy=None, max_catch_up=None, concurrency_key=None,
                  max_concurrency=None, delivery_target=None, now=None) -> Schedule:
    schedule = load_schedule(schedule_id)
    values = {}
    if expression is not None:
        _initial_next(schedule.kind, expression, next_run_at, _dt(now or now_utc()))
        values["expression"] = expression
    if next_run_at is not None:
        values["next_run_at"] = _iso(next_run_at)
    if missed_policy is not None:
        if missed_policy not in MISSED_POLICIES:
            raise ValueError(f"invalid missed-run policy: {missed_policy}")
        values["missed_policy"] = missed_policy
    if max_catch_up is not None:
        values["max_catch_up"] = int(max_catch_up)
    if concurrency_key is not None:
        values["concurrency_key"] = concurrency_key
    if max_concurrency is not None:
        values["max_concurrency"] = int(max_concurrency)
    if delivery_target is not None:
        values["delivery_target"] = delivery_target
    if not values:
        return schedule
    values["revision"] = schedule.revision + 1
    values["updated_at"] = _iso(now or now_utc())
    with transaction(immediate=True) as conn:
        conn.execute("UPDATE schedules SET " + ",".join(f"{key}=?" for key in values) + " WHERE id=?",
                     (*values.values(), schedule_id))
        updated = conn.execute("SELECT * FROM schedules WHERE id=?", (schedule_id,)).fetchone()
        conn.execute("UPDATE tasks SET schedule_expr=?,next_run_at=?,updated_at=? WHERE id=?",
                     (updated["expression"], updated["next_run_at"], values["updated_at"], schedule.task_id))
    notify_scheduler()
    return load_schedule(schedule_id)


def set_enabled(schedule_id: str, enabled: bool) -> Schedule:
    schedule = load_schedule(schedule_id)
    if schedule.removed_at:
        raise RuntimeError("removed schedule cannot be resumed")
    stamp = now_utc()
    with transaction(immediate=True) as conn:
        conn.execute("UPDATE schedules SET enabled=?,revision=revision+1,updated_at=? WHERE id=?",
                     (1 if enabled else 0, stamp, schedule_id))
        conn.execute("UPDATE tasks SET schedule_enabled=?,updated_at=? WHERE id=?",
                     (1 if enabled else 0, stamp, schedule.task_id))
    notify_scheduler()
    return load_schedule(schedule_id)


def remove_schedule(schedule_id: str) -> None:
    schedule = load_schedule(schedule_id)
    with transaction(immediate=True) as conn:
        active = conn.execute("SELECT COUNT(*) FROM schedule_runs WHERE schedule_id=? AND state IN ('claimed','running')",
                              (schedule_id,)).fetchone()[0]
        if active:
            raise RuntimeError("cannot remove a schedule with an active run")
        stamp = now_utc()
        conn.execute("UPDATE schedules SET enabled=0,removed_at=?,updated_at=? WHERE id=?",
                     (stamp, stamp, schedule_id))
        conn.execute("UPDATE tasks SET schedule_kind=NULL,schedule_expr=NULL,next_run_at=NULL,schedule_enabled=0,updated_at=? WHERE id=?",
                     (stamp, schedule.task_id))


def _advance(schedule: Schedule, planned: datetime, now: datetime) -> str | None:
    if schedule.kind == "one-shot":
        return None
    expression = schedule.expression
    if schedule.kind == "condition":
        expression = expression.split("@", 1)[1] if "@" in expression else "every 60s"
        return _iso(now + timedelta(seconds=interval_seconds(expression)))
    seconds = interval_seconds(expression)
    candidate = planned + timedelta(seconds=seconds)
    if schedule.missed_policy == "skip" and candidate <= now:
        jumps = int((now - candidate).total_seconds() // seconds) + 1
        candidate += timedelta(seconds=seconds * jumps)
    elif schedule.missed_policy == "run-once" and candidate <= now:
        candidate = now + timedelta(seconds=seconds)
    return _iso(candidate)


class Scheduler:
    """Model-free durable scheduler. Inference exists only inside the executor."""

    def __init__(self, *, max_concurrency=1, conditions: Mapping[str, Callable[[], bool]] | None = None):
        self.max_concurrency = max(1, int(max_concurrency))
        self.conditions = dict(conditions or {})
        self._wake_socket = None
        ensure_state_store()

    def _open_wake_socket(self):
        path = _wake_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        if path.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                probe.sendto(b"probe", str(path))
            except OSError:
                path.unlink(missing_ok=True)
            else:
                sock.close()
                raise RuntimeError(f"scheduler wake endpoint is already active: {path}")
            finally:
                probe.close()
        sock.bind(str(path))
        sock.setblocking(False)
        self._wake_socket = sock
        return sock

    def _close_wake_socket(self):
        if self._wake_socket is not None:
            self._wake_socket.close()
            self._wake_socket = None
        _wake_path().unlink(missing_ok=True)

    def recover(self) -> int:
        """Reclaim interrupted work with the same idempotency key and checkpoint."""
        stamp = now_utc()
        with transaction(immediate=True) as conn:
            rows = conn.execute("SELECT * FROM schedule_runs WHERE state='running'").fetchall()
            for row in rows:
                conn.execute("UPDATE schedule_runs SET state='claimed',attempt=attempt+1,error=?,claimed_at=?,started_at=NULL WHERE id=?",
                             ("recovered after scheduler restart", stamp, row["id"]))
        return len(rows)

    def _condition_due(self, schedule: Schedule) -> bool:
        name = schedule.expression.split("@", 1)[0].strip()
        predicate = self.conditions.get(name)
        if predicate is None:
            return False
        current = bool(predicate())
        due = current and not schedule.condition_state
        with transaction(immediate=True) as conn:
            conn.execute("UPDATE schedules SET condition_state=?,updated_at=? WHERE id=?",
                         (1 if current else 0, now_utc(), schedule.id))
        return due

    def claim_due(self, *, now=None) -> list[ScheduleRun]:
        current = _dt(now or now_utc())
        claimed = []
        for schedule in list_schedules(enabled=True):
            if len(claimed) >= self.max_concurrency or not schedule.next_run_at:
                break
            first = _dt(schedule.next_run_at)
            if first > current:
                continue
            if schedule.kind == "condition" and not self._condition_due(schedule):
                self._set_next(schedule, _advance(schedule, first, current))
                continue
            planned_slots = [first]
            if schedule.kind == "recurring" and schedule.missed_policy == "catch-up":
                step = timedelta(seconds=interval_seconds(schedule.expression))
                while (planned_slots[-1] + step <= current and
                       len(planned_slots) < schedule.max_catch_up):
                    planned_slots.append(planned_slots[-1] + step)
            for planned in planned_slots:
                if len(claimed) >= self.max_concurrency:
                    break
                checkpoint = latest_checkpoint(schedule.task_id)
                with transaction(immediate=True) as conn:
                    global_active = conn.execute("SELECT COUNT(*) FROM schedule_runs WHERE state IN ('claimed','running')").fetchone()[0]
                    keyed = conn.execute(
                        "SELECT COUNT(*) FROM schedule_runs r JOIN schedules s ON s.id=r.schedule_id "
                        "WHERE r.state IN ('claimed','running') AND s.concurrency_key=?",
                        (schedule.concurrency_key,)).fetchone()[0]
                    if global_active >= self.max_concurrency or keyed >= schedule.max_concurrency:
                        break
                    late = planned < current
                    state = "skipped" if late and schedule.missed_policy == "skip" else "claimed"
                    run_id = "sjr-" + uuid.uuid4().hex
                    key = f"{schedule.id}:{_iso(planned)}:r{schedule.revision}"
                    try:
                        conn.execute(
                            """INSERT INTO schedule_runs(id,schedule_id,task_id,planned_for,
                               idempotency_key,state,attempt,checkpoint_id,claimed_at,finished_at)
                               VALUES(?,?,?,?,?,?,1,?,?,?)""",
                            (run_id, schedule.id, schedule.task_id, _iso(planned), key, state,
                             checkpoint.id if checkpoint else None, _iso(current),
                             _iso(current) if state == "skipped" else None),
                        )
                    except Exception as exc:
                        if "UNIQUE" in str(exc):
                            continue
                        raise
                    next_run = _advance(schedule, planned, current)
                    conn.execute("UPDATE schedules SET next_run_at=?,enabled=?,updated_at=? WHERE id=?",
                                 (next_run, 0 if next_run is None else 1, _iso(current), schedule.id))
                    conn.execute("UPDATE tasks SET next_run_at=?,schedule_enabled=?,updated_at=? WHERE id=?",
                                 (next_run, 0 if next_run is None else 1, _iso(current), schedule.task_id))
                if state == "claimed":
                    claimed.append(load_run(run_id))
                else:
                    append_event(schedule.task_id, "status", "Missed scheduled run skipped",
                                 data={"schedule_id": schedule.id, "planned_for": _iso(planned)})
        return claimed

    @staticmethod
    def _set_next(schedule: Schedule, value: str | None) -> None:
        with transaction(immediate=True) as conn:
            conn.execute("UPDATE schedules SET next_run_at=?,updated_at=? WHERE id=?",
                         (value, now_utc(), schedule.id))
            conn.execute("UPDATE tasks SET next_run_at=?,updated_at=? WHERE id=?",
                         (value, now_utc(), schedule.task_id))

    def execute_claimed(self, executor: Callable[[ScheduleRun], dict], *, deliver=None) -> list[ScheduleRun]:
        with connect() as conn:
            rows = conn.execute("SELECT * FROM schedule_runs WHERE state='claimed' ORDER BY planned_for").fetchall()
        completed = []
        for row in rows[:self.max_concurrency]:
            run = _run(row)
            stamp = now_utc()
            with transaction(immediate=True) as conn:
                changed = conn.execute("UPDATE schedule_runs SET state='running',started_at=? WHERE id=? AND state='claimed'",
                                       (stamp, run.id)).rowcount
            if not changed:
                continue
            append_event(run.task_id, "status", f"Scheduled run {run.id} started",
                         data={"schedule_id": run.schedule_id, "idempotency_key": run.idempotency_key})
            try:
                result = executor(load_run(run.id)) or {}
                checkpoint = latest_checkpoint(run.task_id)
                self._finish(run.id, "succeeded", result=result,
                             checkpoint_id=checkpoint.id if checkpoint else None)
                self._queue_delivery(run.id)
                if deliver is not None:
                    self.deliver_pending(deliver, run_id=run.id)
            except Exception as exc:
                checkpoint = latest_checkpoint(run.task_id)
                self._finish(run.id, "failed", error=str(exc),
                             checkpoint_id=checkpoint.id if checkpoint else None)
            completed.append(load_run(run.id))
        return completed

    def _finish(self, run_id, state, *, result=None, error="", checkpoint_id=None):
        run = load_run(run_id)
        stamp = now_utc()
        with transaction(immediate=True) as conn:
            conn.execute("UPDATE schedule_runs SET state=?,result_json=?,error=?,checkpoint_id=?,finished_at=? WHERE id=?",
                         (state, json_dumps(result or {}), error, checkpoint_id, stamp, run_id))
            conn.execute("UPDATE tasks SET last_run_at=?,last_result_status=?,updated_at=? WHERE id=?",
                         (stamp, state, stamp, run.task_id))
        append_event(run.task_id, "result" if state == "succeeded" else "error",
                     f"Scheduled run {run_id} {state}",
                     data={"schedule_id": run.schedule_id, "checkpoint_id": checkpoint_id,
                           "error": error})

    def _queue_delivery(self, run_id):
        run = load_run(run_id)
        schedule = load_schedule(run.schedule_id)
        state = "pending" if schedule.delivery_target else "suppressed"
        with transaction(immediate=True) as conn:
            conn.execute("INSERT OR IGNORE INTO schedule_deliveries(id,run_id,target,state,updated_at) VALUES(?,?,?,?,?)",
                         ("sjd-" + uuid.uuid4().hex, run_id, schedule.delivery_target, state, now_utc()))

    def deliver_pending(self, deliver: Callable[[str, ScheduleRun], dict], *, run_id=None) -> int:
        ensure_state_store()
        sql = "SELECT * FROM schedule_deliveries WHERE state IN ('pending','failed')"
        params = []
        if run_id:
            sql += " AND run_id=?"
            params.append(run_id)
        with connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        delivered = 0
        for row in rows:
            run = load_run(row["run_id"])
            try:
                result = deliver(row["target"], run) or {}
                state, error = "delivered", ""
                delivered += 1
            except Exception as exc:
                result, state, error = {}, "failed", str(exc)
            with transaction(immediate=True) as conn:
                conn.execute("UPDATE schedule_deliveries SET state=?,attempt=attempt+1,result_json=?,error=?,updated_at=? WHERE id=?",
                             (state, json_dumps(result), error, now_utc(), row["id"]))
        return delivered

    def next_wake_seconds(self, *, now=None, maximum=3600.0) -> float:
        current = _dt(now or now_utc())
        with connect() as conn:
            if conn.execute("SELECT 1 FROM schedule_runs WHERE state='claimed' LIMIT 1").fetchone():
                return 0.0
        enabled = [item for item in list_schedules(enabled=True) if item.next_run_at]
        if not enabled:
            return maximum
        return max(0.0, min(maximum, min((_dt(item.next_run_at) - current).total_seconds()
                                         for item in enabled)))

    def run_forever(self, executor, *, deliver=None, stop: threading.Event | None = None):
        stop = stop or threading.Event()
        wake_socket = self._open_wake_socket()
        try:
            self.recover()
            require_condition_support(self.conditions)
            while not stop.is_set():
                self.claim_due()
                self.execute_claimed(executor, deliver=deliver)
                readable, _, _ = select.select(
                    [wake_socket], [], [], self.next_wake_seconds())
                if readable:
                    try:
                        while wake_socket.recv(4096):
                            pass
                    except BlockingIOError:
                        pass
        finally:
            self._close_wake_socket()

    def wake(self):
        notify_scheduler()


def load_run(run_id: str) -> ScheduleRun:
    ensure_state_store()
    with connect() as conn:
        row = conn.execute("SELECT * FROM schedule_runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        raise KeyError(f"unknown schedule run: {run_id}")
    return _run(row)


def list_runs(schedule_id=None, *, limit=100) -> list[ScheduleRun]:
    ensure_state_store()
    sql = "SELECT * FROM schedule_runs"
    params = []
    if schedule_id:
        sql += " WHERE schedule_id=?"
        params.append(schedule_id)
    sql += " ORDER BY claimed_at DESC LIMIT ?"
    params.append(int(limit))
    with connect() as conn:
        return [_run(row) for row in conn.execute(sql, params).fetchall()]


def health(conditions: Mapping[str, Callable[[], bool]] | None = None) -> dict:
    ensure_state_store()
    conditions = conditions or {}
    schedules = list_schedules()
    missing = sorted({
        item.expression.split("@", 1)[0].strip() for item in schedules
        if item.enabled and item.kind == "condition" and
        item.expression.split("@", 1)[0].strip() not in conditions
    })
    with connect() as conn:
        running = conn.execute("SELECT COUNT(*) FROM schedule_runs WHERE state='running'").fetchone()[0]
        claimed = conn.execute("SELECT COUNT(*) FROM schedule_runs WHERE state='claimed'").fetchone()[0]
        failed_delivery = conn.execute("SELECT COUNT(*) FROM schedule_deliveries WHERE state='failed'").fetchone()[0]
    return {
        "ok": not missing,
        "enabled": sum(1 for item in schedules if item.enabled),
        "paused": sum(1 for item in schedules if not item.enabled),
        "running": running,
        "claimed": claimed,
        "failed_deliveries": failed_delivery,
        "missing_conditions": missing,
        "model_residency_required_while_waiting": False,
    }
