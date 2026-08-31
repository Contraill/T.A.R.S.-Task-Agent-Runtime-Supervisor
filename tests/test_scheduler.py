from datetime import datetime, timedelta, timezone
import threading
import time

import pytest

from tars import checkpoints, cli, scheduler, state_store, tasks


@pytest.fixture
def scheduled_state(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy-tasks")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "legacy-events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "legacy-index.json")
    monkeypatch.setattr(tasks, "resolve_role_id", lambda value: value)
    return tmp_path


def utc(hour=12):
    return datetime(2026, 8, 31, hour, tzinfo=timezone.utc)


def make_task():
    return tasks.create_task("scheduled work", "general", make_active=False)


def test_one_shot_claim_is_deduplicated_and_model_free(scheduled_state):
    task = make_task()
    item = scheduler.create_schedule(
        task.id, "one-shot", "2026-08-31T12:00:00Z", now=utc(11))
    engine = scheduler.Scheduler()
    assert engine.claim_due(now=utc(11)) == []
    claimed = engine.claim_due(now=utc())
    assert len(claimed) == 1 and claimed[0].schedule_id == item.id
    assert engine.claim_due(now=utc()) == []
    assert scheduler.load_schedule(item.id).enabled is False
    assert tasks.load_task(task.id).schedule_enabled is False


def test_recurring_missed_policies_and_bounded_catch_up(scheduled_state):
    skipped_task = make_task()
    skipped = scheduler.create_schedule(
        skipped_task.id, "recurring", "every 10m", next_run_at=utc(10),
        missed_policy="skip", now=utc(9))
    assert scheduler.Scheduler().claim_due(now=utc(12)) == []
    assert scheduler.list_runs(skipped.id)[0].state == "skipped"
    assert scheduler.load_schedule(skipped.id).next_run_at == "2026-08-31T12:10:00Z"

    catch_task = make_task()
    catch = scheduler.create_schedule(
        catch_task.id, "recurring", "every 30m", next_run_at=utc(10),
        missed_policy="catch-up", max_catch_up=3, max_concurrency=3, now=utc(9))
    claimed = scheduler.Scheduler(max_concurrency=3).claim_due(now=utc(12))
    assert len(claimed) == 3
    assert len({run.idempotency_key for run in claimed}) == 3
    assert scheduler.load_schedule(catch.id).next_run_at == "2026-08-31T11:30:00Z"


def test_condition_watch_is_edge_triggered_without_inference(scheduled_state):
    task = make_task()
    state = {"ready": False}
    item = scheduler.create_schedule(
        task.id, "condition", "ready@every 1m", next_run_at=utc(), now=utc(11))
    engine = scheduler.Scheduler(conditions={"ready": lambda: state["ready"]})
    assert engine.claim_due(now=utc()) == []
    state["ready"] = True
    assert len(engine.claim_due(now=utc() + timedelta(minutes=1))) == 1
    engine.execute_claimed(lambda run: {"observed": True})
    assert engine.claim_due(now=utc() + timedelta(minutes=2)) == []
    state["ready"] = False
    assert engine.claim_due(now=utc() + timedelta(minutes=3)) == []
    state["ready"] = True
    assert len(engine.claim_due(now=utc() + timedelta(minutes=4))) == 1
    assert item.expression.startswith("ready@")


def test_checkpoint_recovery_delivery_and_failure_truth(scheduled_state):
    task = make_task()
    checkpoint = checkpoints.create_checkpoint(task.id, {"step": 2}, reason="resume")
    item = scheduler.create_schedule(
        task.id, "recurring", "every 1h", next_run_at=utc(),
        delivery_target="local:conversation", now=utc(11))
    engine = scheduler.Scheduler()
    claimed = engine.claim_due(now=utc())[0]
    assert claimed.checkpoint_id == checkpoint.id
    with state_store.transaction(immediate=True) as conn:
        conn.execute("UPDATE schedule_runs SET state='running' WHERE id=?", (claimed.id,))
    assert engine.recover() == 1
    recovered = scheduler.load_run(claimed.id)
    assert recovered.state == "claimed" and recovered.attempt == 2
    seen = []
    result = engine.execute_claimed(
        lambda run: {"resumed_from": run.checkpoint_id},
        deliver=lambda target, run: seen.append((target, run.id)) or {"ok": True})[0]
    assert result.state == "succeeded" and result.result["resumed_from"] == checkpoint.id
    assert seen == [("local:conversation", claimed.id)]

    failure_task = make_task()
    failure_schedule = scheduler.create_schedule(
        failure_task.id, "one-shot", utc().isoformat(), now=utc(11))
    engine.claim_due(now=utc())
    failed = engine.execute_claimed(
        lambda run: (_ for _ in ()).throw(RuntimeError("real failure")))[0]
    assert failed.state == "failed" and failed.error == "real failure"
    assert tasks.load_task(failure_task.id).last_result_status == "failed"
    assert failure_schedule.id == failed.schedule_id


def test_pause_edit_remove_preserves_run_journal(scheduled_state):
    task = make_task()
    item = scheduler.create_schedule(
        task.id, "recurring", "every 1h", next_run_at=utc(), now=utc(11))
    assert not scheduler.set_enabled(item.id, False).enabled
    edited = scheduler.edit_schedule(item.id, expression="every 2h",
                                     next_run_at=utc(14), missed_policy="catch-up")
    assert edited.expression == "every 2h" and edited.revision == 3
    scheduler.set_enabled(item.id, True)
    engine = scheduler.Scheduler()
    run = engine.claim_due(now=utc(14))[0]
    engine.execute_claimed(lambda value: {"ok": True})
    scheduler.remove_schedule(item.id)
    assert scheduler.schedule_for_task(task.id) is None
    assert scheduler.load_schedule(item.id).removed_at
    assert scheduler.load_run(run.id).state == "succeeded"
    with pytest.raises(RuntimeError, match="removed"):
        scheduler.set_enabled(item.id, True)


def test_wait_calculation_has_no_model_or_busy_polling(scheduled_state):
    task = make_task()
    scheduler.create_schedule(
        task.id, "one-shot", "2026-08-31T12:30:00Z", now=utc())
    engine = scheduler.Scheduler()
    assert engine.next_wake_seconds(now=utc()) == 1800
    assert scheduler.RUNNING_STATES == {"claimed", "running"}
    report = scheduler.health()
    assert report["ok"] and report["model_residency_required_while_waiting"] is False


def test_health_reports_unavailable_condition_without_running_it(scheduled_state):
    task = make_task()
    scheduler.create_schedule(
        task.id, "condition", "network-online@every 1m", next_run_at=utc(), now=utc(11))
    report = scheduler.health()
    assert not report["ok"] and report["missing_conditions"] == ["network-online"]


def test_configured_condition_registry_is_truthful_and_model_free(scheduled_state):
    marker = scheduled_state / "ready"
    cfg = {"scheduler": {"conditions": {
        "workspace-ready": {"type": "path-exists", "path": str(marker)},
    }}}
    registry = scheduler.condition_registry(cfg)
    assert not registry["workspace-ready"]()
    marker.write_text("ready")
    assert registry["workspace-ready"]()
    with pytest.raises(ValueError, match="unsupported type"):
        scheduler.condition_registry({"scheduler": {"conditions": {
            "bad": {"type": "model-prompt"}}}})


def test_unavailable_condition_is_rejected_before_execution(scheduled_state):
    task = make_task()
    scheduler.create_schedule(
        task.id, "condition", "missing@every 1m", next_run_at=utc(), now=utc(11))
    with pytest.raises(RuntimeError, match="unavailable schedule conditions: missing"):
        scheduler.require_condition_support({})
    with pytest.raises(RuntimeError, match="unavailable schedule conditions: missing"):
        cli.command_schedule_run_due({})


def test_cli_condition_add_requires_registered_predicate(scheduled_state):
    task = make_task()
    args = type("Args", (), {
        "task_id": task.id, "kind": "condition",
        "expression": "ready@every 1m", "next": None,
        "missed": "run-once", "max_catch_up": 1,
        "concurrency_key": "default", "max_concurrency": 1,
        "delivery_target": "",
    })()
    with pytest.raises(ValueError, match="not configured"):
        cli.command_schedule_add({}, args)
    marker = scheduled_state / "ready"
    created = cli.command_schedule_add({"scheduler": {"conditions": {
        "ready": {"type": "path-exists", "path": str(marker)}}}}, args)
    assert created is None and scheduler.schedule_for_task(task.id).kind == "condition"


def test_claimed_backlog_wakes_immediately(scheduled_state):
    task = make_task()
    scheduler.create_schedule(task.id, "one-shot", utc().isoformat(), now=utc(11))
    engine = scheduler.Scheduler()
    assert engine.claim_due(now=utc())
    assert engine.next_wake_seconds(now=utc()) == 0.0


def test_external_schedule_change_wakes_long_lived_scheduler(scheduled_state):
    engine = scheduler.Scheduler()
    stopped = threading.Event()
    executed = []

    def execute(run):
        executed.append(run.id)
        stopped.set()
        engine.wake()
        return {"ok": True}

    thread = threading.Thread(
        target=engine.run_forever, args=(execute,), kwargs={"stop": stopped}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while not scheduler._wake_path().exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert scheduler._wake_path().exists()
    task = make_task()
    scheduler.create_schedule(
        task.id, "one-shot", datetime.now(timezone.utc).isoformat())
    thread.join(timeout=2)
    assert executed and not thread.is_alive()
