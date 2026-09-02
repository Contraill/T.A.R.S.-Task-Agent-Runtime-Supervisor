import multiprocessing
import hashlib
import os
from pathlib import Path
import signal
import stat
import time
from types import SimpleNamespace

import pytest

from tars import calibration_engine as engine
from tars import ownership, runtime, state_store


def _hold_model_execution_slot(database, scratch, ready, release):
    state_store.STATE_DB_PATH = Path(database)
    state_store.TASK_ROOT = Path(scratch) / "legacy"
    state_store.TASK_EVENTS_ROOT = Path(scratch) / "events"
    state_store.TASK_INDEX_PATH = Path(scratch) / "index"
    with ownership.model_execution_scope(operation="inference-test-holder"):
        ready.set()
        release.wait(10)


def _run_fenced_benchmark(database, scratch, bench, model, pid_file):
    state_store.STATE_DB_PATH = Path(database)
    state_store.TASK_ROOT = Path(scratch) / "legacy"
    state_store.TASK_EVENTS_ROOT = Path(scratch) / "events"
    state_store.TASK_INDEX_PATH = Path(scratch) / "index"
    engine.LLAMA_BENCH_PATH = Path(bench)
    os.environ["TARS_TEST_BENCH_PIDS"] = str(pid_file)
    available = sorted(os.sched_getaffinity(0))
    candidate = engine.Candidate(context=512, cpus=str(available[0]))
    with ownership.model_execution_scope(operation="calibration-crash-test"):
        engine.LlamaBenchProbe().measure(Path(model), candidate)


class _BlockingCalibrationProbe:
    def __init__(self, ready, release):
        self.ready = ready
        self.release = release

    def fingerprint(self):
        assert ownership.model_execution_owner() is not None
        self.ready.set()
        assert self.release.wait(10)
        return {"digest": "blocked-calibration", "cpu_affinity": [0], "gpus": []}

    def measure(self, model_path, candidate, *, pressure=0.0, fit=False):
        return engine.Measurement(
            True, 10.0, 10.0, placement={"n_gpu_layers": candidate.ngl})

    def zero_idle(self):
        return {"passed": True, "llama_server_processes": [],
                "nvidia_runtime_status": []}


def _run_blocked_calibration(database, scratch, model_path, ready, release):
    state_store.STATE_DB_PATH = Path(database)
    state_store.TASK_ROOT = Path(scratch) / "legacy"
    state_store.TASK_EVENTS_ROOT = Path(scratch) / "events"
    state_store.TASK_INDEX_PATH = Path(scratch) / "index"
    model = SimpleNamespace(
        alias="fixture", path=Path(model_path),
        sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        backend="llama.cpp", native_context=32768,
    )
    engine.get_model = lambda alias: model
    engine.load_calibration = lambda alias: (_ for _ in ()).throw(FileNotFoundError())
    engine.calibration_path = lambda digest: Path(scratch) / "calibration.json"
    engine.CalibrationEngine(
        _BlockingCalibrationProbe(ready, release),
        root=Path(scratch) / "calibration-cache",
    ).calibrate("fixture")


def _process_parent(pid):
    value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return int(value[value.rfind(")") + 2:].split()[1])


def _wait_process_gone(pid, start, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ownership.process_start(pid) != start:
            return True
        time.sleep(0.02)
    return ownership.process_start(pid) != start


def _sleeping_bench(path):
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import time\n"
        "while True:\n"
        "    time.sleep(1)\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture(autouse=True)
def isolated_calibration_state(monkeypatch, tmp_path):
    monkeypatch.setattr(state_store, "STATE_DB_PATH", tmp_path / "state.sqlite3")
    monkeypatch.setattr(state_store, "TASK_ROOT", tmp_path / "legacy")
    monkeypatch.setattr(state_store, "TASK_EVENTS_ROOT", tmp_path / "events")
    monkeypatch.setattr(state_store, "TASK_INDEX_PATH", tmp_path / "index")


class FakeProbe:
    def __init__(self, digest="hardware-a", *, idle=True):
        self.digest = digest
        self.idle = idle
        self.calls = []
        self.fingerprints = 0

    @staticmethod
    def _assert_owned():
        owner = ownership.model_execution_owner()
        assert owner is not None
        assert ownership.held_by("gpu-slot", "local-inference:0", owner)

    def fingerprint(self):
        self._assert_owned()
        self.fingerprints += 1
        return {"digest": self.digest, "cpu_affinity": list(range(8)), "gpus": []}

    def measure(self, model_path, candidate, *, pressure=0.0, fit=False):
        self._assert_owned()
        self.calls.append((candidate, pressure, fit))
        speed = 1000.0 / max(1, candidate.threads)
        return engine.Measurement(True, speed, 40.0 + candidate.threads,
                                  1024, 2048, placement={"n_gpu_layers": candidate.ngl})

    def zero_idle(self):
        self._assert_owned()
        return {"passed": self.idle, "llama_server_processes": [],
                "nvidia_runtime_status": ["suspended"]}


def _model(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"model")
    return SimpleNamespace(
        alias="model", path=path, sha256=hashlib.sha256(b"model").hexdigest(),
        backend="llama.cpp",
                           native_context=131072)


def _setup(monkeypatch, tmp_path, probe, existing=None):
    model = _model(tmp_path)
    monkeypatch.setattr(engine, "get_model", lambda alias: model)
    if existing is None:
        monkeypatch.setattr(engine, "load_calibration", lambda alias: (_ for _ in ()).throw(FileNotFoundError()))
    else:
        monkeypatch.setattr(engine, "load_calibration", lambda alias: existing)
    output = tmp_path / "final.json"
    monkeypatch.setattr(engine, "calibration_path", lambda digest: output)
    return engine.CalibrationEngine(probe, root=tmp_path / "cache"), output


def test_min_calibration_resumes_stage_cache(monkeypatch, tmp_path):
    probe = FakeProbe()
    calibration, output = _setup(monkeypatch, tmp_path, probe)
    first = calibration.calibrate("model")
    call_count = len(probe.calls)
    second = calibration.calibrate("model")
    assert len(probe.calls) == call_count
    assert first["profiles"].keys() == {"compact", "normal", "extended", "reasonable_max"}
    assert second["stage_cache_events"]
    assert output.exists()


def test_fresh_reruns_cached_stages(monkeypatch, tmp_path):
    probe = FakeProbe()
    calibration, _ = _setup(monkeypatch, tmp_path, probe)
    calibration.calibrate("model")
    count = len(probe.calls)
    calibration.calibrate("model", fresh=True)
    assert len(probe.calls) > count


def test_fingerprint_change_uses_new_stage_cache(monkeypatch, tmp_path):
    probe = FakeProbe("hardware-a")
    calibration, _ = _setup(monkeypatch, tmp_path, probe)
    calibration.calibrate("model")
    count = len(probe.calls)
    probe.digest = "hardware-b"
    calibration.calibrate("model")
    assert len(probe.calls) > count


def test_mid_and_max_depth_run_adaptive_stages(monkeypatch, tmp_path):
    probe = FakeProbe()
    calibration, _ = _setup(monkeypatch, tmp_path, probe)
    mid = calibration.calibrate("model", depth="mid")
    maximum = calibration.calibrate("model", depth="max", fresh=True)
    assert mid["depth"] == "mid"
    assert maximum["depth"] == "max"
    assert maximum["profiles"]["extended"]["metrics"]["pressure"]


def test_higher_depth_result_is_protected(monkeypatch, tmp_path):
    existing = {"status": "ready", "depth": "max",
                "model_sha256": hashlib.sha256(b"model").hexdigest(),
                "fingerprint_digest": "hardware-a",
                "profiles": {}, "reasonable_max_context": 131072}
    probe = FakeProbe()
    calibration, _ = _setup(monkeypatch, tmp_path, probe, existing=existing)
    result = calibration.calibrate("model", depth="mid", fresh=True)
    assert result["protected_from_shallower"]
    assert not probe.calls


def test_zero_idle_failure_prevents_promotion(monkeypatch, tmp_path):
    probe = FakeProbe(idle=False)
    calibration, output = _setup(monkeypatch, tmp_path, probe)
    with pytest.raises(RuntimeError, match="Zero-Idle"):
        calibration.calibrate("model")
    assert not output.exists()


def test_stale_stage_cache_is_detected(monkeypatch, tmp_path):
    probe = FakeProbe()
    calibration, _ = _setup(monkeypatch, tmp_path, probe)
    result = calibration.calibrate("model")
    stage = Path(result["stage_cache"]) / "fit.json"
    data = __import__("json").loads(stage.read_text())
    data["input_digest"] = "stale"
    stage.write_text(__import__("json").dumps(data))
    result = calibration.calibrate("model")
    assert {"stage": "fit", "status": "stale"} in result["stage_cache_events"]


def test_calibration_cannot_touch_probe_while_inference_owns_slot(
        monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "CALIBRATION_SLOT_WAIT_SECONDS", 0.1)
    probe = FakeProbe()
    calibration, _ = _setup(monkeypatch, tmp_path, probe)
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_hold_model_execution_slot,
        args=(str(state_store.STATE_DB_PATH), str(tmp_path), ready, release),
    )
    process.start()
    assert ready.wait(5)
    try:
        with pytest.raises(RuntimeError, match="slot is busy"):
            calibration.calibrate("model", fresh=True)
        assert probe.fingerprints == 0
        assert probe.calls == []
    finally:
        release.set()
        process.join(timeout=10)
    assert process.exitcode == 0


def test_inference_cannot_prepare_while_calibration_owns_slot(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "INFERENCE_SLOT_WAIT_SECONDS", 0.1)
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"fixture-not-model-weights")
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(
        target=_run_blocked_calibration,
        args=(str(state_store.STATE_DB_PATH), str(tmp_path), str(model), ready, release),
    )
    process.start()
    assert ready.wait(5)
    route = SimpleNamespace(backend="llama.cpp", runtime_id="fixture")
    router = SimpleNamespace(
        prepare=lambda value: (_ for _ in ()).throw(
            AssertionError("inference prepared while calibration owned the slot")),
        release=lambda value: None,
    )
    try:
        with pytest.raises(RuntimeError, match="slot is busy"):
            with runtime._inference_lifecycle(router, route):
                pass
    finally:
        release.set()
        process.join(timeout=10)
    assert process.exitcode == 0


def test_direct_llama_bench_probe_requires_authoritative_owner(tmp_path):
    candidate = engine.Candidate(context=512, cpus="0")
    with pytest.raises(RuntimeError, match="requires model execution ownership"):
        engine.LlamaBenchProbe().measure(tmp_path / "fixture.gguf", candidate)


def test_llama_bench_executes_verified_descriptors_across_path_replacement(
        monkeypatch, tmp_path):
    bench = tmp_path / "llama-bench"
    bench.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "model = pathlib.Path(sys.argv[sys.argv.index('-m') + 1]).read_bytes()\n"
        "if model != b'verified-model':\n"
        "    raise SystemExit(9)\n"
        "print(json.dumps([{'n_prompt': 512, 'n_gen': 64, 'avg_ts': 10.0}]))\n",
        encoding="utf-8",
    )
    bench.chmod(bench.stat().st_mode | stat.S_IXUSR)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"verified-model")
    monkeypatch.setattr(engine, "LLAMA_BENCH_PATH", bench)
    swapped = False

    def replace_paths_once():
        nonlocal swapped
        if not swapped:
            replacement_bench = tmp_path / "replacement-bench"
            replacement_bench.write_text(
                "#!/usr/bin/env python3\nraise SystemExit(8)\n",
                encoding="utf-8",
            )
            replacement_bench.chmod(
                replacement_bench.stat().st_mode | stat.S_IXUSR)
            replacement_model = tmp_path / "replacement-model"
            replacement_model.write_bytes(b"replacement-model")
            replacement_bench.replace(bench)
            replacement_model.replace(model)
            swapped = True
        return []

    monkeypatch.setattr(engine, "_gpu_snapshot", replace_paths_once)
    cpus = str(min(os.sched_getaffinity(0)))
    with ownership.model_execution_scope(operation="descriptor-race-test"):
        result = engine.LlamaBenchProbe().measure(
            model, engine.Candidate(context=512, cpus=cpus))

    assert result.success
    assert swapped


def test_llama_bench_timeout_kills_supervised_process_and_releases_fence(
        monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "LLAMA_BENCH_PATH", _sleeping_bench(tmp_path / "bench"))
    monkeypatch.setattr(engine, "CALIBRATION_PROCESS_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(engine, "CALIBRATION_POLL_SECONDS", 0.01)
    observed = []

    def snapshot():
        metadata = ownership.active_metadata(*ownership.MODEL_EXECUTION_RESOURCE)
        if metadata:
            observed.extend(metadata.get("fence_processes", ()))
        return []

    monkeypatch.setattr(engine, "_gpu_snapshot", snapshot)
    cpus = str(min(os.sched_getaffinity(0)))
    model_path = tmp_path / "fixture.gguf"
    model_path.write_bytes(b"fixture-model")
    with ownership.model_execution_scope(operation="calibration-timeout-test"):
        result = engine.LlamaBenchProbe().measure(
            model_path, engine.Candidate(context=512, cpus=cpus))
        assert not result.success and "timed out" in result.error
        assert ownership.active_metadata(
            *ownership.MODEL_EXECUTION_RESOURCE).get("fence_processes") == []
    assert observed
    assert all(ownership.process_start(item["pid"]) != item["start"]
               for item in observed)
    assert not ownership.active(*ownership.MODEL_EXECUTION_RESOURCE)


def test_llama_bench_interrupt_kills_process_before_ownership_release(
        monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "LLAMA_BENCH_PATH", _sleeping_bench(tmp_path / "bench"))
    calls = 0
    observed = []

    def interrupting_snapshot():
        nonlocal calls
        calls += 1
        if calls > 1:
            metadata = ownership.active_metadata(*ownership.MODEL_EXECUTION_RESOURCE)
            observed.extend(metadata.get("fence_processes", ()))
            raise KeyboardInterrupt
        return []

    monkeypatch.setattr(engine, "_gpu_snapshot", interrupting_snapshot)
    cpus = str(min(os.sched_getaffinity(0)))
    model_path = tmp_path / "fixture.gguf"
    model_path.write_bytes(b"fixture-model")
    with pytest.raises(KeyboardInterrupt):
        with ownership.model_execution_scope(operation="calibration-interrupt-test"):
            engine.LlamaBenchProbe().measure(
                model_path, engine.Candidate(context=512, cpus=cpus))
    assert observed
    assert all(ownership.process_start(item["pid"]) != item["start"]
               for item in observed)
    assert not ownership.active(*ownership.MODEL_EXECUTION_RESOURCE)


def test_crashed_calibration_remains_fenced_until_benchmark_tree_is_gone(
        monkeypatch, tmp_path):
    bench = tmp_path / "fake-llama-bench"
    bench.write_text(
        "#!/usr/bin/env python3\n"
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        "start_new_session=True)\n"
        "with open(os.environ['TARS_TEST_BENCH_PIDS'], 'w', encoding='utf-8') as handle:\n"
        "    handle.write(f'{os.getpid()} {child.pid}')\n"
        "    handle.flush()\n"
        "while True:\n"
        "    time.sleep(1)\n",
        encoding="utf-8",
    )
    bench.chmod(bench.stat().st_mode | stat.S_IXUSR)
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"fixture-not-model-weights")
    pid_file = tmp_path / "bench-pids"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_run_fenced_benchmark,
        args=(str(state_store.STATE_DB_PATH), str(tmp_path), str(bench),
              str(model), str(pid_file)),
    )
    process.start()

    deadline = time.monotonic() + 10
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pid_file.exists()
    business_pid, descendant_pid = map(int, pid_file.read_text().split())
    supervisor_pid = _process_parent(business_pid)
    identities = {
        pid: ownership.process_start(pid)
        for pid in (supervisor_pid, business_pid, descendant_pid)
    }
    assert all(identities.values())

    os.kill(supervisor_pid, signal.SIGSTOP)
    try:
        process.kill()
        process.join(timeout=10)
        assert process.exitcode is not None
        assert ownership.process_start(business_pid) == identities[business_pid]
        with pytest.raises(RuntimeError, match="slot is busy"):
            with ownership.model_execution_scope(
                    operation="must-remain-fenced", timeout=0.1):
                pass
    finally:
        try:
            os.kill(supervisor_pid, signal.SIGCONT)
        except ProcessLookupError:
            pass

    assert _wait_process_gone(supervisor_pid, identities[supervisor_pid])
    assert _wait_process_gone(business_pid, identities[business_pid])
    assert _wait_process_gone(descendant_pid, identities[descendant_pid])
    with ownership.model_execution_scope(
            operation="post-cleanup-reclaim", timeout=1):
        pass
