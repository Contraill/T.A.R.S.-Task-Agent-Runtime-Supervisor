from pathlib import Path
from types import SimpleNamespace

import pytest

from tars import calibration_engine as engine


class FakeProbe:
    def __init__(self, digest="hardware-a", *, idle=True):
        self.digest = digest
        self.idle = idle
        self.calls = []

    def fingerprint(self):
        return {"digest": self.digest, "cpu_affinity": list(range(8)), "gpus": []}

    def measure(self, model_path, candidate, *, pressure=0.0, fit=False):
        self.calls.append((candidate, pressure, fit))
        speed = 1000.0 / max(1, candidate.threads)
        return engine.Measurement(True, speed, 40.0 + candidate.threads,
                                  1024, 2048, placement={"n_gpu_layers": candidate.ngl})

    def zero_idle(self):
        return {"passed": self.idle, "llama_server_processes": [],
                "nvidia_runtime_status": ["suspended"]}


def _model(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"model")
    return SimpleNamespace(alias="model", path=path, sha256="abc", backend="llama.cpp",
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
    existing = {"status": "ready", "depth": "max", "model_sha256": "abc",
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
