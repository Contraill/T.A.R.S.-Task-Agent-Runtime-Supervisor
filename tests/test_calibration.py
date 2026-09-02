import json
from copy import deepcopy

import pytest

from tars import calibration


SERVER_SHA = "1" * 64
BENCH_SHA = "2" * 64


def _seed():
    return {
        "schema": 1,
        "model_alias": "fixture",
        "model_sha256": "a" * 64,
        "status": "ready",
        "source": "trusted-seed",
        "profiles": {"normal": {"context": 32768}},
    }


def _legacy(seed):
    return {
        **deepcopy(seed),
        "fingerprint": {
            "captured_at": "2026-08-29T12:54:06+00:00",
            "llama_cpp": "version: fixture",
        },
    }


def _configure(monkeypatch, tmp_path, seed, identity=None):
    monkeypatch.setattr(calibration, "MODEL_CALIBRATION_ROOT", tmp_path)
    monkeypatch.setattr(calibration, "ensure_registry", lambda: {"models": {}})
    monkeypatch.setattr(calibration, "seed_payloads", lambda registry: {"fixture": seed})
    monkeypatch.setattr(
        calibration,
        "capture_fingerprint",
        lambda: identity or {
            "captured_at": "2026-09-02T10:00:00+00:00",
            "llama_server_sha256": SERVER_SHA,
            "llama_bench_sha256": BENCH_SHA,
        },
    )


def test_exact_legacy_seed_is_atomically_bound_to_runtime_artifacts(
        monkeypatch, tmp_path):
    seed = _seed()
    _configure(monkeypatch, tmp_path, seed)
    path = tmp_path / f"{seed['model_sha256']}.json"
    path.write_text(json.dumps(_legacy(seed)), encoding="utf-8")

    calibration.ensure_seed_calibrations()

    upgraded = json.loads(path.read_text(encoding="utf-8"))
    assert upgraded["fingerprint"]["captured_at"] == "2026-08-29T12:54:06+00:00"
    assert upgraded["fingerprint"]["llama_cpp"] == "version: fixture"
    assert upgraded["fingerprint"]["llama_server_sha256"] == SERVER_SHA
    assert upgraded["fingerprint"]["llama_bench_sha256"] == BENCH_SHA
    assert upgraded["fingerprint"]["runtime_identity_bound_at"] == (
        "2026-09-02T10:00:00+00:00")


def test_modified_legacy_calibration_is_not_adopted_as_trusted_seed(
        monkeypatch, tmp_path):
    seed = _seed()
    _configure(monkeypatch, tmp_path, seed)
    path = tmp_path / f"{seed['model_sha256']}.json"
    modified = _legacy(seed)
    modified["profiles"]["normal"]["context"] = 65536
    path.write_text(json.dumps(modified), encoding="utf-8")

    calibration.ensure_seed_calibrations()

    assert json.loads(path.read_text(encoding="utf-8")) == modified


def test_bound_seed_does_not_rehash_runtime_artifacts(monkeypatch, tmp_path):
    seed = _seed()
    _configure(monkeypatch, tmp_path, seed)
    path = tmp_path / f"{seed['model_sha256']}.json"
    bound = _legacy(seed)
    bound["fingerprint"].update({
        "llama_server_sha256": SERVER_SHA,
        "llama_bench_sha256": BENCH_SHA,
    })
    path.write_text(json.dumps(bound), encoding="utf-8")
    monkeypatch.setattr(
        calibration, "capture_fingerprint",
        lambda: (_ for _ in ()).throw(AssertionError("runtime was rehashed")),
    )

    calibration.ensure_seed_calibrations()

    assert json.loads(path.read_text(encoding="utf-8")) == bound


def test_legacy_seed_remains_unchanged_when_runtime_identity_is_unavailable(
        monkeypatch, tmp_path):
    seed = _seed()
    _configure(
        monkeypatch, tmp_path, seed,
        identity={
            "captured_at": "2026-09-02T10:00:00+00:00",
            "llama_server_sha256": "",
            "llama_bench_sha256": BENCH_SHA,
        },
    )
    path = tmp_path / f"{seed['model_sha256']}.json"
    legacy = _legacy(seed)
    path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing runtime artifacts"):
        calibration.ensure_seed_calibrations()

    assert json.loads(path.read_text(encoding="utf-8")) == legacy
