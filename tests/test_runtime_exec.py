import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from tars import runtime_exec


class _Executed(Exception):
    pass


def test_verified_launcher_executes_real_descriptor_paths(tmp_path):
    server_path = tmp_path / "llama-server"
    model_path = tmp_path / "model.gguf"
    server_path.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "model = sys.argv[sys.argv.index('-m') + 1]\n"
        "print(Path(model).read_text(encoding='utf-8'))\n"
        "print(' '.join(sys.argv[sys.argv.index(model) + 1:]))\n",
        encoding="utf-8",
    )
    server_path.chmod(server_path.stat().st_mode | stat.S_IXUSR)
    model_path.write_text("verified model", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable, "-m", "tars.runtime_exec",
            "--server", str(server_path),
            "--server-sha256", hashlib.sha256(server_path.read_bytes()).hexdigest(),
            "--model", str(model_path),
            "--model-sha256", hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "--", "-c", "4096",
        ],
        capture_output=True, text=True, check=False, timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["verified model", "-c 4096"]


def test_verified_launcher_executes_exact_open_server_and_model_files(
        monkeypatch, tmp_path):
    server_path = tmp_path / "llama-server"
    model_path = tmp_path / "model.gguf"
    server_path.write_bytes(b"verified-server")
    model_path.write_bytes(b"verified-model")
    server_sha = hashlib.sha256(b"verified-server").hexdigest()
    model_sha = hashlib.sha256(b"verified-model").hexdigest()
    observed = {}

    def execve(path, argv, env):
        replacement_server = tmp_path / "replacement-server"
        replacement_model = tmp_path / "replacement-model"
        replacement_server.write_bytes(b"replacement-server")
        replacement_model.write_bytes(b"replacement-model")
        replacement_server.replace(server_path)
        replacement_model.replace(model_path)
        model_ref = argv[argv.index("-m") + 1]
        observed["server"] = Path(path).read_bytes()
        observed["model"] = Path(model_ref).read_bytes()
        observed["server_inherited"] = os.get_inheritable(
            int(path.rsplit("/", 1)[1]))
        observed["model_inherited"] = os.get_inheritable(
            int(model_ref.rsplit("/", 1)[1]))
        observed["trailing"] = argv[3:]
        raise _Executed

    monkeypatch.setattr(runtime_exec.os, "execve", execve)
    with pytest.raises(_Executed):
        runtime_exec.main([
            "--server", str(server_path),
            "--server-sha256", server_sha,
            "--model", str(model_path),
            "--model-sha256", model_sha,
            "--", "-c", "4096", "--port", "10001",
        ])

    assert observed == {
        "server": b"verified-server",
        "model": b"verified-model",
        "server_inherited": True,
        "model_inherited": True,
        "trailing": ["-c", "4096", "--port", "10001"],
    }


@pytest.mark.parametrize("target", ["server", "model"])
def test_verified_launcher_refuses_changed_artifact_before_exec(
        monkeypatch, tmp_path, target):
    server_path = tmp_path / "llama-server"
    model_path = tmp_path / "model.gguf"
    server_path.write_bytes(b"server")
    model_path.write_bytes(b"model")
    server_sha = hashlib.sha256(b"server").hexdigest()
    model_sha = hashlib.sha256(b"model").hexdigest()
    (server_path if target == "server" else model_path).write_bytes(b"changed")
    monkeypatch.setattr(
        runtime_exec.os, "execve",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("unverified artifact was executed")),
    )

    with pytest.raises(RuntimeError, match="no longer match"):
        runtime_exec.main([
            "--server", str(server_path),
            "--server-sha256", server_sha,
            "--model", str(model_path),
            "--model-sha256", model_sha,
        ])
