from __future__ import annotations

from dataclasses import dataclass
import os
import select
import signal
import subprocess
import sys

from .ownership import process_start


PARENT_DEATH_WRAPPER = r"""
import ctypes
import os
import signal
import subprocess
import sys
import threading

expected_parent = int(sys.argv[1])
ready_fd = int(sys.argv[2])
control_fd = int(sys.argv[3])
response_fd = int(sys.argv[4])
libc = ctypes.CDLL(None, use_errno=True)
owner_death_signal = signal.SIGRTMIN
signal.pthread_sigmask(signal.SIG_BLOCK, {owner_death_signal})
if libc.prctl(1, owner_death_signal) != 0:
    raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
if libc.prctl(36, 1) != 0:
    raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER) failed")
child_pgid = 0
child_identity = ""
def process_identity(pid):
    try:
        value = open(f"/proc/{pid}/stat", encoding="utf-8").read()
        return value[value.rfind(")") + 2:].split()[19]
    except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError):
        return ""
def descendants():
    children = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        entries = ()
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            value = open(f"/proc/{entry}/stat", encoding="utf-8").read()
            fields = value[value.rfind(")") + 2:].split()
            parent = int(fields[1])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
            continue
        children.setdefault(parent, []).append((int(entry), fields[19]))
    found = {}
    pending = [os.getpid()]
    seen = {os.getpid()}
    while pending:
        parent = pending.pop()
        for child, identity in children.get(parent, ()):
            if child not in seen:
                seen.add(child)
                found[child] = identity
                pending.append(child)
    return found
def parent_died(_signum=None, _frame=None):
    signal.signal(owner_death_signal, signal.SIG_IGN)
    owned = {}
    # Stop the whole observed tree before killing it so descendants cannot fork
    # between enumeration and cleanup. Repeat to catch reparenting to the subreaper.
    for _ in range(4):
        current = descendants()
        owned.update(current)
        for pid, identity in current.items():
            if process_identity(pid) != identity:
                continue
            try:
                os.kill(pid, signal.SIGSTOP)
            except ProcessLookupError:
                pass
    if child_pgid and process_identity(child_pgid) == child_identity:
        try:
            os.killpg(child_pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for pid, identity in owned.items():
        if process_identity(pid) != identity:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    os._exit(128 + owner_death_signal)
signal.signal(owner_death_signal, parent_died)
def user_signal(_signum=None, _frame=None):
    pass
for user_signal_number in (
        signal.SIGTERM, signal.SIGINT, signal.SIGHUP,
        signal.SIGUSR1, signal.SIGUSR2):
    signal.signal(user_signal_number, user_signal)
if os.getppid() != expected_parent:
    parent_died()
extra_fds = tuple(int(value) for value in sys.argv[5].split(",") if value)
child = subprocess.Popen(sys.argv[6:], start_new_session=True, pass_fds=extra_fds)
child_pgid = child.pid
def child_start():
    return process_identity(child.pid)
child_identity = child_start()
def control_process():
    allowed = {
        "TERM": signal.SIGTERM, "INT": signal.SIGINT, "HUP": signal.SIGHUP,
        "USR1": signal.SIGUSR1, "USR2": signal.SIGUSR2,
        "STOP": signal.SIGSTOP, "CONT": signal.SIGCONT, "KILL": signal.SIGKILL,
    }
    with os.fdopen(control_fd, "rb", closefd=True) as requests:
        with os.fdopen(response_fd, "wb", closefd=True) as responses:
            for line in requests:
                name = line.rstrip(b"\r\n").decode("ascii", errors="replace")
                number = allowed.get(name)
                if number is None:
                    responses.write(b"invalid\n")
                    responses.flush()
                    continue
                try:
                    os.killpg(child_pgid, number)
                except ProcessLookupError:
                    outcome = b"already-exited\n"
                else:
                    outcome = b"dispatched\n"
                responses.write(outcome)
                responses.flush()
threading.Thread(target=control_process, daemon=True).start()
signal.pthread_sigmask(signal.SIG_UNBLOCK, {owner_death_signal})
os.write(ready_fd, (str(child_pgid) + ":" + child_identity + "\n").encode("ascii"))
os.close(ready_fd)
returncode = child.wait()
while True:
    try:
        os.wait()
    except ChildProcessError:
        break
raise SystemExit(returncode if returncode >= 0 else 128 - returncode)
"""

EXEC_GATE_WRAPPER = r"""
import os
import sys

gate_fd = int(sys.argv[1])
try:
    released = os.read(gate_fd, 1)
finally:
    os.close(gate_fd)
if released != b"1":
    raise SystemExit(125)
os.execvp(sys.argv[2], sys.argv[2:])
"""


def _close_fd(descriptor):
    if descriptor is None or descriptor < 0:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


@dataclass
class SupervisedProcess:
    process: subprocess.Popen
    child_pid: int
    child_start: str
    supervisor_start: str
    control_write: int
    response_read: int
    gate_write: int = -1

    def release_start_gate(self):
        if self.gate_write < 0:
            return
        descriptor = self.gate_write
        self.gate_write = -1
        try:
            os.write(descriptor, b"1")
        finally:
            _close_fd(descriptor)

    def close_control(self):
        _close_fd(self.control_write)
        _close_fd(self.response_read)
        _close_fd(self.gate_write)
        self.control_write = -1
        self.response_read = -1
        self.gate_write = -1

    def stop(self, *, timeout=5.0):
        if self.process.poll() is None:
            current = process_start(self.process.pid)
            if current and current != self.supervisor_start:
                raise PermissionError("supervisor process identity changed")
            try:
                os.kill(self.process.pid, signal.SIGRTMIN)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if process_start(self.child_pid) == self.child_start:
                    try:
                        os.killpg(self.child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if process_start(self.process.pid) == self.supervisor_start:
                    try:
                        os.kill(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                self.process.wait(timeout=timeout)
        return self.process.returncode


def spawn_supervised(argv, *, cwd=None, env=None, popen=subprocess.Popen,
                     startup_timeout=5.0, start_gated=False, inherited_fds=(),
                     **popen_kwargs):
    extra_fds = tuple(int(value) for value in inherited_fds)
    if any(value < 0 for value in extra_fds):
        raise ValueError("inherited process descriptors must be open")
    ready_read, ready_write = os.pipe()
    control_read, control_write = os.pipe()
    response_read, response_write = os.pipe()
    gate_read, gate_write = (-1, -1)
    command = list(map(str, argv))
    if start_gated:
        gate_read, gate_write = os.pipe()
        extra_fds = (*extra_fds, gate_read)
        command = [
            sys.executable, "-c", EXEC_GATE_WRAPPER, str(gate_read), *command,
        ]
    process = None
    managed_argv = [
        sys.executable, "-c", PARENT_DEATH_WRAPPER,
        str(os.getpid()), str(ready_write), str(control_read), str(response_write),
        ",".join(map(str, extra_fds)), *command,
    ]
    try:
        process = popen(
            managed_argv, cwd=cwd, env=env, start_new_session=True,
            pass_fds=(ready_write, control_read, response_write, *extra_fds),
            **popen_kwargs,
        )
        _close_fd(ready_write)
        ready_write = -1
        _close_fd(control_read)
        control_read = -1
        _close_fd(response_write)
        response_write = -1
        _close_fd(gate_read)
        gate_read = -1
        readable, _, _ = select.select((ready_read,), (), (), startup_timeout)
        ready_payload = os.read(ready_read, 128) if readable else b""
        try:
            pid_text, child_start = ready_payload.strip().decode("ascii").split(":", 1)
            child_pid = int(pid_text)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise RuntimeError("process supervisor did not confirm child startup") from exc
        supervisor_start = process_start(process.pid)
        if child_pid <= 1 or not child_start or not supervisor_start:
            raise RuntimeError("process supervisor returned invalid child identity")
        return SupervisedProcess(
            process, child_pid, child_start, supervisor_start,
            control_write, response_read, gate_write,
        )
    except Exception:
        if process is not None and process.poll() is None:
            try:
                os.kill(process.pid, signal.SIGRTMIN)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=startup_timeout)
            except subprocess.TimeoutExpired:
                if process_start(process.pid):
                    try:
                        os.kill(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=startup_timeout)
        _close_fd(control_write)
        _close_fd(response_read)
        _close_fd(gate_write)
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is None:
                    continue
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        raise
    finally:
        _close_fd(ready_read)
        _close_fd(ready_write)
        _close_fd(control_read)
        _close_fd(response_write)
        _close_fd(gate_read)
