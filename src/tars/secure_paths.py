from __future__ import annotations

import errno
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import secrets
import stat


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _components(parts, *, allow_empty=True):
    values = tuple(os.fspath(part) for part in parts)
    if not allow_empty and not values:
        raise PermissionError("the scope root itself is not a mutable child path")
    separators = tuple(value for value in (os.sep, os.altsep) if value)
    for value in values:
        if (not value or value in {".", ".."} or "\0" in value
                or any(separator in value for separator in separators)):
            raise PermissionError(f"unsafe path component: {value!r}")
    return values


def hash_fd(fd, *, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    size = 0
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


class AnchoredRoot:
    """Filesystem operations rooted at an already-open directory descriptor."""

    def __init__(self, root):
        self.fd = -1
        self.path = Path(os.path.abspath(os.fspath(root)))
        current = os.open(os.sep, _DIRECTORY_FLAGS)
        try:
            for component in self.path.parts[1:]:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
                os.close(current)
                current = child
            self.fd = current
        except Exception:
            os.close(current)
            raise

    @classmethod
    def from_fd(cls, fd, *, display=None):
        """Duplicate an already-open directory without resolving its pathname again."""
        instance = cls.__new__(cls)
        instance.fd = os.dup(fd)
        value = os.fstat(instance.fd)
        if not stat.S_ISDIR(value.st_mode):
            os.close(instance.fd)
            instance.fd = -1
            raise NotADirectoryError("anchored root descriptor is not a directory")
        shown = display if display is not None else f"/proc/self/fd/{fd}"
        instance.path = Path(os.path.abspath(os.fspath(shown)))
        return instance

    @classmethod
    def open_or_create(cls, root, *, mode=0o700):
        """Open a directory path without symlinks, durably creating missing levels."""
        absolute = Path(os.path.abspath(os.fspath(root)))
        current = os.open(os.sep, _DIRECTORY_FLAGS)
        try:
            for component in absolute.parts[1:]:
                try:
                    child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
                except FileNotFoundError:
                    created = False
                    try:
                        os.mkdir(component, mode, dir_fd=current)
                        created = True
                    except FileExistsError:
                        pass
                    child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
                    if created:
                        os.fsync(current)
                        os.fsync(child)
                os.close(current)
                current = child
            instance = cls.__new__(cls)
            instance.fd = current
            instance.path = absolute
            return instance
        except Exception:
            os.close(current)
            raise

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def relative(self, target) -> tuple[str, ...]:
        candidate = Path(os.path.abspath(os.fspath(target)))
        try:
            relative = candidate.relative_to(self.path)
        except ValueError as exc:
            raise PermissionError(f"path is outside allowed roots: {target}") from exc
        if any(part in ("", ".", "..") for part in relative.parts):
            raise PermissionError(f"unsafe path component: {target}")
        return relative.parts

    def parent(self, parts, *, create=False):
        parts = _components(parts, allow_empty=False)
        current = os.dup(self.fd)
        try:
            for component in parts[:-1]:
                try:
                    child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, 0o700, dir_fd=current)
                    child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
                os.close(current)
                current = child
            return current, parts[-1]
        except Exception:
            os.close(current)
            raise

    def open(self, parts, flags, mode=0o600, *, create_parents=False):
        parent, name = self.parent(parts, create=create_parents)
        try:
            return os.open(name, flags | os.O_CLOEXEC | os.O_NOFOLLOW, mode, dir_fd=parent)
        finally:
            os.close(parent)

    def open_directory(self, parts=()):
        parts = _components(parts)
        current = os.dup(self.fd)
        try:
            for component in parts:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
                os.close(current)
                current = child
            return current
        except Exception:
            os.close(current)
            raise

    def bind(self, parts):
        if not parts:
            fd = os.dup(self.fd)
            return BoundPath(fd, self.path, os.fstat(fd))
        flags = getattr(os, "O_PATH", os.O_RDONLY) | os.O_CLOEXEC | os.O_NOFOLLOW
        parent, name = self.parent(parts)
        try:
            fd = os.open(name, flags, dir_fd=parent)
        finally:
            os.close(parent)
        value = os.fstat(fd)
        if not (stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)):
            os.close(fd)
            raise ValueError("bound filesystem objects must be regular files or directories")
        return BoundPath(fd, self.path.joinpath(*parts), value)

    @contextmanager
    def reader(self, parts, *, text=False, encoding="utf-8", errors="strict"):
        fd = self.open(parts, os.O_RDONLY)
        try:
            mode = os.fstat(fd).st_mode
            if not stat.S_ISREG(mode):
                raise ValueError("filesystem input must be a regular file")
            if text:
                with os.fdopen(fd, "r", encoding=encoding, errors=errors) as handle:
                    fd = -1
                    yield handle
            else:
                with os.fdopen(fd, "rb") as handle:
                    fd = -1
                    yield handle
        finally:
            if fd >= 0:
                os.close(fd)

    @contextmanager
    def atomic_writer(self, parts, *, text=False, encoding="utf-8",
                      newline=None, require_existing=False, expected_identity=None):
        parent, name = self.parent(parts, create=not require_existing)
        temporary = f".{name}.tars-{secrets.token_hex(8)}"
        fd = -1
        committed = False
        try:
            if require_existing:
                check = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                                dir_fd=parent)
                try:
                    value = os.fstat(check)
                    if not stat.S_ISREG(value.st_mode):
                        raise ValueError("filesystem output must be a regular file")
                    if (expected_identity is not None and
                            (value.st_dev, value.st_ino) != tuple(expected_identity)):
                        raise RuntimeError("filesystem object changed during the operation")
                finally:
                    os.close(check)
            fd = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600, dir_fd=parent,
            )
            if text:
                with os.fdopen(fd, "w", encoding=encoding, newline=newline) as handle:
                    fd = -1
                    yield handle
                    handle.flush()
                    os.fsync(handle.fileno())
            else:
                with os.fdopen(fd, "wb") as handle:
                    fd = -1
                    yield handle
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
            committed = True
        finally:
            if fd >= 0:
                os.close(fd)
            if not committed:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
            os.close(parent)

    def atomic_write(self, parts, payload, *, require_existing=False,
                     expected_identity=None):
        with self.atomic_writer(
                parts, require_existing=require_existing,
                expected_identity=expected_identity) as handle:
            handle.write(payload)

    def read_bytes(self, parts, *, limit=None):
        with self.reader(parts) as handle:
            if limit is None:
                return handle.read()
            return handle.read(int(limit) + 1)

    def hash(self, parts, *, algorithm="sha256"):
        fd = self.open(parts, os.O_RDONLY)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("filesystem input must be a regular file")
            return hash_fd(fd, algorithm=algorithm)
        finally:
            os.close(fd)

    def copy_fd_to(self, source_fd, destination_parts):
        return self._atomic_copy(source_fd, destination_parts)

    def copy_to(self, source_parts, destination_root, destination_parts):
        source_fd = self.open(source_parts, os.O_RDONLY)
        try:
            mode = os.fstat(source_fd).st_mode
            if stat.S_ISREG(mode):
                destination_root._atomic_copy(source_fd, destination_parts)
            elif stat.S_ISDIR(mode):
                destination_root.makedirs(destination_parts)
                self._copy_directory(source_fd, destination_root, destination_parts)
            else:
                raise ValueError("fs.copy supports only regular files and directories")
        finally:
            os.close(source_fd)

    def _copy_directory(self, source_fd, destination_root, destination_parts):
        for name in os.listdir(source_fd):
            try:
                child = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                                dir_fd=source_fd)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise ValueError(
                        "fs.copy does not follow symbolic links or special files"
                    ) from exc
                raise
            try:
                mode = os.fstat(child).st_mode
                child_destination = destination_parts + (name,)
                if stat.S_ISREG(mode):
                    destination_root._atomic_copy(child, child_destination)
                elif stat.S_ISDIR(mode):
                    destination_root.makedirs(child_destination)
                    self._copy_directory(child, destination_root, child_destination)
                else:
                    raise ValueError("fs.copy does not follow symbolic links or special files")
            finally:
                os.close(child)

    def _atomic_copy(self, source_fd, parts):
        parent, name = self.parent(parts, create=True)
        temporary = f".{name}.tars-{secrets.token_hex(8)}"
        destination_fd = -1
        digest = hashlib.sha256()
        size = 0
        try:
            destination_fd = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600, dir_fd=parent,
            )
            os.lseek(source_fd, 0, os.SEEK_SET)
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                view = memoryview(chunk)
                while view:
                    view = view[os.write(destination_fd, view):]
            os.fsync(destination_fd)
            os.close(destination_fd)
            destination_fd = -1
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
            return digest.hexdigest(), size
        finally:
            if destination_fd >= 0:
                os.close(destination_fd)
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)

    def mkdir(self, parts, *, parents=False):
        parent, name = self.parent(parts, create=parents)
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
        finally:
            os.close(parent)

    def makedirs(self, parts):
        current = os.dup(self.fd)
        try:
            for component in parts:
                try:
                    child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
                except FileNotFoundError:
                    os.mkdir(component, 0o700, dir_fd=current)
                    child = os.open(component, _DIRECTORY_FLAGS, dir_fd=current)
                os.close(current)
                current = child
        finally:
            os.close(current)

    def rename(self, source_parts, destination_root, destination_parts):
        source_parent, source_name = self.parent(source_parts)
        destination_parent, destination_name = destination_root.parent(
            destination_parts, create=True
        )
        try:
            os.rename(source_name, destination_name, src_dir_fd=source_parent,
                      dst_dir_fd=destination_parent)
        finally:
            os.close(source_parent)
            os.close(destination_parent)

    def delete(self, parts, *, recursive=False):
        parent, name = self.parent(parts)
        try:
            try:
                child = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
            except OSError as exc:
                if exc.errno not in (errno.ENOTDIR, errno.ELOOP):
                    raise
                os.unlink(name, dir_fd=parent)
                return
            try:
                if recursive:
                    self._clear_directory(child)
                os.rmdir(name, dir_fd=parent)
            finally:
                os.close(child)
        finally:
            os.close(parent)

    def _clear_directory(self, directory_fd):
        for name in os.listdir(directory_fd):
            try:
                child = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            except OSError as exc:
                if exc.errno not in (errno.ENOTDIR, errno.ELOOP):
                    raise
                os.unlink(name, dir_fd=directory_fd)
                continue
            try:
                self._clear_directory(child)
                os.rmdir(name, dir_fd=directory_fd)
            finally:
                os.close(child)

    def stat(self, parts):
        fd = self.open(parts, os.O_RDONLY)
        try:
            return os.fstat(fd)
        finally:
            os.close(fd)

    def lstat(self, parts):
        parent, name = self.parent(parts)
        try:
            return os.stat(name, dir_fd=parent, follow_symlinks=False)
        finally:
            os.close(parent)

    def list(self, parts=()):
        directory = self.open_directory(parts)
        try:
            rows = []
            for name in sorted(os.listdir(directory)):
                value = os.stat(name, dir_fd=directory, follow_symlinks=False)
                rows.append((name, value))
            return rows
        finally:
            os.close(directory)

    def walk_files(self, parts=(), *, reject_symlinks=False):
        """Yield (relative parts, open fd) without following symbolic links."""
        directory = self.open_directory(parts)
        try:
            yield from self._walk_files(
                directory, tuple(parts), reject_symlinks=reject_symlinks
            )
        finally:
            os.close(directory)

    def _walk_files(self, directory, prefix, *, reject_symlinks):
        for name in sorted(os.listdir(directory)):
            try:
                child = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                                dir_fd=directory)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    if reject_symlinks:
                        raise ValueError("symbolic links are not allowed in this operation") from exc
                    continue
                raise
            mode = os.fstat(child).st_mode
            if stat.S_ISDIR(mode):
                try:
                    yield from self._walk_files(
                        child, prefix + (name,), reject_symlinks=reject_symlinks
                    )
                finally:
                    os.close(child)
            elif stat.S_ISREG(mode):
                try:
                    yield prefix + (name,), child
                finally:
                    os.close(child)
            else:
                os.close(child)


def select_anchor(anchors, target):
    candidate = Path(os.path.abspath(os.fspath(target)))
    matches = []
    for anchor in anchors:
        try:
            parts = anchor.relative(candidate)
        except PermissionError:
            continue
        matches.append((len(anchor.path.parts), anchor, parts))
    if not matches:
        raise PermissionError(f"path is outside allowed roots: {target}")
    _, anchor, parts = max(matches, key=lambda item: item[0])
    return anchor, parts, candidate


class BoundPath:
    """An open, immutable reference suitable for a same-host external consumer."""

    def __init__(self, fd, display, value):
        self.fd = int(fd)
        self.display = Path(display)
        self.identity = (value.st_dev, value.st_ino)
        self.is_directory = stat.S_ISDIR(value.st_mode)

    @property
    def proc_path(self):
        if self.fd < 0:
            raise RuntimeError("filesystem binding is closed")
        return f"/proc/{os.getpid()}/fd/{self.fd}"

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
