from __future__ import annotations

import errno
import os
from pathlib import Path
import secrets
import stat


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


class AnchoredRoot:
    """Filesystem operations rooted at an already-open directory descriptor."""

    def __init__(self, root):
        self.fd = -1
        self.path = Path(root).resolve(strict=True)
        self.fd = os.open(self.path, _DIRECTORY_FLAGS)

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __del__(self):
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
        if not parts:
            raise PermissionError("the scope root itself is not a mutable child path")
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

    def atomic_write(self, parts, payload, *, require_existing=False):
        parent, name = self.parent(parts, create=not require_existing)
        temporary = f".{name}.tars-{secrets.token_hex(8)}"
        fd = -1
        try:
            if require_existing:
                check = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent)
                os.close(check)
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                         0o600, dir_fd=parent)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.close(parent)

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
                view = memoryview(chunk)
                while view:
                    view = view[os.write(destination_fd, view):]
            os.fsync(destination_fd)
            os.close(destination_fd)
            destination_fd = -1
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
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
