from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import zipfile

from .policy import ScopeRequest, canonical_path
from .secure_paths import AnchoredRoot, select_anchor
from .tool_core import ToolResult, ToolRuntime
from .evidence import verify_artifact


def _member_parts(name: str) -> tuple[str, ...]:
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe archive member: {name}")
    parts = tuple(part for part in pure.parts if part not in ("", "."))
    if not parts:
        raise ValueError(f"unsafe archive member: {name}")
    return parts


class ArtifactRuntime:
    def __init__(self, roots, *, runtime=None):
        self.roots = tuple(canonical_path(root) for root in roots)
        if not self.roots:
            raise ValueError("artifact tools require at least one scope root")
        self.anchors = tuple(AnchoredRoot(root) for root in self.roots)
        self.runtime = runtime or ToolRuntime()

    def authorize(self, tool, reads=(), writes=(), *, approval_ids=None, destructive=False,
                  task_id=None, session_id=None, arguments=None):
        requests = []
        for index, path in enumerate(reads):
            requests.append((f"read-{index}", ScopeRequest(
                tool, "read", str(path), arguments or {}, task_id=task_id,
                session_id=session_id, allowed_paths=self.roots,
            )))
        for index, path in enumerate(writes):
            effect = "destructive" if destructive else "write"
            requests.append((f"write-{index}", ScopeRequest(
                tool, effect, str(path), arguments or {}, task_id=task_id,
                session_id=session_id, allowed_paths=self.roots, destructive=destructive,
            )))
        actions = self.runtime.authorize(tuple(requests), approval_ids)
        return actions, [Path(canonical_path(path)) for path in reads], [
            Path(canonical_path(path)) for path in writes
        ]

    def result(self, tool, actions, data, *, task_id=None, evidence_source=None,
               evidence_type="artifact", state="succeeded", error=""):
        journal_state = "failed" if state == "unavailable" else state
        self.runtime.finish(actions, state=journal_state, result=data)
        evidence_ids = ()
        if evidence_source is not None and actions:
            evidence = self.runtime.evidence(
                evidence_type, str(evidence_source), repr(data), task_id=task_id,
                event_uuid=actions[0].event_uuid,
                metadata={"tool": tool, "state": state},
            )
            evidence_ids = (evidence.id,)
        return ToolResult(tool, state, data, error=error,
                          action_ids=tuple(action.id for action in actions),
                          evidence_ids=evidence_ids)


class IntegrityTools:
    def __init__(self, roots, *, runtime=None):
        self.artifacts = ArtifactRuntime(roots, runtime=runtime)

    def hash(self, path, *, algorithm="sha256", expected=None, task_id=None,
             session_id=None):
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(f"unsupported hash algorithm: {algorithm}")
        actions, reads, _ = self.artifacts.authorize(
            "artifact.hash", (path,), task_id=task_id, session_id=session_id,
            arguments={"algorithm": algorithm},
        )
        target = reads[0]
        digest = hashlib.new(algorithm)
        size = 0
        try:
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            value = digest.hexdigest()
            verified = expected is None or value.casefold() == expected.casefold()
            data = {"path": str(target), "algorithm": algorithm, "digest": value,
                    "bytes": size, "expected": expected, "verified": verified}
            state = "succeeded" if verified else "failed"
            error = "checksum mismatch" if not verified else ""
        except Exception as exc:
            self.artifacts.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        return self.artifacts.result(
            "artifact.hash", actions, data, task_id=task_id, evidence_source=target,
            state=state, error=error,
        )

    def verify_claims(self, path, claims, *, task_id=None, session_id=None):
        actions, reads, _ = self.artifacts.authorize(
            "artifact.verify", (path,), task_id=task_id, session_id=session_id,
            arguments={"claims": len(claims)},
        )
        target = reads[0]
        try:
            data = verify_artifact(target, claims, task_id=task_id,
                                   event_uuid=actions[0].event_uuid)
        except Exception as exc:
            self.artifacts.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        state = "succeeded" if data["verified"] else "failed"
        self.artifacts.runtime.finish(actions, state=state, result=data)
        return ToolResult("artifact.verify", state, data,
                          error="" if data["verified"] else "one or more claims are unsupported",
                          action_ids=tuple(action.id for action in actions),
                          evidence_ids=(data["evidence_id"],))


class ArchiveTools:
    FORMATS = ("zip", "tar", "tar.gz", "tgz", "tar.bz2", "tar.xz")

    def __init__(self, roots, *, runtime=None):
        self.artifacts = ArtifactRuntime(roots, runtime=runtime)

    @staticmethod
    def _kind(path: Path) -> str:
        if zipfile.is_zipfile(path):
            return "zip"
        if tarfile.is_tarfile(path):
            return "tar"
        raise ValueError(f"unsupported archive format: {path}")

    def list(self, path, *, task_id=None, session_id=None):
        actions, reads, _ = self.artifacts.authorize(
            "archive.list", (path,), task_id=task_id, session_id=session_id,
        )
        target = reads[0]
        try:
            kind = self._kind(target)
            if kind == "zip":
                with zipfile.ZipFile(target) as archive:
                    members = [{"name": item.filename, "size": item.file_size,
                                "compressed_size": item.compress_size,
                                "directory": item.is_dir()} for item in archive.infolist()]
            else:
                with tarfile.open(target) as archive:
                    members = [{"name": item.name, "size": item.size,
                                "directory": item.isdir(), "symlink": item.issym() or item.islnk()}
                               for item in archive.getmembers()]
            data = {"path": str(target), "format": kind, "members": members}
        except Exception as exc:
            self.artifacts.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        return self.artifacts.result("archive.list", actions, data, task_id=task_id,
                                     evidence_source=target)

    def extract(self, path, destination, *, approval_ids=None, task_id=None,
                session_id=None):
        actions, reads, writes = self.artifacts.authorize(
            "archive.extract", (path,), (destination,), approval_ids=approval_ids,
            task_id=task_id, session_id=session_id, arguments={"archive": str(path)},
        )
        source, root = reads[0], writes[0]
        try:
            source_anchor, source_parts, _ = select_anchor(self.artifacts.anchors, path)
            destination_anchor, root_parts, root = select_anchor(
                self.artifacts.anchors, destination
            )
            destination_anchor.makedirs(root_parts)
            extracted = []
            source_fd = source_anchor.open(source_parts, os.O_RDONLY)
            source_handle = os.fdopen(source_fd, "rb")
            kind = "zip" if zipfile.is_zipfile(source_handle) else "tar"
            source_handle.seek(0)
            if kind == "zip":
                with source_handle, zipfile.ZipFile(source_handle) as archive:
                    members = archive.infolist()
                    for member in members:
                        member_parts = _member_parts(member.filename)
                        mode = member.external_attr >> 16
                        if (mode & 0o170000) == 0o120000:
                            raise ValueError(f"archive symlink is not allowed: {member.filename}")
                    for member in members:
                        member_parts = _member_parts(member.filename)
                        target_parts = root_parts + member_parts
                        target = root.joinpath(*member_parts)
                        if member.is_dir():
                            destination_anchor.makedirs(target_parts)
                        else:
                            fd = destination_anchor.open(
                                target_parts, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                                create_parents=True,
                            )
                            with archive.open(member) as src, os.fdopen(fd, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                        extracted.append(str(target))
            else:
                with source_handle, tarfile.open(fileobj=source_handle) as archive:
                    members = archive.getmembers()
                    for member in members:
                        _member_parts(member.name)
                        if member.issym() or member.islnk() or member.isdev():
                            raise ValueError(f"unsafe archive member type: {member.name}")
                    for member in members:
                        member_parts = _member_parts(member.name)
                        target_parts = root_parts + member_parts
                        target = root.joinpath(*member_parts)
                        if member.isdir():
                            destination_anchor.makedirs(target_parts)
                        elif member.isfile():
                            source_handle = archive.extractfile(member)
                            if source_handle is None:
                                raise ValueError(f"cannot extract archive member: {member.name}")
                            fd = destination_anchor.open(
                                target_parts, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                                create_parents=True,
                            )
                            with source_handle, os.fdopen(fd, "wb") as destination_handle:
                                shutil.copyfileobj(source_handle, destination_handle)
                        extracted.append(str(target))
            data = {"archive": str(source), "destination": str(root), "format": kind,
                    "extracted": extracted}
        except Exception as exc:
            self.artifacts.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        return self.artifacts.result("archive.extract", actions, data, task_id=task_id,
                                     evidence_source=root)

    def create(self, output, sources, *, format=None, approval_ids=None, task_id=None,
               session_id=None):
        actions, reads, writes = self.artifacts.authorize(
            "archive.create", tuple(sources), (output,), approval_ids=approval_ids,
            task_id=task_id, session_id=session_id,
            arguments={"sources": [str(item) for item in sources], "format": format},
        )
        destination = writes[0]
        kind = format or ("zip" if destination.suffix.casefold() == ".zip" else "tar")
        if kind not in self.FORMATS:
            self.artifacts.runtime.finish(actions, state="failed",
                                          result={"error": "unsupported archive format"})
            raise ValueError(f"unsupported archive format: {kind}")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            for source in reads:
                if source.is_symlink() or (source.is_dir() and any(
                    item.is_symlink() for item in source.rglob("*")
                )):
                    raise ValueError("archive creation does not include symbolic links")
            if kind == "zip":
                with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
                    for source in reads:
                        if source.is_dir():
                            for item in sorted(source.rglob("*")):
                                if item.is_file() and not item.is_symlink():
                                    archive.write(item, Path(source.name) / item.relative_to(source))
                        else:
                            archive.write(source, source.name)
            else:
                modes = {"tar": "w", "tar.gz": "w:gz", "tgz": "w:gz",
                         "tar.bz2": "w:bz2", "tar.xz": "w:xz"}
                with tarfile.open(destination, modes[kind]) as archive:
                    for source in reads:
                        archive.add(source, arcname=source.name, recursive=True)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            data = {"path": str(destination), "format": kind,
                    "sources": [str(path) for path in reads], "sha256": digest,
                    "bytes": destination.stat().st_size}
        except Exception as exc:
            self.artifacts.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        return self.artifacts.result("archive.create", actions, data, task_id=task_id,
                                     evidence_source=destination)


class PDFTools:
    """Bounded PDF operations with explicit backend capability reporting."""

    def __init__(self, roots, *, runtime=None, runner=subprocess.run):
        self.artifacts = ArtifactRuntime(roots, runtime=runtime)
        self.runner = runner

    def capabilities(self):
        pypdf = bool(importlib.util.find_spec("pypdf"))
        return {
            "info": bool(shutil.which("pdfinfo")), "text": bool(shutil.which("pdftotext")),
            "search": bool(shutil.which("pdftotext")), "render": bool(shutil.which("pdftoppm")),
            "merge": pypdf, "split": pypdf, "reorder": pypdf, "rotate": pypdf,
            "delete_pages": pypdf, "annotation": False, "redaction": False,
            "form_fill": pypdf, "export": pypdf,
            "limitations": {
                "annotation": "no reliable annotation backend installed",
                "redaction": "content-safe redaction requires a supported redaction backend",
                "text_edit": "arbitrary in-place PDF text editing is unsupported",
            },
        }

    def _read_action(self, tool, path, task_id, session_id):
        actions, reads, _ = self.artifacts.authorize(
            tool, (path,), task_id=task_id, session_id=session_id,
        )
        return actions, reads[0]

    def info(self, path, *, task_id=None, session_id=None):
        actions, target = self._read_action("pdf.info", path, task_id, session_id)
        binary = shutil.which("pdfinfo")
        if not binary:
            data = {"path": str(target), "available": False, "reason": "pdfinfo unavailable"}
            return self.artifacts.result("pdf.info", actions, data, task_id=task_id,
                                         evidence_source=target, state="unavailable",
                                         error=data["reason"])
        proc = self.runner([binary, str(target)], capture_output=True, text=True, check=False)
        values = dict(line.split(":", 1) for line in proc.stdout.splitlines() if ":" in line)
        data = {"path": str(target), "exit_code": proc.returncode,
                "metadata": {key.strip(): value.strip() for key, value in values.items()},
                "stderr": proc.stderr}
        state = "succeeded" if proc.returncode == 0 else "failed"
        return self.artifacts.result("pdf.info", actions, data, task_id=task_id,
                                     evidence_source=target, state=state,
                                     error=proc.stderr if proc.returncode else "")

    def text(self, path, *, pages=None, task_id=None, session_id=None, _tool="pdf.text"):
        actions, target = self._read_action(_tool, path, task_id, session_id)
        binary = shutil.which("pdftotext")
        if not binary:
            data = {"path": str(target), "available": False, "reason": "pdftotext unavailable"}
            return self.artifacts.result(_tool, actions, data, task_id=task_id,
                                         evidence_source=target, state="unavailable",
                                         error=data["reason"])
        argv = [binary]
        if pages:
            argv.extend(["-f", str(min(pages)), "-l", str(max(pages))])
        argv.extend([str(target), "-"])
        proc = self.runner(argv, capture_output=True, text=True, check=False)
        data = {"path": str(target), "text": proc.stdout, "exit_code": proc.returncode,
                "pages": list(pages) if pages else None, "stderr": proc.stderr}
        state = "succeeded" if proc.returncode == 0 else "failed"
        return self.artifacts.result(_tool, actions, data, task_id=task_id,
                                     evidence_source=target, state=state,
                                     error=proc.stderr if proc.returncode else "")

    def search(self, path, query, **kwargs):
        result = self.text(path, _tool="pdf.search", **kwargs)
        if not result.succeeded:
            return ToolResult("pdf.search", result.state, result.data, result.error,
                              result.action_ids, result.evidence_ids)
        hits = [{"line": number, "text": line} for number, line in
                enumerate(result.data["text"].splitlines(), 1)
                if query.casefold() in line.casefold()]
        return ToolResult("pdf.search", "succeeded",
                          {"path": result.data["path"], "query": query, "hits": hits},
                          action_ids=result.action_ids, evidence_ids=result.evidence_ids)

    def render(self, path, output_dir, *, pages=None, dpi=144, approval_ids=None,
               task_id=None, session_id=None):
        actions, reads, writes = self.artifacts.authorize(
            "pdf.render", (path,), (output_dir,), approval_ids=approval_ids,
            task_id=task_id, session_id=session_id,
            arguments={"dpi": dpi, "pages": pages},
        )
        target, destination = reads[0], writes[0]
        binary = shutil.which("pdftoppm")
        if not binary:
            data = {"available": False, "reason": "pdftoppm unavailable"}
            return self.artifacts.result("pdf.render", actions, data, state="unavailable",
                                         error=data["reason"])
        destination.mkdir(parents=True, exist_ok=True)
        prefix = destination / "page"
        argv = [binary, "-png", "-r", str(int(dpi))]
        if pages:
            argv.extend(["-f", str(min(pages)), "-l", str(max(pages))])
        argv.extend([str(target), str(prefix)])
        proc = self.runner(argv, capture_output=True, text=True, check=False)
        outputs = sorted(str(path) for path in destination.glob("page-*.png"))
        data = {"source": str(target), "output_dir": str(destination), "files": outputs,
                "exit_code": proc.returncode, "stderr": proc.stderr, "dpi": int(dpi)}
        verified = proc.returncode == 0 and bool(outputs)
        return self.artifacts.result("pdf.render", actions, data, task_id=task_id,
                                     evidence_source=destination,
                                     state="succeeded" if verified else "failed",
                                     error="" if verified else proc.stderr or "no pages rendered")

    @staticmethod
    def _pypdf():
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError as exc:
            raise RuntimeError("pypdf backend is unavailable") from exc
        return PdfReader, PdfWriter

    def transform(self, operation, inputs, output, *, pages=None, rotations=None,
                  form_values=None, approval_ids=None, task_id=None, session_id=None):
        if operation not in {"merge", "split", "reorder", "rotate", "delete_pages",
                             "form_fill", "export"}:
            raise ValueError(f"unsupported PDF transformation: {operation}")
        actions, reads, writes = self.artifacts.authorize(
            f"pdf.{operation}", tuple(inputs), (output,), approval_ids=approval_ids,
            task_id=task_id, session_id=session_id,
            arguments={
                "operation": operation, "pages": pages, "rotations": rotations,
                "form_values_sha256": hashlib.sha256(json.dumps(
                    form_values or {}, sort_keys=True, ensure_ascii=False,
                    separators=(",", ":"), default=str).encode()).hexdigest(),
            },
        )
        destination = writes[0]
        try:
            PdfReader, PdfWriter = self._pypdf()
            readers = [PdfReader(str(path)) for path in reads]
            writer = PdfWriter()
            source_pages = [page for reader in readers for page in reader.pages]
            indexes = list(range(len(source_pages))) if pages is None else [int(i) for i in pages]
            if operation == "delete_pages":
                removed = set(indexes)
                indexes = [i for i in range(len(source_pages)) if i not in removed]
            for index in indexes:
                page = source_pages[index]
                if operation == "rotate":
                    angle = int((rotations or {}).get(index, (rotations or {}).get(str(index), 0)))
                    if angle % 90:
                        raise ValueError("PDF rotation must be a multiple of 90 degrees")
                    page.rotate(angle)
                writer.add_page(page)
            if operation == "form_fill":
                for page in writer.pages:
                    writer.update_page_form_field_values(page, form_values or {}, auto_regenerate=False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                writer.write(handle)
            verified_pages = len(PdfReader(str(destination)).pages)
            data = {"operation": operation, "inputs": [str(path) for path in reads],
                    "output": str(destination), "pages": verified_pages,
                    "regenerated": True,
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
        except Exception as exc:
            self.artifacts.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        return self.artifacts.result(f"pdf.{operation}", actions, data, task_id=task_id,
                                     evidence_source=destination)

    def annotation(self, *args, **kwargs):
        return ToolResult("pdf.annotation", "unavailable", {},
                          "no reliable annotation backend installed")

    def redact(self, *args, **kwargs):
        return ToolResult("pdf.redact", "unavailable", {},
                          "content-safe redaction backend is unavailable")

    def merge(self, inputs, output, **kwargs):
        return self.transform("merge", inputs, output, **kwargs)

    def reorder(self, path, output, pages, **kwargs):
        return self.transform("reorder", (path,), output, pages=pages, **kwargs)

    def rotate(self, path, output, rotations, **kwargs):
        return self.transform("rotate", (path,), output, rotations=rotations, **kwargs)

    def delete_pages(self, path, output, pages, **kwargs):
        return self.transform("delete_pages", (path,), output, pages=pages, **kwargs)

    def form_fill(self, path, output, values, **kwargs):
        return self.transform("form_fill", (path,), output, form_values=values, **kwargs)

    def export(self, path, output, **kwargs):
        return self.transform("export", (path,), output, **kwargs)

    def split(self, path, output_dir, page_groups, *, approval_ids=None,
              task_id=None, session_id=None):
        actions, reads, writes = self.artifacts.authorize(
            "pdf.split", (path,), (output_dir,), approval_ids=approval_ids,
            task_id=task_id, session_id=session_id,
            arguments={"page_groups": page_groups},
        )
        source, output_root = reads[0], writes[0]
        try:
            PdfReader, PdfWriter = self._pypdf()
            reader = PdfReader(str(source))
            output_root.mkdir(parents=True, exist_ok=True)
            outputs = []
            for number, pages in enumerate(page_groups, 1):
                writer = PdfWriter()
                for page in pages:
                    writer.add_page(reader.pages[int(page)])
                output = output_root / f"part-{number}.pdf"
                with output.open("wb") as handle:
                    writer.write(handle)
                verified_pages = len(PdfReader(str(output)).pages)
                outputs.append({"path": str(output), "pages": verified_pages,
                                "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
            data = {"source": str(source), "output_dir": str(output_root),
                    "parts": outputs, "regenerated": True}
        except Exception as exc:
            self.artifacts.runtime.finish(actions, state="failed", result={"error": str(exc)})
            raise
        return self.artifacts.result("pdf.split", actions, data, task_id=task_id,
                                     evidence_source=output_root)
