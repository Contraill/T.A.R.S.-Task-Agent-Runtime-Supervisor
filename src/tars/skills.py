from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

from .config import CONFIG_PATH, ROLE_PERSONA_ROOT
from .roles import resolve_role_id
from .secure_paths import AnchoredRoot


SKILL_FILENAME = "SKILL.md"
MAX_SKILL_BYTES = 256_000
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class SkillDescriptor:
    name: str
    description: str
    version: str
    scope: str
    path: str
    valid: bool
    errors: tuple[str, ...] = ()
    content_sha256: str = ""

    def summary(self):
        return {"name": self.name, "description": self.description,
                "version": self.version, "scope": self.scope}


@dataclass(frozen=True)
class LoadedSkill:
    descriptor: SkillDescriptor
    instructions: str
    resources: tuple[str, ...]


def _frontmatter(text):
    if not text.startswith("---\n"):
        return {}, text
    try:
        header, body = text[4:].split("\n---\n", 1)
    except ValueError:
        return {}, text
    metadata = {}
    for line in header.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            metadata[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            metadata[key.strip()] = value.strip("'\"")
    return metadata, body


def _inspect(anchor, parts, path, scope):
    errors = []
    digest = ""
    try:
        with anchor.reader(parts) as handle:
            payload = handle.read(MAX_SKILL_BYTES + 1)
        if len(payload) > MAX_SKILL_BYTES:
            raise ValueError("skill instructions exceed size limit")
        text = payload.decode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        metadata, body = _frontmatter(text)
    except (OSError, UnicodeError, ValueError) as exc:
        return SkillDescriptor(path.parent.name, "", "", scope, str(path), False,
                               (str(exc),))
    name = str(metadata.get("name") or path.parent.name).strip()
    description = str(metadata.get("description") or "").strip()
    version = str(metadata.get("version") or "").strip()
    if not NAME_RE.fullmatch(name):
        errors.append("name must use lowercase letters, digits, dot, underscore or hyphen")
    if not description:
        errors.append("description is required")
    if not version:
        errors.append("version is required")
    if not body.strip():
        errors.append("instructions are empty")
    return SkillDescriptor(name, description, version, scope, str(path), not errors,
                           tuple(errors), digest)


class SkillRegistry:
    def __init__(self, *, global_root=None):
        self.global_root = Path(global_root or (CONFIG_PATH.parent / "skills")).expanduser().absolute()

    def _roots(self, *, project_path=None, role=None):
        roots = [("global", self.global_root, ())]
        if project_path:
            project_root = Path(project_path).expanduser().resolve(strict=True)
            roots.append(("project", project_root, (".tars", "skills")))
        if role:
            role_id = resolve_role_id(role)
            roots.append((f"role:{role_id}", ROLE_PERSONA_ROOT, (role_id, "skills")))
        return roots

    def discover(self, *, project_path=None, role=None, include_invalid=False):
        # Later/more-specific roots replace matching names without loading bodies.
        found = {}
        for scope, base, prefix in self._roots(project_path=project_path, role=role):
            try:
                anchor = AnchoredRoot(base.resolve(strict=True))
            except (FileNotFoundError, NotADirectoryError, OSError):
                continue
            try:
                directory = anchor.open_directory(prefix)
            except (FileNotFoundError, NotADirectoryError, OSError):
                anchor.close()
                continue
            try:
                for entry in sorted(os.listdir(directory)):
                    parts = prefix + (entry, SKILL_FILENAME)
                    path = anchor.path.joinpath(*parts)
                    descriptor = _inspect(anchor, parts, path, scope)
                    if descriptor.valid or include_invalid:
                        found[descriptor.name] = descriptor
            finally:
                os.close(directory)
                anchor.close()
        return sorted(found.values(), key=lambda item: item.name)

    def load(self, name, *, project_path=None, role=None):
        descriptors = {item.name: item for item in self.discover(
            project_path=project_path, role=role, include_invalid=True)}
        if name not in descriptors:
            raise KeyError(f"unknown skill: {name}")
        descriptor = descriptors[name]
        if not descriptor.valid:
            raise ValueError("invalid skill: " + "; ".join(descriptor.errors))
        path = Path(descriptor.path)
        anchor = None
        try:
            for scope, base, _prefix in self._roots(project_path=project_path, role=role):
                if scope != descriptor.scope:
                    continue
                try:
                    candidate_anchor = AnchoredRoot(base.resolve(strict=True))
                except (FileNotFoundError, NotADirectoryError, OSError):
                    continue
                try:
                    parts = candidate_anchor.relative(path)
                except PermissionError:
                    candidate_anchor.close()
                    continue
                anchor = candidate_anchor
                break
            if anchor is None:
                raise RuntimeError("skill authority root changed during loading")
            with anchor.reader(parts) as handle:
                payload = handle.read(MAX_SKILL_BYTES + 1)
            if (len(payload) > MAX_SKILL_BYTES or
                    hashlib.sha256(payload).hexdigest() != descriptor.content_sha256):
                raise RuntimeError("skill changed between discovery and loading")
            metadata, body = _frontmatter(payload.decode("utf-8"))
            resources = []
            resource_values = (metadata.get("resources", ())
                               if isinstance(metadata.get("resources", ()), list) else ())
            for value in resource_values:
                relative = Path(str(value))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("skill resource escapes the skill directory")
                resource_parts = parts[:-1] + tuple(relative.parts)
                with anchor.reader(resource_parts):
                    pass
                resources.append(str(anchor.path.joinpath(*resource_parts)))
        finally:
            if anchor is not None:
                anchor.close()
        return LoadedSkill(descriptor, body.strip(), tuple(resources))

    def doctor(self, *, project_path=None, role=None):
        records = self.discover(project_path=project_path, role=role, include_invalid=True)
        return {"ok": all(item.valid for item in records), "skills": len(records),
                "invalid": [{"name": item.name, "scope": item.scope,
                             "errors": list(item.errors)} for item in records if not item.valid]}


def prompt_summaries(registry, **scope):
    """Progressive disclosure: summaries are safe to compile before explicit load."""
    return [json.dumps(item.summary(), ensure_ascii=False)
            for item in registry.discover(**scope)]
