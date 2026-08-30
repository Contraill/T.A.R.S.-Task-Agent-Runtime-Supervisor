from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from .config import CONFIG_PATH, ROLE_PERSONA_ROOT
from .roles import resolve_role_id


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


def _inspect(path, scope, root):
    errors = []
    try:
        resolved_root = root.resolve()
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
        relative = path.relative_to(root)
        cursor = root
        if any((cursor := cursor / part).is_symlink() for part in relative.parts):
            raise ValueError("skill entry cannot be a symlink")
        size = resolved.stat().st_size
        if size > MAX_SKILL_BYTES:
            raise ValueError("skill instructions exceed size limit")
        metadata, body = _frontmatter(resolved.read_text(encoding="utf-8"))
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
                           tuple(errors))


class SkillRegistry:
    def __init__(self, *, global_root=None):
        self.global_root = Path(global_root or (CONFIG_PATH.parent / "skills"))

    def _roots(self, *, project_path=None, role=None):
        roots = [("global", self.global_root)]
        if project_path:
            roots.append(("project", Path(project_path).resolve() / ".tars" / "skills"))
        if role:
            role_id = resolve_role_id(role)
            roots.append((f"role:{role_id}", ROLE_PERSONA_ROOT / role_id / "skills"))
        return roots

    def discover(self, *, project_path=None, role=None, include_invalid=False):
        # Later/more-specific roots replace matching names without loading bodies.
        found = {}
        for scope, root in self._roots(project_path=project_path, role=role):
            if not root.is_dir():
                continue
            for path in sorted(root.glob(f"*/{SKILL_FILENAME}")):
                descriptor = _inspect(path, scope, root)
                if descriptor.valid or include_invalid:
                    found[descriptor.name] = descriptor
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
        metadata, body = _frontmatter(path.read_text(encoding="utf-8"))
        resources = []
        root = path.parent.resolve()
        for value in metadata.get("resources", ()) if isinstance(metadata.get("resources", ()), list) else ():
            candidate = (root / str(value)).resolve(strict=True)
            candidate.relative_to(root)
            if candidate.is_file():
                resources.append(str(candidate))
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
