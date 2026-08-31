from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import re


EXTENSION_API_VERSION = 1
KINDS = {"runtime_backend", "tool"}
ENTRY_POINT_GROUPS = {kind: f"tars.{kind}s" for kind in KINDS}
NAME_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class ExtensionDescriptor:
    kind: str
    name: str
    provenance: str
    trusted: bool
    enabled: bool
    distribution: str = ""
    value: str = ""
    api_version: int | None = None


BUILTINS = (
    ExtensionDescriptor("runtime_backend", "llama.cpp", "builtin", True, True),
    ExtensionDescriptor("runtime_backend", "colibri", "builtin", True, True),
    ExtensionDescriptor("tool", "native", "builtin", True, True),
)


def _extension_config(cfg):
    value = cfg.get("extensions", {}) if isinstance(cfg, dict) else {}
    return value if isinstance(value, dict) else {}


class ExtensionLoader:
    """Explicit in-process extension gate; discovery never imports third-party code."""

    def __init__(self, cfg=None, *, entry_points=None):
        self.cfg = cfg or {}
        self._entry_points = entry_points

    def _all_entry_points(self):
        if self._entry_points is not None:
            return tuple(self._entry_points)
        discovered = metadata.entry_points()
        return tuple(discovered) if not hasattr(discovered, "select") else tuple(
            item for group in ENTRY_POINT_GROUPS.values()
            for item in discovered.select(group=group))

    def discover(self, kind=None):
        if kind is not None and kind not in KINDS:
            raise ValueError(f"unknown extension kind: {kind}")
        config = _extension_config(self.cfg)
        enabled = set(map(str, config.get("enabled", ())))
        trusted = set(map(str, config.get("trusted", ())))
        result = [item for item in BUILTINS if kind is None or item.kind == kind]
        groups = ({ENTRY_POINT_GROUPS[kind]: kind} if kind else
                  {group: value for value, group in ENTRY_POINT_GROUPS.items()})
        for point in self._all_entry_points():
            extension_kind = groups.get(getattr(point, "group", ""))
            if extension_kind is None:
                continue
            name = str(point.name)
            identifier = f"{extension_kind}:{name}"
            dist = getattr(point, "dist", None)
            result.append(ExtensionDescriptor(
                extension_kind, name, "third-party", identifier in trusted,
                identifier in enabled, getattr(dist, "name", "") if dist else "",
                str(getattr(point, "value", ""))))
        return tuple(sorted(result, key=lambda item: (item.kind, item.name, item.provenance)))

    def load(self, kind, name):
        if kind not in KINDS or not NAME_RE.fullmatch(str(name)):
            raise ValueError("invalid extension kind or name")
        group = ENTRY_POINT_GROUPS[kind]
        matches = [point for point in self._all_entry_points()
                   if getattr(point, "group", "") == group and point.name == name]
        if not matches:
            raise KeyError(f"unknown {kind} extension: {name}")
        if len(matches) != 1:
            raise RuntimeError(f"ambiguous {kind} extension: {name}")
        identifier = f"{kind}:{name}"
        config = _extension_config(self.cfg)
        if identifier not in set(map(str, config.get("enabled", ()))):
            raise PermissionError(f"third-party extension is not enabled: {identifier}")
        if identifier not in set(map(str, config.get("trusted", ()))):
            raise PermissionError(
                f"in-process extension requires explicit trust: {identifier}; prefer MCP for tools")
        provider = matches[0].load()
        if getattr(provider, "api_version", None) != EXTENSION_API_VERSION:
            raise RuntimeError(f"extension API mismatch: {identifier}")
        if getattr(provider, "kind", None) != kind or getattr(provider, "name", None) != name:
            raise RuntimeError(f"extension identity mismatch: {identifier}")
        if not callable(getattr(provider, "create", None)):
            raise TypeError(f"extension provider has no create factory: {identifier}")
        return provider


def validate_runtime_backend(value):
    required = ("status", "capabilities", "model_capabilities", "load", "unload",
                "complete", "stream", "diagnostics")
    missing = [name for name in required if not callable(getattr(value, name, None))]
    if missing:
        raise TypeError("runtime backend extension is incomplete: " + ", ".join(missing))
    if getattr(value, "local_only", None) is not True:
        raise PermissionError("runtime backend extension must guarantee local-only inference")
    if getattr(value, "zero_idle", None) is not True:
        raise PermissionError("runtime backend extension must guarantee probe-only Zero-Idle")
    return value
