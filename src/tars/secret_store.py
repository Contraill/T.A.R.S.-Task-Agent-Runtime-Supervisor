from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import re


REFERENCE_RE = re.compile(r"^(?P<provider>[a-z][a-z0-9_-]{0,31}):(?P<key>[^\s:][^\s]{0,254})$")


@dataclass(frozen=True)
class SecretReference:
    provider: str
    key: str

    @property
    def value(self):
        return f"{self.provider}:{self.key}"


def parse_reference(value) -> SecretReference:
    match = REFERENCE_RE.fullmatch(str(value or ""))
    if not match:
        raise ValueError("secret values must use provider:key secret references")
    return SecretReference(match.group("provider"), match.group("key"))


class EnvironmentSecrets:
    identity = "env"

    def get(self, key):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("invalid environment secret reference")
        try:
            return os.environ[key]
        except KeyError as exc:
            raise KeyError(f"secret reference is unavailable: env:{key}") from exc


class SecretStore:
    """Resolve opaque references only at an explicitly named consumer boundary."""

    def __init__(self, providers=None, *, scopes=None):
        self.providers = {"env": EnvironmentSecrets(), **(providers or {})}
        self.scopes = {}
        for key, values in (scopes or {}).items():
            reference = parse_reference(key).value
            if not isinstance(values, (list, tuple, set, frozenset)):
                raise ValueError("secret consumer scopes must be arrays")
            self.scopes[reference] = frozenset(map(str, values))

    def __repr__(self):
        return (f"SecretStore(providers={tuple(sorted(self.providers))!r}, "
                f"scoped_references={tuple(sorted(self.scopes))!r})")

    @classmethod
    def from_config(cls, cfg=None, *, providers=None):
        section = cfg.get("secrets", {}) if isinstance(cfg, dict) else {}
        scopes = section.get("scopes", {}) if isinstance(section, dict) else {}
        return cls(providers, scopes=scopes if isinstance(scopes, dict) else {})

    def available(self, reference, *, consumer):
        try:
            with self.resolve(reference, consumer=consumer):
                return True
        except (KeyError, PermissionError, RuntimeError, ValueError):
            return False

    @contextmanager
    def resolve(self, reference, *, consumer):
        if not consumer or not isinstance(consumer, str):
            raise ValueError("secret resolution requires an explicit consumer")
        parsed = parse_reference(reference)
        allowed = self.scopes.get(parsed.value)
        if allowed is not None and consumer not in allowed:
            raise PermissionError(
                f"secret reference is not scoped for consumer {consumer}: {parsed.value}")
        try:
            provider = self.providers[parsed.provider]
        except KeyError as exc:
            raise KeyError(f"secret provider is unavailable: {parsed.provider}") from exc
        value = provider.get(parsed.key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"secret reference resolved to an empty value: {parsed.value}")
        yield value

    def resolve_many(self, references, *, consumer):
        values = {}
        for name, reference in references.items():
            with self.resolve(reference, consumer=consumer) as value:
                values[str(name)] = value
        return values
