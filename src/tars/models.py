from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelRecord:
    alias: str
    name: str
    path: Path
    sha256: str
    backend: str
    quant: str
    native_context: int
    source: str = "seed"
    source_revision: str = ""
    license: str = "unknown"
    size: int = 0
    integrity_verified: bool = False
    runtime_compatible: bool = False

    @classmethod
    def from_dict(cls, alias, data):
        return cls(
            alias=alias,
            name=data["name"],
            path=Path(data["path"]).expanduser(),
            sha256=data["sha256"],
            backend=data.get("backend", "llama.cpp"),
            quant=data.get("quant", "unknown"),
            native_context=int(data.get("native_context", 0)),
            source=data.get("source", "seed"),
            source_revision=data.get("source_revision", ""),
            license=data.get("license", "unknown"),
            size=int(data.get("size", 0)),
            integrity_verified=bool(data.get("integrity_verified", False)),
            runtime_compatible=bool(data.get("runtime_compatible", False)),
        )


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    context: int
    cache_type_k: str
    cache_type_v: str
    cpus: str
    threads: int
    batch_threads: int
    ngl: str | int | None = None
    tensor_overrides: str | None = None

    @classmethod
    def from_dict(cls, name, data):
        return cls(
            name=name,
            context=int(data["context"]),
            cache_type_k=data.get("cache_type_k", "f16"),
            cache_type_v=data.get("cache_type_v", "f16"),
            cpus=data.get("cpus", "0-11"),
            threads=int(data.get("threads", 1)),
            batch_threads=int(data.get("batch_threads", data.get("threads", 1))),
            ngl=data.get("ngl"),
            tensor_overrides=data.get("tensor_overrides"),
        )
