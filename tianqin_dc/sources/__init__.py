from __future__ import annotations

from tianqin_dc.sources.base import SourceFactory
from tianqin_dc.sources.burst import BurstSourceFactory
from tianqin_dc.sources.compact_binary import SBBHSourceFactory, SMBHBSourceFactory
from tianqin_dc.sources.dwd import DWDSourceFactory
from tianqin_dc.sources.emri import EMRISourceFactory
from tianqin_dc.sources.gcb import GCBSourceFactory


SOURCE_FACTORIES: dict[str, SourceFactory] = {
    "burst": BurstSourceFactory(),
    "dwd": DWDSourceFactory(),
    "emri": EMRISourceFactory(),
    "gcb": GCBSourceFactory(),
    "sbbh": SBBHSourceFactory(),
    "smbhb": SMBHBSourceFactory(),
}


def get_source_factory(kind: str) -> SourceFactory:
    try:
        return SOURCE_FACTORIES[kind.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(SOURCE_FACTORIES))
        raise ValueError(f"Unsupported source kind '{kind}'. Supported: {supported}.") from exc


def available_sources() -> tuple[str, ...]:
    return tuple(sorted(SOURCE_FACTORIES))
