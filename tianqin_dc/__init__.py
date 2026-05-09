from __future__ import annotations

from typing import Any

__all__ = [
    "RunConfig",
    "TianQinDatasetBuilder",
    "SimpleBBHBuilder",
    "SimpleEMRIBuilder",
    "build_dataset",
    "load_emri_catalog",
    "load_simple_bbh_config",
    "load_simple_emri_config",
    "load_run_config",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name in {"RunConfig", "load_run_config"}:
        from tianqin_dc.config import RunConfig, load_run_config

        return {"RunConfig": RunConfig, "load_run_config": load_run_config}[name]
    if name in {"TianQinDatasetBuilder", "build_dataset"}:
        from tianqin_dc.builder import TianQinDatasetBuilder, build_dataset

        return {"TianQinDatasetBuilder": TianQinDatasetBuilder, "build_dataset": build_dataset}[name]
    if name in {"SimpleBBHBuilder", "load_simple_bbh_config"}:
        from tianqin_dc.bbh_export import SimpleBBHBuilder, load_simple_bbh_config

        return {"SimpleBBHBuilder": SimpleBBHBuilder, "load_simple_bbh_config": load_simple_bbh_config}[name]
    if name == "load_emri_catalog":
        from tianqin_dc.emri_catalog import load_emri_catalog

        return load_emri_catalog
    if name in {"SimpleEMRIBuilder", "load_simple_emri_config"}:
        from tianqin_dc.emri_export import SimpleEMRIBuilder, load_simple_emri_config

        return {"SimpleEMRIBuilder": SimpleEMRIBuilder, "load_simple_emri_config": load_simple_emri_config}[name]
    raise AttributeError(f"module 'tianqin_dc' has no attribute {name!r}")
