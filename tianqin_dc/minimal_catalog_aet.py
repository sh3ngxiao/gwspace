from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, replace
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from tianqin_dc.bbh_catalog import BBHCatalogEntry, iter_bbh_catalog, redshift_to_luminosity_distance_mpc
from tianqin_dc.bbh_export import BBHCompletionConfig, _parameters_from_catalog_entry as bbh_parameters_from_catalog_entry
from tianqin_dc.config import ObservationConfig
from tianqin_dc.dwd_catalog import DWDCatalogEntry, iter_dwd_catalog
from tianqin_dc.emri_catalog import EMRICatalogEntry, load_emri_catalog
from tianqin_dc.emri_export import EMRICompletionConfig, _parameters_from_catalog_entry as emri_parameters_from_catalog_entry
from tianqin_dc.minimal_aet_io import MinimalOutputConfig, save_minimal_aet_hdf5
from tianqin_dc.mbhb_catalog import MBHBCatalogEntry, iter_mbhb_catalog
from tianqin_dc.smbhb_export import (
    SMBHBCompletionConfig,
    _parameters_from_catalog_entry as smbhb_parameters_from_catalog_entry,
)
from tianqin_dc.sources import get_source_factory
from tianqin_dc.sources.dwd import (
    DWDSourceFactory,
    add_dwd_to_fastgb_frequency_buffers,
    empty_fastgb_xyz_frequency_buffers,
    fastgb_xyz_frequency_buffers_to_aet_channels,
)


SUPPORTED_CATALOG_KINDS = ("dwd", "emri", "sbbh", "smbhb")


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{field_name}' to be a mapping, got {type(value).__name__}.")
    return value


def _normalize_kind(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "bbh": "sbbh",
        "stellar_bbh": "sbbh",
        "stellar-mass-bbh": "sbbh",
        "mbhb": "smbhb",
        "massive_bbh": "smbhb",
        "massive-black-hole-binary": "smbhb",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_CATALOG_KINDS:
        raise ValueError(f"Unsupported catalog source kind '{value}'. Supported: {SUPPORTED_CATALOG_KINDS}.")
    return normalized


@dataclass(frozen=True)
class CatalogSelectionConfig:
    paths: tuple[str, ...]
    select_all: bool = False
    row_numbers: tuple[int, ...] = tuple()
    selection: str = "first"
    rows_per_file: int | None = None
    max_sources: int | None = None

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "CatalogSelectionConfig":
        data = _mapping(value, field_name="catalog")
        resolved_paths: list[str] = []

        path = data.get("path")
        if path is not None:
            resolved_paths.append(str(path))

        paths_raw = data.get("paths")
        if paths_raw is not None:
            if not isinstance(paths_raw, list) or not paths_raw:
                raise ValueError("Config field 'catalog.paths' must be a non-empty list when provided.")
            resolved_paths.extend(str(item) for item in paths_raw)

        path_glob = data.get("path_glob")
        if path_glob is not None:
            matched_paths = sorted(glob.glob(str(path_glob)))
            if not matched_paths:
                raise ValueError(f"Config field 'catalog.path_glob' matched no files: {path_glob}")
            resolved_paths.extend(matched_paths)

        if not resolved_paths:
            raise ValueError("Catalog config must provide at least one of 'path', 'paths', or 'path_glob'.")

        row_numbers_raw = data.get("row_numbers")
        if row_numbers_raw is None:
            row_numbers = tuple()
        else:
            if not isinstance(row_numbers_raw, list) or not row_numbers_raw:
                raise ValueError("Config field 'catalog.row_numbers' must be a non-empty list when provided.")
            row_numbers = tuple(int(item) for item in row_numbers_raw)
            if any(item <= 0 for item in row_numbers):
                raise ValueError("Config field 'catalog.row_numbers' must contain only positive integers.")

        selection = str(data.get("selection", "first")).lower()
        if selection not in ("first", "random"):
            raise ValueError("Config field 'catalog.selection' must be either 'first' or 'random'.")

        rows_per_file = None if data.get("rows_per_file") is None else int(data["rows_per_file"])
        if rows_per_file is not None and rows_per_file <= 0:
            raise ValueError("Config field 'catalog.rows_per_file' must be positive when provided.")

        max_sources = None if data.get("max_sources") is None else int(data["max_sources"])
        if max_sources is not None and max_sources <= 0:
            raise ValueError("Config field 'catalog.max_sources' must be positive when provided.")

        select_all = bool(data.get("select_all", False))
        if select_all and (row_numbers or rows_per_file is not None):
            raise ValueError("Config field 'catalog.select_all' cannot be combined with row_numbers or rows_per_file.")
        if not select_all and not row_numbers and rows_per_file is None:
            raise ValueError(
                "Catalog configs for minimal A/E/T export must set 'catalog.select_all=true', "
                "'catalog.row_numbers', or 'catalog.rows_per_file'."
            )

        return cls(
            paths=tuple(dict.fromkeys(resolved_paths)),
            select_all=select_all,
            row_numbers=row_numbers,
            selection=selection,
            rows_per_file=rows_per_file,
            max_sources=max_sources,
        )


@dataclass(frozen=True)
class SourceMetadataOutputConfig:
    path: str
    overwrite: bool = False

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "SourceMetadataOutputConfig":
        data = _mapping(value, field_name="source_metadata_output")
        return cls(
            path=str(data["path"]),
            overwrite=bool(data.get("overwrite", False)),
        )


@dataclass(frozen=True)
class MinimalCatalogAETConfig:
    kind: str
    seed: int
    catalog: CatalogSelectionConfig
    observation: ObservationConfig
    output: MinimalOutputConfig
    source_metadata_output: SourceMetadataOutputConfig | None
    bbh: BBHCompletionConfig
    emri: EMRICompletionConfig
    smbhb: SMBHBCompletionConfig
    progress_interval: int
    workers: int

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> "MinimalCatalogAETConfig":
        source_raw = _mapping(data.get("source", {}), field_name="source")
        kind = _normalize_kind(str(source_raw.get("kind", data.get("kind", ""))))
        observation_data = dict(_mapping(data["observation"], field_name="observation"))
        observation_data["channels"] = ["A", "E", "T"]
        return cls(
            kind=kind,
            seed=int(data.get("seed", 123456789)),
            catalog=CatalogSelectionConfig.from_config(_mapping(data["catalog"], field_name="catalog")),
            observation=ObservationConfig.from_config(observation_data),
            output=MinimalOutputConfig.from_config(_mapping(data["output"], field_name="output")),
            source_metadata_output=(
                None
                if data.get("source_metadata_output") is None
                else SourceMetadataOutputConfig.from_config(
                    _mapping(data["source_metadata_output"], field_name="source_metadata_output")
                )
            ),
            bbh=BBHCompletionConfig.from_config(data.get("bbh")),
            emri=EMRICompletionConfig.from_config(data.get("emri")),
            smbhb=SMBHBCompletionConfig.from_config(data.get("smbhb")),
            progress_interval=max(0, int(data.get("progress_interval", 1000))),
            workers=max(1, int(data.get("workers", 1))),
        )


def load_minimal_catalog_aet_config(path: str | Path) -> tuple[MinimalCatalogAETConfig, dict[str, Any]]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a JSON object.")
    return MinimalCatalogAETConfig.from_config(raw), raw


def _reservoir_sample[T](items: Iterable[T], count: int, rng: np.random.Generator) -> list[T]:
    reservoir: list[T] = []
    for index, item in enumerate(items):
        if index < count:
            reservoir.append(item)
            continue
        slot = int(rng.integers(0, index + 1))
        if slot < count:
            reservoir[slot] = item
    return reservoir


def _select_by_row_numbers[T](items: Iterable[T], row_numbers: tuple[int, ...], path: str) -> list[T]:
    wanted = set(row_numbers)
    found: dict[int, T] = {}
    for item in items:
        row_number = int(getattr(item, "row_number"))
        if row_number in wanted:
            found[row_number] = item
            if len(found) == len(wanted):
                break
    missing = [row_number for row_number in row_numbers if row_number not in found]
    if missing:
        raise IndexError(f"Catalog '{path}' does not contain requested row numbers: {missing}.")
    return [found[row_number] for row_number in row_numbers]


def _select_rows_per_file[T](
    items: Iterable[T],
    rows_per_file: int,
    selection: str,
    rng: np.random.Generator,
) -> Iterable[T]:
    if selection == "random":
        return _reservoir_sample(items, rows_per_file, rng)

    def first_rows() -> Iterator[T]:
        for index, item in enumerate(items):
            if index >= rows_per_file:
                break
            yield item

    return first_rows()


def _iter_entries_for_path(kind: str, path: str) -> Iterable[Any]:
    if kind == "dwd":
        return iter_dwd_catalog(path)
    if kind == "sbbh":
        return iter_bbh_catalog(path)
    if kind == "smbhb":
        return iter_mbhb_catalog(path)
    if kind == "emri":
        return iter(load_emri_catalog(path))
    raise ValueError(f"Unsupported catalog source kind '{kind}'.")


def _iter_selected_entries(
    kind: str,
    selection: CatalogSelectionConfig,
    rng: np.random.Generator,
) -> Iterable[Any]:
    def selected_by_file() -> Iterator[Any]:
        for path in selection.paths:
            entries = _iter_entries_for_path(kind, path)
            if selection.row_numbers:
                yield from _select_by_row_numbers(entries, selection.row_numbers, path)
            elif selection.rows_per_file is not None:
                yield from _select_rows_per_file(entries, selection.rows_per_file, selection.selection, rng)
            else:
                yield from entries

    entries = selected_by_file()
    if selection.max_sources is None:
        return entries
    if selection.selection == "random":
        return _reservoir_sample(entries, selection.max_sources, rng)

    def limited_entries() -> Iterator[Any]:
        for index, item in enumerate(entries):
            if index >= selection.max_sources:
                break
            yield item

    return limited_entries()


def _source_parameters_from_entry(
    kind: str,
    entry: DWDCatalogEntry | BBHCatalogEntry | MBHBCatalogEntry | EMRICatalogEntry,
    config: MinimalCatalogAETConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if kind == "dwd":
        return entry.to_source_parameters()
    if kind == "sbbh":
        distance_mpc = redshift_to_luminosity_distance_mpc(entry.z)
        return bbh_parameters_from_catalog_entry(entry, float(distance_mpc), config.bbh, rng)
    if kind == "smbhb":
        return smbhb_parameters_from_catalog_entry(entry, config.smbhb, rng)
    if kind == "emri":
        return emri_parameters_from_catalog_entry(entry, config.emri, rng)
    raise ValueError(f"Unsupported catalog source kind '{kind}'.")


def _entry_label(entry: Any) -> str:
    file_name = getattr(entry, "file_name", "<unknown>")
    row_number = getattr(entry, "row_number", "?")
    return f"{file_name}:row{row_number}"


def _child_seed(seed_sequence: np.random.SeedSequence) -> int:
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_mapping"):
        return _jsonable(value.to_mapping())
    return value


class SourceMetadataWriter:
    def __init__(
        self,
        output: SourceMetadataOutputConfig,
        *,
        config: MinimalCatalogAETConfig,
        raw_config: Mapping[str, Any] | None,
        selection_seed: int,
        generation_seed: int,
    ) -> None:
        self.output = output
        self.config = config
        self.raw_config = raw_config
        self.selection_seed = selection_seed
        self.generation_seed = generation_seed
        self.path = Path(output.path)
        self._handle: Any | None = None

    def __enter__(self) -> "SourceMetadataWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self.output.overwrite:
            raise FileExistsError(
                f"Source metadata file '{self.path}' already exists. "
                "Set source_metadata_output.overwrite=true to replace it."
            )
        self._handle = self.path.open("w", encoding="utf-8")
        self.write(
            {
                "record_type": "manifest",
                "format_name": "tianqin-dc-minimal-source-metadata-jsonl",
                "format_version": "0.1.0",
                "kind": self.config.kind,
                "root_seed": self.config.seed,
                "selection_seed": self.selection_seed,
                "generation_seed": self.generation_seed,
                "signal_output_path": self.config.output.path,
                "catalog": {
                    "paths": list(self.config.catalog.paths),
                    "select_all": self.config.catalog.select_all,
                    "row_numbers": list(self.config.catalog.row_numbers),
                    "selection": self.config.catalog.selection,
                    "rows_per_file": self.config.catalog.rows_per_file,
                    "max_sources": self.config.catalog.max_sources,
                },
                "observation": {
                    "duration_s": self.config.observation.duration_s,
                    "sample_rate_hz": self.config.observation.sample_rate_hz,
                    "sample_spacing_s": self.config.observation.sample_spacing_s,
                    "num_samples": self.config.observation.num_samples,
                    "detector": self.config.observation.detector,
                    "tdi_generation": self.config.observation.tdi_generation,
                    "channels": list(self.config.observation.channels),
                    "use_gpu": self.config.observation.use_gpu,
                },
                "raw_config": deepcopy(dict(self.raw_config)) if self.raw_config is not None else None,
            }
        )
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, payload: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("SourceMetadataWriter is not open.")
        self._handle.write(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")))
        self._handle.write("\n")

    def append_jsonl_file(self, path: str | Path) -> None:
        if self._handle is None:
            raise RuntimeError("SourceMetadataWriter is not open.")
        with Path(path).open("r", encoding="utf-8") as source:
            for line in source:
                self._handle.write(line)

    def write_source(self, payload: Mapping[str, Any]) -> None:
        self.write(payload)

    def write_summary(self, *, processed: int) -> None:
        self.write(
            {
                "record_type": "summary",
                "kind": self.config.kind,
                "processed": processed,
            }
        )


def _source_metadata_payload(
    *,
    kind: str,
    source_index: int,
    entry: Any,
    source_parameters: Mapping[str, Any],
    generated: Any,
) -> dict[str, Any]:
    return {
        "record_type": "source",
        "source_index": source_index,
        "source_id": f"{kind}_{source_index:08d}",
        "kind": kind,
        "catalog_label": _entry_label(entry),
        "catalog_entry": entry.to_mapping() if hasattr(entry, "to_mapping") else None,
        "source_parameters": dict(source_parameters),
        "waveform_parameters": generated.parameters,
        "generation": {
            "family": generated.family,
            "engine": generated.engine,
            "implementation": generated.implementation,
            "domain": generated.domain,
            "notes": list(generated.notes),
            "metadata": generated.metadata,
        },
    }


def _write_jsonl_line(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def _metadata_writer(
    config: MinimalCatalogAETConfig,
    raw_config: Mapping[str, Any] | None,
    *,
    selection_seed: int,
    generation_seed: int,
) -> SourceMetadataWriter | None:
    if config.source_metadata_output is None:
        return None
    return SourceMetadataWriter(
        config.source_metadata_output,
        config=config,
        raw_config=raw_config,
        selection_seed=selection_seed,
        generation_seed=generation_seed,
    )


def _generate_minimal_dwd_fastgb_aet_shard(
    args: tuple[int, int, MinimalCatalogAETConfig, int, int, str | None],
) -> dict[str, Any]:
    shard_index, shard_count, config, selection_seed, generation_seed, metadata_part_path = args
    observation = config.observation
    buffers = empty_fastgb_xyz_frequency_buffers(observation)
    factory = DWDSourceFactory()
    selection_rng = np.random.default_rng(selection_seed)
    generation_rng = np.random.default_rng(generation_seed)
    entries = _iter_selected_entries(config.kind, config.catalog, selection_rng)

    processed = 0
    visited = 0
    if metadata_part_path is None:
        metadata_handle_context: Any = nullcontext(None)
    else:
        metadata_handle_context = Path(metadata_part_path).open("w", encoding="utf-8")

    with metadata_handle_context as metadata_handle:
        for entry in entries:
            visited += 1
            parameters = _source_parameters_from_entry(config.kind, entry, config, generation_rng)
            if (visited - 1) % shard_count != shard_index:
                continue
            try:
                prepared, catalog_parameterization = add_dwd_to_fastgb_frequency_buffers(
                    factory,
                    parameters,
                    observation,
                    buffers,
                )
                generated = factory.make_fastgb_result(
                    {},
                    prepared,
                    observation,
                    catalog_parameterization=catalog_parameterization,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to generate {config.kind} source #{visited} from {_entry_label(entry)} "
                    f"in shard {shard_index + 1}/{shard_count}."
                ) from exc
            processed += 1
            if metadata_handle is not None:
                _write_jsonl_line(
                    metadata_handle,
                    _source_metadata_payload(
                        kind=config.kind,
                        source_index=visited,
                        entry=entry,
                        source_parameters=parameters,
                        generated=generated,
                    ),
                )
            if config.progress_interval and processed % config.progress_interval == 0:
                print(
                    f"Processed {processed} {config.kind} catalog sources "
                    f"in shard {shard_index + 1}/{shard_count}.",
                    flush=True,
                )

    partial = fastgb_xyz_frequency_buffers_to_aet_channels(buffers, observation)
    return {
        "shard_index": shard_index,
        "processed": processed,
        "partial": partial,
        "metadata_part_path": metadata_part_path,
    }


def _generate_minimal_aet_shard(args: tuple[int, int, MinimalCatalogAETConfig, int, int, str | None]) -> dict[str, Any]:
    shard_index, shard_count, config, selection_seed, generation_seed, metadata_part_path = args
    if config.kind == "dwd":
        return _generate_minimal_dwd_fastgb_aet_shard(args)

    observation = config.observation
    partial = {channel: np.zeros(observation.num_samples, dtype=np.float64) for channel in ("A", "E", "T")}
    factory = get_source_factory(config.kind)
    selection_rng = np.random.default_rng(selection_seed)
    generation_rng = np.random.default_rng(generation_seed)
    entries = _iter_selected_entries(config.kind, config.catalog, selection_rng)

    processed = 0
    visited = 0
    if metadata_part_path is None:
        metadata_handle_context: Any = nullcontext(None)
    else:
        metadata_handle_context = Path(metadata_part_path).open("w", encoding="utf-8")

    with metadata_handle_context as metadata_handle:
        for entry in entries:
            visited += 1
            parameters = _source_parameters_from_entry(config.kind, entry, config, generation_rng)
            if (visited - 1) % shard_count != shard_index:
                continue
            try:
                generated = factory.generate(parameters, observation)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to generate {config.kind} source #{visited} from {_entry_label(entry)} "
                    f"in shard {shard_index + 1}/{shard_count}."
                ) from exc
            for channel in ("A", "E", "T"):
                partial[channel] += np.asarray(generated.channels[channel], dtype=np.float64)
            processed += 1
            if metadata_handle is not None:
                _write_jsonl_line(
                    metadata_handle,
                    _source_metadata_payload(
                        kind=config.kind,
                        source_index=visited,
                        entry=entry,
                        source_parameters=parameters,
                        generated=generated,
                    ),
                )
            if config.progress_interval and processed % config.progress_interval == 0:
                print(
                    f"Processed {processed} {config.kind} catalog sources "
                    f"in shard {shard_index + 1}/{shard_count}.",
                    flush=True,
                )

    return {
        "shard_index": shard_index,
        "processed": processed,
        "partial": partial,
        "metadata_part_path": metadata_part_path,
    }


def _multiprocessing_context() -> mp.context.BaseContext:
    method = os.environ.get("MINIMAL_AET_MP_START_METHOD", "").strip()
    if method:
        return mp.get_context(method)
    return mp.get_context()


def _build_minimal_catalog_aet_parallel(
    config: MinimalCatalogAETConfig,
    raw_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    root_sequence = np.random.SeedSequence(config.seed)
    selection_sequence, generation_sequence = root_sequence.spawn(2)
    selection_seed = _child_seed(selection_sequence)
    generation_seed = _child_seed(generation_sequence)
    observation = config.observation
    time_s = observation.time_array()
    summed = {channel: np.zeros_like(time_s, dtype=np.float64) for channel in ("A", "E", "T")}
    processed = 0

    writer = _metadata_writer(
        config,
        raw_config,
        selection_seed=selection_seed,
        generation_seed=generation_seed,
    )
    metadata_output_parent = (
        Path(config.source_metadata_output.path).parent if config.source_metadata_output is not None else None
    )
    temp_dir_context: Any
    if metadata_output_parent is None:
        temp_dir_context = nullcontext(None)
    else:
        metadata_output_parent.mkdir(parents=True, exist_ok=True)
        temp_dir_context = TemporaryDirectory(
            dir=metadata_output_parent,
            prefix=f".{Path(config.source_metadata_output.path).name}.parts.",
        )

    with temp_dir_context as temp_dir, writer if writer is not None else nullcontext():
        tasks: list[tuple[int, int, MinimalCatalogAETConfig, int, int, str | None]] = []
        for shard_index in range(config.workers):
            part_path = None
            if temp_dir is not None:
                part_path = str(Path(temp_dir) / f"shard_{shard_index:05d}.jsonl")
            tasks.append(
                (
                    shard_index,
                    config.workers,
                    config,
                    selection_seed,
                    generation_seed,
                    part_path,
                )
            )

        metadata_results: list[dict[str, Any]] = []
        context = _multiprocessing_context()
        with context.Pool(processes=config.workers) as pool:
            for result in pool.imap_unordered(_generate_minimal_aet_shard, tasks):
                partial = result["partial"]
                for channel in ("A", "E", "T"):
                    summed[channel] += np.asarray(partial[channel], dtype=np.float64)
                processed += int(result["processed"])
                metadata_results.append(
                    {
                        "shard_index": result["shard_index"],
                        "metadata_part_path": result["metadata_part_path"],
                    }
                )
                print(
                    f"Finished shard {int(result['shard_index']) + 1}/{config.workers}: "
                    f"{int(result['processed'])} {config.kind} catalog sources.",
                    flush=True,
                )
                del partial
                del result
            print("All shard results received; closing worker pool.", flush=True)

        print(
            f"Worker pool closed; merged {processed} {config.kind} catalog sources into memory.",
            flush=True,
        )

        if processed == 0:
            raise ValueError(f"No {config.kind} catalog entries were selected.")
        if writer is not None:
            print(f"Merging source metadata parts into {config.source_metadata_output.path}.", flush=True)
            for result in sorted(metadata_results, key=lambda item: int(item["shard_index"])):
                part_path = result.get("metadata_part_path")
                if part_path is not None:
                    writer.append_jsonl_file(part_path)
            writer.write_summary(processed=processed)
            print("Finished source metadata merge.", flush=True)

    return time_s, summed, processed


def _build_minimal_dwd_fastgb_aet_serial(
    config: MinimalCatalogAETConfig,
    raw_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    root_sequence = np.random.SeedSequence(config.seed)
    selection_sequence, generation_sequence = root_sequence.spawn(2)
    selection_seed = _child_seed(selection_sequence)
    generation_seed = _child_seed(generation_sequence)
    selection_rng = np.random.default_rng(selection_seed)
    generation_rng = np.random.default_rng(generation_seed)

    observation = config.observation
    time_s = observation.time_array()
    buffers = empty_fastgb_xyz_frequency_buffers(observation)
    factory = DWDSourceFactory()

    processed = 0
    entries = _iter_selected_entries(config.kind, config.catalog, selection_rng)
    writer = _metadata_writer(
        config,
        raw_config,
        selection_seed=selection_seed,
        generation_seed=generation_seed,
    )
    with writer if writer is not None else nullcontext():
        for entry in entries:
            processed += 1
            parameters = _source_parameters_from_entry(config.kind, entry, config, generation_rng)
            try:
                prepared, catalog_parameterization = add_dwd_to_fastgb_frequency_buffers(
                    factory,
                    parameters,
                    observation,
                    buffers,
                )
                generated = factory.make_fastgb_result(
                    {},
                    prepared,
                    observation,
                    catalog_parameterization=catalog_parameterization,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to generate {config.kind} source #{processed} from {_entry_label(entry)}."
                ) from exc
            if writer is not None:
                writer.write_source(
                    _source_metadata_payload(
                        kind=config.kind,
                        source_index=processed,
                        entry=entry,
                        source_parameters=parameters,
                        generated=generated,
                    )
                )
            if config.progress_interval and processed % config.progress_interval == 0:
                print(f"Processed {processed} {config.kind} catalog sources.")

        if processed == 0:
            raise ValueError(f"No {config.kind} catalog entries were selected.")
        if writer is not None:
            writer.write_summary(processed=processed)

    return time_s, fastgb_xyz_frequency_buffers_to_aet_channels(buffers, observation), processed


def _build_minimal_catalog_aet_serial(
    config: MinimalCatalogAETConfig,
    raw_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    if config.kind == "dwd":
        return _build_minimal_dwd_fastgb_aet_serial(config, raw_config)

    root_sequence = np.random.SeedSequence(config.seed)
    selection_sequence, generation_sequence = root_sequence.spawn(2)
    selection_seed = _child_seed(selection_sequence)
    generation_seed = _child_seed(generation_sequence)
    selection_rng = np.random.default_rng(selection_seed)
    generation_rng = np.random.default_rng(generation_seed)

    observation = config.observation
    time_s = observation.time_array()
    summed = {channel: np.zeros_like(time_s, dtype=np.float64) for channel in ("A", "E", "T")}
    factory = get_source_factory(config.kind)

    processed = 0
    entries = _iter_selected_entries(config.kind, config.catalog, selection_rng)
    writer = _metadata_writer(
        config,
        raw_config,
        selection_seed=selection_seed,
        generation_seed=generation_seed,
    )
    with writer if writer is not None else nullcontext():
        for entry in entries:
            processed += 1
            parameters = _source_parameters_from_entry(config.kind, entry, config, generation_rng)
            try:
                generated = factory.generate(parameters, observation)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to generate {config.kind} source #{processed} from {_entry_label(entry)}."
                ) from exc
            for channel in ("A", "E", "T"):
                summed[channel] += np.asarray(generated.channels[channel], dtype=np.float64)
            if writer is not None:
                writer.write_source(
                    _source_metadata_payload(
                        kind=config.kind,
                        source_index=processed,
                        entry=entry,
                        source_parameters=parameters,
                        generated=generated,
                    )
                )
            if config.progress_interval and processed % config.progress_interval == 0:
                print(f"Processed {processed} {config.kind} catalog sources.")

        if processed == 0:
            raise ValueError(f"No {config.kind} catalog entries were selected.")
        if writer is not None:
            writer.write_summary(processed=processed)
    return time_s, summed, processed


def build_minimal_catalog_aet(
    config: MinimalCatalogAETConfig,
    raw_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    if config.workers <= 1:
        return _build_minimal_catalog_aet_serial(config, raw_config)
    return _build_minimal_catalog_aet_parallel(config, raw_config)


def run_minimal_catalog_aet(
    config: MinimalCatalogAETConfig,
    raw_config: Mapping[str, Any] | None = None,
) -> Path:
    time_s, summed, processed = build_minimal_catalog_aet(config, raw_config)
    print(f"Saving {config.kind} signal-only minimal A/E/T file to {config.output.path}.", flush=True)
    output_path = save_minimal_aet_hdf5(
        config.output,
        time_s=time_s,
        a=summed["A"],
        e=summed["E"],
        t=summed["T"],
    )
    print(f"Wrote {config.kind} signal-only minimal A/E/T file to {output_path}")
    print(f"Catalog sources processed: {processed}")
    if config.source_metadata_output is not None:
        print(f"Wrote source metadata JSONL to {config.source_metadata_output.path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a signal-only minimal HDF5 file with one /data dataset containing time,a,e,t."
    )
    parser.add_argument("--config", required=True, help="Path to a minimal catalog A/E/T JSON config.")
    parser.add_argument("--output", help="Override output.path from the config.")
    parser.add_argument("--source-metadata-output", help="Override source_metadata_output.path from the config.")
    parser.add_argument("--no-source-metadata-output", action="store_true", help="Do not write source metadata JSONL.")
    parser.add_argument("--max-sources", type=int, help="Override catalog.max_sources for a short smoke run.")
    parser.add_argument("--workers", type=int, help="Number of source-generation worker processes.")
    parser.add_argument("--dry-run", action="store_true", help="Parse the config and print what would be generated.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config, raw = load_minimal_catalog_aet_config(args.config)
    if args.output:
        config = replace(config, output=replace(config.output, path=args.output))
    if args.no_source_metadata_output:
        config = replace(config, source_metadata_output=None)
    elif args.source_metadata_output:
        if config.source_metadata_output is None:
            config = replace(
                config,
                source_metadata_output=SourceMetadataOutputConfig(path=args.source_metadata_output, overwrite=True),
            )
        else:
            config = replace(
                config,
                source_metadata_output=replace(config.source_metadata_output, path=args.source_metadata_output),
            )
    if args.max_sources is not None:
        if args.max_sources <= 0:
            raise ValueError("--max-sources must be positive.")
        config = replace(config, catalog=replace(config.catalog, max_sources=args.max_sources))
    if args.workers is not None:
        if args.workers <= 0:
            raise ValueError("--workers must be positive.")
        config = replace(config, workers=args.workers)

    if args.dry_run:
        print(f"kind: {config.kind}")
        print(f"paths: {len(config.catalog.paths)}")
        print(f"select_all: {config.catalog.select_all}")
        print(f"row_numbers: {list(config.catalog.row_numbers)}")
        print(f"rows_per_file: {config.catalog.rows_per_file}")
        print(f"max_sources: {config.catalog.max_sources}")
        print(f"num_samples: {config.observation.num_samples}")
        print(f"sample_spacing_s: {config.observation.sample_spacing_s}")
        print(f"workers: {config.workers}")
        print(f"output: {config.output.path}")
        print(
            "source_metadata_output: "
            f"{None if config.source_metadata_output is None else config.source_metadata_output.path}"
        )
        return 0

    run_minimal_catalog_aet(config, raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
