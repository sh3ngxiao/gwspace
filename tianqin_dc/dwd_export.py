from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np

from tianqin_dc.config import DatasetConfig, ObservationConfig
from tianqin_dc.dwd_catalog import DWDCatalogEntry, iter_dwd_catalog
from tianqin_dc.response import generate_tdi_xyz_td
from tianqin_dc.sources.dwd import CatalogDWDWaveform, DWDSourceFactory


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{field_name}' to be a mapping, got {type(value).__name__}.")
    return value


def _json_dataset(group: h5py.Group, name: str, payload: Any) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=json.dumps(payload, indent=2, sort_keys=True), dtype=dtype)


def _array_dataset(
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    *,
    compression: str | None,
    compression_level: int,
) -> None:
    kwargs: dict[str, Any] = {}
    if compression:
        kwargs["compression"] = compression
        kwargs["compression_opts"] = compression_level
    group.create_dataset(name, data=data, **kwargs)


def _child_seed(seed_sequence: np.random.SeedSequence) -> int:
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


@dataclass(frozen=True)
class SimpleOutputConfig:
    path: str
    overwrite: bool = False
    compression: str | None = "gzip"
    compression_level: int = 4

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "SimpleOutputConfig":
        data = _mapping(value, field_name="output")
        return cls(
            path=str(data["path"]),
            overwrite=bool(data.get("overwrite", False)),
            compression=data.get("compression", "gzip"),
            compression_level=int(data.get("compression_level", 4)),
        )


@dataclass(frozen=True)
class DWDCatalogSelectionConfig:
    paths: tuple[str, ...]
    row_numbers: tuple[int, ...] = tuple()
    selection: str = "first"
    rows_per_file: int | None = None
    max_sources: int | None = None

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "DWDCatalogSelectionConfig":
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

        deduplicated_paths = tuple(dict.fromkeys(resolved_paths))

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

        if not row_numbers and rows_per_file is None:
            raise ValueError(
                "DWD catalog configs must set either 'catalog.row_numbers' or 'catalog.rows_per_file' "
                "to avoid accidentally loading the full multi-million-row source table."
            )

        return cls(
            paths=deduplicated_paths,
            row_numbers=row_numbers,
            selection=selection,
            rows_per_file=rows_per_file,
            max_sources=max_sources,
        )


@dataclass(frozen=True)
class SimpleDWDConfig:
    dataset: DatasetConfig
    seed: int
    observation: ObservationConfig
    output: SimpleOutputConfig
    catalog: DWDCatalogSelectionConfig

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> "SimpleDWDConfig":
        dataset = DatasetConfig.from_config(data.get("dataset"))
        observation = ObservationConfig.from_config(_mapping(data["observation"], field_name="observation"))
        output = SimpleOutputConfig.from_config(_mapping(data["output"], field_name="output"))
        catalog = DWDCatalogSelectionConfig.from_config(_mapping(data["catalog"], field_name="catalog"))
        seed = int(data.get("seed", 123456789))
        return cls(
            dataset=dataset,
            seed=seed,
            observation=observation,
            output=output,
            catalog=catalog,
        )


@dataclass(frozen=True)
class SimpleDWDBundle:
    time_s: np.ndarray
    tdi_xyz: dict[str, np.ndarray]
    selected_sources: list[dict[str, Any]]
    run_config: dict[str, Any]


def load_simple_dwd_config(path: str | Path) -> tuple[SimpleDWDConfig, dict[str, Any]]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a JSON object.")
    return SimpleDWDConfig.from_config(raw), raw


def _select_subset[T](items: list[T], count: int, selection: str, rng: np.random.Generator) -> list[T]:
    if count >= len(items):
        return list(items)
    if selection == "first":
        return list(items[:count])
    indices = np.sort(rng.choice(len(items), size=count, replace=False))
    return [items[int(index)] for index in indices]


def _reservoir_sample[T](
    iterable: Iterable[T],
    count: int,
    rng: np.random.Generator,
) -> list[T]:
    reservoir: list[T] = []
    for index, item in enumerate(iterable):
        if index < count:
            reservoir.append(item)
            continue
        slot = int(rng.integers(0, index + 1))
        if slot < count:
            reservoir[slot] = item
    return reservoir


def _resolve_row_numbers(path: str, row_numbers: tuple[int, ...]) -> list[DWDCatalogEntry]:
    wanted = set(row_numbers)
    found: dict[int, DWDCatalogEntry] = {}

    for entry in iter_dwd_catalog(path):
        if entry.row_number in wanted:
            found[entry.row_number] = entry
            if len(found) == len(wanted):
                break

    missing = [row_number for row_number in row_numbers if row_number not in found]
    if missing:
        raise IndexError(f"Catalog '{path}' does not contain requested row numbers: {missing}.")
    return [found[row_number] for row_number in row_numbers]


def _resolve_rows_per_file(
    path: str,
    rows_per_file: int,
    selection: str,
    rng: np.random.Generator,
) -> list[DWDCatalogEntry]:
    iterator = iter_dwd_catalog(path)
    if selection == "first":
        selected: list[DWDCatalogEntry] = []
        for entry in iterator:
            selected.append(entry)
            if len(selected) >= rows_per_file:
                break
        return selected
    return _reservoir_sample(iterator, rows_per_file, rng)


def _resolve_catalog_entries(
    selection: DWDCatalogSelectionConfig,
    rng: np.random.Generator,
) -> list[DWDCatalogEntry]:
    selected: list[DWDCatalogEntry] = []

    for path in selection.paths:
        if selection.row_numbers:
            file_entries = _resolve_row_numbers(path, selection.row_numbers)
        else:
            if selection.rows_per_file is None:
                raise ValueError("Internal error: rows_per_file is required when row_numbers is empty.")
            file_entries = _resolve_rows_per_file(path, selection.rows_per_file, selection.selection, rng)

        if not file_entries:
            raise ValueError(f"No rows were selected from DWD catalog '{path}'.")
        selected.extend(file_entries)

    if selection.max_sources is not None:
        selected = _select_subset(selected, selection.max_sources, selection.selection, rng)

    return selected


def build_simple_dwd_bundle(config: SimpleDWDConfig, raw_config: dict[str, Any]) -> SimpleDWDBundle:
    root_sequence = np.random.SeedSequence(config.seed)
    selection_sequence, generation_sequence = root_sequence.spawn(2)
    selection_rng = np.random.default_rng(_child_seed(selection_sequence))
    selected_entries = _resolve_catalog_entries(config.catalog, selection_rng)

    observation = config.observation
    time_s = observation.time_array()
    tdi_xyz = {channel: np.zeros_like(time_s, dtype=np.float64) for channel in ("X", "Y", "Z")}

    factory = DWDSourceFactory()
    source_sequences = generation_sequence.spawn(len(selected_entries))
    source_records: list[dict[str, Any]] = []

    for entry, source_sequence in zip(selected_entries, source_sequences, strict=True):
        source_seed = _child_seed(source_sequence)
        source_parameters = entry.to_source_parameters()
        prepared_parameters = factory.prepare_parameters(source_parameters, observation)
        waveform = CatalogDWDWaveform(**prepared_parameters)
        channels = generate_tdi_xyz_td(waveform, time_s, observation)

        for channel, series in channels.items():
            tdi_xyz[channel] += np.asarray(series, dtype=np.float64)

        source_records.append(
            {
                "seed": source_seed,
                "catalog_entry": entry.to_mapping(),
                "waveform_parameters": deepcopy(prepared_parameters),
            }
        )

    return SimpleDWDBundle(
        time_s=time_s,
        tdi_xyz=tdi_xyz,
        selected_sources=source_records,
        run_config=deepcopy(raw_config),
    )


def save_simple_dwd_hdf5(bundle: SimpleDWDBundle, config: SimpleDWDConfig) -> Path:
    output_path = Path(config.output.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Output file '{output_path}' already exists. Set output.overwrite=true to replace it."
        )

    with h5py.File(output_path, "w") as handle:
        handle.attrs["dataset_name"] = config.dataset.name
        handle.attrs["dataset_description"] = config.dataset.description
        handle.attrs["format_name"] = "tianqin-dc-dwd-simple"
        handle.attrs["format_version"] = "0.1.0"
        handle.attrs["source_class"] = "dwd"
        handle.attrs["detector"] = config.observation.detector
        handle.attrs["tdi_generation"] = config.observation.tdi_generation
        handle.attrs["sample_rate_hz"] = config.observation.sample_rate_hz
        handle.attrs["sample_spacing_s"] = config.observation.sample_spacing_s
        handle.attrs["effective_duration_s"] = config.observation.effective_duration_s
        handle.attrs["num_samples"] = config.observation.num_samples
        handle.attrs["num_sources"] = len(bundle.selected_sources)
        handle.attrs["num_catalog_files"] = len(
            {record["catalog_entry"]["file_path"] for record in bundle.selected_sources}
        )
        handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

        _array_dataset(
            handle,
            "time",
            bundle.time_s,
            compression=config.output.compression,
            compression_level=config.output.compression_level,
        )

        tdi_group = handle.create_group("tdi")
        for channel in ("X", "Y", "Z"):
            _array_dataset(
                tdi_group,
                channel,
                bundle.tdi_xyz[channel],
                compression=config.output.compression,
                compression_level=config.output.compression_level,
            )

        meta_group = handle.create_group("meta")
        _json_dataset(meta_group, "sources_json", bundle.selected_sources)
        _json_dataset(meta_group, "config_json", bundle.run_config)

    return output_path


class SimpleDWDBuilder:
    def __init__(self, config: SimpleDWDConfig, raw_config: dict[str, Any]) -> None:
        self.config = config
        self.raw_config = deepcopy(raw_config)

    def build(self) -> SimpleDWDBundle:
        return build_simple_dwd_bundle(self.config, self.raw_config)

    def build_and_save(self) -> Path:
        bundle = self.build()
        return save_simple_dwd_hdf5(bundle, self.config)
