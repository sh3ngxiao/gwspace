from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from gwspace.constants import YRSID_SI

from tianqin_dc.bbh_catalog import BBHCatalogEntry, iter_bbh_catalog, redshift_to_luminosity_distance_mpc
from tianqin_dc.config import DatasetConfig, ObservationConfig, SamplerConfig
from tianqin_dc.response import generate_tdi_xyz_fd
from tianqin_dc.sampling import sample_value
from tianqin_dc.sources.compact_binary import SBBHSourceFactory


_TWO_PI = float(2.0 * np.pi)
_DEFAULT_SAMPLER_CONFIGS: dict[str, Any] = {
    "Lambda": {"distribution": "uniform", "low": 0.0, "high": _TWO_PI},
    "Beta": {"distribution": "isotropic_latitude"},
    "iota": {"distribution": "isotropic_polar"},
    "var_phi": {"distribution": "uniform", "low": 0.0, "high": _TWO_PI},
    "phi_c": {"distribution": "uniform", "low": 0.0, "high": _TWO_PI},
}
_DEFAULT_FIXED_PARAMETERS: dict[str, Any] = {
    "engine": "bhb_EccFD",
    "eccentricity": 0.0,
}
_FALLBACK_ENGINE_EXCEPTIONS = (
    ImportError,
    ModuleNotFoundError,
    AttributeError,
    NameError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{field_name}' to be a mapping, got {type(value).__name__}.")
    return value


def _maybe_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    return dict(_mapping(value, field_name=field_name))


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
class BBHCatalogSelectionConfig:
    paths: tuple[str, ...]
    row_numbers: tuple[int, ...] = tuple()
    selection: str = "first"
    rows_per_file: int | None = None
    max_sources: int | None = None

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "BBHCatalogSelectionConfig":
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
                "BBH catalog configs must set either 'catalog.row_numbers' or 'catalog.rows_per_file' "
                "to avoid accidentally loading the full source table."
            )

        return cls(
            paths=deduplicated_paths,
            row_numbers=row_numbers,
            selection=selection,
            rows_per_file=rows_per_file,
            max_sources=max_sources,
        )


@dataclass(frozen=True)
class BBHCompletionConfig:
    fixed: dict[str, Any] = field(default_factory=dict)
    sampler: dict[str, SamplerConfig] = field(default_factory=dict)
    use_catalog_psi_for_polarization: bool = True
    seconds_per_year: float = float(YRSID_SI)

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "BBHCompletionConfig":
        data = {} if value is None else _maybe_mapping(value, field_name="bbh")
        fixed = deepcopy(_DEFAULT_FIXED_PARAMETERS)
        fixed.update(_maybe_mapping(data.get("fixed"), field_name="bbh.fixed"))

        sampler = {
            name: SamplerConfig.from_config(spec) for name, spec in deepcopy(_DEFAULT_SAMPLER_CONFIGS).items()
        }
        sampler.update(
            {
                name: SamplerConfig.from_config(spec)
                for name, spec in _maybe_mapping(data.get("sampler"), field_name="bbh.sampler").items()
            }
        )

        use_catalog_psi_for_polarization = bool(data.get("use_catalog_psi_for_polarization", True))
        seconds_per_year = float(data.get("seconds_per_year", YRSID_SI))
        if seconds_per_year <= 0.0:
            raise ValueError("Config field 'bbh.seconds_per_year' must be positive.")

        for name in tuple(sampler):
            if name in fixed:
                sampler.pop(name)

        return cls(
            fixed=fixed,
            sampler=sampler,
            use_catalog_psi_for_polarization=use_catalog_psi_for_polarization,
            seconds_per_year=seconds_per_year,
        )


@dataclass(frozen=True)
class SimpleBBHConfig:
    dataset: DatasetConfig
    seed: int
    observation: ObservationConfig
    output: SimpleOutputConfig
    catalog: BBHCatalogSelectionConfig
    bbh: BBHCompletionConfig

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> "SimpleBBHConfig":
        dataset = DatasetConfig.from_config(data.get("dataset"))
        observation = ObservationConfig.from_config(_mapping(data["observation"], field_name="observation"))
        output = SimpleOutputConfig.from_config(_mapping(data["output"], field_name="output"))
        catalog = BBHCatalogSelectionConfig.from_config(_mapping(data["catalog"], field_name="catalog"))
        bbh = BBHCompletionConfig.from_config(data.get("bbh"))
        seed = int(data.get("seed", 123456789))
        return cls(
            dataset=dataset,
            seed=seed,
            observation=observation,
            output=output,
            catalog=catalog,
            bbh=bbh,
        )


@dataclass(frozen=True)
class SimpleBBHBundle:
    time_s: np.ndarray
    tdi_xyz: dict[str, np.ndarray]
    selected_sources: list[dict[str, Any]]
    run_config: dict[str, Any]


def load_simple_bbh_config(path: str | Path) -> tuple[SimpleBBHConfig, dict[str, Any]]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a JSON object.")
    return SimpleBBHConfig.from_config(raw), raw


def _select_subset(items: list[Any], count: int, selection: str, rng: np.random.Generator) -> list[Any]:
    if count >= len(items):
        return list(items)
    if selection == "first":
        return list(items[:count])
    indices = np.sort(rng.choice(len(items), size=count, replace=False))
    return [items[int(index)] for index in indices]


def _reservoir_sample(
    iterable: list[Any] | Any,
    count: int,
    rng: np.random.Generator,
) -> list[Any]:
    reservoir: list[Any] = []
    for index, item in enumerate(iterable):
        if index < count:
            reservoir.append(item)
            continue
        slot = int(rng.integers(0, index + 1))
        if slot < count:
            reservoir[slot] = item
    return reservoir


def _resolve_row_numbers(path: str, row_numbers: tuple[int, ...]) -> list[BBHCatalogEntry]:
    wanted = set(row_numbers)
    found: dict[int, BBHCatalogEntry] = {}

    for entry in iter_bbh_catalog(path):
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
) -> list[BBHCatalogEntry]:
    iterator = iter_bbh_catalog(path)
    if selection == "first":
        selected: list[BBHCatalogEntry] = []
        for entry in iterator:
            selected.append(entry)
            if len(selected) >= rows_per_file:
                break
        return selected
    return _reservoir_sample(iterator, rows_per_file, rng)


def _resolve_catalog_entries(
    selection: BBHCatalogSelectionConfig,
    rng: np.random.Generator,
) -> list[BBHCatalogEntry]:
    selected: list[BBHCatalogEntry] = []

    for path in selection.paths:
        if selection.row_numbers:
            file_entries = _resolve_row_numbers(path, selection.row_numbers)
        else:
            if selection.rows_per_file is None:
                raise ValueError("Internal error: rows_per_file is required when row_numbers is empty.")
            file_entries = _resolve_rows_per_file(path, selection.rows_per_file, selection.selection, rng)

        if not file_entries:
            raise ValueError(f"No rows were selected from BBH catalog '{path}'.")
        selected.extend(file_entries)

    if selection.max_sources is not None:
        selected = _select_subset(selected, selection.max_sources, selection.selection, rng)

    return selected


def _parameters_from_catalog_entry(
    entry: BBHCatalogEntry,
    distance_mpc: float,
    completion: BBHCompletionConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "mass1": entry.m1_msun,
        "mass2": entry.m2_msun,
        "DL": float(distance_mpc),
        "tc": float(entry.t_c_yr * completion.seconds_per_year),
    }

    if completion.use_catalog_psi_for_polarization and "psi" not in completion.fixed and "psi" not in completion.sampler:
        parameters["psi"] = entry.psi_rad

    parameters.update(deepcopy(completion.fixed))
    for name, spec in completion.sampler.items():
        parameters[name] = sample_value(spec, rng)
    parameters.setdefault("psi", 0.0)
    return parameters


def _build_bbh_waveform_and_channels(
    factory: SBBHSourceFactory,
    prepared_parameters: dict[str, Any],
    observation: ObservationConfig,
) -> tuple[dict[str, np.ndarray], str]:
    engine_request = str(prepared_parameters.get("engine", "auto"))
    last_error: Exception | None = None

    for engine in factory._resolve_engine_candidates(engine_request):
        try:
            waveform = factory._build_waveform(engine, prepared_parameters)
            channels = generate_tdi_xyz_fd(waveform, observation)
            return channels, engine
        except _FALLBACK_ENGINE_EXCEPTIONS as exc:
            last_error = exc
            continue

    if last_error is None:
        raise RuntimeError(f"Failed to resolve any compact-binary engine for request '{engine_request}'.")
    raise RuntimeError(
        f"BBH catalog source could not be generated with engine request '{engine_request}'."
    ) from last_error


def build_simple_bbh_bundle(config: SimpleBBHConfig, raw_config: dict[str, Any]) -> SimpleBBHBundle:
    root_sequence = np.random.SeedSequence(config.seed)
    selection_sequence, generation_sequence = root_sequence.spawn(2)
    selection_rng = np.random.default_rng(_child_seed(selection_sequence))
    selected_entries = _resolve_catalog_entries(config.catalog, selection_rng)

    observation = config.observation
    time_s = observation.time_array()
    tdi_xyz = {channel: np.zeros_like(time_s, dtype=np.float64) for channel in ("X", "Y", "Z")}

    factory = SBBHSourceFactory()
    source_sequences = generation_sequence.spawn(len(selected_entries))
    source_records: list[dict[str, Any]] = []
    luminosity_distances = redshift_to_luminosity_distance_mpc([entry.z for entry in selected_entries])

    for entry, source_sequence, distance_mpc in zip(
        selected_entries, source_sequences, luminosity_distances, strict=True
    ):
        source_seed = _child_seed(source_sequence)
        rng = np.random.default_rng(source_seed)
        source_parameters = _parameters_from_catalog_entry(entry, float(distance_mpc), config.bbh, rng)
        prepared_parameters = factory.prepare_parameters(source_parameters, observation)
        channels, engine_resolved = _build_bbh_waveform_and_channels(factory, prepared_parameters, observation)

        for channel, series in channels.items():
            tdi_xyz[channel] += np.asarray(series, dtype=np.float64)

        source_records.append(
            {
                "seed": source_seed,
                "catalog_entry": entry.to_mapping(),
                "luminosity_distance_Mpc": float(distance_mpc),
                "waveform_parameters": deepcopy(prepared_parameters),
                "engine_request": str(prepared_parameters.get("engine", "auto")),
                "engine_resolved": engine_resolved,
                "notes": factory._engine_specific_notes(engine_resolved, factory.kind),
            }
        )

    return SimpleBBHBundle(
        time_s=time_s,
        tdi_xyz=tdi_xyz,
        selected_sources=source_records,
        run_config=deepcopy(raw_config),
    )


def save_simple_bbh_hdf5(bundle: SimpleBBHBundle, config: SimpleBBHConfig) -> Path:
    output_path = Path(config.output.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Output file '{output_path}' already exists. Set output.overwrite=true to replace it."
        )

    with h5py.File(output_path, "w") as handle:
        handle.attrs["dataset_name"] = config.dataset.name
        handle.attrs["dataset_description"] = config.dataset.description
        handle.attrs["format_name"] = "tianqin-dc-bbh-simple"
        handle.attrs["format_version"] = "0.1.0"
        handle.attrs["source_class"] = "sbbh"
        handle.attrs["catalog_format"] = "bp2p-csv"
        handle.attrs["detector"] = config.observation.detector
        handle.attrs["tdi_generation"] = config.observation.tdi_generation
        handle.attrs["sample_rate_hz"] = config.observation.sample_rate_hz
        handle.attrs["sample_spacing_s"] = config.observation.sample_spacing_s
        handle.attrs["effective_duration_s"] = config.observation.effective_duration_s
        handle.attrs["num_samples"] = config.observation.num_samples
        handle.attrs["num_sources"] = len(bundle.selected_sources)
        handle.attrs["num_catalog_files"] = len({record["catalog_entry"]["file_path"] for record in bundle.selected_sources})
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


class SimpleBBHBuilder:
    def __init__(self, config: SimpleBBHConfig, raw_config: dict[str, Any]) -> None:
        self.config = config
        self.raw_config = deepcopy(raw_config)

    def build(self) -> SimpleBBHBundle:
        return build_simple_bbh_bundle(self.config, self.raw_config)

    def build_and_save(self) -> Path:
        bundle = self.build()
        return save_simple_bbh_hdf5(bundle, self.config)
