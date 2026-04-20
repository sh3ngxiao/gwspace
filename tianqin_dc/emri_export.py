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

from gwspace.Waveform import waveforms

from tianqin_dc.config import DatasetConfig, ObservationConfig, SamplerConfig
from tianqin_dc.emri_catalog import EMRICatalogEntry, load_emri_catalog
from tianqin_dc.response import generate_tdi_xyz_td
from tianqin_dc.sampling import sample_value
from tianqin_dc.sources.emri import EMRISourceFactory


_TWO_PI = float(2.0 * np.pi)
_DEFAULT_SAMPLER_CONFIGS: dict[str, Any] = {
    "e0": {"distribution": "uniform", "low": 0.0, "high": 0.2},
    "qS": {"distribution": "isotropic_polar"},
    "phiS": {"distribution": "uniform", "low": 0.0, "high": _TWO_PI},
    "qK": {"distribution": "isotropic_polar"},
    "phiK": {"distribution": "uniform", "low": 0.0, "high": _TWO_PI},
    "Phi_phi0": {"distribution": "uniform", "low": 0.0, "high": _TWO_PI},
    "Phi_theta0": {"distribution": "uniform", "low": 0.0, "high": _TWO_PI},
    "Phi_r0": {"distribution": "uniform", "low": 0.0, "high": _TWO_PI},
}
_DEFAULT_FIXED_PARAMETERS: dict[str, Any] = {
    "p0": 12.0,
    "backend": "cpu",
}


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
class EMRICatalogSelectionConfig:
    paths: tuple[str, ...]
    row_numbers: tuple[int, ...] = tuple()
    selection: str = "first"
    rows_per_file: int | None = None
    max_sources: int | None = None

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "EMRICatalogSelectionConfig":
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

        # Preserve the first occurrence order while removing duplicates.
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

        return cls(
            paths=deduplicated_paths,
            row_numbers=row_numbers,
            selection=selection,
            rows_per_file=rows_per_file,
            max_sources=max_sources,
        )


@dataclass(frozen=True)
class EMRICompletionConfig:
    fixed: dict[str, Any] = field(default_factory=dict)
    sampler: dict[str, SamplerConfig] = field(default_factory=dict)
    use_catalog_inclination_for_x0: bool = True

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "EMRICompletionConfig":
        data = {} if value is None else _maybe_mapping(value, field_name="emri")
        fixed = deepcopy(_DEFAULT_FIXED_PARAMETERS)
        fixed.update(_maybe_mapping(data.get("fixed"), field_name="emri.fixed"))

        sampler = {
            name: SamplerConfig.from_config(spec) for name, spec in deepcopy(_DEFAULT_SAMPLER_CONFIGS).items()
        }
        sampler.update(
            {
                name: SamplerConfig.from_config(spec)
                for name, spec in _maybe_mapping(data.get("sampler"), field_name="emri.sampler").items()
            }
        )

        use_catalog_inclination_for_x0 = bool(data.get("use_catalog_inclination_for_x0", True))
        if not use_catalog_inclination_for_x0 and "x0" not in fixed and "x0" not in sampler:
            fixed["x0"] = 1.0

        for name in tuple(sampler):
            if name in fixed:
                sampler.pop(name)

        return cls(
            fixed=fixed,
            sampler=sampler,
            use_catalog_inclination_for_x0=use_catalog_inclination_for_x0,
        )


@dataclass(frozen=True)
class SimpleEMRIConfig:
    dataset: DatasetConfig
    seed: int
    observation: ObservationConfig
    output: SimpleOutputConfig
    catalog: EMRICatalogSelectionConfig
    emri: EMRICompletionConfig

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> "SimpleEMRIConfig":
        dataset = DatasetConfig.from_config(data.get("dataset"))
        observation = ObservationConfig.from_config(_mapping(data["observation"], field_name="observation"))
        output = SimpleOutputConfig.from_config(_mapping(data["output"], field_name="output"))
        catalog = EMRICatalogSelectionConfig.from_config(_mapping(data["catalog"], field_name="catalog"))
        emri = EMRICompletionConfig.from_config(data.get("emri"))
        seed = int(data.get("seed", 123456789))
        return cls(
            dataset=dataset,
            seed=seed,
            observation=observation,
            output=output,
            catalog=catalog,
            emri=emri,
        )


@dataclass(frozen=True)
class SimpleEMRIBundle:
    time_s: np.ndarray
    tdi_xyz: dict[str, np.ndarray]
    selected_sources: list[dict[str, Any]]
    run_config: dict[str, Any]


def load_simple_emri_config(path: str | Path) -> tuple[SimpleEMRIConfig, dict[str, Any]]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a JSON object.")
    return SimpleEMRIConfig.from_config(raw), raw


def _select_subset[T](items: list[T], count: int, selection: str, rng: np.random.Generator) -> list[T]:
    if count >= len(items):
        return list(items)
    if selection == "first":
        return list(items[:count])
    indices = np.sort(rng.choice(len(items), size=count, replace=False))
    return [items[int(index)] for index in indices]


def _resolve_catalog_entries(
    selection: EMRICatalogSelectionConfig,
    rng: np.random.Generator,
) -> list[EMRICatalogEntry]:
    selected: list[EMRICatalogEntry] = []

    for path in selection.paths:
        entries = load_emri_catalog(path)
        if selection.row_numbers:
            file_entries: list[EMRICatalogEntry] = []
            for row_number in selection.row_numbers:
                try:
                    file_entries.append(entries[row_number - 1])
                except IndexError as exc:
                    raise IndexError(
                        f"Requested row_number={row_number} but catalog '{path}' has only {len(entries)} rows."
                    ) from exc
        elif selection.rows_per_file is not None:
            file_entries = _select_subset(entries, selection.rows_per_file, selection.selection, rng)
        else:
            file_entries = list(entries)

        selected.extend(file_entries)

    if not selected:
        raise ValueError("No EMRI catalog entries were selected. Check the catalog config and selection limits.")

    if selection.max_sources is not None:
        selected = _select_subset(selected, selection.max_sources, selection.selection, rng)

    return selected


def _parameters_from_catalog_entry(
    entry: EMRICatalogEntry,
    completion: EMRICompletionConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "M": entry.mbh_mass_msun,
        "mu": entry.compact_object_mass_msun,
        "a": entry.mbh_spin,
        "dist": entry.distance_gpc,
    }
    if completion.use_catalog_inclination_for_x0 and "x0" not in completion.fixed and "x0" not in completion.sampler:
        parameters["x0"] = float(np.clip(np.cos(entry.inclination), -1.0, 1.0))

    parameters.update(deepcopy(completion.fixed))
    for name, spec in completion.sampler.items():
        parameters[name] = sample_value(spec, rng)
    return parameters


def build_simple_emri_bundle(config: SimpleEMRIConfig, raw_config: dict[str, Any]) -> SimpleEMRIBundle:
    root_sequence = np.random.SeedSequence(config.seed)
    selection_sequence, generation_sequence = root_sequence.spawn(2)
    selection_rng = np.random.default_rng(_child_seed(selection_sequence))
    selected_entries = _resolve_catalog_entries(config.catalog, selection_rng)
    observation = config.observation
    time_s = observation.time_array()
    tdi_xyz = {channel: np.zeros_like(time_s, dtype=np.float64) for channel in ("X", "Y", "Z")}

    factory = EMRISourceFactory()
    source_sequences = generation_sequence.spawn(len(selected_entries))
    source_records: list[dict[str, Any]] = []

    for entry, source_sequence in zip(selected_entries, source_sequences, strict=True):
        source_seed = _child_seed(source_sequence)
        rng = np.random.default_rng(source_seed)
        source_parameters = _parameters_from_catalog_entry(entry, config.emri, rng)
        prepared_parameters = factory.prepare_parameters(source_parameters, observation)
        waveform = waveforms["emri"](**prepared_parameters)
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

    return SimpleEMRIBundle(
        time_s=time_s,
        tdi_xyz=tdi_xyz,
        selected_sources=source_records,
        run_config=deepcopy(raw_config),
    )


def save_simple_emri_hdf5(bundle: SimpleEMRIBundle, config: SimpleEMRIConfig) -> Path:
    output_path = Path(config.output.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not config.output.overwrite:
        raise FileExistsError(
            f"Output file '{output_path}' already exists. Set output.overwrite=true to replace it."
        )

    with h5py.File(output_path, "w") as handle:
        handle.attrs["dataset_name"] = config.dataset.name
        handle.attrs["dataset_description"] = config.dataset.description
        handle.attrs["format_name"] = "tianqin-dc-emri-simple"
        handle.attrs["format_version"] = "0.1.0"
        handle.attrs["source_class"] = "emri"
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


class SimpleEMRIBuilder:
    def __init__(self, config: SimpleEMRIConfig, raw_config: dict[str, Any]) -> None:
        self.config = config
        self.raw_config = deepcopy(raw_config)

    def build(self) -> SimpleEMRIBundle:
        return build_simple_emri_bundle(self.config, self.raw_config)

    def build_and_save(self) -> Path:
        bundle = self.build()
        return save_simple_emri_hdf5(bundle, self.config)
