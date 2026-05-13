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

from tianqin_dc.config import DatasetConfig, ObservationConfig, SamplerConfig
from tianqin_dc.emri_catalog import EMRICatalogEntry, load_emri_catalog
from tianqin_dc.plotting import save_time_domain_preview
from tianqin_dc.response import generate_tdi_channels_td, generate_tdi_xyz_td
from tianqin_dc.sampling import sample_value
from tianqin_dc.sources.emri import EMRISourceFactory, build_interpolated_emri_waveform


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
_DEFAULT_PLACEMENT_TIME_SAMPLER: dict[str, Any] = {
    "distribution": "uniform",
    "low": 0.05,
    "high": 0.95,
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
class EMRIPlacementConfig:
    enabled: bool = False
    anchor: str = "peak"
    time_sampler: SamplerConfig = field(
        default_factory=lambda: SamplerConfig.from_config(deepcopy(_DEFAULT_PLACEMENT_TIME_SAMPLER))
    )
    time_unit: str = "fraction"
    allow_outside: bool = False
    threshold_fraction: float = 1e-6
    threshold_abs: float = 0.0

    @classmethod
    def from_config(cls, value: Any) -> "EMRIPlacementConfig":
        if value is None:
            return cls()
        if isinstance(value, bool):
            return cls(enabled=value)

        data = _mapping(value, field_name="emri.placement")
        anchor = str(data.get("anchor", "peak")).lower()
        if anchor not in {"start", "peak", "end", "trajectory_end", "plunge"}:
            raise ValueError(
                "Config field 'emri.placement.anchor' must be one of: start, peak, end, trajectory_end, plunge."
            )

        time_unit = str(data.get("time_unit", "fraction")).lower()
        if time_unit not in {"fraction", "duration_fraction", "seconds", "s"}:
            raise ValueError(
                "Config field 'emri.placement.time_unit' must be 'fraction' or 'seconds'."
            )

        sampler_raw = data.get("time_sampler", data.get("sampler", _DEFAULT_PLACEMENT_TIME_SAMPLER))
        threshold_fraction = float(data.get("threshold_fraction", 1e-6))
        threshold_abs = float(data.get("threshold_abs", 0.0))
        if threshold_fraction < 0.0:
            raise ValueError("Config field 'emri.placement.threshold_fraction' must be non-negative.")
        if threshold_abs < 0.0:
            raise ValueError("Config field 'emri.placement.threshold_abs' must be non-negative.")

        return cls(
            enabled=bool(data.get("enabled", True)),
            anchor=anchor,
            time_sampler=SamplerConfig.from_config(sampler_raw),
            time_unit=time_unit,
            allow_outside=bool(data.get("allow_outside", False)),
            threshold_fraction=threshold_fraction,
            threshold_abs=threshold_abs,
        )


@dataclass(frozen=True)
class EMRICompletionConfig:
    fixed: dict[str, Any] = field(default_factory=dict)
    sampler: dict[str, SamplerConfig] = field(default_factory=dict)
    use_catalog_inclination_for_x0: bool = True
    placement: EMRIPlacementConfig = field(default_factory=EMRIPlacementConfig)

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "EMRICompletionConfig":
        data = {} if value is None else _maybe_mapping(value, field_name="emri")
        explicit_fixed = _maybe_mapping(data.get("fixed"), field_name="emri.fixed")
        fixed = deepcopy(_DEFAULT_FIXED_PARAMETERS)
        fixed.update(explicit_fixed)

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
                if name in explicit_fixed:
                    sampler.pop(name)
                else:
                    fixed.pop(name)

        return cls(
            fixed=fixed,
            sampler=sampler,
            use_catalog_inclination_for_x0=use_catalog_inclination_for_x0,
            placement=EMRIPlacementConfig.from_config(data.get("placement")),
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


def sample_emri_placement_target_time_s(
    completion: EMRICompletionConfig,
    observation: ObservationConfig,
    rng: np.random.Generator,
) -> float | None:
    placement = completion.placement
    if not placement.enabled:
        return None

    value = float(sample_value(placement.time_sampler, rng))
    if placement.time_unit in {"fraction", "duration_fraction"}:
        target_time_s = value * observation.effective_duration_s
    else:
        target_time_s = value

    if not placement.allow_outside:
        upper = max(0.0, observation.effective_duration_s - observation.sample_spacing_s)
        target_time_s = float(np.clip(target_time_s, 0.0, upper))
    return float(target_time_s)


def apply_emri_waveform_time_placement(
    waveform: Any,
    observation: ObservationConfig,
    placement: EMRIPlacementConfig,
    target_time_s: float | None,
) -> dict[str, Any] | None:
    """Place an EMRI by shifting source waveform time before TDI response.

    Shifting already-generated TDI channels is not equivalent to moving the
    source in detector time: it also shifts response boundary artifacts and uses
    the wrong detector orbit time. This helper aligns the source waveform anchor
    to the requested target before GWspace evaluates delayed TDI samples.
    """

    if not placement.enabled or target_time_s is None:
        return None

    activity = waveform.source_activity(
        threshold_fraction=placement.threshold_fraction,
        threshold_abs=placement.threshold_abs,
    )
    trajectory_end = waveform.trajectory_end()
    trajectory_end_index = int(trajectory_end["trajectory_end_index"])
    anchor_indices = {
        "start": int(activity["original_active_start_index"]),
        "peak": int(activity["original_peak_index"]),
        "end": trajectory_end_index,
        "trajectory_end": trajectory_end_index,
        "plunge": trajectory_end_index,
    }
    anchor_index = anchor_indices[placement.anchor]
    target_index = int(round(float(target_time_s) / observation.sample_spacing_s))
    shift_samples = target_index - anchor_index
    shift_s = float(shift_samples * observation.sample_spacing_s)
    waveform.set_time_shift_s(shift_s)

    placed_active_start_index = int(activity["original_active_start_index"]) + shift_samples
    active_end_index = trajectory_end_index if placement.anchor in {"end", "trajectory_end", "plunge"} else int(
        activity["original_active_end_index"]
    )
    placed_active_end_index = active_end_index + shift_samples
    visible_active_start_index = max(0, placed_active_start_index)
    visible_active_end_index = min(observation.num_samples - 1, placed_active_end_index)
    visible_active_samples = max(0, visible_active_end_index - visible_active_start_index + 1)

    return {
        "enabled": True,
        "domain": "source_waveform_time",
        "anchor": placement.anchor,
        "anchor_source": "few_trajectory_end" if placement.anchor in {"end", "trajectory_end", "plunge"} else "waveform_amplitude",
        "allow_outside": bool(placement.allow_outside),
        "target_time_s": float(target_index * observation.sample_spacing_s),
        "target_index": target_index,
        "shift_samples": int(shift_samples),
        "shift_s": shift_s,
        "threshold": activity["threshold"],
        "threshold_fraction": activity["threshold_fraction"],
        "threshold_abs": activity["threshold_abs"],
        "original_peak_time_s": activity["original_peak_time_s"],
        "original_peak_index": activity["original_peak_index"],
        "original_peak_value": activity["original_peak_value"],
        "original_active_start_time_s": activity["original_active_start_time_s"],
        "original_active_end_time_s": float(active_end_index * observation.sample_spacing_s),
        "original_active_samples": int(active_end_index - int(activity["original_active_start_index"]) + 1),
        "waveform_threshold_active_end_time_s": activity["original_active_end_time_s"],
        "waveform_threshold_active_samples": activity["original_active_samples"],
        "trajectory_end": trajectory_end,
        "placed_peak_time_s": float((int(activity["original_peak_index"]) + shift_samples) * observation.sample_spacing_s),
        "placed_active_start_time_s": float(placed_active_start_index * observation.sample_spacing_s),
        "placed_active_end_time_s": float(placed_active_end_index * observation.sample_spacing_s),
        "visible_active_start_time_s": (
            None if visible_active_samples == 0 else float(visible_active_start_index * observation.sample_spacing_s)
        ),
        "visible_active_end_time_s": (
            None if visible_active_samples == 0 else float(visible_active_end_index * observation.sample_spacing_s)
        ),
        "visible_active_samples": int(visible_active_samples),
        "cropped_left_samples": int(max(0, -placed_active_start_index)),
        "cropped_right_samples": int(max(0, placed_active_end_index - (observation.num_samples - 1))),
    }


def generate_emri_aet_with_waveform_placement(
    factory: EMRISourceFactory,
    parameters: dict[str, Any],
    observation: ObservationConfig,
    placement: EMRIPlacementConfig,
    target_time_s: float | None,
) -> tuple[Any, dict[str, Any] | None]:
    prepared = factory.prepare_parameters(parameters, observation)
    waveform = build_interpolated_emri_waveform(prepared, observation)
    placement_diagnostics = apply_emri_waveform_time_placement(waveform, observation, placement, target_time_s)
    channels = generate_tdi_channels_td(waveform, observation.time_array(), observation)
    generated = factory.make_result(
        channels,
        prepared,
        notes=[
            "EMRI source waveform is cached on the observation grid and interpolated at delayed TDI sample times.",
            "EMRI time placement is applied to source waveform time before TDI response generation.",
        ],
        metadata={
            "few_backend_request": prepared.get("backend", "cpu"),
            "time_interpolation": "linear",
            "time_taper_duration_s": waveform.taper_duration_s,
            "time_placement_domain": "source_waveform_time" if placement_diagnostics is not None else None,
        },
    )
    return generated, placement_diagnostics


def apply_emri_time_placement(
    channels: Mapping[str, np.ndarray],
    observation: ObservationConfig,
    placement: EMRIPlacementConfig,
    target_time_s: float | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    if not placement.enabled or target_time_s is None:
        return {name: np.asarray(values, dtype=np.float64) for name, values in channels.items()}, None

    n_rows = observation.num_samples
    if n_rows <= 0:
        raise ValueError("Observation must contain at least one sample for EMRI placement.")

    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in channels.items()}
    amplitude = np.zeros(n_rows, dtype=np.float64)
    for name, values in arrays.items():
        if values.shape != (n_rows,):
            raise ValueError(f"Channel '{name}' has shape {values.shape}, expected {(n_rows,)}.")
        np.maximum(amplitude, np.abs(values), out=amplitude)

    peak_index = int(np.argmax(amplitude))
    peak_value = float(amplitude[peak_index])
    threshold = max(float(placement.threshold_abs), peak_value * float(placement.threshold_fraction))
    active = np.flatnonzero(amplitude > threshold)
    if active.size:
        active_start_index = int(active[0])
        active_end_index = int(active[-1])
    else:
        active_start_index = peak_index
        active_end_index = peak_index

    anchor_indices = {
        "start": active_start_index,
        "peak": peak_index,
        "end": active_end_index,
    }
    anchor_index = anchor_indices[placement.anchor]
    target_index = int(round(float(target_time_s) / observation.sample_spacing_s))
    shift_samples = target_index - anchor_index

    src_start = max(0, -shift_samples)
    dst_start = max(0, shift_samples)
    span = max(0, min(n_rows - src_start, n_rows - dst_start))
    placed: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        output = np.zeros_like(values)
        if span:
            output[dst_start : dst_start + span] = values[src_start : src_start + span]
        placed[name] = output

    placed_active_start_index = active_start_index + shift_samples
    placed_active_end_index = active_end_index + shift_samples
    visible_active_start_index = max(0, placed_active_start_index)
    visible_active_end_index = min(n_rows - 1, placed_active_end_index)
    visible_active_samples = max(0, visible_active_end_index - visible_active_start_index + 1)

    diagnostics = {
        "enabled": True,
        "anchor": placement.anchor,
        "allow_outside": bool(placement.allow_outside),
        "target_time_s": float(target_index * observation.sample_spacing_s),
        "target_index": target_index,
        "shift_samples": int(shift_samples),
        "shift_s": float(shift_samples * observation.sample_spacing_s),
        "threshold": threshold,
        "threshold_fraction": float(placement.threshold_fraction),
        "threshold_abs": float(placement.threshold_abs),
        "original_peak_time_s": float(peak_index * observation.sample_spacing_s),
        "original_peak_index": peak_index,
        "original_peak_value": peak_value,
        "original_active_start_time_s": float(active_start_index * observation.sample_spacing_s),
        "original_active_end_time_s": float(active_end_index * observation.sample_spacing_s),
        "original_active_samples": int(active_end_index - active_start_index + 1),
        "placed_peak_time_s": float((peak_index + shift_samples) * observation.sample_spacing_s),
        "placed_active_start_time_s": float(placed_active_start_index * observation.sample_spacing_s),
        "placed_active_end_time_s": float(placed_active_end_index * observation.sample_spacing_s),
        "visible_active_start_time_s": (
            None if visible_active_samples == 0 else float(visible_active_start_index * observation.sample_spacing_s)
        ),
        "visible_active_end_time_s": (
            None if visible_active_samples == 0 else float(visible_active_end_index * observation.sample_spacing_s)
        ),
        "visible_active_samples": int(visible_active_samples),
        "cropped_left_samples": int(src_start),
        "cropped_right_samples": int(n_rows - (src_start + span)),
    }
    return placed, diagnostics


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
        placement_target_time_s = sample_emri_placement_target_time_s(config.emri, observation, rng)
        prepared_parameters = factory.prepare_parameters(source_parameters, observation)
        waveform = build_interpolated_emri_waveform(prepared_parameters, observation)
        placement = apply_emri_waveform_time_placement(
            waveform,
            observation,
            config.emri.placement,
            placement_target_time_s,
        )
        channels = generate_tdi_xyz_td(waveform, time_s, observation)

        for channel, series in channels.items():
            tdi_xyz[channel] += np.asarray(series, dtype=np.float64)

        source_records.append(
            {
                "seed": source_seed,
                "catalog_entry": entry.to_mapping(),
                "waveform_parameters": deepcopy(prepared_parameters),
                "placement": placement,
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

    save_time_domain_preview(
        output_path,
        bundle.time_s,
        bundle.tdi_xyz,
        title=f"{config.dataset.name} EMRI X/Y/Z",
    )
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
