from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Mapping

import h5py
import numpy as np
from PIL import Image, ImageDraw

from tianqin_dc.bbh_catalog import redshift_to_luminosity_distance_mpc
from tianqin_dc.bbh_export import (
    BBHCompletionConfig,
    BBHCatalogSelectionConfig,
    _parameters_from_catalog_entry as bbh_parameters_from_catalog_entry,
    _resolve_catalog_entries as resolve_bbh_catalog_entries,
)
from tianqin_dc.config import ObservationConfig, RunConfig, load_run_config
from tianqin_dc.dwd_export import DWDCatalogSelectionConfig, _resolve_catalog_entries as resolve_dwd_catalog_entries
from tianqin_dc.emri_export import (
    EMRICompletionConfig,
    EMRIPlacementConfig,
    EMRICatalogSelectionConfig,
    apply_emri_time_placement,
    sample_emri_placement_target_time_s,
    _parameters_from_catalog_entry as emri_parameters_from_catalog_entry,
    _resolve_catalog_entries as resolve_emri_catalog_entries,
)
from tianqin_dc.plotting import save_time_domain_preview
from tianqin_dc.sampling import sample_population_parameters
from tianqin_dc.smbhb_export import (
    SMBHBCompletionConfig,
    SMBHBCatalogSelectionConfig,
    _parameters_from_catalog_entry as smbhb_parameters_from_catalog_entry,
    _resolve_catalog_entries as resolve_smbhb_catalog_entries,
)
from tianqin_dc.sources import get_source_factory


DEFAULT_DETECTORS = ("TQ", "Taiji", "LISA")
DETECTOR_COLORS = {
    "TQ": (31, 119, 180),
    "TianQin": (31, 119, 180),
    "Taiji": (214, 39, 40),
    "LISA": (44, 160, 44),
}


@dataclass(frozen=True)
class MultiDetectorJobConfig:
    source_config_path: Path
    source_config: dict[str, Any]
    run_config: dict[str, Any]
    output_dir: Path
    prefix: str
    detectors: tuple[str, ...]
    duration_s: float | None
    sample_rate_hz: float | None
    max_plot_points: int


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    population_name: str
    parameters: dict[str, Any]
    seed: int | None = None
    catalog_entry: dict[str, Any] | None = None
    placement: EMRIPlacementConfig | None = None
    placement_target_time_s: float | None = None

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "population_name": self.population_name,
            "parameters": deepcopy(self.parameters),
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.catalog_entry is not None:
            payload["catalog_entry"] = deepcopy(self.catalog_entry)
        if self.placement_target_time_s is not None:
            payload["placement_target_time_s"] = self.placement_target_time_s
        return payload


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{field_name}' to be a mapping, got {type(value).__name__}.")
    return value


def _child_seed(seed_sequence: np.random.SeedSequence) -> int:
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def _load_raw_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a JSON object.")
    return raw


def _deep_merge_dict(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_config_path(path_value: str | Path, *, base_path: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    relative_to_config = base_path.parent / candidate
    if relative_to_config.exists():
        return relative_to_config
    return candidate


def _load_job_config(args: argparse.Namespace) -> MultiDetectorJobConfig:
    config_path = Path(args.config)
    raw_config = _load_raw_config(config_path)

    if "source_config" in raw_config:
        source_config_path = _resolve_config_path(str(raw_config["source_config"]), base_path=config_path)
        source_config = _load_raw_config(source_config_path)
        overrides = raw_config.get("source_config_overrides")
        if overrides is not None:
            source_config = _deep_merge_dict(
                source_config,
                _mapping(overrides, field_name="source_config_overrides"),
            )
        output_raw = dict(_mapping(raw_config.get("output", {}), field_name="output"))
        observation_raw = dict(_mapping(raw_config.get("observation", {}), field_name="observation"))
        plot_raw = dict(_mapping(raw_config.get("plot", {}), field_name="plot"))
        dataset_name = str(raw_config.get("dataset", {}).get("name", source_config_path.stem))
        run_config = {
            "multi_detector_config": deepcopy(raw_config),
            "source_config_path": str(source_config_path),
            "source_config": deepcopy(source_config),
        }
        return MultiDetectorJobConfig(
            source_config_path=source_config_path,
            source_config=source_config,
            run_config=run_config,
            output_dir=Path(args.output_dir or output_raw.get("directory", "outputs/multi_detector_aet")),
            prefix=str(args.prefix or output_raw.get("prefix", dataset_name)),
            detectors=tuple(str(item) for item in (args.detectors or raw_config.get("detectors", DEFAULT_DETECTORS))),
            duration_s=(
                args.duration_s
                if args.duration_s is not None
                else (None if observation_raw.get("duration_s") is None else float(observation_raw["duration_s"]))
            ),
            sample_rate_hz=(
                args.sample_rate_hz
                if args.sample_rate_hz is not None
                else (
                    None
                    if observation_raw.get("sample_rate_hz") is None
                    else float(observation_raw["sample_rate_hz"])
                )
            ),
            max_plot_points=int(args.max_plot_points or plot_raw.get("max_points", 20000)),
        )

    dataset_name = str(raw_config.get("dataset", {}).get("name", config_path.stem))
    return MultiDetectorJobConfig(
        source_config_path=config_path,
        source_config=raw_config,
        run_config=deepcopy(raw_config),
        output_dir=Path(args.output_dir or "outputs/multi_detector_aet"),
        prefix=str(args.prefix or dataset_name),
        detectors=tuple(str(item) for item in (args.detectors or DEFAULT_DETECTORS)),
        duration_s=args.duration_s if args.duration_s is not None else 63072000.0,
        sample_rate_hz=args.sample_rate_hz if args.sample_rate_hz is not None else 0.2,
        max_plot_points=int(args.max_plot_points or 20000),
    )


def _observation_from_raw(raw_config: Mapping[str, Any], *, duration_s: float | None, sample_rate_hz: float | None) -> ObservationConfig:
    observation_raw = dict(_mapping(raw_config["observation"], field_name="observation"))
    if duration_s is not None:
        observation_raw["duration_s"] = duration_s
    if sample_rate_hz is not None:
        observation_raw["sample_rate_hz"] = sample_rate_hz
    observation_raw["channels"] = ["A", "E", "T"]
    return ObservationConfig.from_config(observation_raw)


def _resolve_source_population_specs(config: RunConfig) -> list[SourceSpec]:
    active_populations = [population for population in config.sources if population.enabled]
    root_sequence = np.random.SeedSequence(config.seed)
    population_sequences = root_sequence.spawn(len(active_populations) + 1)[:-1]

    specs: list[SourceSpec] = []
    for population, population_sequence in zip(active_populations, population_sequences, strict=True):
        population_rng = np.random.default_rng(_child_seed(population_sequence))
        sampled_parameters = sample_population_parameters(population, population_rng)
        source_sequences = population_sequence.spawn(len(sampled_parameters))
        population_name = population.name or f"{population.kind}_population"

        for parameters, source_sequence in zip(sampled_parameters, source_sequences, strict=True):
            specs.append(
                SourceSpec(
                    kind=population.kind,
                    population_name=population_name,
                    parameters=deepcopy(parameters),
                    seed=_child_seed(source_sequence),
                )
            )
    return specs


def _resolve_emri_catalog_specs(raw_config: Mapping[str, Any], observation: ObservationConfig | None) -> list[SourceSpec]:
    root_sequence = np.random.SeedSequence(int(raw_config.get("seed", 123456789)))
    selection_sequence, generation_sequence = root_sequence.spawn(2)
    selection_rng = np.random.default_rng(_child_seed(selection_sequence))
    selection = EMRICatalogSelectionConfig.from_config(_mapping(raw_config["catalog"], field_name="catalog"))
    completion = EMRICompletionConfig.from_config(raw_config.get("emri"))
    entries = resolve_emri_catalog_entries(selection, selection_rng)
    source_sequences = generation_sequence.spawn(len(entries))

    specs: list[SourceSpec] = []
    for entry, source_sequence in zip(entries, source_sequences, strict=True):
        source_seed = _child_seed(source_sequence)
        rng = np.random.default_rng(source_seed)
        parameters = emri_parameters_from_catalog_entry(entry, completion, rng)
        placement_target_time_s = (
            sample_emri_placement_target_time_s(completion, observation, rng)
            if observation is not None
            else None
        )
        specs.append(
            SourceSpec(
                kind="emri",
                population_name="emri_catalog",
                parameters=parameters,
                seed=source_seed,
                catalog_entry=entry.to_mapping(),
                placement=completion.placement,
                placement_target_time_s=placement_target_time_s,
            )
        )
    return specs


def _resolve_bbh_catalog_specs(raw_config: Mapping[str, Any]) -> list[SourceSpec]:
    root_sequence = np.random.SeedSequence(int(raw_config.get("seed", 123456789)))
    selection_sequence, generation_sequence = root_sequence.spawn(2)
    selection_rng = np.random.default_rng(_child_seed(selection_sequence))
    selection = BBHCatalogSelectionConfig.from_config(_mapping(raw_config["catalog"], field_name="catalog"))
    completion = BBHCompletionConfig.from_config(raw_config.get("bbh"))
    entries = resolve_bbh_catalog_entries(selection, selection_rng)
    distances = redshift_to_luminosity_distance_mpc([entry.z for entry in entries])
    source_sequences = generation_sequence.spawn(len(entries))

    specs: list[SourceSpec] = []
    for entry, source_sequence, distance_mpc in zip(entries, source_sequences, distances, strict=True):
        source_seed = _child_seed(source_sequence)
        rng = np.random.default_rng(source_seed)
        specs.append(
            SourceSpec(
                kind="sbbh",
                population_name="bbh_catalog",
                parameters=bbh_parameters_from_catalog_entry(entry, float(distance_mpc), completion, rng),
                seed=source_seed,
                catalog_entry=entry.to_mapping(),
            )
        )
    return specs


def _resolve_dwd_catalog_specs(raw_config: Mapping[str, Any]) -> list[SourceSpec]:
    root_sequence = np.random.SeedSequence(int(raw_config.get("seed", 123456789)))
    selection_sequence, generation_sequence = root_sequence.spawn(2)
    selection_rng = np.random.default_rng(_child_seed(selection_sequence))
    selection = DWDCatalogSelectionConfig.from_config(_mapping(raw_config["catalog"], field_name="catalog"))
    entries = resolve_dwd_catalog_entries(selection, selection_rng)
    source_sequences = generation_sequence.spawn(len(entries))

    return [
        SourceSpec(
            kind="dwd",
            population_name="dwd_catalog",
            parameters=entry.to_source_parameters(),
            seed=_child_seed(source_sequence),
            catalog_entry=entry.to_mapping(),
        )
        for entry, source_sequence in zip(entries, source_sequences, strict=True)
    ]


def _resolve_smbhb_catalog_specs(raw_config: Mapping[str, Any]) -> list[SourceSpec]:
    root_sequence = np.random.SeedSequence(int(raw_config.get("seed", 123456789)))
    selection_sequence, generation_sequence = root_sequence.spawn(2)
    selection_rng = np.random.default_rng(_child_seed(selection_sequence))
    selection = SMBHBCatalogSelectionConfig.from_config(_mapping(raw_config["catalog"], field_name="catalog"))
    completion = SMBHBCompletionConfig.from_config(raw_config.get("smbhb"))
    entries = resolve_smbhb_catalog_entries(selection, selection_rng)
    source_sequences = generation_sequence.spawn(len(entries))

    specs: list[SourceSpec] = []
    for entry, source_sequence in zip(entries, source_sequences, strict=True):
        source_seed = _child_seed(source_sequence)
        rng = np.random.default_rng(source_seed)
        specs.append(
            SourceSpec(
                kind="smbhb",
                population_name="smbhb_catalog",
                parameters=smbhb_parameters_from_catalog_entry(entry, completion, rng),
                seed=source_seed,
                catalog_entry=entry.to_mapping(),
            )
        )
    return specs


def resolve_source_specs(
    config_path: str | Path,
    raw_config: Mapping[str, Any],
    observation: ObservationConfig | None = None,
) -> list[SourceSpec]:
    if "sources" in raw_config:
        config, _ = load_run_config(config_path)
        return _resolve_source_population_specs(config)

    if "catalog" not in raw_config:
        raise ValueError("Config must contain either 'sources' or a supported source-specific 'catalog' section.")
    if "emri" in raw_config:
        return _resolve_emri_catalog_specs(raw_config, observation)
    if "bbh" in raw_config:
        return _resolve_bbh_catalog_specs(raw_config)
    if "smbhb" in raw_config:
        return _resolve_smbhb_catalog_specs(raw_config)
    return _resolve_dwd_catalog_specs(raw_config)


def generate_detector_response(
    source_specs: list[SourceSpec],
    observation: ObservationConfig,
    detector: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    detector_observation = ObservationConfig.from_config(
        {
            "duration_s": observation.duration_s,
            "sample_rate_hz": observation.sample_rate_hz,
            "detector": detector,
            "tdi_generation": observation.tdi_generation,
            "channels": ["A", "E", "T"],
            "use_gpu": observation.use_gpu,
        }
    )
    response = {channel: np.zeros(detector_observation.num_samples, dtype=np.float64) for channel in ("A", "E", "T")}
    generated_records: list[dict[str, Any]] = []

    for index, source_spec in enumerate(source_specs):
        factory = get_source_factory(source_spec.kind)
        generated = factory.generate(source_spec.parameters, detector_observation)
        generated_channels = generated.channels
        placement = None
        if source_spec.kind == "emri" and source_spec.placement is not None:
            generated_channels, placement = apply_emri_time_placement(
                generated.channels,
                detector_observation,
                source_spec.placement,
                source_spec.placement_target_time_s,
            )
        for channel in ("A", "E", "T"):
            response[channel] += np.asarray(generated_channels[channel], dtype=np.float64)
        record = source_spec.to_mapping()
        record["source_index"] = index
        record["prepared_parameters"] = deepcopy(generated.parameters)
        record["engine"] = generated.engine
        record["detector"] = detector
        record["placement"] = placement
        generated_records.append(record)

    return response, generated_records


def save_aet_hdf5(path: Path, time_s: np.ndarray, channels: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("time", data=time_s)
        handle.create_dataset("a", data=np.asarray(channels["A"], dtype=np.float64))
        handle.create_dataset("e", data=np.asarray(channels["E"], dtype=np.float64))
        handle.create_dataset("t", data=np.asarray(channels["T"], dtype=np.float64))
    save_time_domain_preview(
        path,
        time_s,
        {"A": channels["A"], "E": channels["E"], "T": channels["T"]},
        title=f"{path.name} A/E/T",
    )


def save_parameters_hdf5(
    path: Path,
    *,
    raw_config: Mapping[str, Any],
    source_specs: list[SourceSpec],
    detector_records: Mapping[str, list[dict[str, Any]]],
    response_files: Mapping[str, str],
    plot_files: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = h5py.string_dtype(encoding="utf-8")
    payload = {
        "config": deepcopy(dict(raw_config)),
        "selected_sources": [source.to_mapping() for source in source_specs],
        "detector_records": detector_records,
        "response_files": dict(response_files),
        "plot_files": dict(plot_files),
    }
    with h5py.File(path, "w") as handle:
        handle.create_dataset("parameters_json", data=json.dumps(payload, indent=2, sort_keys=True), dtype=dtype)


def plot_channel_comparison(
    path: Path,
    time_s: np.ndarray,
    detector_responses: Mapping[str, Mapping[str, np.ndarray]],
    channel: str,
    *,
    max_points: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(np.ceil(time_s.size / max_points)))
    time_days = time_s[::stride] / 86400.0
    series_by_detector = {
        detector: np.asarray(response[channel], dtype=np.float64)[::stride]
        for detector, response in detector_responses.items()
    }
    finite_chunks = [values[np.isfinite(values)] for values in series_by_detector.values() if np.any(np.isfinite(values))]
    finite_values = np.concatenate(finite_chunks) if finite_chunks else np.array([], dtype=np.float64)
    y_min = float(np.min(finite_values)) if finite_values.size else -1.0
    y_max = float(np.max(finite_values)) if finite_values.size else 1.0
    if y_min == y_max:
        padding = abs(y_min) * 0.1 or 1.0
        y_min -= padding
        y_max += padding

    width, height = 1600, 720
    left, right, top, bottom = 96, 40, 56, 92
    plot_width = width - left - right
    plot_height = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    axis_color = (55, 65, 81)
    grid_color = (226, 232, 240)
    text_color = (17, 24, 39)

    for i in range(6):
        y = top + int(round(plot_height * i / 5))
        draw.line((left, y, width - right, y), fill=grid_color, width=1)
    for i in range(6):
        x = left + int(round(plot_width * i / 5))
        draw.line((x, top, x, height - bottom), fill=grid_color, width=1)
    draw.rectangle((left, top, width - right, height - bottom), outline=axis_color, width=2)

    x_min = float(time_days[0]) if time_days.size else 0.0
    x_max = float(time_days[-1]) if time_days.size else 1.0
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0

    def point(x_value: float, y_value: float) -> tuple[int, int]:
        x = left + int(round((x_value - x_min) / x_span * plot_width))
        y = top + int(round((y_max - y_value) / y_span * plot_height))
        return x, y

    for detector, values in series_by_detector.items():
        points = [point(float(x), float(y)) for x, y in zip(time_days, values, strict=True) if np.isfinite(y)]
        if len(points) >= 2:
            draw.line(points, fill=DETECTOR_COLORS.get(detector, (0, 0, 0)), width=2)

    title = f"{channel} channel time-domain response"
    draw.text((left, 20), title, fill=text_color)
    draw.text((left, height - 48), "time [day]", fill=text_color)
    draw.text((12, top + 8), f"{channel} response", fill=text_color)
    draw.text((left, height - 72), f"{x_min:.3g}", fill=text_color)
    draw.text((width - right - 96, height - 72), f"{x_max:.3g}", fill=text_color)
    draw.text((8, top), f"{y_max:.3e}", fill=text_color)
    draw.text((8, height - bottom - 12), f"{y_min:.3e}", fill=text_color)

    legend_x = width - right - 220
    legend_y = top + 16
    for index, detector in enumerate(detector_responses):
        y = legend_y + index * 28
        color = DETECTOR_COLORS.get(detector, (0, 0, 0))
        draw.line((legend_x, y + 8, legend_x + 36, y + 8), fill=color, width=4)
        draw.text((legend_x + 46, y), detector, fill=text_color)

    image.save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TQ/Taiji/LISA AET responses, comparison plots, and selected-parameter HDF5.")
    parser.add_argument("--config", required=True, help="Path to a multi-detector job config or an existing TianQin DC source config.")
    parser.add_argument("--output-dir", help="Directory for generated HDF5 and PNG files.")
    parser.add_argument("--prefix", help="Output file prefix. Defaults to dataset.name or config stem.")
    parser.add_argument("--duration-s", type=float, help="Observation duration in seconds.")
    parser.add_argument("--sample-rate-hz", type=float, help="Sample rate in Hz.")
    parser.add_argument("--detectors", nargs="+", help="Detector names to generate.")
    parser.add_argument("--max-plot-points", type=int, help="Maximum plotted points per detector per channel.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    job_config = _load_job_config(args)

    observation = _observation_from_raw(
        job_config.source_config,
        duration_s=job_config.duration_s,
        sample_rate_hz=job_config.sample_rate_hz,
    )
    source_specs = resolve_source_specs(job_config.source_config_path, job_config.source_config, observation)
    time_s = observation.time_array()

    detector_responses: dict[str, dict[str, np.ndarray]] = {}
    detector_records: dict[str, list[dict[str, Any]]] = {}
    response_files: dict[str, str] = {}

    for detector in job_config.detectors:
        response, records = generate_detector_response(source_specs, observation, detector)
        detector_responses[detector] = response
        detector_records[detector] = records
        response_path = job_config.output_dir / f"{job_config.prefix}_{detector.lower()}_aet.h5"
        save_aet_hdf5(response_path, time_s, response)
        response_files[detector] = str(response_path)
        print(f"Wrote {detector} AET response to {response_path}")

    plot_files: dict[str, str] = {}
    for channel in ("A", "E", "T"):
        plot_path = job_config.output_dir / f"{job_config.prefix}_{channel.lower()}_comparison.png"
        plot_channel_comparison(
            plot_path,
            time_s,
            detector_responses,
            channel,
            max_points=job_config.max_plot_points,
        )
        plot_files[channel] = str(plot_path)
        print(f"Wrote {channel} comparison plot to {plot_path}")

    parameters_path = job_config.output_dir / f"{job_config.prefix}_selected_parameters.h5"
    save_parameters_hdf5(
        parameters_path,
        raw_config=job_config.run_config,
        source_specs=source_specs,
        detector_records=detector_records,
        response_files=response_files,
        plot_files=plot_files,
    )
    print(f"Wrote selected parameters to {parameters_path}")
    print(f"Selected sources: {len(source_specs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
