from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from tianqin_dc.config import NoiseConfig, ObservationConfig
from tianqin_dc.minimal_aet_io import MinimalOutputConfig, read_minimal_aet_hdf5, save_minimal_aet_hdf5
from tianqin_dc.noise import generate_noise
from tianqin_dc.plotting import (
    DEFAULT_MAX_TIME_DOMAIN_PLOT_POINTS,
    _extrema_preserving_sample_indices,
    _finite_min_max,
    _load_pyplot,
    _time_scale_and_unit,
    save_time_domain_preview,
)


BASE_SOURCE_KEYS = ("dwd", "emri", "sbbh", "sgwb")
BASE_SOURCE_LABELS = {
    "dwd": "DWD",
    "emri": "EMRI",
    "sbbh": "SBBH",
    "sgwb": "SGWB",
}
SOURCE_COLORS = {
    "dwd": "#ff7f0e",
    "emri": "#2ca02c",
    "sbbh": "#d62728",
    "sgwb": "#9467bd",
    "smbhb_popIII": "#17becf",
    "smbhb_Q3d": "#8c564b",
    "smbhb_Q3nod": "#e377c2",
}
DATA_COLOR = "#32388f"


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{field_name}' to be a mapping, got {type(value).__name__}.")
    return value


@dataclass(frozen=True)
class PlotConfig:
    max_points: int = DEFAULT_MAX_TIME_DOMAIN_PLOT_POINTS
    dpi: int = 180

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "PlotConfig":
        if value is None:
            return cls()
        data = _mapping(value, field_name="plot")
        return cls(
            max_points=int(data.get("max_points", DEFAULT_MAX_TIME_DOMAIN_PLOT_POINTS)),
            dpi=int(data.get("dpi", 180)),
        )


@dataclass(frozen=True)
class OutputDefaults:
    overwrite: bool = True
    compression: str | None = "gzip"
    compression_level: int = 4
    chunk_rows: int = 65536

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "OutputDefaults":
        if value is None:
            return cls()
        data = _mapping(value, field_name="output")
        return cls(
            overwrite=bool(data.get("overwrite", True)),
            compression=data.get("compression", "gzip"),
            compression_level=int(data.get("compression_level", 4)),
            chunk_rows=int(data.get("chunk_rows", 65536)),
        )

    def minimal_output(self, path: str | Path) -> MinimalOutputConfig:
        return MinimalOutputConfig(
            path=str(path),
            overwrite=self.overwrite,
            compression=self.compression,
            compression_level=self.compression_level,
            chunk_rows=self.chunk_rows,
        )


@dataclass(frozen=True)
class SMBHBPopulationConfig:
    key: str
    source_key: str
    label: str

    @classmethod
    def from_config(cls, value: Any) -> "SMBHBPopulationConfig":
        if isinstance(value, str):
            return cls(key=value, source_key=value, label=value.replace("_", " "))
        data = _mapping(value, field_name="smbhb_populations[]")
        key = str(data["key"])
        return cls(
            key=key,
            source_key=str(data.get("source_key", key)),
            label=str(data.get("label", key.replace("_", " "))),
        )


@dataclass(frozen=True)
class DetectorConfig:
    key: str
    title: str
    observation: ObservationConfig
    noise: NoiseConfig
    sources: Mapping[str, str]
    noise_seed: int | None = None

    @classmethod
    def from_config(cls, key: str, value: Mapping[str, Any]) -> "DetectorConfig":
        data = _mapping(value, field_name=f"detectors.{key}")
        observation_data = dict(_mapping(data["observation"], field_name=f"detectors.{key}.observation"))
        observation_data["channels"] = ["A", "E", "T"]
        source_data = _mapping(data["sources"], field_name=f"detectors.{key}.sources")
        return cls(
            key=key,
            title=str(data.get("title", key)),
            observation=ObservationConfig.from_config(observation_data),
            noise=NoiseConfig.from_config(data.get("noise")),
            sources={str(source_key): str(path) for source_key, path in source_data.items()},
            noise_seed=None if data.get("noise_seed") is None else int(data["noise_seed"]),
        )


@dataclass(frozen=True)
class BatchConfig:
    seed: int
    output_root: Path
    plot_dir: Path
    output: OutputDefaults
    plot: PlotConfig
    smbhb_populations: tuple[SMBHBPopulationConfig, ...]
    detectors: tuple[DetectorConfig, ...]

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> "BatchConfig":
        output_root = Path(str(data.get("output_root", "/public/home/zhuangzhenye/jobs/gwspace_runs/multi_source_mixtures")))
        plot_dir = Path(str(data.get("plot_dir", output_root / "plots")))
        population_data = data.get("smbhb_populations")
        if not isinstance(population_data, list) or not population_data:
            raise ValueError("Config field 'smbhb_populations' must be a non-empty list.")
        detectors_data = _mapping(data.get("detectors"), field_name="detectors")
        return cls(
            seed=int(data.get("seed", 20260514)),
            output_root=output_root,
            plot_dir=plot_dir,
            output=OutputDefaults.from_config(data.get("output")),
            plot=PlotConfig.from_config(data.get("plot")),
            smbhb_populations=tuple(SMBHBPopulationConfig.from_config(item) for item in population_data),
            detectors=tuple(DetectorConfig.from_config(str(key), _mapping(value, field_name=f"detectors.{key}")) for key, value in detectors_data.items()),
        )


@dataclass(frozen=True)
class SourceInput:
    key: str
    label: str
    path: str
    color: str


@dataclass(frozen=True)
class MixtureJob:
    detector: DetectorConfig
    detector_index: int
    population: SMBHBPopulationConfig
    source_inputs: tuple[SourceInput, ...]
    output_path: Path
    mixture_plot_path: Path
    source_plot_path: Path

    @property
    def title(self) -> str:
        return f"{self.detector.title} Multi-source Mixture ({self.population.label})"

    @property
    def source_title(self) -> str:
        return f"{self.detector.title} Multi-source Mixture Source Contributions ({self.population.label}, A channel)"


def load_batch_config(path: str | Path) -> BatchConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a JSON object.")
    return BatchConfig.from_config(raw)


def iter_jobs(config: BatchConfig) -> Iterable[MixtureJob]:
    for detector_index, detector in enumerate(config.detectors):
        missing_base = [key for key in BASE_SOURCE_KEYS if key not in detector.sources]
        if missing_base:
            raise ValueError(f"Detector '{detector.key}' is missing source paths for: {missing_base}")

        for population in config.smbhb_populations:
            if population.source_key not in detector.sources:
                raise ValueError(
                    f"Detector '{detector.key}' is missing SMBHB source path '{population.source_key}'."
                )
            source_inputs = tuple(
                SourceInput(
                    key=key,
                    label=BASE_SOURCE_LABELS[key],
                    path=detector.sources[key],
                    color=SOURCE_COLORS[key],
                )
                for key in BASE_SOURCE_KEYS
            ) + (
                SourceInput(
                    key=population.source_key,
                    label=population.label,
                    path=detector.sources[population.source_key],
                    color=SOURCE_COLORS.get(population.source_key, "#17becf"),
                ),
            )
            stem = f"{detector.key}_{population.key}_multisource_mixture_aet"
            yield MixtureJob(
                detector=detector,
                detector_index=detector_index,
                population=population,
                source_inputs=source_inputs,
                output_path=config.output_root / f"{stem}.h5",
                mixture_plot_path=config.plot_dir / f"{stem}.png",
                source_plot_path=config.plot_dir / f"{stem}_source_contributions_A.png",
            )


def _assert_same_time(reference: np.ndarray, candidate: np.ndarray, *, label: str) -> None:
    if candidate.shape != reference.shape or not np.array_equal(candidate, reference):
        raise ValueError(f"Time column in '{label}' does not match the detector observation time grid.")


def _noise_seed(config: BatchConfig, job: MixtureJob) -> int:
    if job.detector.noise_seed is not None:
        return job.detector.noise_seed
    return config.seed + job.detector_index


def build_mixture(config: BatchConfig, job: MixtureJob) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    observation = job.detector.observation
    time_s = observation.time_array()
    summed = {channel: np.zeros_like(time_s, dtype=np.float64) for channel in ("A", "E", "T")}
    components_a: dict[str, np.ndarray] = {}

    for source in job.source_inputs:
        input_time_s, channels = read_minimal_aet_hdf5(source.path)
        _assert_same_time(time_s, input_time_s, label=source.path)
        components_a[source.key] = np.asarray(channels["A"], dtype=np.float64)
        for channel in ("A", "E", "T"):
            summed[channel] += np.asarray(channels[channel], dtype=np.float64)

    if job.detector.noise.enabled:
        noise = generate_noise(observation, job.detector.noise, seed=_noise_seed(config, job)).series
        for channel in ("A", "E", "T"):
            summed[channel] += noise[channel]

    return time_s, summed, components_a


def run_job(config: BatchConfig, job: MixtureJob) -> Path:
    print(f"Running {job.detector.key} / {job.population.key}", flush=True)
    for source in job.source_inputs:
        print(f"  {source.label}: {source.path}", flush=True)
    time_s, channels, components_a = build_mixture(config, job)

    output_path = save_minimal_aet_hdf5(
        config.output.minimal_output(job.output_path),
        time_s=time_s,
        a=channels["A"],
        e=channels["E"],
        t=channels["T"],
        preview=False,
    )
    save_time_domain_preview(
        output_path,
        time_s,
        channels,
        title=job.title,
        output_path=job.mixture_plot_path,
        max_points=config.plot.max_points,
        show_stats=False,
        fail_on_error=True,
    )
    save_a_channel_source_contribution_plot(
        job.source_plot_path,
        time_s=time_s,
        mixture_a=channels["A"],
        components=[
            (source.label, components_a[source.key], source.color)
            for source in job.source_inputs
        ],
        title=job.source_title,
        max_points=config.plot.max_points,
        dpi=config.plot.dpi,
    )
    print(f"Wrote mixture output to {output_path}", flush=True)
    return output_path


def _combined_sample_indices(data: np.ndarray, component: np.ndarray, max_points: int) -> np.ndarray:
    points_per_series = max(2, int(max_points) // 2)
    data_indices = _extrema_preserving_sample_indices(data, points_per_series)
    component_indices = _extrema_preserving_sample_indices(component, points_per_series)
    return np.unique(np.concatenate((data_indices, component_indices)))


def _padded_limits(*arrays: np.ndarray) -> tuple[float, float]:
    limits = [_finite_min_max(values) for values in arrays]
    y_min = min(item[0] for item in limits)
    y_max = max(item[1] for item in limits)
    if y_min == y_max:
        pad = max(abs(y_min) * 0.05, 1.0)
    else:
        pad = 0.04 * (y_max - y_min)
    return y_min - pad, y_max + pad


def save_a_channel_source_contribution_plot(
    output_path: str | Path,
    *,
    time_s: np.ndarray,
    mixture_a: np.ndarray,
    components: list[tuple[str, np.ndarray, str]],
    title: str,
    max_points: int,
    dpi: int,
) -> Path:
    if not components:
        raise ValueError("At least one source component is required.")
    time_s = np.asarray(time_s, dtype=np.float64)
    mixture_a = np.asarray(mixture_a, dtype=np.float64)
    if time_s.shape != mixture_a.shape:
        raise ValueError(f"mixture_a has shape {mixture_a.shape}, expected {time_s.shape}.")

    try:
        plt = _load_pyplot()
    except Exception:
        return _save_a_channel_source_contribution_plot_pil(
            output_path,
            time_s=time_s,
            mixture_a=mixture_a,
            components=components,
            title=title,
            max_points=max_points,
        )

    time_scale, time_unit = _time_scale_and_unit(time_s)
    fig, axes = plt.subplots(
        len(components),
        1,
        figsize=(13.0, max(7.0, 2.1 * len(components))),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)

    for axis, (label, values, color) in zip(axes_array, components, strict=True):
        component = np.asarray(values, dtype=np.float64)
        if component.shape != time_s.shape:
            raise ValueError(f"Component '{label}' has shape {component.shape}, expected {time_s.shape}.")
        indices = _combined_sample_indices(mixture_a, component, max_points)
        x_values = time_s[indices] / time_scale
        axis.plot(
            x_values,
            mixture_a[indices],
            color=DATA_COLOR,
            linewidth=0.45,
            alpha=0.78,
            label="Data",
        )
        axis.plot(
            x_values,
            component[indices],
            color=color,
            linewidth=0.85,
            alpha=0.95,
            label=label,
        )
        axis.set_title(label, fontsize=10, loc="left")
        axis.set_ylabel("A strain")
        axis.set_ylim(*_padded_limits(mixture_a, component))
        axis.grid(True, linewidth=0.4, alpha=0.32)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        axis.legend(loc="upper right", fontsize=8, frameon=True)

    axes_array[-1].set_xlabel(f"Time [{time_unit}]")
    fig.suptitle(title, fontsize=13)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote A-channel source contribution plot to {path}", flush=True)
    return path


def _save_a_channel_source_contribution_plot_pil(
    output_path: str | Path,
    *,
    time_s: np.ndarray,
    mixture_a: np.ndarray,
    components: list[tuple[str, np.ndarray, str]],
    title: str,
    max_points: int,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    time_scale, time_unit = _time_scale_and_unit(time_s)
    max_points = min(int(max_points), 50_000)
    width = 1400
    panel_height = 230
    top_margin = 58
    bottom_margin = 58
    gap = 18
    left_margin = 96
    right_margin = 36
    height = top_margin + bottom_margin + len(components) * panel_height + (len(components) - 1) * gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_color = (38, 38, 38)
    grid_color = (220, 220, 220)
    axis_color = (80, 80, 80)
    data_color = _hex_to_rgb(DATA_COLOR)

    draw.text((left_margin, 20), title, fill=text_color, font=font)
    x_min = float(time_s[0] / time_scale)
    x_max = float(time_s[-1] / time_scale)
    if x_max == x_min:
        x_max = x_min + 1.0

    plot_left = left_margin
    plot_right = width - right_margin
    plot_width = plot_right - plot_left

    for index, (label, values, color) in enumerate(components):
        component = np.asarray(values, dtype=np.float64)
        panel_top = top_margin + index * (panel_height + gap)
        panel_bottom = panel_top + panel_height
        plot_top = panel_top + 28
        plot_bottom = panel_bottom - 30
        plot_height = plot_bottom - plot_top
        indices = _combined_sample_indices(mixture_a, component, max_points)
        x_values = time_s[indices] / time_scale
        data_values = mixture_a[indices]
        component_values = component[indices]
        y_min, y_max = _padded_limits(mixture_a, component)

        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=axis_color)
        for tick in range(1, 5):
            x = plot_left + tick * plot_width / 5.0
            draw.line((x, plot_top, x, plot_bottom), fill=grid_color)
        for tick in range(1, 4):
            y = plot_top + tick * plot_height / 4.0
            draw.line((plot_left, y, plot_right, y), fill=grid_color)

        _draw_series(
            draw,
            x_values,
            data_values,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            color=data_color,
            width=1,
        )
        _draw_series(
            draw,
            x_values,
            component_values,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            color=_hex_to_rgb(color),
            width=2,
        )
        draw.text((12, panel_top + 30), "A strain", fill=text_color, font=font)
        draw.text((plot_left, panel_top + 8), label, fill=text_color, font=font)
        draw.text((plot_left, panel_bottom - 22), f"{y_min:.2e}", fill=text_color, font=font)
        draw.text((plot_right - 86, panel_bottom - 22), f"{y_max:.2e}", fill=text_color, font=font)
        legend_y = panel_top + 8
        draw.line((plot_right - 170, legend_y + 6, plot_right - 145, legend_y + 6), fill=data_color, width=2)
        draw.text((plot_right - 140, legend_y), "Data", fill=text_color, font=font)
        draw.line((plot_right - 96, legend_y + 6, plot_right - 71, legend_y + 6), fill=_hex_to_rgb(color), width=2)
        draw.text((plot_right - 66, legend_y), label, fill=text_color, font=font)

    draw.text((plot_left, height - 35), f"Time [{time_unit}]", fill=text_color, font=font)
    draw.text((plot_left, height - 20), f"{x_min:.3g}", fill=text_color, font=font)
    draw.text((plot_right - 86, height - 20), f"{x_max:.3g}", fill=text_color, font=font)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"Wrote A-channel source contribution plot to {path}", flush=True)
    return path


def _draw_series(
    draw: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    points: list[tuple[float, float]] = []
    for x_value, y_value in zip(x_values, y_values, strict=True):
        if not np.isfinite(y_value):
            if len(points) > 1:
                draw.line(points, fill=color, width=width)
            points = []
            continue
        x_pixel = plot_left + (float(x_value) - x_min) / (x_max - x_min) * plot_width
        y_pixel = plot_bottom - (float(y_value) - y_min) / (y_max - y_min) * plot_height
        points.append((x_pixel, y_pixel))
    if len(points) > 1:
        draw.line(points, fill=color, width=width)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    stripped = value.strip().lstrip("#")
    if len(stripped) != 6:
        return (40, 40, 40)
    return tuple(int(stripped[index : index + 2], 16) for index in (0, 2, 4))


def _filter_jobs(
    jobs: Iterable[MixtureJob],
    *,
    detector_keys: set[str] | None,
    population_keys: set[str] | None,
) -> list[MixtureJob]:
    selected = []
    for job in jobs:
        if detector_keys is not None and job.detector.key not in detector_keys:
            continue
        if population_keys is not None and job.population.key not in population_keys:
            continue
        selected.append(job)
    return selected


def print_dry_run(config: BatchConfig, jobs: list[MixtureJob]) -> None:
    print(f"output_root: {config.output_root}")
    print(f"plot_dir: {config.plot_dir}")
    print(f"jobs: {len(jobs)}")
    for job in jobs:
        print(f"- {job.detector.key} / {job.population.key}")
        print(f"  output: {job.output_path}")
        print(f"  mixture_plot: {job.mixture_plot_path}")
        print(f"  source_plot: {job.source_plot_path}")
        print(f"  noise_enabled: {job.detector.noise.enabled}")
        print(f"  noise_model: {job.detector.noise.model}")
        print(f"  noise_seed: {_noise_seed(config, job)}")
        for source in job.source_inputs:
            print(f"  input {source.label}: {source.path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build noisy multi-source A/E/T mixtures for TianQin, LISA, and Taiji SMBHB populations."
    )
    parser.add_argument("--config", required=True, help="Path to a multi-source mixture JSON config.")
    parser.add_argument("--detector", action="append", help="Run only this detector key. Can be repeated.")
    parser.add_argument("--population", action="append", help="Run only this SMBHB population key. Can be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected jobs without opening inputs or writing files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_batch_config(args.config)
    jobs = _filter_jobs(
        iter_jobs(config),
        detector_keys=None if args.detector is None else set(args.detector),
        population_keys=None if args.population is None else set(args.population),
    )
    if not jobs:
        raise ValueError("No jobs matched the requested detector/population filters.")
    if args.dry_run:
        print_dry_run(config, jobs)
        return 0

    for job in jobs:
        run_job(config, job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
