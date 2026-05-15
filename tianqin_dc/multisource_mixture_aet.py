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
    "noise": "#7f7f7f",
}
DATA_COLOR = "#32388f"


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected '{field_name}' to be a mapping, got {type(value).__name__}.")
    return value


@dataclass(frozen=True)
class PlotConfig:
    max_points: int = DEFAULT_MAX_TIME_DOMAIN_PLOT_POINTS
    frequency_max_points: int = DEFAULT_MAX_TIME_DOMAIN_PLOT_POINTS
    frequency_min_hz: float = 1.0e-4
    frequency_max_hz: float | None = None
    dpi: int = 180

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "PlotConfig":
        if value is None:
            return cls()
        data = _mapping(value, field_name="plot")
        return cls(
            max_points=int(data.get("max_points", DEFAULT_MAX_TIME_DOMAIN_PLOT_POINTS)),
            frequency_max_points=int(data.get("frequency_max_points", data.get("max_points", DEFAULT_MAX_TIME_DOMAIN_PLOT_POINTS))),
            frequency_min_hz=float(data.get("frequency_min_hz", 1.0e-4)),
            frequency_max_hz=None if data.get("frequency_max_hz") is None else float(data["frequency_max_hz"]),
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
    no_noise_output_path: Path
    mixture_plot_path: Path
    no_noise_mixture_plot_path: Path
    frequency_plot_path: Path
    no_noise_frequency_plot_path: Path
    source_plot_path: Path
    source_frequency_plot_path: Path

    @property
    def title(self) -> str:
        return f"{self.detector.title} Multi-source Mixture ({self.population.label})"

    @property
    def source_title(self) -> str:
        return f"{self.detector.title} Multi-source Mixture Source Contributions ({self.population.label}, A channel)"

    @property
    def source_frequency_title(self) -> str:
        return (
            f"{self.detector.title} Multi-source Mixture Source Contributions "
            f"({self.population.label}, A channel, frequency domain)"
        )

    @property
    def no_noise_title(self) -> str:
        return f"{self.detector.title} Multi-source Mixture ({self.population.label}, no noise)"

    @property
    def frequency_title(self) -> str:
        return f"{self.detector.title} {self.population.label} A/E/T frequency-domain signals"

    @property
    def no_noise_frequency_title(self) -> str:
        return f"{self.detector.title} {self.population.label} A/E/T frequency-domain signals (no noise)"


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
                no_noise_output_path=config.output_root / f"{stem}_no_noise.h5",
                mixture_plot_path=config.plot_dir / f"{stem}.png",
                no_noise_mixture_plot_path=config.plot_dir / f"{stem}_no_noise.png",
                frequency_plot_path=config.plot_dir / f"{stem}_frequency.png",
                no_noise_frequency_plot_path=config.plot_dir / f"{stem}_no_noise_frequency.png",
                source_plot_path=config.plot_dir / f"{stem}_source_contributions_A.png",
                source_frequency_plot_path=config.plot_dir / f"{stem}_source_contributions_A_frequency.png",
            )


def _assert_same_time(reference: np.ndarray, candidate: np.ndarray, *, label: str) -> None:
    if candidate.shape != reference.shape or not np.array_equal(candidate, reference):
        raise ValueError(f"Time column in '{label}' does not match the detector observation time grid.")


def _noise_seed(config: BatchConfig, job: MixtureJob) -> int:
    if job.detector.noise_seed is not None:
        return job.detector.noise_seed
    return config.seed + job.detector_index


def build_signal_mixture(job: MixtureJob) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
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

    return time_s, summed, components_a


def add_noise_to_mixture(
    config: BatchConfig,
    job: MixtureJob,
    channels: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    noisy = {channel: np.asarray(channels[channel], dtype=np.float64).copy() for channel in ("A", "E", "T")}
    reference = noisy["A"]
    noise_channels = {channel: np.zeros_like(reference, dtype=np.float64) for channel in ("A", "E", "T")}
    if job.detector.noise.enabled:
        observation = job.detector.observation
        noise_channels = generate_noise(observation, job.detector.noise, seed=_noise_seed(config, job)).series
        for channel in ("A", "E", "T"):
            noisy[channel] += noise_channels[channel]

    return noisy, noise_channels


def run_job(config: BatchConfig, job: MixtureJob) -> Path:
    print(f"Running {job.detector.key} / {job.population.key}", flush=True)
    for source in job.source_inputs:
        print(f"  {source.label}: {source.path}", flush=True)
    time_s, signal_channels, components_a = build_signal_mixture(job)
    noisy_channels, noise_channels = add_noise_to_mixture(config, job, signal_channels)
    contribution_components = [
        (source.label, components_a[source.key], source.color)
        for source in job.source_inputs
    ]
    if job.detector.noise.enabled:
        contribution_components.append(("Noise", noise_channels["A"], SOURCE_COLORS["noise"]))

    no_noise_output_path = save_minimal_aet_hdf5(
        config.output.minimal_output(job.no_noise_output_path),
        time_s=time_s,
        a=signal_channels["A"],
        e=signal_channels["E"],
        t=signal_channels["T"],
        preview=False,
    )
    save_time_domain_preview(
        no_noise_output_path,
        time_s,
        signal_channels,
        title=job.no_noise_title,
        output_path=job.no_noise_mixture_plot_path,
        max_points=config.plot.max_points,
        show_stats=False,
        fail_on_error=True,
    )
    save_frequency_domain_preview(
        job.no_noise_frequency_plot_path,
        time_s=time_s,
        channels=signal_channels,
        title=job.no_noise_frequency_title,
        detector_title=job.detector.title,
        population_label=job.population.label,
        max_points=config.plot.frequency_max_points,
        f_min_hz=config.plot.frequency_min_hz,
        f_max_hz=config.plot.frequency_max_hz,
        dpi=config.plot.dpi,
    )

    output_path = save_minimal_aet_hdf5(
        config.output.minimal_output(job.output_path),
        time_s=time_s,
        a=noisy_channels["A"],
        e=noisy_channels["E"],
        t=noisy_channels["T"],
        preview=False,
    )
    save_time_domain_preview(
        output_path,
        time_s,
        noisy_channels,
        title=job.title,
        output_path=job.mixture_plot_path,
        max_points=config.plot.max_points,
        show_stats=False,
        fail_on_error=True,
    )
    save_frequency_domain_preview(
        job.frequency_plot_path,
        time_s=time_s,
        channels=noisy_channels,
        title=job.frequency_title,
        detector_title=job.detector.title,
        population_label=job.population.label,
        max_points=config.plot.frequency_max_points,
        f_min_hz=config.plot.frequency_min_hz,
        f_max_hz=config.plot.frequency_max_hz,
        dpi=config.plot.dpi,
    )
    save_a_channel_source_contribution_plot(
        job.source_plot_path,
        time_s=time_s,
        mixture_a=noisy_channels["A"],
        components=contribution_components,
        title=job.source_title,
        max_points=config.plot.max_points,
        dpi=config.plot.dpi,
    )
    save_a_channel_source_frequency_contribution_plot(
        job.source_frequency_plot_path,
        time_s=time_s,
        mixture_a=noisy_channels["A"],
        components=contribution_components,
        title=job.source_frequency_title,
        max_points=config.plot.frequency_max_points,
        f_min_hz=config.plot.frequency_min_hz,
        f_max_hz=config.plot.frequency_max_hz,
        dpi=config.plot.dpi,
    )
    print(f"Wrote mixture output to {output_path}", flush=True)
    print(f"Wrote no-noise mixture output to {no_noise_output_path}", flush=True)
    return output_path


def _frequency_plot_sample_indices(values: np.ndarray, max_points: int) -> np.ndarray:
    values = np.asarray(values)
    finite = np.isfinite(values) & (values > 0.0)
    valid_indices = np.flatnonzero(finite)
    if valid_indices.size == 0:
        return valid_indices
    if valid_indices.size <= max_points:
        return valid_indices
    reduced = _extrema_preserving_sample_indices(values[valid_indices], max_points)
    return valid_indices[reduced]


def _format_frequency_range(f_min_hz: float, f_max_hz: float) -> str:
    return f"{f_min_hz:.0e}-{f_max_hz:.0e} Hz"


def save_frequency_domain_preview(
    output_path: str | Path,
    *,
    time_s: np.ndarray,
    channels: Mapping[str, np.ndarray],
    title: str,
    detector_title: str,
    population_label: str,
    max_points: int,
    f_min_hz: float,
    f_max_hz: float | None,
    dpi: int,
) -> Path:
    time_s = np.asarray(time_s, dtype=np.float64)
    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must contain at least two samples.")
    dt = float(time_s[1] - time_s[0])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time_s must be strictly increasing.")

    frequencies = np.fft.rfftfreq(time_s.size, d=dt)
    upper = float(frequencies[-1]) if f_max_hz is None else min(float(f_max_hz), float(frequencies[-1]))
    lower = max(float(f_min_hz), float(frequencies[1]) if frequencies.size > 1 else 0.0)
    if upper <= lower:
        raise ValueError(f"Invalid frequency plot range: {lower}--{upper} Hz.")
    frequency_mask = (frequencies >= lower) & (frequencies <= upper)
    if not np.any(frequency_mask):
        raise ValueError(f"No FFT bins fall in frequency range {lower}--{upper} Hz.")

    ordered_channels = [channel for channel in ("A", "E", "T") if channel in channels]
    if not ordered_channels:
        raise ValueError("No A/E/T channels were provided.")

    try:
        plt = _load_pyplot()
    except Exception:
        return _save_frequency_domain_preview_pil(
            output_path,
            time_s=time_s,
            channels=channels,
            frequencies=frequencies,
            frequency_mask=frequency_mask,
            title=title,
            detector_title=detector_title,
            population_label=population_label,
            ordered_channels=ordered_channels,
            max_points=max_points,
            lower=lower,
            upper=upper,
        )

    fig, axes = plt.subplots(
        len(ordered_channels),
        1,
        figsize=(13.0, max(7.0, 2.8 * len(ordered_channels))),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)

    for axis, channel in zip(axes_array, ordered_channels, strict=True):
        values = np.asarray(channels[channel], dtype=np.float64)
        if values.shape != time_s.shape:
            raise ValueError(f"Channel {channel} has shape {values.shape}, expected {time_s.shape}.")
        freq_values, amp_values = _sample_frequency_channel(
            frequencies,
            frequency_mask,
            values,
            max_points=max(2, int(max_points)),
        )
        if amp_values.size == 0:
            continue
        axis.loglog(freq_values, amp_values, linewidth=0.45, color="#1f77b4")
        axis.set_title(f"{detector_title} {population_label} {channel} channel", fontsize=12)
        axis.set_ylabel(rf"$|\tilde{{h}}_{{{channel}}}(f)|$")
        axis.grid(True, which="both", linewidth=0.35, alpha=0.32)

    axes_array[-1].set_xlabel("Frequency [Hz]")
    axes_array[-1].set_xlim(lower, upper)
    fig.suptitle(f"{title}, {_format_frequency_range(lower, upper)}", fontsize=13)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote frequency-domain preview to {path}", flush=True)
    return path


def _sample_frequency_channel(
    frequencies: np.ndarray,
    frequency_mask: np.ndarray,
    values: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = np.abs(np.fft.rfft(values))
    freq_values = frequencies[frequency_mask]
    amp_values = amplitude[frequency_mask]
    sample_indices = _frequency_plot_sample_indices(amp_values, max_points)
    if sample_indices.size == 0:
        return freq_values[:0], amp_values[:0]
    return freq_values[sample_indices], amp_values[sample_indices]


def _save_frequency_domain_preview_pil(
    output_path: str | Path,
    *,
    time_s: np.ndarray,
    channels: Mapping[str, np.ndarray],
    frequencies: np.ndarray,
    frequency_mask: np.ndarray,
    title: str,
    detector_title: str,
    population_label: str,
    ordered_channels: list[str],
    max_points: int,
    lower: float,
    upper: float,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    width = 1400
    panel_height = 300
    top_margin = 62
    bottom_margin = 58
    gap = 28
    left_margin = 104
    right_margin = 44
    height = top_margin + bottom_margin + len(ordered_channels) * panel_height + (len(ordered_channels) - 1) * gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_color = (38, 38, 38)
    grid_color = (220, 220, 220)
    axis_color = (80, 80, 80)
    line_color = (31, 119, 180)

    draw.text((left_margin, 20), f"{title}, {_format_frequency_range(lower, upper)}", fill=text_color, font=font)
    plot_left = left_margin
    plot_right = width - right_margin
    plot_width = plot_right - plot_left
    log_x_min = np.log10(lower)
    log_x_max = np.log10(upper)

    for index, channel in enumerate(ordered_channels):
        values = np.asarray(channels[channel], dtype=np.float64)
        if values.shape != time_s.shape:
            raise ValueError(f"Channel {channel} has shape {values.shape}, expected {time_s.shape}.")
        freq_values, amp_values = _sample_frequency_channel(
            frequencies,
            frequency_mask,
            values,
            max_points=max(2, min(int(max_points), 120_000)),
        )
        panel_top = top_margin + index * (panel_height + gap)
        panel_bottom = panel_top + panel_height
        plot_top = panel_top + 34
        plot_bottom = panel_bottom - 34
        plot_height = plot_bottom - plot_top

        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=axis_color)
        for tick in range(1, 5):
            x = plot_left + tick * plot_width / 5.0
            draw.line((x, plot_top, x, plot_bottom), fill=grid_color)
        for tick in range(1, 4):
            y = plot_top + tick * plot_height / 4.0
            draw.line((plot_left, y, plot_right, y), fill=grid_color)

        if amp_values.size:
            y_min = float(np.nanmin(amp_values))
            y_max = float(np.nanmax(amp_values))
            if y_min <= 0.0 or not np.isfinite(y_min):
                y_min = float(np.nanmin(amp_values[amp_values > 0.0]))
            if y_min == y_max:
                y_min *= 0.8
                y_max *= 1.2
            log_y_min = np.floor(np.log10(y_min))
            log_y_max = np.ceil(np.log10(y_max))
            if log_y_min == log_y_max:
                log_y_min -= 1.0
                log_y_max += 1.0
            _draw_log_frequency_series(
                draw,
                freq_values,
                amp_values,
                log_x_min=log_x_min,
                log_x_max=log_x_max,
                log_y_min=log_y_min,
                log_y_max=log_y_max,
                plot_left=plot_left,
                plot_right=plot_right,
                plot_top=plot_top,
                plot_bottom=plot_bottom,
                color=line_color,
            )
            draw.text((plot_left, panel_bottom - 24), f"1e{int(log_y_min)}", fill=text_color, font=font)
            draw.text((plot_right - 62, panel_bottom - 24), f"1e{int(log_y_max)}", fill=text_color, font=font)

        draw.text((plot_left, panel_top + 10), f"{detector_title} {population_label} {channel} channel", fill=text_color, font=font)
        draw.text((12, panel_top + 44), f"|h~_{channel}(f)|", fill=text_color, font=font)

    draw.text((plot_left, height - 34), "Frequency [Hz]", fill=text_color, font=font)
    draw.text((plot_left, height - 20), f"{lower:.1e}", fill=text_color, font=font)
    draw.text((plot_right - 72, height - 20), f"{upper:.1e}", fill=text_color, font=font)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"Wrote frequency-domain preview to {path}", flush=True)
    return path


def _draw_log_frequency_series(
    draw: Any,
    freq_values: np.ndarray,
    amp_values: np.ndarray,
    *,
    log_x_min: float,
    log_x_max: float,
    log_y_min: float,
    log_y_max: float,
    plot_left: int,
    plot_right: int,
    plot_top: int,
    plot_bottom: int,
    color: tuple[int, int, int],
) -> None:
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    points: list[tuple[float, float]] = []
    for freq, amp in zip(freq_values, amp_values, strict=True):
        if not np.isfinite(freq) or not np.isfinite(amp) or freq <= 0.0 or amp <= 0.0:
            if len(points) > 1:
                draw.line(points, fill=color, width=1)
            points = []
            continue
        x_pixel = plot_left + (np.log10(float(freq)) - log_x_min) / (log_x_max - log_x_min) * plot_width
        y_pixel = plot_bottom - (np.log10(float(amp)) - log_y_min) / (log_y_max - log_y_min) * plot_height
        points.append((x_pixel, y_pixel))
    if len(points) > 1:
        draw.line(points, fill=color, width=1)


def _positive_min_max(*arrays: np.ndarray) -> tuple[float, float]:
    positive_values = []
    for values in arrays:
        array = np.asarray(values, dtype=np.float64)
        mask = np.isfinite(array) & (array > 0.0)
        if np.any(mask):
            positive_values.append(array[mask])
    if not positive_values:
        return 1.0e-30, 1.0
    merged = np.concatenate(positive_values)
    y_min = float(np.nanmin(merged))
    y_max = float(np.nanmax(merged))
    if y_min == y_max:
        y_min *= 0.8
        y_max *= 1.2
    return y_min, y_max


def _log_padded_limits(*arrays: np.ndarray) -> tuple[float, float]:
    y_min, y_max = _positive_min_max(*arrays)
    log_y_min = np.floor(np.log10(y_min))
    log_y_max = np.ceil(np.log10(y_max))
    if log_y_min == log_y_max:
        log_y_min -= 1.0
        log_y_max += 1.0
    return float(10.0**log_y_min), float(10.0**log_y_max)


def _frequency_plot_range(
    time_s: np.ndarray,
    *,
    f_min_hz: float,
    f_max_hz: float | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    time_s = np.asarray(time_s, dtype=np.float64)
    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must contain at least two samples.")
    dt = float(time_s[1] - time_s[0])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("time_s must be strictly increasing.")
    frequencies = np.fft.rfftfreq(time_s.size, d=dt)
    upper = float(frequencies[-1]) if f_max_hz is None else min(float(f_max_hz), float(frequencies[-1]))
    lower = max(float(f_min_hz), float(frequencies[1]) if frequencies.size > 1 else 0.0)
    if upper <= lower:
        raise ValueError(f"Invalid frequency plot range: {lower}--{upper} Hz.")
    frequency_mask = (frequencies >= lower) & (frequencies <= upper)
    if not np.any(frequency_mask):
        raise ValueError(f"No FFT bins fall in frequency range {lower}--{upper} Hz.")
    return frequencies, frequency_mask, lower, upper


def _sample_frequency_pair(
    frequencies: np.ndarray,
    frequency_mask: np.ndarray,
    data_amplitude: np.ndarray,
    component: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_amplitude = np.asarray(data_amplitude, dtype=np.float64)
    component_amplitude = np.abs(np.fft.rfft(np.asarray(component, dtype=np.float64)))
    freq_values = frequencies[frequency_mask]
    data_values = data_amplitude[frequency_mask]
    component_values = component_amplitude[frequency_mask]
    points_per_series = max(2, int(max_points) // 2)
    sample_indices = np.unique(
        np.concatenate(
            (
                _frequency_plot_sample_indices(data_values, points_per_series),
                _frequency_plot_sample_indices(component_values, points_per_series),
            )
        )
    )
    if sample_indices.size == 0:
        return freq_values[:0], data_values[:0], component_values[:0]
    return freq_values[sample_indices], data_values[sample_indices], component_values[sample_indices]


def _positive_log_series(
    freq_values: np.ndarray,
    amp_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(freq_values) & np.isfinite(amp_values) & (freq_values > 0.0) & (amp_values > 0.0)
    return freq_values[mask], amp_values[mask]


def save_a_channel_source_frequency_contribution_plot(
    output_path: str | Path,
    *,
    time_s: np.ndarray,
    mixture_a: np.ndarray,
    components: list[tuple[str, np.ndarray, str]],
    title: str,
    max_points: int,
    f_min_hz: float,
    f_max_hz: float | None,
    dpi: int,
) -> Path:
    if not components:
        raise ValueError("At least one source component is required.")
    time_s = np.asarray(time_s, dtype=np.float64)
    mixture_a = np.asarray(mixture_a, dtype=np.float64)
    if time_s.shape != mixture_a.shape:
        raise ValueError(f"mixture_a has shape {mixture_a.shape}, expected {time_s.shape}.")
    frequencies, frequency_mask, lower, upper = _frequency_plot_range(
        time_s,
        f_min_hz=f_min_hz,
        f_max_hz=f_max_hz,
    )
    mixture_amplitude = np.abs(np.fft.rfft(mixture_a))

    try:
        plt = _load_pyplot()
    except Exception:
        return _save_a_channel_source_frequency_contribution_plot_pil(
            output_path,
            time_s=time_s,
            mixture_a=mixture_a,
            components=components,
            frequencies=frequencies,
            frequency_mask=frequency_mask,
            title=title,
            max_points=max_points,
            lower=lower,
            upper=upper,
        )

    fig, axes = plt.subplots(
        len(components),
        1,
        figsize=(13.0, max(7.0, 2.25 * len(components))),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)

    for axis, (label, values, color) in zip(axes_array, components, strict=True):
        component = np.asarray(values, dtype=np.float64)
        if component.shape != time_s.shape:
            raise ValueError(f"Component '{label}' has shape {component.shape}, expected {time_s.shape}.")
        freq_values, data_values, component_values = _sample_frequency_pair(
            frequencies,
            frequency_mask,
            mixture_amplitude,
            component,
            max_points=max(2, int(max_points)),
        )
        data_freq, data_amp = _positive_log_series(freq_values, data_values)
        component_freq, component_amp = _positive_log_series(freq_values, component_values)
        if data_amp.size:
            axis.loglog(data_freq, data_amp, color=DATA_COLOR, linewidth=0.45, alpha=0.78, label="Data")
        if component_amp.size:
            axis.loglog(component_freq, component_amp, color=color, linewidth=0.85, alpha=0.95, label=label)
        axis.set_title(label, fontsize=10, loc="left")
        axis.set_ylabel(r"$|\tilde{h}_A(f)|$")
        axis.set_ylim(*_log_padded_limits(data_values, component_values))
        axis.grid(True, which="both", linewidth=0.35, alpha=0.32)
        axis.legend(loc="upper right", fontsize=8, frameon=True)

    axes_array[-1].set_xlabel("Frequency [Hz]")
    axes_array[-1].set_xlim(lower, upper)
    fig.suptitle(f"{title}, {_format_frequency_range(lower, upper)}", fontsize=13)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote A-channel source frequency contribution plot to {path}", flush=True)
    return path


def _save_a_channel_source_frequency_contribution_plot_pil(
    output_path: str | Path,
    *,
    time_s: np.ndarray,
    mixture_a: np.ndarray,
    components: list[tuple[str, np.ndarray, str]],
    frequencies: np.ndarray,
    frequency_mask: np.ndarray,
    title: str,
    max_points: int,
    lower: float,
    upper: float,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    width = 1400
    panel_height = 265
    top_margin = 62
    bottom_margin = 58
    gap = 24
    left_margin = 104
    right_margin = 44
    height = top_margin + bottom_margin + len(components) * panel_height + (len(components) - 1) * gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_color = (38, 38, 38)
    grid_color = (220, 220, 220)
    axis_color = (80, 80, 80)
    data_color = _hex_to_rgb(DATA_COLOR)
    mixture_amplitude = np.abs(np.fft.rfft(mixture_a))

    draw.text((left_margin, 20), f"{title}, {_format_frequency_range(lower, upper)}", fill=text_color, font=font)
    plot_left = left_margin
    plot_right = width - right_margin
    plot_width = plot_right - plot_left
    log_x_min = np.log10(lower)
    log_x_max = np.log10(upper)

    for index, (label, values, color) in enumerate(components):
        component = np.asarray(values, dtype=np.float64)
        if component.shape != time_s.shape:
            raise ValueError(f"Component '{label}' has shape {component.shape}, expected {time_s.shape}.")
        freq_values, data_values, component_values = _sample_frequency_pair(
            frequencies,
            frequency_mask,
            mixture_amplitude,
            component,
            max_points=max(2, min(int(max_points), 120_000)),
        )
        y_min, y_max = _log_padded_limits(data_values, component_values)
        log_y_min = np.log10(y_min)
        log_y_max = np.log10(y_max)

        panel_top = top_margin + index * (panel_height + gap)
        panel_bottom = panel_top + panel_height
        plot_top = panel_top + 34
        plot_bottom = panel_bottom - 34
        plot_height = plot_bottom - plot_top

        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=axis_color)
        for tick in range(1, 5):
            x = plot_left + tick * plot_width / 5.0
            draw.line((x, plot_top, x, plot_bottom), fill=grid_color)
        for tick in range(1, 4):
            y = plot_top + tick * plot_height / 4.0
            draw.line((plot_left, y, plot_right, y), fill=grid_color)

        _draw_log_frequency_series(
            draw,
            freq_values,
            data_values,
            log_x_min=log_x_min,
            log_x_max=log_x_max,
            log_y_min=log_y_min,
            log_y_max=log_y_max,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            color=data_color,
        )
        _draw_log_frequency_series(
            draw,
            freq_values,
            component_values,
            log_x_min=log_x_min,
            log_x_max=log_x_max,
            log_y_min=log_y_min,
            log_y_max=log_y_max,
            plot_left=plot_left,
            plot_right=plot_right,
            plot_top=plot_top,
            plot_bottom=plot_bottom,
            color=_hex_to_rgb(color),
        )

        draw.text((12, panel_top + 44), "|h~_A(f)|", fill=text_color, font=font)
        draw.text((plot_left, panel_top + 10), label, fill=text_color, font=font)
        draw.text((plot_left, panel_bottom - 24), f"{y_min:.1e}", fill=text_color, font=font)
        draw.text((plot_right - 72, panel_bottom - 24), f"{y_max:.1e}", fill=text_color, font=font)
        legend_y = panel_top + 10
        draw.line((plot_right - 170, legend_y + 6, plot_right - 145, legend_y + 6), fill=data_color, width=2)
        draw.text((plot_right - 140, legend_y), "Data", fill=text_color, font=font)
        draw.line((plot_right - 96, legend_y + 6, plot_right - 71, legend_y + 6), fill=_hex_to_rgb(color), width=2)
        draw.text((plot_right - 66, legend_y), label, fill=text_color, font=font)

    draw.text((plot_left, height - 34), "Frequency [Hz]", fill=text_color, font=font)
    draw.text((plot_left, height - 20), f"{lower:.1e}", fill=text_color, font=font)
    draw.text((plot_right - 72, height - 20), f"{upper:.1e}", fill=text_color, font=font)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"Wrote A-channel source frequency contribution plot to {path}", flush=True)
    return path


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
        print(f"  no_noise_output: {job.no_noise_output_path}")
        print(f"  mixture_plot: {job.mixture_plot_path}")
        print(f"  no_noise_mixture_plot: {job.no_noise_mixture_plot_path}")
        print(f"  frequency_plot: {job.frequency_plot_path}")
        print(f"  no_noise_frequency_plot: {job.no_noise_frequency_plot_path}")
        print(f"  source_plot: {job.source_plot_path}")
        print(f"  source_frequency_plot: {job.source_frequency_plot_path}")
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
