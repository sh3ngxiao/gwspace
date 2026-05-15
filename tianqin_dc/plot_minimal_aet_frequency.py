from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


DEFAULT_INPUT = (
    "/public/home/zhuangzhenye/jobs/gwspace_runs/"
    "minimal_aet_tianqin_tdi2_dt1s/"
    "smbhb_Q3nod_K16_100_uniformtc_eccfd_aet.h5"
)

CHANNEL_FIELDS = {
    "A": ("a", 1),
    "E": ("e", 2),
    "T": ("t", 3),
}


@dataclass(frozen=True)
class MinimalAETDataset:
    path: Path
    rows: int
    sample_spacing_s: float
    duration_s: float
    compound: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a minimal A/E/T time-domain HDF5 file and save a one-sided "
            "frequency-domain plot."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input minimal A/E/T HDF5 file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to '<input_stem>_frequency.png' next to the input file.",
    )
    parser.add_argument(
        "--channels",
        default="A,E,T",
        help="Comma-separated channels to plot, chosen from A,E,T.",
    )
    parser.add_argument(
        "--quantity",
        choices=("asd", "psd"),
        default="asd",
        help="Plot amplitude spectral density or power spectral density.",
    )
    parser.add_argument(
        "--window",
        choices=("hann", "boxcar"),
        default="hann",
        help="Window applied before FFT.",
    )
    parser.add_argument(
        "--detrend",
        choices=("mean", "none"),
        default="mean",
        help="Remove the channel mean before FFT.",
    )
    parser.add_argument(
        "--sample-spacing-s",
        type=float,
        default=None,
        help="Override the time spacing in seconds. If omitted, infer it from /data time.",
    )
    parser.add_argument("--f-min", type=float, default=None, help="Minimum frequency to plot in Hz.")
    parser.add_argument("--f-max", type=float, default=None, help="Maximum frequency to plot in Hz.")
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=50_000,
        help="Maximum plotted points per channel after logarithmic binning.",
    )
    parser.add_argument(
        "--bin-stat",
        choices=("max", "mean", "median"),
        default="max",
        help="Statistic used when compressing many frequency bins for plotting.",
    )
    parser.add_argument("--title", default=None, help="Optional figure title.")
    parser.add_argument("--dpi", type=int, default=180, help="Output figure DPI.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    channels = parse_channels(args.channels)

    with h5py.File(input_path, "r") as handle:
        dataset = require_minimal_dataset(handle, input_path)
        info = inspect_dataset(
            input_path,
            dataset,
            sample_spacing_s=args.sample_spacing_s,
        )
        frequency_hz = np.fft.rfftfreq(info.rows, d=info.sample_spacing_s)
        print(
            "Input summary: "
            f"rows={info.rows}, dt={info.sample_spacing_s:g} s, "
            f"duration={info.duration_s / 86400.0:g} day, "
            f"Nyquist={frequency_hz[-1]:g} Hz",
            flush=True,
        )

        plot_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for channel in channels:
            print(f"Reading channel {channel}.", flush=True)
            values = read_channel(dataset, channel)
            print(f"Computing {args.quantity.upper()} for channel {channel}.", flush=True)
            spectrum_values = one_sided_spectral_quantity(
                values,
                sample_spacing_s=info.sample_spacing_s,
                quantity=args.quantity,
                window=args.window,
                detrend=args.detrend,
            )
            del values
            plot_series[channel] = select_plot_points(
                frequency_hz,
                spectrum_values,
                f_min=args.f_min,
                f_max=args.f_max,
                max_points=args.max_plot_points,
                bin_stat=args.bin_stat,
            )
            del spectrum_values
            print(
                f"Prepared {plot_series[channel][0].size} plotted frequency points for channel {channel}.",
                flush=True,
            )

    save_frequency_plot(
        plot_series,
        output_path,
        title=args.title or f"{input_path.name} A/E/T {args.quantity.upper()}",
        quantity=args.quantity,
        f_min=args.f_min,
        f_max=args.f_max,
        sample_spacing_s=info.sample_spacing_s,
        rows=info.rows,
        dpi=args.dpi,
    )
    print(f"Wrote frequency-domain plot to {output_path}", flush=True)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_frequency.png")


def parse_channels(value: str) -> tuple[str, ...]:
    channels = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not channels:
        raise ValueError("At least one channel must be requested.")
    invalid = [channel for channel in channels if channel not in CHANNEL_FIELDS]
    if invalid:
        raise ValueError(f"Unknown channel(s): {invalid}. Expected a comma-separated subset of A,E,T.")
    return channels


def require_minimal_dataset(handle: h5py.File, path: Path) -> h5py.Dataset:
    if "data" not in handle:
        raise ValueError(f"Minimal AET file '{path}' must contain the root dataset '/data'.")
    dataset = handle["data"]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"'{path}:/data' must be a dataset.")
    if dataset.ndim == 1 and dataset.dtype.fields is not None:
        fields = dataset.dtype.fields
        required = ("time", "a", "e", "t")
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"Compound minimal AET dataset is missing fields: {missing}.")
        return dataset
    if dataset.ndim == 2 and dataset.shape[1] == 4:
        return dataset
    raise ValueError(
        f"'{path}:/data' must be either a compound row dataset with time/a/e/t "
        f"fields or a two-dimensional array with four columns; got shape={dataset.shape}, dtype={dataset.dtype}."
    )


def inspect_dataset(
    path: Path,
    dataset: h5py.Dataset,
    *,
    sample_spacing_s: float | None,
) -> MinimalAETDataset:
    rows = int(dataset.shape[0])
    if rows < 2:
        raise ValueError(f"'{path}:/data' must contain at least two samples.")
    compound = dataset.dtype.fields is not None
    dt = float(sample_spacing_s) if sample_spacing_s is not None else infer_sample_spacing(dataset)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Invalid sample spacing inferred from '{path}:/data': {dt}.")
    return MinimalAETDataset(
        path=path,
        rows=rows,
        sample_spacing_s=dt,
        duration_s=(rows - 1) * dt,
        compound=compound,
    )


def infer_sample_spacing(dataset: h5py.Dataset, *, probe_rows: int = 10_000) -> float:
    rows = int(dataset.shape[0])
    probe_stop = min(rows, max(2, int(probe_rows)))
    time_sample = read_time_values(dataset, slice(0, probe_stop))
    finite = np.isfinite(time_sample)
    if np.count_nonzero(finite) >= 2:
        diffs = np.diff(time_sample[finite])
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size:
            return float(np.median(diffs))

    first = float(read_time_values(dataset, 0))
    last = float(read_time_values(dataset, rows - 1))
    return (last - first) / float(rows - 1)


def read_time_values(dataset: h5py.Dataset, selection):
    if dataset.dtype.fields is not None:
        return dataset.fields("time")[selection]
    return dataset[selection, 0]


def read_channel(dataset: h5py.Dataset, channel: str) -> np.ndarray:
    field_name, column_index = CHANNEL_FIELDS[channel]
    if dataset.dtype.fields is not None:
        values = dataset.fields(field_name)[:]
    else:
        values = dataset[:, column_index]
    return np.asarray(values, dtype=np.float64)


def one_sided_spectral_quantity(
    values: np.ndarray,
    *,
    sample_spacing_s: float,
    quantity: str,
    window: str,
    detrend: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    rows = values.size
    if rows < 2:
        raise ValueError("At least two samples are required for an FFT.")

    finite = np.isfinite(values)
    if np.all(finite):
        mean_value = float(np.mean(values))
    elif np.any(finite):
        mean_value = float(np.mean(values[finite]))
        values[~finite] = mean_value
    else:
        raise ValueError("Channel contains no finite samples.")

    if detrend == "mean":
        values -= mean_value
    elif detrend != "none":
        raise ValueError(f"Unsupported detrend option: {detrend}")

    window_power = apply_window_in_place(values, window)
    spectrum = np.fft.rfft(values)
    spectral_density = np.abs(spectrum)
    del spectrum
    spectral_density *= spectral_density
    spectral_density *= sample_spacing_s / (rows * window_power)
    if spectral_density.size > 1:
        if rows % 2 == 0:
            spectral_density[1:-1] *= 2.0
        else:
            spectral_density[1:] *= 2.0

    if quantity == "psd":
        return spectral_density
    if quantity == "asd":
        return np.sqrt(spectral_density, out=spectral_density)
    raise ValueError(f"Unsupported quantity: {quantity}")


def apply_window_in_place(values: np.ndarray, window: str) -> float:
    if window == "boxcar":
        return 1.0
    if window == "hann":
        weights = np.hanning(values.size)
        values *= weights
        return float(np.mean(weights * weights))
    raise ValueError(f"Unsupported window: {window}")


def select_plot_points(
    frequency_hz: np.ndarray,
    values: np.ndarray,
    *,
    f_min: float | None,
    f_max: float | None,
    max_points: int,
    bin_stat: str,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(frequency_hz) & np.isfinite(values) & (frequency_hz > 0.0) & (values > 0.0)
    if f_min is not None:
        mask &= frequency_hz >= float(f_min)
    if f_max is not None:
        mask &= frequency_hz <= float(f_max)

    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if selected.size <= max(2, int(max_points)):
        return frequency_hz[selected], values[selected]
    return log_bin_series(frequency_hz[selected], values[selected], max_points=max_points, bin_stat=bin_stat)


def log_bin_series(
    frequency_hz: np.ndarray,
    values: np.ndarray,
    *,
    max_points: int,
    bin_stat: str,
) -> tuple[np.ndarray, np.ndarray]:
    max_points = max(2, int(max_points))
    edges = np.geomspace(float(frequency_hz[0]), float(frequency_hz[-1]), num=max_points + 1)
    starts = np.searchsorted(frequency_hz, edges[:-1], side="left")
    stops = np.searchsorted(frequency_hz, edges[1:], side="right")
    out_frequency: list[float] = []
    out_values: list[float] = []
    for start, stop, left, right in zip(starts, stops, edges[:-1], edges[1:], strict=True):
        if stop <= start:
            continue
        segment = values[start:stop]
        finite = np.isfinite(segment)
        if not np.any(finite):
            continue
        if bin_stat == "max":
            finite_offsets = np.flatnonzero(finite)
            offset = finite_offsets[int(np.argmax(segment[finite]))]
            out_frequency.append(float(frequency_hz[start + offset]))
            out_values.append(float(segment[offset]))
        elif bin_stat == "mean":
            out_frequency.append(float(np.sqrt(left * right)))
            out_values.append(float(np.mean(segment[finite])))
        elif bin_stat == "median":
            out_frequency.append(float(np.sqrt(left * right)))
            out_values.append(float(np.median(segment[finite])))
        else:
            raise ValueError(f"Unsupported bin statistic: {bin_stat}")
    return np.asarray(out_frequency, dtype=np.float64), np.asarray(out_values, dtype=np.float64)


def save_frequency_plot(
    series_by_channel: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    title: str,
    quantity: str,
    f_min: float | None,
    f_max: float | None,
    sample_spacing_s: float,
    rows: int,
    dpi: int,
) -> None:
    plt = load_pyplot()
    ordered = [channel for channel in ("A", "E", "T") if channel in series_by_channel]
    ylabel = "ASD [channel / sqrt(Hz)]" if quantity == "asd" else "PSD [channel^2 / Hz]"
    duration_days = (rows - 1) * sample_spacing_s / 86400.0
    subtitle = f"N={rows}, dt={sample_spacing_s:g} s, T={duration_days:g} day"
    if plt is None:
        save_frequency_plot_pil(
            series_by_channel,
            ordered,
            output_path,
            title=title,
            subtitle=subtitle,
            ylabel=ylabel,
            f_min=f_min,
            f_max=f_max,
        )
        return

    fig, axes = plt.subplots(
        len(ordered),
        1,
        figsize=(11.5, max(3.2, 2.5 * len(ordered))),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    colors = {
        "A": "#1f77b4",
        "E": "#d62728",
        "T": "#2ca02c",
    }

    for axis, channel in zip(axes_array, ordered, strict=True):
        frequency, values = series_by_channel[channel]
        if frequency.size:
            axis.loglog(frequency, values, linewidth=0.75, color=colors.get(channel, "#333333"))
        axis.set_ylabel(f"{channel}\n{ylabel}")
        axis.grid(True, which="both", linewidth=0.4, alpha=0.35)

    axes_array[-1].set_xlabel("Frequency [Hz]")
    if f_min is not None or f_max is not None:
        axes_array[-1].set_xlim(left=f_min, right=f_max)

    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.965, subtitle, ha="center", va="top", fontsize=9, color="0.35")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi))
    plt.close(fig)


def load_pyplot():
    import contextlib
    import io

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Matplotlib is unavailable; falling back to Pillow plot: {exc}", flush=True)
        return None

    return plt


def save_frequency_plot_pil(
    series_by_channel: dict[str, tuple[np.ndarray, np.ndarray]],
    ordered: list[str],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    ylabel: str,
    f_min: float | None,
    f_max: float | None,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Neither Matplotlib nor Pillow is available for plotting.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1400
    panel_height = 250
    left = 120
    right = width - 45
    top = 84
    bottom_margin = 70
    gap = 24
    height = top + bottom_margin + len(ordered) * panel_height + max(0, len(ordered) - 1) * gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = {
        "A": (31, 119, 180),
        "E": (214, 39, 40),
        "T": (44, 160, 44),
    }
    axis_color = (70, 70, 70)
    grid_color = (224, 224, 224)
    text_color = (35, 35, 35)

    draw.text((left, 22), title, fill=text_color, font=font)
    draw.text((left, 44), subtitle, fill=(80, 80, 80), font=font)
    x_limits = frequency_limits(series_by_channel, f_min=f_min, f_max=f_max)

    for panel_index, channel in enumerate(ordered):
        y_top = top + panel_index * (panel_height + gap)
        y_bottom = y_top + panel_height
        plot_top = y_top + 15
        plot_bottom = y_bottom - 32
        frequency, values = series_by_channel[channel]
        valid = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0.0) & (values > 0.0)
        if x_limits is not None:
            valid &= frequency >= x_limits[0]
            valid &= frequency <= x_limits[1]
        y_limits = value_limits(values[valid])

        draw.rectangle((left, plot_top, right, plot_bottom), outline=axis_color)
        for grid_index in range(1, 5):
            x_grid = left + grid_index * (right - left) / 5.0
            draw.line((x_grid, plot_top, x_grid, plot_bottom), fill=grid_color)
        for grid_index in range(1, 4):
            y_grid = plot_top + grid_index * (plot_bottom - plot_top) / 4.0
            draw.line((left, y_grid, right, y_grid), fill=grid_color)
        draw.text((14, y_top + 18), f"{channel}", fill=text_color, font=font)
        draw.text((14, y_top + 36), ylabel, fill=(80, 80, 80), font=font)

        if x_limits is None or y_limits is None:
            continue
        x_min, x_max = x_limits
        y_min, y_max = y_limits
        draw.text((left, y_bottom - 20), f"{x_min:.2e}", fill=text_color, font=font)
        draw.text((right - 72, y_bottom - 20), f"{x_max:.2e}", fill=text_color, font=font)
        draw.text((left + 6, plot_top + 5), f"{y_max:.2e}", fill=text_color, font=font)
        draw.text((left + 6, plot_bottom - 16), f"{y_min:.2e}", fill=text_color, font=font)

        log_x_min = np.log10(x_min)
        log_x_span = max(np.log10(x_max) - log_x_min, np.finfo(float).eps)
        log_y_min = np.log10(y_min)
        log_y_span = max(np.log10(y_max) - log_y_min, np.finfo(float).eps)
        x_values = left + (np.log10(frequency[valid]) - log_x_min) / log_x_span * (right - left)
        y_values = plot_bottom - (np.log10(values[valid]) - log_y_min) / log_y_span * (plot_bottom - plot_top)
        points = list(zip(x_values.astype(int), y_values.astype(int)))
        if len(points) > 1:
            draw.line(points, fill=colors.get(channel, (40, 40, 40)), width=1)

    draw.text((int((left + right) / 2) - 48, height - 42), "Frequency [Hz]", fill=text_color, font=font)
    image.save(output_path)


def frequency_limits(
    series_by_channel: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    f_min: float | None,
    f_max: float | None,
) -> tuple[float, float] | None:
    minimum = float(f_min) if f_min is not None else np.inf
    maximum = float(f_max) if f_max is not None else 0.0
    for frequency, values in series_by_channel.values():
        valid = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0.0) & (values > 0.0)
        if not np.any(valid):
            continue
        if f_min is None:
            minimum = min(minimum, float(np.min(frequency[valid])))
        if f_max is None:
            maximum = max(maximum, float(np.max(frequency[valid])))
    if not np.isfinite(minimum) or maximum <= minimum:
        return None
    return minimum, maximum


def value_limits(values: np.ndarray) -> tuple[float, float] | None:
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return None
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if y_max <= y_min:
        y_min = max(y_min * 0.5, np.nextafter(0.0, 1.0))
        y_max *= 2.0
    return y_min, y_max


if __name__ == "__main__":
    main()
