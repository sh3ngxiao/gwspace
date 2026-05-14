from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


DEFAULT_MAX_TIME_DOMAIN_PLOT_POINTS = 200_000


def default_time_domain_plot_path(hdf5_path: str | Path) -> Path:
    path = Path(hdf5_path)
    return path.with_name(f"{path.stem}_time.png")


def save_time_domain_preview(
    hdf5_path: str | Path,
    time_s: np.ndarray,
    channels: Mapping[str, np.ndarray],
    *,
    title: str | None = None,
    output_path: str | Path | None = None,
    max_points: int = DEFAULT_MAX_TIME_DOMAIN_PLOT_POINTS,
    show_stats: bool = True,
    fail_on_error: bool = False,
) -> Path | None:
    """Save a compact time-domain preview next to an HDF5 output file."""

    plot_path = Path(output_path) if output_path is not None else default_time_domain_plot_path(hdf5_path)
    try:
        return _save_time_domain_preview(
            plot_path,
            np.asarray(time_s, dtype=np.float64),
            channels,
            title=title or Path(hdf5_path).name,
            max_points=max_points,
            show_stats=show_stats,
        )
    except Exception as exc:
        if fail_on_error:
            raise
        print(f"Skipping time-domain preview for {hdf5_path}: {exc}", flush=True)
        return None


def _save_time_domain_preview(
    plot_path: Path,
    time_s: np.ndarray,
    channels: Mapping[str, np.ndarray],
    *,
    title: str,
    max_points: int,
    show_stats: bool,
) -> Path:
    if time_s.ndim != 1:
        raise ValueError(f"time_s must be one-dimensional, got shape {time_s.shape}.")
    if time_s.size == 0:
        raise ValueError("time_s is empty.")

    ordered_channels = _ordered_channels(channels)
    if not ordered_channels:
        raise ValueError("No channels were provided.")

    max_points = max(2, int(max_points))
    time_scale, time_unit = _time_scale_and_unit(time_s)

    try:
        plt = _load_pyplot()
    except Exception:
        return _save_time_domain_preview_pil(
            plot_path,
            time_s,
            channels,
            ordered_channels,
            title=title,
            time_scale=time_scale,
            time_unit=time_unit,
            max_points=max_points,
            show_stats=show_stats,
        )

    fig, axes = plt.subplots(
        len(ordered_channels),
        1,
        figsize=(12.0, max(3.0, 2.2 * len(ordered_channels))),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    colors = {
        "A": "#1f77b4",
        "E": "#d62728",
        "T": "#2ca02c",
        "X": "#1f77b4",
        "Y": "#d62728",
        "Z": "#2ca02c",
    }

    for axis, channel in zip(axes_array, ordered_channels, strict=True):
        values = np.asarray(channels[channel], dtype=np.float64)
        if values.shape != time_s.shape:
            raise ValueError(f"Channel {channel} has shape {values.shape}, expected {time_s.shape}.")

        sample_indices = _extrema_preserving_sample_indices(values, max_points)
        x_values = time_s[sample_indices] / time_scale
        sampled = values[sample_indices]
        axis.plot(x_values, sampled, linewidth=0.55, color=colors.get(channel, "#333333"))
        axis.set_ylabel(channel)
        axis.grid(True, linewidth=0.4, alpha=0.35)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        if show_stats:
            axis.text(
                0.99,
                0.86,
                _channel_stats(values),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
            )

    axes_array[-1].set_xlabel(f"Time [{time_unit}]")
    fig.suptitle(title, fontsize=12)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"Wrote time-domain preview to {plot_path}", flush=True)
    return plot_path


def _load_pyplot():
    import contextlib
    import io

    with contextlib.redirect_stderr(io.StringIO()):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

    return plt


def _save_time_domain_preview_pil(
    plot_path: Path,
    time_s: np.ndarray,
    channels: Mapping[str, np.ndarray],
    ordered_channels: list[str],
    *,
    title: str,
    time_scale: float,
    time_unit: str,
    max_points: int,
    show_stats: bool,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    width = 1200
    panel_height = 230
    top_margin = 54
    bottom_margin = 54
    gap = 20
    left_margin = 92
    right_margin = 34
    height = top_margin + bottom_margin + len(ordered_channels) * panel_height + (len(ordered_channels) - 1) * gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = {
        "A": (31, 119, 180),
        "E": (214, 39, 40),
        "T": (44, 160, 44),
        "X": (31, 119, 180),
        "Y": (214, 39, 40),
        "Z": (44, 160, 44),
    }
    text_color = (38, 38, 38)
    grid_color = (220, 220, 220)
    axis_color = (80, 80, 80)

    draw.text((left_margin, 20), title, fill=text_color, font=font)
    x_min = float(time_s[0] / time_scale)
    x_max = float(time_s[-1] / time_scale)
    if x_max == x_min:
        x_max = x_min + 1.0

    plot_left = left_margin
    plot_right = width - right_margin
    plot_width = plot_right - plot_left

    for index, channel in enumerate(ordered_channels):
        panel_top = top_margin + index * (panel_height + gap)
        panel_bottom = panel_top + panel_height
        plot_top = panel_top + 18
        plot_bottom = panel_bottom - 28
        plot_height = plot_bottom - plot_top

        values = np.asarray(channels[channel], dtype=np.float64)
        sample_indices = _extrema_preserving_sample_indices(values, max_points)
        x_values = time_s[sample_indices] / time_scale
        sampled = values[sample_indices]
        y_min, y_max = _finite_min_max(values)
        if y_min == y_max:
            pad = max(abs(y_min) * 0.05, 1.0)
            y_min -= pad
            y_max += pad

        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=axis_color)
        for tick in range(1, 5):
            x = plot_left + tick * plot_width / 5.0
            draw.line((x, plot_top, x, plot_bottom), fill=grid_color)
        for tick in range(1, 4):
            y = plot_top + tick * plot_height / 4.0
            draw.line((plot_left, y, plot_right, y), fill=grid_color)

        points: list[tuple[float, float]] = []
        color = colors.get(channel, (40, 40, 40))
        for x_value, y_value in zip(x_values, sampled, strict=True):
            if not np.isfinite(y_value):
                if len(points) > 1:
                    draw.line(points, fill=color, width=1)
                points = []
                continue
            x_pixel = plot_left + (float(x_value) - x_min) / (x_max - x_min) * plot_width
            y_pixel = plot_bottom - (float(y_value) - y_min) / (y_max - y_min) * plot_height
            points.append((x_pixel, y_pixel))
        if len(points) > 1:
            draw.line(points, fill=color, width=1)

        draw.text((12, panel_top + 20), channel, fill=text_color, font=font)
        draw.text((plot_left, panel_bottom - 20), f"{y_min:.2e}", fill=text_color, font=font)
        draw.text((plot_right - 80, panel_bottom - 20), f"{y_max:.2e}", fill=text_color, font=font)
        if show_stats:
            draw.text((plot_right - 170, panel_top + 20), _channel_stats(values), fill=text_color, font=font)

    draw.text((plot_left, height - 34), f"Time [{time_unit}]", fill=text_color, font=font)
    draw.text((plot_left, height - 20), f"{x_min:.3g}", fill=text_color, font=font)
    draw.text((plot_right - 80, height - 20), f"{x_max:.3g}", fill=text_color, font=font)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(plot_path)
    print(f"Wrote time-domain preview to {plot_path}", flush=True)
    return plot_path


def _ordered_channels(channels: Mapping[str, np.ndarray]) -> list[str]:
    known = [channel for channel in ("A", "E", "T", "X", "Y", "Z") if channel in channels]
    extra = sorted(channel for channel in channels if channel not in known)
    return known + extra


def _extrema_preserving_sample_indices(values: np.ndarray, max_points: int) -> np.ndarray:
    size = values.size
    if size <= max_points:
        return np.arange(size)
    if max_points < 4:
        return np.linspace(0, size - 1, num=max_points, dtype=np.int64)

    bucket_count = max(1, (max_points - 2) // 2)
    edges = np.linspace(0, size, num=bucket_count + 1, dtype=np.int64)
    indices = np.empty(2 * bucket_count + 2, dtype=np.int64)
    cursor = 0
    indices[cursor] = 0
    cursor += 1

    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if stop <= start:
            continue

        segment = values[start:stop]
        finite_mask = np.isfinite(segment)
        if np.any(finite_mask):
            finite_offsets = np.flatnonzero(finite_mask)
            finite_values = segment[finite_mask]
            indices[cursor] = start + finite_offsets[int(np.argmin(finite_values))]
            cursor += 1
            indices[cursor] = start + finite_offsets[int(np.argmax(finite_values))]
            cursor += 1
        else:
            indices[cursor] = start
            cursor += 1
            indices[cursor] = stop - 1
            cursor += 1

    indices[cursor] = size - 1
    cursor += 1
    return np.unique(indices[:cursor])


def _time_scale_and_unit(time_s: np.ndarray) -> tuple[float, str]:
    duration = float(np.nanmax(time_s) - np.nanmin(time_s))
    day_s = 86400.0
    year_s = 365.25 * day_s
    hour_s = 3600.0
    if duration >= 2.0 * year_s:
        return year_s, "yr"
    if duration >= 2.0 * day_s:
        return day_s, "day"
    if duration >= 2.0 * hour_s:
        return hour_s, "hr"
    return 1.0, "s"


def _channel_stats(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "no finite samples"
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    max_abs = max(abs(minimum), abs(maximum))
    rms = float(np.linalg.norm(finite) / np.sqrt(finite.size))
    return f"max|h|={max_abs:.3e}\nrms={rms:.3e}"


def _finite_min_max(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    return float(np.min(finite)), float(np.max(finite))
