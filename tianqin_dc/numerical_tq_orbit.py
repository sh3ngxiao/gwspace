from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from gwspace.Orbit import detectors


ORBIT_PATH_ENV = "TQ_NUMERICAL_ORBIT_PATH"
ORBIT_TIME_OFFSET_ENV = "TQ_NUMERICAL_ORBIT_TIME_OFFSET_S"
ORBIT_OUT_OF_RANGE_ENV = "TQ_NUMERICAL_ORBIT_OUT_OF_RANGE"
ORBIT_L_T_ENV = "TQ_NUMERICAL_ORBIT_L_T_S"

DEFAULT_ORBIT_PATH = Path(__file__).resolve().parent / "satellite_positions_ssb_tq_1s_from_oem.npy"


@lru_cache(maxsize=8)
def _load_orbit_table(path: str) -> np.ndarray:
    orbit_path = Path(path).expanduser().resolve()
    if not orbit_path.exists():
        raise FileNotFoundError(f"TianQin numerical orbit file does not exist: {orbit_path}")

    table = np.load(orbit_path, mmap_mode="r")
    if table.ndim != 2 or table.shape[1] != 10:
        raise ValueError(
            "TianQin numerical orbit file must have shape (N, 10): "
            "time_s plus x/y/z for three spacecraft."
        )
    if table.shape[0] < 2:
        raise ValueError("TianQin numerical orbit file must contain at least two time samples.")
    return table


def _default_orbit_path() -> str:
    return os.environ.get(ORBIT_PATH_ENV, str(DEFAULT_ORBIT_PATH))


def _time_offset_s() -> float:
    return float(os.environ.get(ORBIT_TIME_OFFSET_ENV, "0.0"))


def _out_of_range_mode() -> str:
    mode = os.environ.get(ORBIT_OUT_OF_RANGE_ENV, "raise").strip().lower()
    if mode not in {"raise", "clip", "wrap"}:
        raise ValueError(
            f"{ORBIT_OUT_OF_RANGE_ENV} must be one of 'raise', 'clip', or 'wrap', got {mode!r}."
        )
    return mode


@lru_cache(maxsize=8)
def _orbit_time_bounds(path: str) -> tuple[float, float, float]:
    table = _load_orbit_table(path)
    start = float(table[0, 0])
    second = float(table[1, 0])
    stop = float(table[-1, 0])
    step = second - start
    if step <= 0.0:
        raise ValueError("TianQin numerical orbit time column must be strictly increasing.")
    expected_stop = start + step * (table.shape[0] - 1)
    if not np.isclose(stop, expected_stop, rtol=0.0, atol=max(1.0e-9, abs(step) * 1.0e-9)):
        raise ValueError("TianQin numerical orbit time column must be uniformly sampled.")
    return start, stop, step


@lru_cache(maxsize=8)
def _mean_arm_length_s(path: str) -> float:
    override = os.environ.get(ORBIT_L_T_ENV)
    if override is not None and override.strip():
        value = float(override)
        if value <= 0.0:
            raise ValueError(f"{ORBIT_L_T_ENV} must be positive.")
        return value

    table = _load_orbit_table(path)
    sample_count = min(table.shape[0], 4096)
    indices = np.linspace(0, table.shape[0] - 1, sample_count, dtype=np.int64)
    positions = np.asarray(table[indices, 1:10], dtype=np.float64).reshape(sample_count, 3, 3)
    p1 = positions[:, 0, :]
    p2 = positions[:, 1, :]
    p3 = positions[:, 2, :]
    lengths = np.concatenate(
        [
            np.linalg.norm(p2 - p1, axis=1),
            np.linalg.norm(p3 - p2, axis=1),
            np.linalg.norm(p1 - p3, axis=1),
        ]
    )
    mean_length = float(np.mean(lengths))
    if mean_length <= 0.0:
        raise ValueError("TianQin numerical orbit mean arm length must be positive.")
    return mean_length


def _prepare_times(time: np.ndarray, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start, stop, step = _orbit_time_bounds(path)
    adjusted = np.asarray(time, dtype=np.float64) + _time_offset_s()
    mode = _out_of_range_mode()

    if mode == "wrap":
        period = stop - start
        if period <= 0.0:
            raise ValueError("Cannot wrap TianQin numerical orbit with non-positive time span.")
        adjusted = start + np.mod(adjusted - start, period)
    elif mode == "clip":
        adjusted = np.clip(adjusted, start, stop)
    else:
        tolerance = max(1.0e-9, abs(step) * 1.0e-9)
        if np.any((adjusted < start - tolerance) | (adjusted > stop + tolerance)):
            min_time = float(np.min(adjusted))
            max_time = float(np.max(adjusted))
            raise ValueError(
                "TianQin numerical orbit time request is outside the file coverage: "
                f"requested [{min_time}, {max_time}] s, available [{start}, {stop}] s. "
                f"Set {ORBIT_OUT_OF_RANGE_ENV}=clip or wrap only if that is intended."
            )
        adjusted = np.clip(adjusted, start, stop)

    index_float = (adjusted - start) / step
    lower = np.floor(index_float).astype(np.int64, copy=False)
    lower = np.clip(lower, 0, _load_orbit_table(path).shape[0] - 2)
    fraction = index_float - lower
    return lower, lower + 1, fraction


def numerical_orbit_time_window_s(orbit_path: str | None = None) -> tuple[float, float]:
    """Return the valid detector-time request window for the numerical orbit."""

    path = str(Path(orbit_path or _default_orbit_path()).expanduser().resolve())
    start, stop, _step = _orbit_time_bounds(path)
    offset = _time_offset_s()
    return start - offset, stop - offset


def _interpolate_column(
    table: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    fraction: np.ndarray,
    column: int,
) -> np.ndarray:
    lower_values = np.asarray(table[lower, column], dtype=np.float64)
    upper_values = np.asarray(table[upper, column], dtype=np.float64)
    return lower_values + fraction * (upper_values - lower_values)


class NumericalTianQinOrbit:
    """TianQin orbit backed by sampled SSB spacecraft positions.

    The GWspace response code expects coordinates in light seconds. The OEM-derived
    orbit file used here is already in that unit.
    """

    def __init__(self, time: np.ndarray, *args: object, orbit_path: str | None = None, **kwargs: object) -> None:
        del args, kwargs
        path = str(Path(orbit_path or _default_orbit_path()).expanduser().resolve())
        table = _load_orbit_table(path)
        lower, upper, fraction = _prepare_times(np.asarray(time, dtype=np.float64), path)

        spacecraft: list[np.ndarray] = []
        for spacecraft_index in range(3):
            base_column = 1 + spacecraft_index * 3
            spacecraft.append(
                np.array(
                    [
                        _interpolate_column(table, lower, upper, fraction, base_column),
                        _interpolate_column(table, lower, upper, fraction, base_column + 1),
                        _interpolate_column(table, lower, upper, fraction, base_column + 2),
                    ],
                    dtype=np.float64,
                )
            )

        self.orbits = tuple(spacecraft)
        self._p_0 = (self.orbits[0] + self.orbits[1] + self.orbits[2]) / 3.0
        self._L_T = _mean_arm_length_s(path)

    @property
    def L_T(self) -> float:
        return self._L_T

    @property
    def p_0(self) -> np.ndarray:
        return self._p_0

    def uni_vec_ij(self, i: int, j: int) -> np.ndarray:
        return (self.orbits[j - 1] - self.orbits[i - 1]) / self.L_T


def register_numerical_tianqin_orbit(detector_names: Iterable[str] = ("TQ", "TianQin")) -> None:
    """Register the numerical TianQin orbit for the current Python process."""

    # Validate eagerly so a missing or malformed path fails before workers start.
    path = str(Path(_default_orbit_path()).expanduser().resolve())
    _load_orbit_table(path)
    _orbit_time_bounds(path)
    _mean_arm_length_s(path)
    for detector_name in detector_names:
        detectors[detector_name] = NumericalTianQinOrbit
