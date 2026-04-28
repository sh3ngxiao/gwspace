from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.constants import c
from scipy.integrate import cumulative_trapezoid


_OMEGA_LAMBDA = 0.689
_OMEGA_MATTER = 0.311
_OMEGA_RADIATION_AND_NU = 2.47e-5 * pow(0.677, -2) + 0.1 * pow(93.12, -1) * pow(0.677, -2)
_H0_KM_S_MPC = 67.7
_REQUIRED_FIELDS = ("z", "m1_Msun", "m2_Msun", "t_c_yr", "psi_rad")


@dataclass(frozen=True)
class BBHCatalogEntry:
    file_path: str
    file_name: str
    row_number: int
    z: float
    m1_msun: float
    m2_msun: float
    t_c_yr: float
    psi_rad: float

    @property
    def total_mass_msun(self) -> float:
        return self.m1_msun + self.m2_msun

    @property
    def mass_ratio(self) -> float:
        return self.m2_msun / self.m1_msun

    @property
    def chirp_mass_msun(self) -> float:
        total_mass = self.total_mass_msun
        return float((self.m1_msun * self.m2_msun) ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "row_number": self.row_number,
            "z": self.z,
            "m1_Msun": self.m1_msun,
            "m2_Msun": self.m2_msun,
            "t_c_yr": self.t_c_yr,
            "psi_rad": self.psi_rad,
            "total_mass_Msun": self.total_mass_msun,
            "chirp_mass_Msun": self.chirp_mass_msun,
            "q": self.mass_ratio,
        }


def redshift_to_luminosity_distance_mpc(redshift: float | np.ndarray) -> float | np.ndarray:
    values = np.asarray(redshift, dtype=np.float64)
    scalar_input = values.ndim == 0
    if scalar_input:
        values = values.reshape(1)

    if np.any(values < 0.0):
        raise ValueError("Redshift must be non-negative.")
    if values.size == 0:
        return np.array([], dtype=np.float64)

    z_max = float(np.max(values))
    if z_max == 0.0:
        distances = np.zeros_like(values, dtype=np.float64)
    else:
        grid_size = max(2, int(np.ceil(z_max * 5000.0)) + 1)
        z_grid = np.linspace(0.0, z_max, grid_size, dtype=np.float64)
        expansion = np.sqrt(
            _OMEGA_LAMBDA
            + _OMEGA_MATTER * (1.0 + z_grid) ** 3
            + _OMEGA_RADIATION_AND_NU * (1.0 + z_grid) ** 4
        )
        comoving_integral = cumulative_trapezoid(1.0 / expansion, z_grid, initial=0.0)
        luminosity_distance_mpc = comoving_integral * (1.0 + z_grid) * c / _H0_KM_S_MPC / 1e3
        distances = np.interp(values, z_grid, luminosity_distance_mpc)

    if scalar_input:
        return float(distances[0])
    return distances


def _parse_catalog_row(path: Path, row_number: int, row: dict[str, str]) -> BBHCatalogEntry:
    missing = tuple(field for field in _REQUIRED_FIELDS if field not in row)
    if missing:
        raise ValueError(f"BBH catalog '{path}' is missing required columns: {missing}.")

    try:
        z = float(row["z"])
        m1_msun = float(row["m1_Msun"])
        m2_msun = float(row["m2_Msun"])
        t_c_yr = float(row["t_c_yr"])
        psi_rad = float(row["psi_rad"])
    except ValueError as exc:
        raise ValueError(f"BBH catalog '{path}' row {row_number} contains non-numeric values.") from exc

    if z < 0.0:
        raise ValueError(f"BBH catalog '{path}' row {row_number} has negative redshift {z}.")
    if m1_msun <= 0.0 or m2_msun <= 0.0:
        raise ValueError(f"BBH catalog '{path}' row {row_number} must have positive component masses.")
    if m2_msun > m1_msun:
        raise ValueError(
            f"BBH catalog '{path}' row {row_number} requires m1_Msun >= m2_Msun, got {m1_msun} < {m2_msun}."
        )

    return BBHCatalogEntry(
        file_path=str(path.resolve()),
        file_name=path.name,
        row_number=row_number,
        z=z,
        m1_msun=m1_msun,
        m2_msun=m2_msun,
        t_c_yr=t_c_yr,
        psi_rad=psi_rad,
    )


def iter_bbh_catalog(path: str | Path) -> Iterator[BBHCatalogEntry]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"BBH catalog file '{catalog_path}' does not exist.")

    with catalog_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"BBH catalog '{catalog_path}' is empty or missing a CSV header.")
        for row_number, row in enumerate(reader, start=1):
            yield _parse_catalog_row(catalog_path, row_number, row)


def load_bbh_catalog(path: str | Path, *, limit: int | None = None) -> list[BBHCatalogEntry]:
    entries: list[BBHCatalogEntry] = []
    for entry in iter_bbh_catalog(path):
        entries.append(entry)
        if limit is not None and len(entries) >= limit:
            break
    return entries
