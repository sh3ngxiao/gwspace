from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


_REQUIRED_FIELDS = (
    "source_id",
    "z",
    "m1_Msun",
    "m2_Msun",
    "DL_Mpc",
    "t_c_yr",
    "Lambda",
    "Beta",
    "iota",
    "psi_rad",
    "var_phi",
    "phi_c",
    "eccentricity",
    "engine",
)


@dataclass(frozen=True)
class MBHBCatalogEntry:
    file_path: str
    file_name: str
    row_number: int
    source_id: str
    z: float
    m1_msun: float
    m2_msun: float
    luminosity_distance_mpc: float
    t_c_yr: float
    lambda_rad: float
    beta_rad: float
    iota_rad: float
    psi_rad: float
    var_phi_rad: float
    phi_c_rad: float
    eccentricity: float
    engine: str
    metadata: dict[str, Any]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "row_number": self.row_number,
            "source_id": self.source_id,
            "z": self.z,
            "m1_Msun": self.m1_msun,
            "m2_Msun": self.m2_msun,
            "DL_Mpc": self.luminosity_distance_mpc,
            "t_c_yr": self.t_c_yr,
            "Lambda": self.lambda_rad,
            "Beta": self.beta_rad,
            "iota": self.iota_rad,
            "psi_rad": self.psi_rad,
            "var_phi": self.var_phi_rad,
            "phi_c": self.phi_c_rad,
            "eccentricity": self.eccentricity,
            "engine": self.engine,
            **self.metadata,
        }

    def to_waveform_parameters(self, *, seconds_per_year: float) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "mass1": self.m1_msun,
            "mass2": self.m2_msun,
            "DL": self.luminosity_distance_mpc,
            "Lambda": self.lambda_rad,
            "Beta": self.beta_rad,
            "iota": self.iota_rad,
            "psi": self.psi_rad,
            "var_phi": self.var_phi_rad,
            "phi_c": self.phi_c_rad,
            "tc": self.t_c_yr * seconds_per_year,
            "eccentricity": self.eccentricity,
            "engine": self.engine,
        }
        if self.metadata.get("f_gw_start_hz") not in (None, ""):
            parameters["response_f_min_hz"] = float(self.metadata["f_gw_start_hz"])
        return parameters


def _float_field(path: Path, row_number: int, row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except ValueError as exc:
        raise ValueError(f"MBHB catalog '{path}' row {row_number} has non-numeric field '{field}'.") from exc


def _parse_catalog_row(path: Path, row_number: int, row: dict[str, str]) -> MBHBCatalogEntry:
    missing = tuple(field for field in _REQUIRED_FIELDS if field not in row)
    if missing:
        raise ValueError(f"MBHB catalog '{path}' is missing required columns: {missing}.")

    m1_msun = _float_field(path, row_number, row, "m1_Msun")
    m2_msun = _float_field(path, row_number, row, "m2_Msun")
    distance_mpc = _float_field(path, row_number, row, "DL_Mpc")
    t_c_yr = _float_field(path, row_number, row, "t_c_yr")
    eccentricity = _float_field(path, row_number, row, "eccentricity")

    if m1_msun <= 0.0 or m2_msun <= 0.0:
        raise ValueError(f"MBHB catalog '{path}' row {row_number} must have positive component masses.")
    if m2_msun > m1_msun:
        raise ValueError(
            f"MBHB catalog '{path}' row {row_number} requires m1_Msun >= m2_Msun, got {m1_msun} < {m2_msun}."
        )
    if distance_mpc <= 0.0:
        raise ValueError(f"MBHB catalog '{path}' row {row_number} must have positive DL_Mpc.")
    if t_c_yr <= 0.0:
        raise ValueError(f"MBHB catalog '{path}' row {row_number} must have positive t_c_yr.")
    if eccentricity < 0.0:
        raise ValueError(f"MBHB catalog '{path}' row {row_number} has negative eccentricity.")

    metadata = {
        key: value
        for key, value in row.items()
        if key not in _REQUIRED_FIELDS and value is not None and value != ""
    }

    return MBHBCatalogEntry(
        file_path=str(path.resolve()),
        file_name=path.name,
        row_number=row_number,
        source_id=str(row["source_id"]),
        z=_float_field(path, row_number, row, "z"),
        m1_msun=m1_msun,
        m2_msun=m2_msun,
        luminosity_distance_mpc=distance_mpc,
        t_c_yr=t_c_yr,
        lambda_rad=_float_field(path, row_number, row, "Lambda"),
        beta_rad=_float_field(path, row_number, row, "Beta"),
        iota_rad=_float_field(path, row_number, row, "iota"),
        psi_rad=_float_field(path, row_number, row, "psi_rad"),
        var_phi_rad=_float_field(path, row_number, row, "var_phi"),
        phi_c_rad=_float_field(path, row_number, row, "phi_c"),
        eccentricity=eccentricity,
        engine=str(row["engine"]),
        metadata=metadata,
    )


def iter_mbhb_catalog(path: str | Path) -> Iterator[MBHBCatalogEntry]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"MBHB catalog file '{catalog_path}' does not exist.")

    with catalog_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"MBHB catalog '{catalog_path}' is empty or missing a CSV header.")
        for row_number, row in enumerate(reader, start=1):
            yield _parse_catalog_row(catalog_path, row_number, row)


def load_mbhb_catalog(path: str | Path, *, limit: int | None = None) -> list[MBHBCatalogEntry]:
    entries: list[MBHBCatalogEntry] = []
    for entry in iter_mbhb_catalog(path):
        entries.append(entry)
        if limit is not None and len(entries) >= limit:
            break
    return entries
